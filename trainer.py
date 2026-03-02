


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainer.py

CityscapesTrainer (mask-conditioned) that works correctly with:
- DDP (DistributedDataParallel)
- EMA (tracks raw/non-DDP weights)
- AMP + GradScaler
- Gradient accumulation
- Condition mask stays in pixel space; ONLY the model downsamples internally.

Assumptions:
- train_loader yields (mask_onehot_img, img)
    mask_onehot_img: [B,19,H,W] float/bool one-hot (0/1)
    img:             [B,3,H,W] in [-1,1]
- model is a wrapper like MaskCondDiTWithVAE:
    - encode_image_to_latent(img) -> z_scaled [B,4,H/8,W/8]
    - forward(z_noisy_scaled, t, mask_onehot_img) -> pred (v or eps etc. depending on gen_model)
    - decode_latent_to_image(z_scaled) -> img [-1,1]
- gen_model is a CLASS (FlowMatching or GaussianDiffusion), instantiated once:
    self.gen_model = gen_model(cfg=config, device=self.device)
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, Any, Optional, Union

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from ema_pytorch import EMA

from diffusion.Rectified_flow_matching import FlowMatchingConfig, FlowMatching
from diffusion.guassian_diffusion import DiffusionConfig, GaussianDiffusion


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_rank0() -> bool:
    return _rank() == 0


class CityscapesTrainer:
    def __init__(
        self,
        model: nn.Module,  # raw model (not DDP)
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader,
        local_rank: int,
        global_rank: int,
        config: Union[FlowMatchingConfig, DiffusionConfig],
        gen_model: Union[FlowMatching, GaussianDiffusion],  # CLASS
        args,
        exp_log_dir: str,
        palette_19: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ):
        self.local_rank = int(local_rank)
        self.global_rank = int(global_rank)

        if device is None:
            self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Distributed info
        self.is_distributed = _is_dist()
        self.world_size = _world_size()

        # --- IMPORTANT: keep raw model separate; DDP wraps a handle ---
        self.raw_model: nn.Module = model.to(self.device)

        if self.is_distributed:
            # Ensure correct device placement for DDP
            torch.cuda.set_device(self.local_rank)
            self.model: nn.Module = DDP(self.raw_model, device_ids=[self.local_rank], output_device=self.local_rank)
            if _is_rank0():
                print(f"[DDP] Wrapped model. world_size={self.world_size}")
        else:
            self.model = self.raw_model
            if _is_rank0():
                print("[Single GPU] No DDP wrapping.")

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_iter = iter(train_loader)
        self.val_loader = val_loader
        self.args = args
        self.exp_log_dir = exp_log_dir
        self.palette_19 = palette_19

        os.makedirs(self.exp_log_dir, exist_ok=True)

        # W&B
        self.use_wandb = wandb.run is not None
        if _is_rank0():
            print("W&B logging enabled" if self.use_wandb else "W&B not initialized - console logging only")

        # Instantiate generative helper (FlowMatching or GaussianDiffusion)
        self.gen_model = gen_model(cfg=config, device=self.device)

        # EMA MUST track RAW model weights (not DDP wrapper)
        self.ema_model: Optional[EMA] = None
        if getattr(self.args, "use_ema", False):
            # EMA expects a module; using raw_model avoids DDP wrapper params
            self.ema_model = EMA(
                self.raw_model,
                beta=float(getattr(self.args, "ema_beta", 0.9999)),
                update_after_step=int(getattr(self.args, "ema_update_after_step", 0)),
                update_every=int(getattr(self.args, "ema_update_every", 1)),
            ).to(self.device)
            if _is_rank0():
                print("[EMA] Enabled. Tracking raw_model parameters.")

        # AMP + GradScaler
        self.use_amp = (self.device.type == "cuda") and bool(getattr(self.args, "use_amp", True))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Grad accumulation
        self.grad_accum_steps = max(1, int(getattr(self.args, "grad_accum_steps", 1)))

        # State
        self.current_step = 0
        self.start_step = 0

        # Running window metrics
        self.running_loss = 0.0
        self.running_grad_norm = 0.0
        self.running_optim_steps = 0
        self.window_start_time = time.time()

        # Zero grads once
        self.optimizer.zero_grad(set_to_none=True)

    # ----------------------------
    # Core training
    # ----------------------------
    def training_step(self) -> Dict[str, float]:
        self.model.train()

        # 1) Data
        try:
            mask_onehot_img, img = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            mask_onehot_img, img = next(self.train_iter)

        mask_onehot_img = mask_onehot_img.to(self.device, non_blocking=True)
        img = img.to(self.device, non_blocking=True)

        # 2) Encode with VAE (no grad through VAE)
        with torch.no_grad():
            # ALWAYS use raw_model for encode/decode access; it also works under DDP training
            z = self.raw_model.encode_image_to_latent(img)  # scaled latents [B,4,H/8,W/8]

        # 3) Forward + loss under autocast (forward only)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            loss = self.gen_model.loss(
                model=self.model,          # use DDP wrapper here so grads sync properly
                x=z,
                cond=mask_onehot_img,      # pixel mask; ONLY model downsamples internally
            )
            loss_to_backward = loss / float(self.grad_accum_steps)

        # 4) Backward (scaled)
        self.scaler.scale(loss_to_backward).backward()

        # 5) Optim step only every grad_accum_steps
        do_step = ((self.current_step + 1) % self.grad_accum_steps == 0)

        grad_norm_val = 0.0
        if do_step:
            self.scaler.unscale_(self.optimizer)

            # Clip grads
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(getattr(self.args, "max_grad_norm", 1.0)),
            )
            grad_norm_val = float(grad_norm.item())

            # Step optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # EMA update on real optimizer steps
            if self.ema_model is not None:
                self.ema_model.update()

            self.running_optim_steps += 1
            self.running_grad_norm += grad_norm_val

        # 6) Metrics
        loss_val = float(loss.detach().item())
        self.running_loss += loss_val
        self.current_step += 1

        return {"loss": loss_val, "did_step": float(do_step), "grad_norm": grad_norm_val}

    # ----------------------------
    # Logging
    # ----------------------------
    def log_metrics(self, step: int):
        if step % int(self.args.log_every) != 0:
            return
        if self.is_distributed and not _is_rank0():
            return

        avg_loss = self.running_loss / float(self.args.log_every)
        avg_gn = self.running_grad_norm / max(1, self.running_optim_steps)
        lr = float(self.optimizer.param_groups[0]["lr"])

        print(
            f"Step {step:08d}/{self.args.train_steps} ({100*step/self.args.train_steps:.1f}%) | "
            f"Loss {avg_loss:.6f} | GradNorm {avg_gn:.4f} | LR {lr:.2e} | "
            f"accum {self.grad_accum_steps} (opt_steps {self.running_optim_steps})"
        )

        if self.use_wandb:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/grad_norm": avg_gn,
                    "train/lr": lr,
                    "train/step": step,
                    "train/grad_accum_steps": self.grad_accum_steps,
                    "train/opt_steps_in_window": self.running_optim_steps,
                },
                step=step,
            )

        self.running_loss = 0.0
        self.running_grad_norm = 0.0
        self.running_optim_steps = 0
        self.window_start_time = time.time()

    # ----------------------------
    # Visualization
    # ----------------------------
    @torch.no_grad()
    def visualize(self, step: int):
        if not bool(getattr(self.args, "enable_visualization", False)):
            return
        if not (
            (bool(getattr(self.args, "visualize_first_step", False)) and step == 1)
            or (step % int(getattr(self.args, "visualize_every", 1000)) == 0)
        ):
            return
        if self.is_distributed and not _is_rank0():
            return
        if self.val_loader is None:
            return

        # Choose sampling model (EMA preferred)
        if self.ema_model is not None:
            sampling_model = self.ema_model.ema_model
        else:
            sampling_model = self.raw_model

        sampling_model.eval()

        # Fetch a val batch
        if hasattr(self.val_loader, "__next__"):
            mask_onehot_val, img_val = next(self.val_loader)
        else:
            val_iter = iter(self.val_loader)
            mask_onehot_val, img_val = next(val_iter)

        mask_onehot_val = mask_onehot_val.to(self.device, non_blocking=True)
        img_val = img_val.to(self.device, non_blocking=True)

        # Encode RGB to latents for comparison
        z_gt = self.raw_model.encode_image_to_latent(img_val)
        Hl, Wl = int(z_gt.shape[2]), int(z_gt.shape[3])
        B = int(z_gt.shape[0])

        # Sample latents
        z_gen = self.gen_model.sample(
            model=sampling_model,  # NOT DDP wrapper
            shape=(B, 4, Hl, Wl),
            cond=mask_onehot_val,
            device=self.device,
        )

        # Decode
        img_pred = self.raw_model.decode_latent_to_image(z_gen)

        # Log/Save
        save_images = bool(getattr(self.args, "save_visualization_images", True))
        num_samples = min(B, int(getattr(self.args, "visualize_num", min(B, 4))))

        if save_images:
            viz_dir = os.path.join(self.args.log_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            step_dir = os.path.join(viz_dir, f"step_{step:06d}")
            os.makedirs(step_dir, exist_ok=True)

        samples_to_log = {}
        for i in range(num_samples):
            cond_pil = self._mask_to_pil(mask_onehot_val[i])
            gen_pil = self._tensor_to_pil(img_pred[i])
            gt_pil = self._tensor_to_pil(img_val[i])

            if save_images:
                cond_pil.save(os.path.join(step_dir, f"sample_{i}_condition.png"))
                gen_pil.save(os.path.join(step_dir, f"sample_{i}_generated.png"))
                gt_pil.save(os.path.join(step_dir, f"sample_{i}_ground_truth.png"))

            if self.use_wandb:
                samples_to_log[f"viz/{i}"] = [
                    wandb.Image(cond_pil, caption="condition"),
                    wandb.Image(gen_pil, caption="generated"),
                    wandb.Image(gt_pil, caption="ground truth"),
                ]

        if self.use_wandb and samples_to_log:
            wandb.log(samples_to_log, step=step)

        # Optional debug
        print(f"[viz] z_gen vs z_gt diff norm: {torch.norm(z_gen - z_gt).item():.4f}")
        print(f"[viz] z_gen mean/std: {float(z_gen.mean()):.4f}/{float(z_gen.std()):.4f}")
        print(f"[viz] z_gt  mean/std: {float(z_gt.mean()):.4f}/{float(z_gt.std()):.4f}")

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        # tensor: [3,H,W] in [-1,1]
        img_np = tensor.permute(1, 2, 0).detach().float().cpu().numpy()
        img_np = np.clip((img_np + 1.0) / 2.0, 0.0, 1.0)
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def _mask_to_pil(self, mask_tensor: torch.Tensor) -> Image.Image:
        # mask_tensor: [19,H,W] one-hot
        mask_indices = torch.argmax(mask_tensor, dim=0).detach().cpu().numpy().astype(np.uint8)
        if self.palette_19 is not None:
            colored = self.palette_19[mask_indices]
            return Image.fromarray(colored)
        mask_vis = (mask_indices * 255 // 19).astype(np.uint8)
        return Image.fromarray(mask_vis)

    # ----------------------------
    # Checkpointing
    # ----------------------------
    def save_checkpoint(self, step: int):
        if self.is_distributed and not _is_rank0():
            return
        if step % int(self.args.ckpt_every) != 0:
            return

        checkpoint_dir = os.path.join(self.args.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pt")

        # Save RAW model weights (portable; no DDP wrapper keys)
        ckpt = {
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "args": vars(self.args),
            "global_rank": self.global_rank,
            "world_size": self.world_size,
        }
        if self.ema_model is not None:
            ckpt["ema_state_dict"] = self.ema_model.state_dict()
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(ckpt, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

        self._cleanup_old_checkpoints(current_step=step)

    def _cleanup_old_checkpoints(self, current_step: int):
        keep_last_k = int(getattr(self.args, "keep_last_k", 10))
        if keep_last_k <= 0:
            return

        import glob
        checkpoint_dir = os.path.join(self.args.checkpoint_dir)
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))

        checkpoints = []
        for f in ckpt_files:
            m = re.search(r"checkpoint_step_(\d+)\.pt", os.path.basename(f))
            if m:
                checkpoints.append((int(m.group(1)), f))
        checkpoints.sort(key=lambda x: x[0], reverse=True)

        for i, (_, f) in enumerate(checkpoints):
            if i >= keep_last_k:
                try:
                    os.remove(f)
                except OSError:
                    pass

    def save_final_checkpoint(self):
        if self.is_distributed and not _is_rank0():
            return
        checkpoint_dir = os.path.join(self.args.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{self.current_step:06d}_final.pt")

        ckpt = {
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.current_step,
            "args": vars(self.args),
            "final": True,
        }
        if self.ema_model is not None:
            ckpt["ema_state_dict"] = self.ema_model.state_dict()
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(ckpt, ckpt_path)
        print(f"🏁 Saved final checkpoint: {ckpt_path}")

    # ----------------------------
    # Train loop
    # ----------------------------
    def set_start_step(self, start_step: int):
        self.start_step = int(start_step)
        self.current_step = int(start_step)
        if _is_rank0():
            print(f"📍 Resuming from step {self.start_step}")

    def train(self):
        if _is_rank0():
            print(f"Starting training for {self.args.train_steps} steps...")
            print(f"Logging every {self.args.log_every} steps")
            print(f"Grad accumulation steps: {self.grad_accum_steps}")
            print("Visualization enabled" if getattr(self.args, "enable_visualization", False) else "Visualization disabled")

        # IMPORTANT: don't consume a batch here (would shift the iterator)
        # If you want to test dataloader, do it with a separate iterator:
        if _is_rank0():
            try:
                _ = next(iter(self.train_loader))
                print("✅ DataLoader sanity check passed.")
            except Exception as e:
                print(f"❌ DataLoader sanity check failed: {e}")
                raise

        while self.current_step < int(self.args.train_steps):
            self.training_step()
            self.log_metrics(self.current_step)
            self.visualize(self.current_step)
            self.save_checkpoint(self.current_step)

        self.save_final_checkpoint()
        if _is_rank0():
            print("Training completed!")

    def print_model_info(self):
        """
        Print detailed model information including parameters and GPU memory requirements.
        """
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        
        # Count parameters
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        # Analyze different components
        components = {
            "Full Model": self.model,
            "DiT (trainable)": getattr(self.model, 'dit', None),
            "VAE (frozen)": getattr(self.model, 'vae', None),
        }
        
        print(f"{'Component':<20} {'Total Params':<12} {'Trainable':<12} {'Frozen':<12} {'Status':<10}")
        print("-" * 70)
        
        for name, component in components.items():
            if component is None:
                continue
                
            comp_total = sum(p.numel() for p in component.parameters())
            comp_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
            comp_frozen = comp_total - comp_trainable
            
            status = "TRAIN" if comp_trainable > 0 else "FROZEN"
            if name == "Full Model":
                total_params = comp_total
                trainable_params = comp_trainable
                frozen_params = comp_frozen
            
            print(f"{name:<20} {comp_total/1e6:>8.2f}M {comp_trainable/1e6:>8.2f}M {comp_frozen/1e6:>8.2f}M {status:<10}")
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_params/1e6:>8.2f}M {trainable_params/1e6:>8.2f}M {frozen_params/1e6:>8.2f}M")
        
        # Memory calculations (rough estimates)
        print(f"\nGPU MEMORY REQUIREMENTS (Estimates)")
        print("-" * 40)
        
        # Model weights (fp16 for inference, fp32 for training)
        model_memory_fp32 = total_params * 4 / (1024**3)  # 4 bytes per fp32 parameter
        model_memory_fp16 = total_params * 2 / (1024**3)  # 2 bytes per fp16 parameter
        
        # Gradients (only for trainable params, fp32)
        grad_memory = trainable_params * 4 / (1024**3)
        
        # Optimizer states (Adam: 2 states per param)
        optimizer_memory = trainable_params * 2 * 4 / (1024**3)
        
        # EMA model (if used)
        ema_memory = 0
        if self.args.use_ema:
            ema_memory = total_params * 4 / (1024**3)
        
        print(f"Model weights (FP32):     {model_memory_fp32:.2f} GB")
        print(f"Model weights (FP16):     {model_memory_fp16:.2f} GB")
        print(f"Gradients:                {grad_memory:.2f} GB")
        print(f"Optimizer states:         {optimizer_memory:.2f} GB")
        if ema_memory > 0:
            print(f"EMA model:                {ema_memory:.2f} GB")
        
        # Total training memory
        total_training_memory = model_memory_fp32 + grad_memory + optimizer_memory + ema_memory
        total_inference_memory = model_memory_fp16
        
        print(f"{'─' * 40}")
        print(f"Est. training memory:     {total_training_memory:.2f} GB")
        print(f"Est. inference memory:    {total_inference_memory:.2f} GB")
        
        # Additional info
        print(f"\nTRAINING CONFIGURATION")
        print("-" * 25)
        print(f"Batch size:               {self.args.bs}")
        print(f"Mixed precision:          {self.device.type == 'cuda'}")
        print(f"EMA enabled:              {self.args.use_ema}")
        print(f"Gradient checkpointing:   {getattr(self.args, 'gradient_checkpointing', False)}")
        
        # Activation memory (very rough estimate)
        if hasattr(self.args, 'bs'):
            # Rough estimate: batch_size * channels * height * width * layers * bytes_per_element
            latent_h = getattr(self.args, 'image_h', 256) // 8
            latent_w = getattr(self.args, 'image_w', 512) // 8
            activation_memory = self.args.bs * 4 * latent_h * latent_w * 8 * 4 / (1024**3)  # Very rough
            print(f"Est. activations:         ~{activation_memory:.2f} GB")
            
            total_with_activations = total_training_memory + activation_memory
            print(f"{'─' * 40}")
            print(f"Est. total GPU usage:     ~{total_with_activations:.2f} GB")
        
        # GPU info if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\nAvailable GPU memory:     {gpu_memory:.2f} GB")
            
            if hasattr(self.args, 'bs'):
                memory_usage_percent = (total_with_activations / gpu_memory) * 100
                print(f"Estimated usage:          {memory_usage_percent:.1f}%")
                
                if memory_usage_percent > 90:
                    print("⚠️  WARNING: High memory usage! Consider reducing batch size.")
                elif memory_usage_percent > 100:
                    print("❌ ERROR: Estimated memory exceeds GPU capacity!")
        
        print("="*60 + "\n")    