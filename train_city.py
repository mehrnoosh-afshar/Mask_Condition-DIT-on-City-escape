#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cityscapes_latent.py

Cityscapes mask-conditioned DiT in VAE latent space.

Key points:
- Dataset yields (mask_onehot_img, img) where:
    mask_onehot_img: [B, 19, H, W] float (Cityscapes trainId one-hot, ignore pixels can be all-zeros)
    img:            [B,  3, H, W] float in [-1, 1]
- VAE encodes img -> z (scaled by latent_scale=0.18215) with shape [B,4,H/8,W/8]
- Mask is downsampled to latent resolution and concatenated with latents BEFORE patchify (your conditioning method)
- Diffusion process unchanged:
    z_t = t * z + (1 - t) * eps
  with the same pred/loss conversions you used before.
- IMPORTANT STABILITY:
  VAE encode/decode forced to FP32 (outside autocast).
- IMPORTANT DDP/Accelerate:
  EMA is created AFTER accelerator.prepare().

Expected model signature:
    TinyDiTLatent.forward(z_noisy_scaled, t, mask_onehot_lat) -> model_out
"""

import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn.functional as F
import wandb

from torch.utils.data import DataLoader, random_split
from PIL import Image
from ema_pytorch import EMA
from diffusers.models import AutoencoderKL

from models.vae_dit import TinyDiTLatent
from dataset.datasets import CityscapesDataset


LOSS_SPACES = {"x", "v"}
PRED_SPACES = {"x", "v"}


def clear_gpu_memory():
    """Clear GPU memory cache to free up space"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cycle(dl: DataLoader):
    while True:
        for data in dl:
            yield data


def sample_t(shape, device, sampling_type="uniform", mu=0.0, sigma=1.0):
    if sampling_type == "uniform":
        return torch.rand(shape, device=device)
    if sampling_type == "logit_normal":
        return torch.sigmoid(torch.randn(shape, device=device) * sigma + mu)
    raise ValueError(f"Invalid sampling type {sampling_type}")


@torch.no_grad()
def encode_to_latents_fp32(vae: AutoencoderKL, img: torch.Tensor, latent_scale: float) -> torch.Tensor:
    """
    img: [B,3,H,W] float in [-1,1]
    returns: z_scaled [B,4,H/8,W/8] float32
    """
    # Force fp32 stability, use appropriate device
    device_type = "cuda" if img.device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        z = vae.encode(img.float()).latent_dist.sample()
        z = z * float(latent_scale)
    return z


@torch.no_grad()
def decode_from_latents_fp32(vae: AutoencoderKL, z_scaled: torch.Tensor, latent_scale: float) -> torch.Tensor:
    """
    z_scaled: [B,4,Hl,Wl] scaled, float
    returns:  img [B,3,H,W] float32 in [-1,1]
    """
    device_type = "cuda" if z_scaled.device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        z = z_scaled.float() / float(latent_scale)
        x = vae.decode(z).sample
    return x


def downsample_mask_to_latent(mask_onehot_img: torch.Tensor, latent_hw: tuple[int, int]) -> torch.Tensor:
    """
    mask_onehot_img: [B,19,H,W] float
    returns:         [B,19,Hl,Wl] float (nearest keeps discrete labels)
    """
    return F.interpolate(mask_onehot_img, size=latent_hw, mode="nearest")


@torch.no_grad()
def euler_sample_latents(
    model: torch.nn.Module,
    mask_onehot_img: torch.Tensor,   # [B,19,H,W]
    latent_hw: tuple[int, int],
    steps: int,
    pred_space: str,                 # "x" or "v"
    device: torch.device,
) -> torch.Tensor:
    """
    Simple Euler integration consistent with your existing algebra.

    Starts from noise at t=0 and integrates to t=1.
    If model predicts x, convert to v via v = (x - z) / (1 - t).
    """
    assert pred_space in {"x", "v"}
    B = mask_onehot_img.shape[0]
    Hl, Wl = latent_hw

    z = torch.randn(B, 4, Hl, Wl, device=device)
    mask_lat = downsample_mask_to_latent(mask_onehot_img, latent_hw).to(device)

    dt = 1.0 / float(steps)
    for i in range(steps):
        t = torch.full((B,), float(i) / float(steps), device=device)

        model_out = model(z, t, mask_lat)  # [B,4,Hl,Wl]

        if pred_space == "v":
            v = model_out
        else:
            one_minus_t = (1.0 - t).view(B, 1, 1, 1).clamp(min=0.05)
            v = (model_out - z) / one_minus_t

        z = z + dt * v

    return z


def _make_fixed_palette_19() -> np.ndarray:
    """
    Simple palette for visualizing 19-class trainId masks (for W&B display).
    """
    rng = np.random.default_rng(0)
    pal = rng.integers(low=0, high=255, size=(19, 3), dtype=np.uint8)
    pal[0] = np.array([128, 64, 128], dtype=np.uint8)   # road-ish
    pal[10] = np.array([70, 130, 180], dtype=np.uint8)  # sky-ish
    pal[13] = np.array([0, 0, 142], dtype=np.uint8)     # car-ish
    pal[11] = np.array([220, 20, 60], dtype=np.uint8)   # person-ish
    return pal


def main(args):
    # Debug CUDA availability
    print("=== CUDA DEBUG INFO ===")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count() if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("========================")
    
    # Temporarily comment out the assertion to continue debugging
    # assert torch.cuda.is_available(), "No GPU detected"
    
    # Use CPU if CUDA is not available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")
    
    assert args.loss in LOSS_SPACES, f"Invalid loss {args.loss}"
    assert args.pred in PRED_SPACES, f"Invalid pred {args.pred}"

    # Set device and seed
    # device = torch.device("cuda")  # This was hardcoded before
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Simplified batch size handling
    per_gpu_bs = args.bs
    
    exp_name = f"{args.dataset_name}__{time.time()}"
    exp_log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)

    # -------------------------
    # Dataset
    # -------------------------
    dataset = CityscapesDataset(
        root=args.dataset_path,
        split="train",
        image_size=(args.image_h, args.image_w),
        num_classes=19,
        augment=True,
    )

    val_size = min(args.val_size, max(1, len(dataset) // 5))
    rest_size = len(dataset) - val_size
    val_dataset, train_dataset = random_split(dataset, [val_size, rest_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_bs,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,  # Disable pin_memory to avoid potential issues
        persistent_workers=False,  # Disable persistent workers
        drop_last=True,
    )
    train_loader = cycle(train_loader)

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(per_gpu_bs, args.visualize_num),
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,  # Disable pin_memory to avoid potential issues
        persistent_workers=False,  # Disable persistent workers
        drop_last=True,
    )
    val_loader = iter(cycle(val_loader))

    print(f"Train dataset contains {len(train_dataset)} samples.")
    print(f"Val dataset contains {len(val_dataset)} samples.")

    # Test data loading before training
    print("Testing data loading...")
    try:
        test_batch = next(iter(train_loader))
        print(f"Successfully loaded test batch: {type(test_batch)}")
        if isinstance(test_batch, (list, tuple)) and len(test_batch) >= 2:
            print(f"Test batch shapes: mask={test_batch[0].shape}, img={test_batch[1].shape}")
        else:
            print(f"Unexpected batch format: {test_batch}")
        print("Data loading test successful!")
    except Exception as e:
        print(f"Error during data loading test: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit early if data loading fails

    # -------------------------
    # VAE (frozen)
    # -------------------------
    assert args.image_h % 8 == 0 and args.image_w % 8 == 0, "image_h/image_w must be divisible by 8"
    latent_hw = (args.image_h // 8, args.image_w // 8)

    vae = AutoencoderKL.from_pretrained(args.vae_name)
    vae.requires_grad_(False)
    vae.eval()
    
    # Handle VAE device placement
    if args.vae_cpu:
        print("VAE running on CPU to avoid cuDNN issues")
        vae.to("cpu")
        vae_device = torch.device("cpu")
    else:
        # Check if we have enough GPU memory for VAE
        if device.type == "cuda":
            torch.cuda.empty_cache()  # Clear cache first
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            cached_memory = torch.cuda.memory_reserved(0)
            free_memory = total_memory - allocated_memory
            
            print(f"GPU Memory Status:")
            print(f"  Total: {total_memory / 1024**3:.2f} GB")
            print(f"  Allocated: {allocated_memory / 1024**3:.2f} GB")
            print(f"  Cached: {cached_memory / 1024**3:.2f} GB")
            print(f"  Free: {free_memory / 1024**3:.2f} GB")
            
            # Estimate VAE memory requirement (rough estimate: ~2-4GB for SD VAE)
            estimated_vae_memory = 3.0 * 1024**3  # 3GB estimate
            
            if free_memory < estimated_vae_memory:
                print(f"WARNING: Insufficient GPU memory for VAE ({free_memory / 1024**3:.2f} GB free < {estimated_vae_memory / 1024**3:.2f} GB needed)")
                print("Automatically switching VAE to CPU")
                vae.to("cpu")
                vae_device = torch.device("cpu")
                args.vae_cpu = True  # Update flag
            else:
                vae.to(device)
                vae_device = device
        else:
            vae.to(device)
            vae_device = device

    # -------------------------
    # Model (latent DiT)
    # -------------------------
    model = TinyDiTLatent(
        model_dim=args.model_dim,
        n_layers=args.n_dit_layers,
        patch_size=args.latent_patch_size,  # patch size in LATENT space
        latent_channels=4,
        cond_channels=19,
        latent_hw=latent_hw,
        n_heads=args.n_attn_heads,
        mlp_dim=args.feed_fwd_dim,
        n_adaln_cond_cls=None,
    )
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.adam_b1, args.adam_b2),
        weight_decay=0.05,
    )

    # Remove accelerator.prepare - just use the objects directly
    # model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # EMA (no accelerator register needed)
    ema_model = None
    if args.use_ema:
        ema_model = EMA(model, beta=args.ema_beta, update_every=10)
        ema_model.to(device)

    # Print model info (no accelerator.is_main_process check needed)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Latent DiT model has {n_params:,} total parameters")
    print(f"Latent DiT model has {n_trainable:,} trainable parameters")
    print(f"Latent HW: {latent_hw} | latent patch size: {args.latent_patch_size}")
    print(f"Model: {args.n_dit_layers} layers, {args.model_dim} dim, {args.n_attn_heads} heads")
    print(f"Logging every {args.log_every} steps")

    # -------------------------
    # W&B (simplified without accelerator)
    # -------------------------
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    # Initialize W&B directly
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=exp_name,
        dir=exp_log_dir,
    )

    palette_19 = _make_fixed_palette_19()

    # -------------------------
    # Train loop
    # -------------------------
    running_loss, running_gn, start_time = 0.0, 0.0, time.time()
    
    # Time profiling variables
    time_data_loading = 0.0
    time_vae_encoding = 0.0
    time_forward_pass = 0.0
    time_loss_computation = 0.0
    time_backward_pass = 0.0
    time_optimizer_step = 0.0
    time_ema_update = 0.0
    
    cur_step = 0

    while cur_step < args.train_steps:
        # print(f"Starting step {cur_step}...")
        step_start_time = time.time()
        
        # Clear GPU memory before each step
        if cur_step % 10 == 0:  # Clear every 10 steps
            clear_gpu_memory()
        
        # Use torch.autocast with appropriate device
        if device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_context = torch.autocast(device_type="cpu", dtype=torch.float32)
            
        with autocast_context:
            model.train()

            # Data loading timing
            data_start = time.time()
            # print(f"About to load data for step {cur_step}...")
            try:
                mask_onehot_img, img = next(train_loader)
                # print(f"Data loaded successfully. Shapes: mask={mask_onehot_img.shape}, img={img.shape}")
            except Exception as e:
                print(f"Error loading data: {e}")
                raise
            
            mask_onehot_img = mask_onehot_img.to(device)  # [B,19,H,W]
            img = img.to(device)                          # [B,3,H,W] in [-1,1]
            data_end = time.time()
            time_data_loading += (data_end - data_start)

            # print(f"Data loaded for step {cur_step} in {data_end - data_start:.3f}s. Starting VAE encoding...")

            # VAE encoding timing with error handling
            vae_start = time.time()
            try:
                with torch.no_grad():
                    # Handle VAE device placement
                    if args.vae_cpu:
                        img_for_vae = img.cpu()
                        z = encode_to_latents_fp32(vae, img_for_vae, args.latent_scale)
                        z = z.to(device)  # Move result back to main device
                    else:
                        z = encode_to_latents_fp32(vae, img, args.latent_scale)
                # print(f"VAE encoding successful. Latent shape: {z.shape}")
            except Exception as e:
                print(f"VAE encoding failed with error: {e}")
                if not args.vae_cpu:  # Only try CPU fallback if not already using CPU
                    print("Trying VAE encoding on CPU...")
                    try:
                        # Try moving VAE to CPU temporarily
                        vae_cpu = vae.cpu()
                        img_cpu = img.cpu()
                        with torch.no_grad():
                            z = encode_to_latents_fp32(vae_cpu, img_cpu, args.latent_scale)
                            z = z.to(device)  # Move result back to GPU
                        vae.to(device)  # Move VAE back to GPU
                        print(f"VAE encoding on CPU successful. Latent shape: {z.shape}")
                    except Exception as e2:
                        print(f"VAE encoding also failed on CPU: {e2}")
                        raise e  # Re-raise original error
                else:
                    raise e  # Re-raise if already using CPU
            vae_end = time.time()
            time_vae_encoding += (vae_end - vae_start)

            # print(f"VAE encoding completed for step {cur_step}. Starting forward pass...")

            mask_lat = downsample_mask_to_latent(mask_onehot_img, latent_hw)  # [B,19,Hl,Wl]

            # print(f"Mask downsampled for step {cur_step}. Starting forward pass...")

            B = z.shape[0]
            t = sample_t((B,), device, args.t_sampling, args.t_logit_normal_mu, args.t_logit_normal_sigma)
            eps = torch.randn_like(z)

            t4 = t.view(B, 1, 1, 1)
            z_noisy = t4 * z + (1.0 - t4) * eps

            # Forward pass timing
            forward_start = time.time()
            model_out = model(z_noisy, t, mask_lat)  # [B,4,Hl,Wl]
            forward_end = time.time()
            time_forward_pass += (forward_end - forward_start)

            # Loss computation timing  
            loss_start = time.time()
            # target
            target = z if args.loss == "x" else (z - eps)

            # pred
            if args.pred == args.loss:
                pred = model_out
            elif args.pred == "x" and args.loss == "v":
                pred = (model_out - z_noisy) / (1.0 - t4).clamp(min=0.05)
            elif args.pred == "v" and args.loss == "x":
                pred = (1.0 - t4) * model_out + z_noisy
            else:
                raise ValueError(f"Unsupported pred/loss combo: pred={args.pred} loss={args.loss}")

            loss = F.mse_loss(pred, target)
            loss_end = time.time()
            time_loss_computation += (loss_end - loss_start)

            # Backward pass timing
            backward_start = time.time()
            # print(f"Starting backward pass for step {cur_step}...")
            loss.backward()
            backward_end = time.time()
            time_backward_pass += (backward_end - backward_start)

            # print(f"Backward pass completed for step {cur_step}. Starting optimizer step...")

            # Gradient clipping (standard PyTorch)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            # Optimizer step timing
            optim_start = time.time()
            optimizer.step()
            optim_end = time.time()
            time_optimizer_step += (optim_end - optim_start)
            
            # EMA update timing
            ema_start = time.time()
            if ema_model is not None:
                ema_model.update()
            ema_end = time.time()
            time_ema_update += (ema_end - ema_start)
            
            optimizer.zero_grad()

            running_loss += loss.item()
            running_gn += grad_norm.item()

            cur_step += 1
            step_end_time = time.time()
            # print(f"Finished step {cur_step}. Total step time: {step_end_time - step_start_time:.4f}s")

        # -------------------------
        # Logging / sampling (simplified without accelerator)
        if True:  # Remove accelerator.is_main_process and accelerator.sync_gradients checks
            if cur_step % args.log_every == 0:
                avg_loss = running_loss / float(args.log_every)
                avg_gn = running_gn / float(args.log_every)
                elapsed = time.time() - start_time
                avg_time = elapsed / float(args.log_every)

                # Calculate average times for each component
                avg_data_time = time_data_loading / float(args.log_every)
                avg_vae_time = time_vae_encoding / float(args.log_every)
                avg_forward_time = time_forward_pass / float(args.log_every)
                avg_loss_time = time_loss_computation / float(args.log_every)
                avg_backward_time = time_backward_pass / float(args.log_every)
                avg_optim_time = time_optimizer_step / float(args.log_every)
                avg_ema_time = time_ema_update / float(args.log_every)

                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {cur_step:08d}/{args.train_steps} ({100*cur_step/args.train_steps:.1f}%) | "
                    f"Loss {avg_loss:.6f} | Time {avg_time:.3f}s/step | GradNorm {avg_gn:.4f} | LR {lr:.2e}"
                )
                
                # Detailed timing breakdown
                print(f"  Time Breakdown (avg/step): "
                      f"Data={avg_data_time:.3f}s ({100*avg_data_time/avg_time:.1f}%) | "
                      f"VAE={avg_vae_time:.3f}s ({100*avg_vae_time/avg_time:.1f}%) | "
                      f"Forward={avg_forward_time:.3f}s ({100*avg_forward_time/avg_time:.1f}%) | "
                      f"Loss={avg_loss_time:.3f}s ({100*avg_loss_time/avg_time:.1f}%) | "
                      f"Backward={avg_backward_time:.3f}s ({100*avg_backward_time/avg_time:.1f}%) | "
                      f"Optim={avg_optim_time:.3f}s ({100*avg_optim_time/avg_time:.1f}%) | "
                      f"EMA={avg_ema_time:.3f}s ({100*avg_ema_time/avg_time:.1f}%)")

                # Log to W&B directly
                wandb.log(
                    {
                        "loss": avg_loss,
                        "grad_norm": avg_gn,
                        "learning_rate": lr,
                        "steps_per_second": 1.0 / max(avg_time, 1e-9),
                        "time_data_loading": avg_data_time,
                        "time_vae_encoding": avg_vae_time,
                        "time_forward_pass": avg_forward_time,
                        "time_loss_computation": avg_loss_time,
                        "time_backward_pass": avg_backward_time,
                        "time_optimizer_step": avg_optim_time,
                        "time_ema_update": avg_ema_time,
                    },
                    step=cur_step,
                )

                running_loss = 0.0
                running_gn = 0.0
                start_time = time.time()
                
                # Reset timing counters
                time_data_loading = 0.0
                time_vae_encoding = 0.0
                time_forward_pass = 0.0
                time_loss_computation = 0.0
                time_backward_pass = 0.0
                time_optimizer_step = 0.0
                time_ema_update = 0.0

            if (args.visualize_first_step and cur_step == 1) or (cur_step % args.visualize_every == 0):
                print("Visualizing...")
                with torch.no_grad():
                    sampling_model = ema_model.ema_model if (ema_model is not None) else model
                    sampling_model.eval()
                    sampling_model.to(device)

                    samples_to_log = {}
                    val_generated_count = 0

                    while val_generated_count < args.visualize_num:
                        mask_onehot_val, img_val = next(val_loader)
                        mask_onehot_val = mask_onehot_val.to(device)
                        img_val = img_val.to(device)

                        z_gen = euler_sample_latents(
                            sampling_model,
                            mask_onehot_val,
                            latent_hw=latent_hw,
                            steps=args.max_sampling_t,
                            pred_space=args.pred,
                            device=device,
                        )

                        img_pred = decode_from_latents_fp32(vae, z_gen, args.latent_scale)

                        Bv = img_pred.shape[0]
                        for i in range(Bv):
                            orig = img_val[i].permute(1, 2, 0).float().cpu().numpy()
                            orig = np.clip((orig + 1.0) / 2.0, 0.0, 1.0)
                            orig = (orig * 255).astype(np.uint8)

                            gen = img_pred[i].permute(1, 2, 0).float().cpu().numpy()
                            gen = np.clip((gen + 1.0) / 2.0, 0.0, 1.0)
                            gen = (gen * 255).astype(np.uint8)

                            cls = torch.argmax(mask_onehot_val[i], dim=0).cpu().numpy().astype(np.uint8)
                            cond_vis = palette_19[cls]

                            cond_pil = Image.fromarray(cond_vis).resize((args.image_w, args.image_h), Image.NEAREST)
                            gen_pil = Image.fromarray(gen).resize((args.image_w, args.image_h), Image.BILINEAR)
                            orig_pil = Image.fromarray(orig).resize((args.image_w, args.image_h), Image.BILINEAR)

                            samples_to_log[f"gen_{val_generated_count + i}"] = [
                                wandb.Image(cond_pil, caption="cond (Cityscapes mask)"),
                                wandb.Image(gen_pil, caption="generated"),
                                wandb.Image(orig_pil, caption="original"),
                            ]

                        val_generated_count += Bv

                    wandb.log(samples_to_log, step=cur_step)

            if cur_step % args.ckpt_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, exp_name, f"iters_{cur_step:08d}")
                os.makedirs(ckpt_path, exist_ok=True)
                
                # Save model and optimizer state using standard PyTorch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict() if ema_model else None,
                    'step': cur_step,
                    'args': vars(args)
                }, os.path.join(ckpt_path, 'checkpoint.pt'))
                
                print(f"Saved Iter {cur_step} checkpoint to {ckpt_path}")

                exp_ckpt_root = os.path.join(args.checkpoint_dir, exp_name)
                for ckpt_dir in os.listdir(exp_ckpt_root):
                    if ckpt_dir.startswith("iters") and ckpt_dir != f"iters_{cur_step:08d}":
                        save_iter = int(ckpt_dir.split("_")[-1])
                        if save_iter < cur_step - args.keep_last_k * args.ckpt_every:
                            shutil.rmtree(os.path.join(exp_ckpt_root, ckpt_dir), ignore_errors=True)

        # Remove accelerator.wait_for_everyone()

    # final checkpoint
    final_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name, f"iters_{cur_step:08d}_final")
    os.makedirs(final_ckpt_dir, exist_ok=True)
    
    # Save final checkpoint using standard PyTorch
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_model_state_dict': ema_model.state_dict() if ema_model else None,
        'step': cur_step,
        'args': vars(args)
    }, os.path.join(final_ckpt_dir, 'checkpoint.pt'))
    
    print(f"Saved Final Iter {cur_step} checkpoint to {final_ckpt_dir}")

    print("Training Done.")
    wandb.finish()  # Close W&B run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--train_steps", type=int, default=300000)
    parser.add_argument("--bs", type=int, default=16)  # Reduced from 64 to save memory
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam_b1", type=float, default=0.9)
    parser.add_argument("--adam_b2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # dataset
    parser.add_argument("--dataset_name", type=str, default="cityscapes")
    parser.add_argument("--dataset_path", type=str, default="/media/mehrnoosh/MEHRNOOSH2/citydata")
    parser.add_argument("--image_h", type=int, default=256)
    parser.add_argument("--image_w", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=4)

    # diffusion (UNCHANGED)
    parser.add_argument("--pred", type=str, choices=["x", "v"], default="v")
    parser.add_argument("--loss", type=str, choices=["x", "v"], default="v")
    parser.add_argument("--t_sampling", type=str, choices=["uniform", "logit_normal"], default="uniform")
    parser.add_argument("--t_logit_normal_mu", type=float, default=-0.8)
    parser.add_argument("--t_logit_normal_sigma", type=float, default=1.0)

    # EMA
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_beta", type=float, default=0.9999)

    # VAE / latent
    parser.add_argument("--vae_name", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--latent_scale", type=float, default=0.18215)
    parser.add_argument("--vae_cpu", action="store_true", default=False, help="Force VAE to run on CPU to avoid cuDNN issues")

    # architecture (LATENT DiT)
    parser.add_argument("--model_dim", type=int, default=384)
    parser.add_argument("--n_dit_layers", type=int, default=8)
    parser.add_argument("--latent_patch_size", type=int, default=2)  # patch size in LATENT space
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--feed_fwd_dim", type=int, default=4 * 384)

    # sampling
    parser.add_argument("--max_sampling_t", type=int, default=20)

    # logging
    parser.add_argument("--log_dir", type=str, default="out")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--visualize_first_step", action="store_true", default=True)
    parser.add_argument("--visualize_every", type=int, default=500)
    parser.add_argument("--visualize_num", type=int, default=2)
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_project_name", type=str, default="Cityscapes_Latent_TinyDiT")

    # checkpoints
    parser.add_argument("--ckpt_every", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="ckpts")
    parser.add_argument("--keep_last_k", type=int, default=10)

    args = parser.parse_args()
    main(args)