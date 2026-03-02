# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# trainer.py

# Trainer for mask-conditioned training (diffusion or flow-matching).

# Fixes included:
# - gen_model is treated as a CLASS and instantiated once.
# - Correct AMP usage: autocast for forward/loss only.
# - Proper GradScaler usage for fp16 stability.
# - Gradient accumulation support.
# - Removed VAE CPU fallback/check.
# - Removed all detailed time metrics.
# """

# import os
# import shutil
# import time
# import re
# from typing import Dict, Any, Optional, Union

# import numpy as np
# from PIL import Image

# import torch
# import wandb
# from ema_pytorch import EMA

# from diffusion.Rectified_flow_matching import FlowMatchingConfig , FlowMatching  # if you use it elsewhere
# from diffusion.guassian_diffusion import DiffusionConfig, GaussianDiffusion  # your diffusion
# from torch import nn

# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP



# class CityscapesTrainer:
#     """
#     Trainer for mask-conditioned generation (Cityscapes).

#     Assumptions:
#     - train_loader yields (mask_onehot_img, img)
#         mask_onehot_img: [B,19,H,W] float/bool one-hot
#         img:             [B,3,H,W] in [-1,1]
#     - model is MaskCondDiTWithVAE-like:
#         - model.encode_image_to_latent(img) -> z [B,4,H/8,W/8] (scaled latents)
#         - model.forward(z_noisy, t_long, mask_onehot_img) -> eps_pred  (for diffusion eps objective)
#         - model.decode_latent_to_image(z) (optional, for visualization)
#     - gen_model is a CLASS (e.g., GaussianDiffusion), so we instantiate:
#         self.gen_model = gen_model(cfg=config, device=device)
#     """

#     def __init__(
#         self,
#         model: nn.Module,
#         optimizer: torch.optim.Optimizer,
#         train_loader,
#         val_loader,
#         local_rank: int,
#         global_rank: int,
#         config: Union[FlowMatchingConfig, DiffusionConfig],
#         gen_model: Union[FlowMatching, GaussianDiffusion],  # <-- CLASS, not instance
#         args,
#         exp_log_dir: str,
#         palette_19: Optional[np.ndarray] = None,
#         device: Optional[torch.device]=None,
#     ):
#         if device is None:
#             self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = device

#         self.local_rank = local_rank
#         self.global_rank = global_rank
        
#         # Determine if we're in distributed mode
#         self.is_distributed = dist.is_initialized() if dist.is_available() else False
#         self.world_size = dist.get_world_size() if self.is_distributed else 1
        
#         self.model = model.to(self.device)
        
#         # Wrap model in DDP if distributed training
#         if self.is_distributed:
#             print(f"Wrapping model in DDP for rank {self.global_rank}/{self.world_size}")
#             self.model = DDP(self.model, device_ids=[self.local_rank])
#         else:
#             print("Single GPU training - no DDP wrapping")
        
#         self.optimizer = optimizer
#         self.train_loader = train_loader
#         self.train_iter = iter(train_loader)  # Create iterator for manual epoch handling
#         self.val_loader = val_loader
#         self.args = args
#         self.exp_log_dir = exp_log_dir
#         self.palette_19 = palette_19


#         # W&B
#         self.use_wandb = wandb.run is not None
#         print("W&B logging enabled" if self.use_wandb else "W&B not initialized - console logging only")

#         # Instantiate generative training helper (diffusion or flow matching)
#         self.gen_model = gen_model(cfg=config, device=self.device)

#         # EMA
#         self.ema_model = None
#         if getattr(args, "use_ema", False):
#             self.ema_model = EMA(self.model)
#             self.ema_model.to(device)

#         # AMP + GradScaler
#         # autocast: runs forward pass in fp16 to save memory + increase throughput
#         # GradScaler: rescales loss to avoid fp16 gradient underflow
#         self.use_amp = (self.device.type == "cuda") and getattr(args, "use_amp", True)
#         self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

#         # Gradient accumulation
#         # effective_batch = args.bs * grad_accum_steps
#         self.grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))

#         # State
#         self.current_step = 0
#         self.start_step = 0  # For resume functionality

#         # Make sure output dirs exist
#         os.makedirs(self.exp_log_dir, exist_ok=True)

#         # Tracking (no time metrics)
#         self.running_loss = 0.0
#         self.running_grad_norm = 0.0
#         self.running_optim_steps = 0
#         self.window_start_time = time.time()

#         # Recommended: zero grads once here
#         self.optimizer.zero_grad(set_to_none=True)

#     # ----------------------------
#     # Core training
#     # ----------------------------

#     def training_step(self) -> Dict[str, float]:
#         """
#         One *micro* step. Optimizer updates only every grad_accum_steps.
#         Returns per-step metrics.
#         """
#         # Debug: Add step tracing for first few steps
#         if self.current_step < 3:
#             print(f"    🔄 training_step() called for step {self.current_step}")
        
#         # Remove DDP wrapping from here - should be in __init__
#         self.model.train()

#         # 1) Data
#         if self.current_step < 3:
#             print(f"    📥 About to get next batch from train_loader...")
        
#         try:
#             mask_onehot_img, img = next(self.train_iter)
#             if self.current_step < 3:
#                 print(f"    ✅ Got batch: mask={mask_onehot_img.shape}, img={img.shape}")
#         except StopIteration:
#             # Restart iterator when epoch ends
#             print(f"    🔄 Restarting train_loader iterator at step {self.current_step}")
#             self.train_iter = iter(self.train_loader)
#             mask_onehot_img, img = next(self.train_iter)
#             if self.current_step < 3:
#                 print(f"    ✅ Got batch after restart: mask={mask_onehot_img.shape}, img={img.shape}")
#         except Exception as e:
#             print(f"    ❌ Failed to get batch: {e}")
#             raise e
            
#         mask_onehot_img = mask_onehot_img.to(self.device, non_blocking=True)
#         img = img.to(self.device, non_blocking=True)

#         if self.current_step < 3:
#             print(f"    🔧 About to encode image to latent...")
#             # DEBUG: Check mask conditioning in training
#             print(f"    🔍 TRAINING MASK DEBUG:")
#             print(f"      - mask_onehot_img shape: {mask_onehot_img.shape}")
#             print(f"      - mask_onehot_img dtype: {mask_onehot_img.dtype}")
#             print(f"      - mask_onehot_img range: [{mask_onehot_img.min().item():.3f}, {mask_onehot_img.max().item():.3f}]")
#             mask_sums_train = mask_onehot_img.sum(dim=1)
#             print(f"      - mask channel sums: min={mask_sums_train.min().item():.3f}, max={mask_sums_train.max().item():.3f}")
        
#         print("img range:", float(img.min()), float(img.max()))
#         # 2) Encode with VAE (no grad through VAE)
#         with torch.no_grad():
#             # Encode RGB to latents - handle DDP wrapping consistently
#             if hasattr(self.model, 'module'):
#                 # Model is wrapped in DDP
#                 z = self.model.module.encode_image_to_latent(img)  # [B,4,Hl,Wl] scaled latents
#             else:
#                 # Model is not wrapped
#                 z = self.model.encode_image_to_latent(img)  # [B,4,Hl,Wl] scaled latents

#         print("z_scaled std:", float(z.std()), "mean:", float(z.mean()))

#         # Check reconstruction quality of VAE on first few steps (optional, can be noisy)
#         # if self.current_step < 3:
#         #     with torch.no_grad():
#         #         z = self.model.module.vae.encode(img).latent_dist.mode()  # not sample()
#         #         z_scaled = z * 0.13025
#         #         x_rec = self.model.module.vae.decode(z).sample  # decode unscaled z
#         #     print(f"    🔍 VAE Reconstruction Quality:")
#         #     print(f"      - Recon img range: [{x_rec.min().item():.3f}, {x_rec.max().item():.3f}]")
#         #     print(f"      - Recon img std: {x_rec.std().item():.3f}, mean: {x_rec.mean().item():.3f}")
#         #     print(f"      - Recon img vs input img MSE: {torch.nn.functional.mse_loss(x_rec, img).item():.6f}")
            

#         if self.current_step < 3:
#             print(f"    ✅ Encoded to latent: {z.shape}")
#             print(f"    📊 About to compute loss...")

#         # 3) Forward + loss under autocast (forward only!)
#         with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
#             loss = self.gen_model.loss(
#                 model=self.model,
#                 x=z,
#                 cond=mask_onehot_img,
#             )
#             loss_to_backward = loss / float(self.grad_accum_steps)

#         # if self.current_step < 3:
#         #     print(f"    ✅ Loss computed: {loss.item():.6f}")
#         #     print(f"    🔄 About to backward...")
        
#         print(f"    ✅ Loss computed: {loss.item():.6f}")
#         print(f"    🔄 About to backward...")

#         # 4) Backward (scaled)
#         self.scaler.scale(loss_to_backward).backward()

#         # 5) Optim step only every grad_accum_steps
#         do_step = ((self.current_step + 1) % self.grad_accum_steps == 0)

#         grad_norm_val = 0.0
#         if do_step:
#             # Unscale grads before clipping
#             self.scaler.unscale_(self.optimizer)

#             # Check for invalid gradients before clipping
#             has_inf_grad = False
#             has_nan_grad = False
#             total_norm = 0.0
            
#             for param in self.model.parameters():
#                 if param.grad is not None:
#                     if torch.isinf(param.grad).any():
#                         has_inf_grad = True
#                     if torch.isnan(param.grad).any():
#                         has_nan_grad = True
#                     param_norm = param.grad.data.norm(2)
#                     total_norm += param_norm.item() ** 2
            
#             total_norm = total_norm ** (1. / 2)
            
#             if has_inf_grad or has_nan_grad or total_norm > 1e6:
#                 # Log the problematic gradients
#                 if self.global_rank == 0:
#                     print(f"⚠️  Step {self.current_step}: Problematic gradients detected!")
#                     print(f"   Has inf grad: {has_inf_grad}")
#                     print(f"   Has nan grad: {has_nan_grad}")  
#                     print(f"   Total norm: {total_norm}")
#                     print(f"   Loss: {loss.item()}")
                
#                 # Skip this optimization step
#                 self.optimizer.zero_grad(set_to_none=True)
#                 grad_norm_val = float('inf')
                
#                 # Reset scaler if gradients are problematic
#                 if has_inf_grad or has_nan_grad:
#                     self.scaler.update()  # This will reduce the scale
#             else:
#                 # Normal gradient clipping and optimization
#                 grad_norm = torch.nn.utils.clip_grad_norm_(
#                     self.model.parameters(),
#                     max_norm=getattr(self.args, "max_grad_norm", 1.0),
#                 )
#                 grad_norm_val = float(grad_norm.item())

#                 # Step optimizer with scaler
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()

#                 # Zero grad after step
#                 self.optimizer.zero_grad(set_to_none=True)

#                 # EMA update only on real optimizer steps
#                 if self.ema_model is not None:
#                     self.ema_model.update()

#                 self.running_optim_steps += 1
                
#             self.running_grad_norm += grad_norm_val

#         # 6) Metrics
#         loss_val = float(loss.detach().item())
#         self.running_loss += loss_val

#         self.current_step += 1

#         return {
#             "loss": loss_val,
#             "did_step": float(do_step),
#             "grad_norm": grad_norm_val,
#         }

#     # ----------------------------
#     # Logging (no time metrics)
#     # ----------------------------

#     def log_metrics(self, step: int):
#         if step % self.args.log_every != 0:
#             return
            
#         # Only log on rank 0 in distributed training
#         if self.is_distributed and self.global_rank != 0:
#             return

#         # Average loss over the last log window (micro steps)
#         avg_loss = self.running_loss / float(self.args.log_every)

#         # Average grad norm over optimizer steps (not micro steps)
#         avg_gn = self.running_grad_norm / max(1, self.running_optim_steps)

#         lr = self.optimizer.param_groups[0]["lr"]

#         print(
#             f"Step {step:08d}/{self.args.train_steps} ({100*step/self.args.train_steps:.1f}%) | "
#             f"Loss {avg_loss:.6f} | GradNorm {avg_gn:.4f} | LR {lr:.2e} | "
#             f"accum {self.grad_accum_steps} (opt_steps {self.running_optim_steps})"
#         )

#         if self.use_wandb:
#             try:
#                 wandb.log(
#                     {
#                         "train/loss": avg_loss,
#                         "train/grad_norm": avg_gn,
#                         "train/lr": lr,
#                         "train/step": step,
#                         "train/grad_accum_steps": self.grad_accum_steps,
#                         "train/opt_steps_in_window": self.running_optim_steps,
#                     },
#                     step=step,
#                 )
#             except Exception as e:
#                 print(f"W&B logging failed: {e}")

#         # Reset window metrics
#         self.running_loss = 0.0
#         self.running_grad_norm = 0.0
#         self.running_optim_steps = 0
#         self.window_start_time = time.time()

#     # ----------------------------
#     # Visualization (optional)
#     # If your gen_model is GaussianDiffusion, sample() expects (model, shape, cond).
#     # ----------------------------

#     def visualize(self, step: int):
#         if not getattr(self.args, "enable_visualization", False):
#             return
#         if not ((getattr(self.args, "visualize_first_step", False) and step == 1) or (step % self.args.visualize_every == 0)):
#             return
            
#         # Only visualize on rank 0 in distributed training
#         if self.is_distributed and self.global_rank != 0:
#             return
            
#         if self.val_loader is None:
#             print("Warning: No validation loader available for visualization")
#             return

#         print("Visualizing...")
        
#         if self.val_loader is None:
#             return
            
#         with torch.no_grad():
#             # Get the underlying model for sampling (handle DDP wrapping)
#             if self.ema_model is not None:
#                 # Use EMA model for better quality samples
#                 ema_model = self.ema_model.ema_model
#                 if hasattr(ema_model, 'module'):
#                     # EMA model is also DDP-wrapped
#                     sampling_model = ema_model.module
#                     print("  - Using DDP wrapped EMA model (.module)")
#                 else:
#                     # EMA model is not wrapped
#                     sampling_model = ema_model
#                     print("  - Using unwrapped EMA model")
#             else:
#                 # Use main model, but handle DDP wrapping
#                 if hasattr(self.model, 'module'):
#                     # Model is wrapped in DDP
#                     sampling_model = self.model.module
#                     print("  - Using DDP wrapped model (.module)")
#                 else:
#                     # Model is not wrapped
#                     sampling_model = self.model
#                     print("  - Using unwrapped model")
            
#             sampling_model.eval()

#             # Debug: Get validation data with error handling and timeout
#             print("  - Attempting to get next batch from val_loader...")
            
          
#             if hasattr(self.val_loader, '__next__'):
#                 # It's an iterator (cycled)
#                 mask_onehot_val, img_val = next(self.val_loader)
#             else:
#                 # It's a regular DataLoader - create a fresh iterator
#                 val_iter = iter(self.val_loader)
#                 mask_onehot_val, img_val = next(val_iter)
                                
#             mask_onehot_val = mask_onehot_val.to(self.device, non_blocking=True)
#             img_val = img_val.to(self.device, non_blocking=True)

#             B = img_val.shape[0]
            
#             # Check if we got RGB images (3 channels) or latents (4 channels)
#             if img_val.shape[1] == 3:
#                 print("📸 Got RGB images - need to encode to latents for visualization")
#                 # Encode RGB to latents for proper visualization - ensure we access the VAE correctly
#                 if hasattr(self.model, 'module'):
#                     # Model is wrapped in DDP - use the wrapped model for VAE access
#                     img_latents = self.model.module.encode_image_to_latent(img_val)
#                 else:
#                     # Model is not wrapped
#                     img_latents = self.model.encode_image_to_latent(img_val)
#                 latent_hw = (img_latents.shape[2], img_latents.shape[3])
#                 print(f"  - Encoded latent shape: {img_latents.shape}")
#             elif img_val.shape[1] == 4:
#                 print("🎯 Got latents directly")
#                 img_latents = img_val
#                 latent_hw = (img_val.shape[2], img_val.shape[3])
#             else:
#                 print(f"❌ Unexpected image channels: {img_val.shape[1]}")
#                 return
                
         

#             # GaussianDiffusion-style sampling:
#             #   z_gen = self.gen_model.sample(model=sampling_model, shape=(B,4,Hl,Wl), cond=mask_onehot_val)
            
#             # Time the sampling operation
#             print("Sampling...")
#             sampling_start_time = time.time()
#             z_gen = self.gen_model.sample(
#                 model=sampling_model,
#                 shape=(B, 4, latent_hw[0], latent_hw[1]),
#                 cond=mask_onehot_val,
#                 device=self.device,
#             )
#             sampling_end_time = time.time()
#             sampling_duration = sampling_end_time - sampling_start_time

#             print("compare z_gen and img_latents:")
#             print(f"  - z_gen shape: {z_gen.shape}")
#             print(f"  - img_latents shape: {img_latents.shape}")
#             print(f"  - z_gen vs img_latents diff norm: {torch.norm(z_gen - img_latents).item():.4f}")
#             print("z_gen std:", float(z_gen.std()), "mean:", float(z_gen.mean()))
#             print("z_gen std:", float(img_latents.std()), "mean:", float(img_latents.mean()))


#             # Track sampling times for averaging
#             if not hasattr(self, 'sampling_times'):
#                 self.sampling_times = []
#             self.sampling_times.append(sampling_duration)
            
#             # Print current sampling time
#             print(f"  → Sampling completed in {sampling_duration:.2f}s")
            
#             # Print average every 5 samples
#             if len(self.sampling_times) >= 5:
#                 avg_sampling_time = sum(self.sampling_times[-5:]) / 5
#                 print(f"  → Average sampling time over last 5 runs: {avg_sampling_time:.2f}s")

#             # Decode generated latents to images - ensure we access the VAE correctly
#             if hasattr(self.model, 'module'):
#                 # Model is wrapped in DDP - use the wrapped model for VAE access
#                 img_pred = self.model.module.decode_latent_to_image(z_gen)
#             else:
#                 # Model is not wrapped
#                 img_pred = self.model.decode_latent_to_image(z_gen)

#             # Save images locally and log to W&B
#             save_images = getattr(self.args, "save_visualization_images", True)
#             num_samples_to_visualize = min(B, getattr(self.args, "visualize_num", min(B, 4)))
            
#             samples_to_log = {}
            
#             # Create visualization directory if saving images locally
#             if save_images:
#                 viz_dir = os.path.join(self.args.log_dir, "visualizations")
#                 os.makedirs(viz_dir, exist_ok=True)
#                 step_dir = os.path.join(viz_dir, f"step_{step:06d}")
#                 os.makedirs(step_dir, exist_ok=True)
#                 print(f"  → Saving images to: {step_dir}")
            
#             for i in range(num_samples_to_visualize):
#                 # Convert tensors to PIL images
#                 orig_pil = self._tensor_to_pil(img_val[i])
#                 gen_pil = self._tensor_to_pil(img_pred[i])
#                 cond_pil = self._mask_to_pil(mask_onehot_val[i])
                
#                 # Save images locally if enabled
#                 if save_images:
#                     orig_pil.save(os.path.join(step_dir, f"sample_{i}_ground_truth.png"))
#                     gen_pil.save(os.path.join(step_dir, f"sample_{i}_generated.png"))
#                     cond_pil.save(os.path.join(step_dir, f"sample_{i}_condition_mask.png"))
                
#                 # Prepare for W&B logging
#                 if self.use_wandb:
#                     samples_to_log[f"viz/{i}"] = [
#                         wandb.Image(cond_pil, caption="condition (mask)"),
#                         wandb.Image(gen_pil, caption="generated"),
#                         wandb.Image(orig_pil, caption="ground truth"),
#                     ]
            
#             # Log to W&B if enabled
#             if self.use_wandb and samples_to_log:
#                 try:
#                     wandb.log(samples_to_log, step=step)
#                     print(f"  → Logged {len(samples_to_log)} sample sets to W&B")
#                 except Exception as e:
#                     print(f"W&B visualization logging failed: {e}")
            
#             if save_images:
#                 print(f"  → Saved {num_samples_to_visualize} sample sets to disk")

#     def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
#         img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
#         img_np = np.clip((img_np + 1.0) / 2.0, 0.0, 1.0)
#         img_np = (img_np * 255).astype(np.uint8)
#         return Image.fromarray(img_np)

#     def _mask_to_pil(self, mask_tensor: torch.Tensor) -> Image.Image:
#         mask_indices = torch.argmax(mask_tensor, dim=0).detach().cpu().numpy().astype(np.uint8)
#         if self.palette_19 is not None:
#             colored_mask = self.palette_19[mask_indices]
#             return Image.fromarray(colored_mask)
#         mask_vis = (mask_indices * 255 // 19).astype(np.uint8)
#         return Image.fromarray(mask_vis)

#     # ----------------------------
#     # Checkpointing
#     # ----------------------------

#     def save_checkpoint(self, step: int):
#         # Only save checkpoints on rank 0 to avoid conflicts
#         if self.global_rank != 0:
#             return
            
#         if step % self.args.ckpt_every != 0:
#             return

#         # Create checkpoint directory
#         checkpoint_dir = os.path.join(self.args.checkpoint_dir)
#         os.makedirs(checkpoint_dir, exist_ok=True)

#         # Save with consistent naming pattern: checkpoint_step_XXXXXX.pt
#         ckpt_filename = f"checkpoint_step_{step:06d}.pt"
#         ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)

#         # Get model state dict (handle DDP wrapping)
#         if hasattr(self.model, 'module'):
#             model_state_dict = self.model.module.state_dict()
#         else:
#             model_state_dict = self.model.state_dict()

#         ckpt_dict = {
#             "model_state_dict": model_state_dict,
#             "optimizer_state_dict": self.optimizer.state_dict(),
#             "step": step,
#             "loss": self.running_loss / max(1, self.running_optim_steps),
#             "args": vars(self.args),
#             "global_rank": self.global_rank,
#             "world_size": self.world_size,
#         }

#         # Add optional states
#         if self.ema_model is not None:
#             ckpt_dict["ema_state_dict"] = self.ema_model.state_dict()
        
#         if self.scaler is not None:
#             ckpt_dict["scaler_state_dict"] = self.scaler.state_dict()

#         try:
#             torch.save(ckpt_dict, ckpt_path)
#             print(f"💾 Saved checkpoint: {ckpt_path} (step {step})")
            
#             # Cleanup old checkpoints
#             self._cleanup_old_checkpoints(step)
#         except Exception as e:
#             print(f"❌ Error saving checkpoint: {e}")

#     def _cleanup_old_checkpoints(self, current_step: int):
#         keep_last_k = int(getattr(self.args, "keep_last_k", 10))
#         if keep_last_k <= 0:
#             return

#         try:
#             checkpoint_dir = os.path.join(self.args.checkpoint_dir)
#             if not os.path.exists(checkpoint_dir):
#                 return

#             # Find all checkpoint files
#             import glob
#             ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
            
#             # Extract step numbers and sort
#             checkpoints = []
#             for ckpt_file in ckpt_files:
#                 match = re.search(r'checkpoint_step_(\d+)\.pt', os.path.basename(ckpt_file))
#                 if match:
#                     step = int(match.group(1))
#                     checkpoints.append((step, ckpt_file))
            
#             # Sort by step number (newest first)
#             checkpoints.sort(key=lambda x: x[0], reverse=True)
            
#             # Remove old checkpoints beyond keep_last_k
#             for i, (step, ckpt_file) in enumerate(checkpoints):
#                 if i >= keep_last_k:
#                     try:
#                         os.remove(ckpt_file)
#                         print(f"🗑️  Cleaned up old checkpoint: {os.path.basename(ckpt_file)}")
#                     except Exception as e:
#                         print(f"Warning: Could not remove {ckpt_file}: {e}")
                        
#         except Exception as e:
#             print(f"Warning: Error during checkpoint cleanup: {e}")

#     def save_final_checkpoint(self):
#         # Only save on rank 0
#         if self.global_rank != 0:
#             return
            
#         checkpoint_dir = os.path.join(self.args.checkpoint_dir)
#         os.makedirs(checkpoint_dir, exist_ok=True)

#         # Save with final marker
#         final_ckpt_filename = f"checkpoint_step_{self.current_step:06d}_final.pt"
#         final_ckpt_path = os.path.join(checkpoint_dir, final_ckpt_filename)

#         # Get model state dict (handle DDP wrapping)
#         if hasattr(self.model, 'module'):
#             model_state_dict = self.model.module.state_dict()
#         else:
#             model_state_dict = self.model.state_dict()

#         final_ckpt_dict = {
#             "model_state_dict": model_state_dict,
#             "optimizer_state_dict": self.optimizer.state_dict(),
#             "step": self.current_step,
#             "loss": self.running_loss / max(1, self.running_optim_steps),
#             "args": vars(self.args),
#             "global_rank": self.global_rank,
#             "world_size": self.world_size,
#             "final": True,
#         }

#         # Add optional states
#         if self.ema_model is not None:
#             final_ckpt_dict["ema_state_dict"] = self.ema_model.state_dict()
        
#         if self.scaler is not None:
#             final_ckpt_dict["scaler_state_dict"] = self.scaler.state_dict()

#         try:
#             torch.save(final_ckpt_dict, final_ckpt_path)
#             print(f"🏁 Saved final checkpoint: {final_ckpt_path}")
#         except Exception as e:
#             print(f"❌ Error saving final checkpoint: {e}")


#     def print_model_info(self):
#         """
#         Print detailed model information including parameters and GPU memory requirements.
#         """
#         print("\n" + "="*60)
#         print("MODEL INFORMATION")
#         print("="*60)
        
#         # Count parameters
#         total_params = 0
#         trainable_params = 0
#         frozen_params = 0
        
#         # Analyze different components
#         components = {
#             "Full Model": self.model,
#             "DiT (trainable)": getattr(self.model, 'dit', None),
#             "VAE (frozen)": getattr(self.model, 'vae', None),
#         }
        
#         print(f"{'Component':<20} {'Total Params':<12} {'Trainable':<12} {'Frozen':<12} {'Status':<10}")
#         print("-" * 70)
        
#         for name, component in components.items():
#             if component is None:
#                 continue
                
#             comp_total = sum(p.numel() for p in component.parameters())
#             comp_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
#             comp_frozen = comp_total - comp_trainable
            
#             status = "TRAIN" if comp_trainable > 0 else "FROZEN"
#             if name == "Full Model":
#                 total_params = comp_total
#                 trainable_params = comp_trainable
#                 frozen_params = comp_frozen
            
#             print(f"{name:<20} {comp_total/1e6:>8.2f}M {comp_trainable/1e6:>8.2f}M {comp_frozen/1e6:>8.2f}M {status:<10}")
        
#         print("-" * 70)
#         print(f"{'TOTAL':<20} {total_params/1e6:>8.2f}M {trainable_params/1e6:>8.2f}M {frozen_params/1e6:>8.2f}M")
        
#         # Memory calculations (rough estimates)
#         print(f"\nGPU MEMORY REQUIREMENTS (Estimates)")
#         print("-" * 40)
        
#         # Model weights (fp16 for inference, fp32 for training)
#         model_memory_fp32 = total_params * 4 / (1024**3)  # 4 bytes per fp32 parameter
#         model_memory_fp16 = total_params * 2 / (1024**3)  # 2 bytes per fp16 parameter
        
#         # Gradients (only for trainable params, fp32)
#         grad_memory = trainable_params * 4 / (1024**3)
        
#         # Optimizer states (Adam: 2 states per param)
#         optimizer_memory = trainable_params * 2 * 4 / (1024**3)
        
#         # EMA model (if used)
#         ema_memory = 0
#         if self.args.use_ema:
#             ema_memory = total_params * 4 / (1024**3)
        
#         print(f"Model weights (FP32):     {model_memory_fp32:.2f} GB")
#         print(f"Model weights (FP16):     {model_memory_fp16:.2f} GB")
#         print(f"Gradients:                {grad_memory:.2f} GB")
#         print(f"Optimizer states:         {optimizer_memory:.2f} GB")
#         if ema_memory > 0:
#             print(f"EMA model:                {ema_memory:.2f} GB")
        
#         # Total training memory
#         total_training_memory = model_memory_fp32 + grad_memory + optimizer_memory + ema_memory
#         total_inference_memory = model_memory_fp16
        
#         print(f"{'─' * 40}")
#         print(f"Est. training memory:     {total_training_memory:.2f} GB")
#         print(f"Est. inference memory:    {total_inference_memory:.2f} GB")
        
#         # Additional info
#         print(f"\nTRAINING CONFIGURATION")
#         print("-" * 25)
#         print(f"Batch size:               {self.args.bs}")
#         print(f"Mixed precision:          {self.device.type == 'cuda'}")
#         print(f"EMA enabled:              {self.args.use_ema}")
#         print(f"Gradient checkpointing:   {getattr(self.args, 'gradient_checkpointing', False)}")
        
#         # Activation memory (very rough estimate)
#         if hasattr(self.args, 'bs'):
#             # Rough estimate: batch_size * channels * height * width * layers * bytes_per_element
#             latent_h = getattr(self.args, 'image_h', 256) // 8
#             latent_w = getattr(self.args, 'image_w', 512) // 8
#             activation_memory = self.args.bs * 4 * latent_h * latent_w * 8 * 4 / (1024**3)  # Very rough
#             print(f"Est. activations:         ~{activation_memory:.2f} GB")
            
#             total_with_activations = total_training_memory + activation_memory
#             print(f"{'─' * 40}")
#             print(f"Est. total GPU usage:     ~{total_with_activations:.2f} GB")
        
#         # GPU info if available
#         if torch.cuda.is_available():
#             gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
#             print(f"\nAvailable GPU memory:     {gpu_memory:.2f} GB")
            
#             if hasattr(self.args, 'bs'):
#                 memory_usage_percent = (total_with_activations / gpu_memory) * 100
#                 print(f"Estimated usage:          {memory_usage_percent:.1f}%")
                
#                 if memory_usage_percent > 90:
#                     print("⚠️  WARNING: High memory usage! Consider reducing batch size.")
#                 elif memory_usage_percent > 100:
#                     print("❌ ERROR: Estimated memory exceeds GPU capacity!")
        
#         print("="*60 + "\n")    

#     # ----------------------------
#     # Main loop
#     # ----------------------------

#     def train(self):
#         start_msg = f"Starting training for {self.args.train_steps} steps..."
#         if self.start_step > 0:
#             start_msg += f" (resuming from step {self.start_step})"
#         print(start_msg)
#         print(f"Logging every {self.args.log_every} steps")
#         print(f"Grad accumulation steps: {self.grad_accum_steps} (effective batch = bs * accum)")
#         print("Visualization enabled" if getattr(self.args, "enable_visualization", False) else "Visualization disabled")
        
#         # Debug: Test data loading before starting the main loop
#         print("🔍 Testing data loading before starting training...")
#         try:
#             print("  - Attempting to get first batch from train_iter...")
#             test_batch = next(self.train_iter)
#             print(f"  ✅ Successfully got batch: {type(test_batch)}")
#             if isinstance(test_batch, (list, tuple)) and len(test_batch) >= 2:
#                 print(f"  - Batch shapes: mask={test_batch[0].shape}, img={test_batch[1].shape}")
#         except Exception as e:
#             print(f"  ❌ Failed to get first batch: {e}")
#             print("  💡 This is likely a DataLoader configuration issue")
#             import traceback
#             traceback.print_exc()
#             return

#         print("🚀 Starting main training loop...")
        
#         step_count = 0
#         while self.current_step < self.args.train_steps:
#             if step_count == 0:
#                 print(f"  - About to call training_step() for step {self.current_step}")
            
#             try:
#                 self.training_step()
#                 if step_count == 0:
#                     print(f"  ✅ First training_step() completed successfully")
#             except Exception as e:
#                 print(f"❌ Error in training_step(): {e}")
#                 import traceback
#                 traceback.print_exc()
#                 break
                
#             self.log_metrics(self.current_step)
#             self.visualize(self.current_step)
#             self.save_checkpoint(self.current_step)
            
#             step_count += 1

#         self.save_final_checkpoint()
#         print("Training completed!")

#     def skip_visualization_if_stuck(self):
#         """
#         Emergency method to disable visualization if DataLoader keeps hanging.
#         Call this to continue training without visualization.
#         """
#         if hasattr(self.args, 'enable_visualization'):
#             self.args.enable_visualization = False
#             print("🚫 Visualization disabled due to DataLoader issues")
#             print("   Training will continue without visualization")

#     def debug_val_loader(self):
#         """
#         Comprehensive debugging for validation DataLoader issues.
#         This will help identify why the DataLoader is hanging/timing out.
#         """
#         print("\n🔍 DEBUGGING VALIDATION DATALOADER")
#         print("="*50)
        
#         if self.val_loader is None:
#             print("❌ val_loader is None!")
#             return
            
#         # Check DataLoader properties
#         print(f"DataLoader info:")
#         print(f"  - Type: {type(self.val_loader)}")
#         print(f"  - Dataset size: {len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') else 'Unknown'}")
#         print(f"  - Batch size: {getattr(self.val_loader, 'batch_size', 'Unknown')}")
#         print(f"  - Num workers: {getattr(self.val_loader, 'num_workers', 'Unknown')}")
#         print(f"  - Pin memory: {getattr(self.val_loader, 'pin_memory', 'Unknown')}")
#         print(f"  - Persistent workers: {getattr(self.val_loader, 'persistent_workers', 'Unknown')}")
        
#         # Test dataset directly (bypass DataLoader)
#         try:
#             print("\n🧪 Testing dataset directly (bypassing DataLoader)...")
#             dataset = self.val_loader.dataset
#             print(f"  - Dataset type: {type(dataset)}")
#             print(f"  - Dataset length: {len(dataset)}")
            
#             # Try to get one sample directly from dataset
#             print("  - Attempting to get sample 0 from dataset...")
#             sample = dataset[0]
#             print(f"  ✅ Direct dataset access works!")
#             if isinstance(sample, (list, tuple)):
#                 print(f"  - Sample shapes: {[s.shape if hasattr(s, 'shape') else type(s) for s in sample]}")
#             else:
#                 print(f"  - Sample type: {type(sample)}")
                
#         except Exception as e:
#             print(f"  ❌ Direct dataset access failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return
            
#         # Test simple DataLoader iteration
#         try:
#             print("\n🔄 Testing DataLoader with timeout...")
            
#             # Create a simple test loader with minimal config
#             from torch.utils.data import DataLoader
#             test_loader = DataLoader(
#                 dataset, 
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=0,  # Use main process only
#                 pin_memory=False,
#                 persistent_workers=False
#             )
            
#             print("  - Created test DataLoader (num_workers=0)")
            
#             import signal
#             def timeout_handler(signum, frame):
#                 raise TimeoutError("Test DataLoader timeout")
                
#             signal.signal(signal.SIGALRM, timeout_handler)
#             signal.alarm(5)  # 5 second timeout
            
#             test_batch = next(iter(test_loader))
#             signal.alarm(0)
            
#             print("  ✅ Test DataLoader works!")
#             if isinstance(test_batch, (list, tuple)):
#                 print(f"  - Test batch shapes: {[b.shape if hasattr(b, 'shape') else type(b) for b in test_batch]}")
                
#         except TimeoutError:
#             signal.alarm(0)
#             print("  ❌ Test DataLoader also times out - dataset issue!")
#         except Exception as e:
#             signal.alarm(0)
#             print(f"  ❌ Test DataLoader failed: {e}")
            
#         # Check GPU memory
#         if torch.cuda.is_available():
#             print(f"\n💾 GPU Memory status:")
#             print(f"  - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
#             print(f"  - Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
#             print(f"  - Total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
            
#         # Check if it's an iterator exhaustion issue
#         try:
#             print(f"\n🔁 Testing iterator state...")
#             # Check if the iterator still has data
#             if hasattr(self.val_loader, '__iter__'):
#                 print("  - val_loader is iterable")
#             if hasattr(self.val_loader, '__next__'):
#                 print("  - val_loader is an iterator (might be exhausted)")
#             else:
#                 print("  - val_loader is not an iterator")
                
#         except Exception as e:
#             print(f"  - Iterator test failed: {e}")
            
#         print("="*50)
#         print("🔍 DEBUGGING COMPLETE\n")

#     def test_val_loader_manually(self):
#         """
#         Quick test method you can call to debug val_loader issues.
#         Run this in a Python shell or add a call in your training script.
#         """
#         print("🔧 MANUAL VALIDATION LOADER TEST")
#         print("-" * 40)
        
#         if self.val_loader is None:
#             print("❌ val_loader is None - check training script setup")
#             return False
            
#         try:
#             # Try with timeout
#             import signal
#             def timeout_handler(signum, frame):
#                 raise TimeoutError("Manual test timeout")
                
#             signal.signal(signal.SIGALRM, timeout_handler)
#             signal.alarm(3)  # 3 second timeout
            
#             print("Attempting next(val_loader)...")
#             batch = next(self.val_loader)
#             signal.alarm(0)
            
#             print("✅ SUCCESS! Val loader works")
#             print(f"Batch type: {type(batch)}")
#             if isinstance(batch, (list, tuple)):
#                 print(f"Batch shapes: {[b.shape if hasattr(b, 'shape') else type(b) for b in batch]}")
#             return True
            
#         except TimeoutError:
#             signal.alarm(0)
#             print("❌ TIMEOUT - val_loader is hanging")
#             print("Running full diagnostics...")
#             self.debug_val_loader()
#             return False
            
#         except Exception as e:
#             signal.alarm(0)
#             print(f"❌ ERROR: {e}")
#             return False
    
#     def set_start_step(self, start_step: int):
#         """Set the starting step for resuming training"""
#         self.start_step = start_step
#         self.current_step = start_step
#         if self.global_rank == 0:
#             print(f"📍 Set starting step to: {start_step}")
            
#     def debug_mask_conditioning(self, mask_onehot_val):
#         """
#         Debugging for mask conditioning consistency between training and inference.
#         Check if the mask is one-hot encoded and has the expected properties.
#         """
#         print(f"🔍 MASK CONDITIONING DEBUG:")
#         print(f"  - mask_onehot_val shape: {mask_onehot_val.shape}")
#         print(f"  - mask_onehot_val dtype: {mask_onehot_val.dtype}")
#         print(f"  - mask_onehot_val device: {mask_onehot_val.device}")
#         print(f"  - mask_onehot_val range: [{mask_onehot_val.min().item():.3f}, {mask_onehot_val.max().item():.3f}]")
        
#         # Check if mask is one-hot encoded
#         mask_sums = mask_onehot_val.sum(dim=1)  # Sum over channel dimension
#         print(f"  - mask channel sums: min={mask_sums.min().item():.3f}, max={mask_sums.max().item():.3f}")
        
#         # Check unique values in each channel
#         unique_vals = torch.unique(mask_onehot_val)
#         print(f"  - unique values in mask: {unique_vals.tolist()}")
        
#         # Check mask statistics
#         non_zero_channels = (mask_onehot_val.sum(dim=(2,3)) > 0).sum(dim=1)
#         print(f"  - non-zero channels per sample: {non_zero_channels.tolist()}")
    
#     # ----------------------------
#     # Inference (separate from training)
#     # ----------------------------

#     def inference(self, checkpoint_path: str, num_samples: int = 8, save_dir: str = "inference_results"):
#         """
#         Load a checkpoint and run inference to generate samples using validation masks.
        
#         Args:
#             checkpoint_path: Path to the checkpoint file
#             num_samples: Number of samples to generate
#             save_dir: Directory to save inference results
#         """
#         print(f"🚀 STARTING INFERENCE")
#         print(f"Checkpoint: {checkpoint_path}")
#         print(f"Samples to generate: {num_samples}")
#         print(f"Save directory: {save_dir}")
        
#         # Create save directory
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Load checkpoint
#         if not os.path.exists(checkpoint_path):
#             print(f"❌ Checkpoint not found: {checkpoint_path}")
#             return
            
#         print(f"📦 Loading checkpoint...")
#         try:
#             checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
#             # Load model state
#             if hasattr(self.model, 'module'):
#                 # DDP wrapped model
#                 missing_keys, unexpected_keys = self.model.module.load_state_dict(
#                     checkpoint['model_state_dict'], strict=False
#                 )
#             else:
#                 # Regular model
#                 missing_keys, unexpected_keys = self.model.load_state_dict(
#                     checkpoint['model_state_dict'], strict=False
#                 )
                
#             if missing_keys:
#                 print(f"⚠️  Missing keys: {len(missing_keys)}")
#             if unexpected_keys:
#                 print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                
#             # Load EMA if available
#             if self.ema_model is not None and 'ema_state_dict' in checkpoint:
#                 try:
#                     self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
#                     print("✅ EMA model loaded")
#                 except Exception as e:
#                     print(f"⚠️  Could not load EMA: {e}")
                    
#             step = checkpoint.get('step', 0)
#             print(f"✅ Checkpoint loaded successfully (step {step})")
            
#         except Exception as e:
#             print(f"❌ Error loading checkpoint: {e}")
#             return
        
#         # Set model to eval mode
#         self.model.eval()
        
#         # Get the sampling model (prefer EMA if available)
#         if self.ema_model is not None:
#             ema_model = self.ema_model.ema_model
#             if hasattr(ema_model, 'module'):
#                 sampling_model = ema_model.module
#                 print("🎯 Using EMA model (DDP wrapped) for inference")
#             else:
#                 sampling_model = ema_model
#                 print("🎯 Using EMA model for inference")
#         else:
#             if hasattr(self.model, 'module'):
#                 sampling_model = self.model.module
#                 print("🎯 Using main model (DDP wrapped) for inference")
#             else:
#                 sampling_model = self.model
#                 print("🎯 Using main model for inference")
        
#         sampling_model.eval()
        
#         # Check validation loader
#         if self.val_loader is None:
#             print("❌ No validation loader available for inference")
#             return
            
#         print(f"📊 Starting inference generation...")
        
#         with torch.no_grad():
#             # Get validation data
#             if hasattr(self.val_loader, '__next__'):
#                 # It's an iterator
#                 mask_onehot_val, img_val = next(self.val_loader)
#             else:
#                 # It's a regular DataLoader
#                 val_iter = iter(self.val_loader)
#                 mask_onehot_val, img_val = next(val_iter)
            
#             # Limit to requested number of samples
#             batch_size = min(num_samples, mask_onehot_val.shape[0])
#             mask_onehot_val = mask_onehot_val[:batch_size]
#             img_val = img_val[:batch_size]
            
#             mask_onehot_val = mask_onehot_val.to(self.device, non_blocking=True)
#             img_val = img_val.to(self.device, non_blocking=True)
            
#             print(f"  - Batch size: {batch_size}")
#             print(f"  - Mask shape: {mask_onehot_val.shape}")
#             print(f"  - Image shape: {img_val.shape}")
            
#             # Handle RGB vs latent input
#             if img_val.shape[1] == 3:
#                 print("  - Input is RGB, encoding to latents...")
#                 if hasattr(self.model, 'module'):
#                     img_latents = self.model.module.encode_image_to_latent(img_val)
#                 else:
#                     img_latents = self.model.encode_image_to_latent(img_val)
#                 latent_hw = (img_latents.shape[2], img_latents.shape[3])
#                 print(f"  - Latent shape: {img_latents.shape}")
#             else:
#                 print("  - Input is already latents")
#                 img_latents = img_val
#                 latent_hw = (img_val.shape[2], img_val.shape[3])
            
#             # Generate samples
#             print("  - Generating samples...")
#             start_time = time.time()
            
#             z_gen = self.gen_model.sample(
#                 model=sampling_model,
#                 shape=(batch_size, 4, latent_hw[0], latent_hw[1]),
#                 cond=mask_onehot_val,
#                 device=self.device,
#             )
            
#             generation_time = time.time() - start_time
#             print(f"  - Generation completed in {generation_time:.2f}s")
            
#             # Decode to images
#             print("  - Decoding latents to images...")
#             if hasattr(self.model, 'module'):
#                 img_generated = self.model.module.decode_latent_to_image(z_gen)
#             else:
#                 img_generated = self.model.decode_latent_to_image(z_gen)
            
#             # Save results
#             print(f"  - Saving results to {save_dir}...")
            
#             for i in range(batch_size):
#                 # Convert tensors to PIL images
#                 generated_pil = self._tensor_to_pil(img_generated[i])
#                 ground_truth_pil = self._tensor_to_pil(img_val[i])
#                 condition_pil = self._mask_to_pil(mask_onehot_val[i])
                
#                 # Save individual images
#                 generated_pil.save(os.path.join(save_dir, f"sample_{i:02d}_generated.png"))
#                 ground_truth_pil.save(os.path.join(save_dir, f"sample_{i:02d}_ground_truth.png"))
#                 condition_pil.save(os.path.join(save_dir, f"sample_{i:02d}_condition.png"))
                
#                 # Create comparison grid (side by side)
#                 grid_width = condition_pil.width + generated_pil.width + ground_truth_pil.width
#                 grid_height = max(condition_pil.height, generated_pil.height, ground_truth_pil.height)
                
#                 from PIL import Image
#                 grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
                
#                 # Paste images side by side: condition | generated | ground_truth
#                 grid_img.paste(condition_pil, (0, 0))
#                 grid_img.paste(generated_pil, (condition_pil.width, 0))
#                 grid_img.paste(ground_truth_pil, (condition_pil.width + generated_pil.width, 0))
                
#                 grid_img.save(os.path.join(save_dir, f"sample_{i:02d}_comparison.png"))
            
#             # Create summary info
#             summary_file = os.path.join(save_dir, "inference_summary.txt")
#             with open(summary_file, 'w') as f:
#                 f.write(f"INFERENCE SUMMARY\n")
#                 f.write(f"================\n")
#                 f.write(f"Checkpoint: {checkpoint_path}\n")
#                 f.write(f"Step: {step}\n")
#                 f.write(f"Samples generated: {batch_size}\n")
#                 f.write(f"Generation time: {generation_time:.2f}s\n")
#                 f.write(f"Time per sample: {generation_time/batch_size:.2f}s\n")
#                 f.write(f"Model: {'EMA' if self.ema_model is not None else 'Main'}\n")
#                 f.write(f"Device: {self.device}\n")
#                 f.write(f"Image size: {img_val.shape[2:]}\n")
#                 f.write(f"Latent size: {latent_hw}\n")
                
#             print(f"✅ Inference completed!")
#             print(f"   Generated {batch_size} samples")
#             print(f"   Saved to: {save_dir}")
#             print(f"   Files: sample_XX_generated.png, sample_XX_ground_truth.png, sample_XX_condition.png")
#             print(f"   Comparisons: sample_XX_comparison.png")
#             print(f"   Summary: inference_summary.txt")




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