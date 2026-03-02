
"""
train_city_modular.py

DDP + EMA + Trainer-compatible main script.

Key fixes vs your script:
1) DO NOT override device after setup_ddp(). Always use cuda:{local_rank}.
2) If you flip args.enable_visualization later (overfit mode), you MUST build val_loader after that decision.
3) Optimizer should ONLY include trainable params (exclude frozen VAE).
4) For DDP, call train_sampler.set_epoch(epoch) so each rank shuffles differently across epochs.
   (We emulate "epoch" from step count since you're running by steps.)
5) Resume loads into trainer.raw_model (NOT DDP wrapper). Trainer checkpoints already store raw_model weights.
6) --use_ema argument: use store_true with default False (your current definition is inconsistent).

This script assumes:
- dataset yields (mask_onehot_img, img)
- model is MaskCondDiTWithVAE (VAE frozen inside the model)
- trainer is the fixed CityscapesTrainer I sent (raw_model + DDP wrapper + EMA on raw_model)
"""

import argparse
import glob
import os
import random
import re
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

from dataset.datasets import CityscapesDataset
from diffusion.Rectified_flow_matching import FlowMatchingConfig, FlowMatching
from diffusion.guassian_diffusion import DiffusionConfig, GaussianDiffusion
from models.vae_dit import MaskCondDiTWithVAE, TinyDiTLatent
from trainer import CityscapesTrainer


def set_global_seed(seed: int, global_rank: int = 0):
    seed = seed + global_rank  # ensure different streams per rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seed] Using seed {seed} on rank {global_rank}")

# -------------------------
# DDP helpers
# -------------------------
def setup_ddp() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        global_rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        if global_rank == 0:
            print(f"[DDP] init ok: rank {global_rank}/{world_size}, local_rank {local_rank}")
        return local_rank, global_rank, world_size

    # single process
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(global_rank: int) -> bool:
    return global_rank == 0


# -------------------------
# misc helpers
# -------------------------
def _make_fixed_palette_19() -> np.ndarray:
    rng = np.random.default_rng(0)
    pal = rng.integers(low=0, high=255, size=(19, 3), dtype=np.uint8)
    pal[0] = np.array([128, 64, 128], dtype=np.uint8)
    pal[10] = np.array([70, 130, 180], dtype=np.uint8)
    pal[13] = np.array([0, 0, 142], dtype=np.uint8)
    pal[11] = np.array([220, 20, 60], dtype=np.uint8)
    return pal


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    if not os.path.exists(checkpoint_dir):
        return ""
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    if not ckpt_files:
        return ""
    best_step = -1
    best_path = ""
    for p in ckpt_files:
        m = re.search(r"checkpoint_step_(\d+)\.pt", os.path.basename(p))
        if m:
            s = int(m.group(1))
            if s > best_step:
                best_step = s
                best_path = p
    return best_path


def determine_resume_checkpoint(args, global_rank: int) -> str:
    if args.resume_from:
        if is_rank0(global_rank):
            print(f"📁 resume_from: {args.resume_from}")
        return args.resume_from

    if args.resume:
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if is_rank0(global_rank):
            if latest:
                print(f"📁 resume latest: {latest}")
            else:
                print("⚠️  resume requested but no checkpoint found.")
        return latest

    return ""


def load_checkpoint_into_trainer(checkpoint_path: str, trainer: CityscapesTrainer, global_rank: int) -> int:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0

    if is_rank0(global_rank):
        print(f"🔄 Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # trainer.raw_model is the source of truth for weights
    missing, unexpected = trainer.raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if is_rank0(global_rank):
        if missing:
            print(f"⚠️  missing keys: {len(missing)} (showing first 5)")
            for k in missing[:5]:
                print("  -", k)
        if unexpected:
            print(f"⚠️  unexpected keys: {len(unexpected)} (showing first 5)")
            for k in unexpected[:5]:
                print("  -", k)

    # optimizer (optional)
    if "optimizer_state_dict" in ckpt:
        try:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if is_rank0(global_rank):
                print("✅ optimizer state loaded")
        except Exception as e:
            if is_rank0(global_rank):
                print(f"⚠️  optimizer state not loaded: {e}")

    # scaler (optional)
    if trainer.scaler is not None and "scaler_state_dict" in ckpt:
        try:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
            if is_rank0(global_rank):
                print("✅ scaler state loaded")
        except Exception as e:
            if is_rank0(global_rank):
                print(f"⚠️  scaler state not loaded: {e}")

    # EMA (optional)
    if trainer.ema_model is not None and "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        try:
            trainer.ema_model.load_state_dict(ckpt["ema_state_dict"])
            if is_rank0(global_rank):
                print("✅ EMA state loaded")
        except Exception as e:
            if is_rank0(global_rank):
                print(f"⚠️  EMA state not loaded: {e}")

    start_step = int(ckpt.get("step", 0))
    if is_rank0(global_rank):
        print(f"✅ resume step = {start_step}")
    return start_step


def set_optimizer_lr(optimizer, lr: float, global_rank: int):
    for g in optimizer.param_groups:
        g["lr"] = lr
    if is_rank0(global_rank):
        print(f"🔄 LR set to {lr:.2e}")


# -------------------------
# main
# -------------------------
def main(args):
    
    local_rank, global_rank, world_size = setup_ddp()
    set_global_seed(args.seed , global_rank=global_rank)


    # IMPORTANT: device must be cuda:{local_rank} in DDP
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_rank0(global_rank):
        print("=== TRAINING SETUP ===")
        print(f"world_size: {world_size}")
        print(f"global_rank: {global_rank}")
        print(f"local_rank:  {local_rank}")
        print(f"device:      {device}")
        print("======================")

    # seed
    torch.manual_seed(args.seed + global_rank)
    np.random.seed(args.seed + global_rank)
    random.seed(args.seed + global_rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + global_rank)
    
    # For better reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # experiment dirs
    exp_name = f"{args.dataset_name}_modular_{int(time.time())}"
    exp_log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if is_rank0(global_rank):
        print(f"Experiment: {exp_name}")
        print(f"Log dir:    {exp_log_dir}")
        print(f"CKPT dir:   {args.checkpoint_dir}")

    # -------------------------
    # dataset
    # -------------------------
    dataset = CityscapesDataset(
        root=args.dataset_path,
        split="train",
        image_size=(args.image_h, args.image_w),
        num_classes=19,
        augment=True,
    )

    if args.overfit_test:
        n = min(args.overfit_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
        if is_rank0(global_rank):
            print(f"🔬 OVERFIT mode: using {n} samples")

        # In overfit mode, force visualization ON early (before creating loaders)
        args.enable_visualization = True
        args.visualize_every = min(args.visualize_every, 50)
        args.log_every = min(args.log_every, 10)

    val_size = min(args.val_size, max(1, len(dataset) // 5))
    if args.overfit_test:
        val_size = min(2, max(1, len(dataset) // 2))

    rest_size = len(dataset) - val_size
    val_dataset, train_dataset = random_split(dataset, [val_size, rest_size])

    # DDP sampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    # val loader only if visualization enabled
    val_loader = None
    if args.enable_visualization:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.bs, args.visualize_num),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
        )

    if is_rank0(global_rank):
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples:   {len(val_dataset)}")
        # quick sanity check
        m, x = next(iter(train_loader))
        print("Sanity batch:", m.shape, x.shape)

    # -------------------------
    # model
    # -------------------------
    latent_h = args.image_h // 8
    latent_w = args.image_w // 8
    latent_hw = (latent_h, latent_w)

    model = MaskCondDiTWithVAE(
        dit_latent=TinyDiTLatent(
            model_dim=args.model_dim,
            n_layers=args.n_dit_layers,
            patch_size=args.latent_patch_size,
            latent_channels=4,
            cond_channels=19,
            latent_hw=latent_hw,
            n_heads=args.n_attn_heads,
            mlp_dim=args.feed_fwd_dim,
            n_adaln_cond_cls=None,
        ),
        vae_name=args.vae_name,
        freeze_vae=True,  # keep VAE frozen
    ).to(device)

    if is_rank0(global_rank):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model params: total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M")

    # Optimizer: ONLY trainable params
    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(args.adam_b1, args.adam_b2),
        weight_decay=args.weight_decay,
    )

    # -------------------------
    # gen model config
    # -------------------------
    if args.generative_model == "flow_matching":
        cfg = FlowMatchingConfig(
            pred="v",
            loss_space="v",
            t_sampling="uniform",
            mse_reduction="mean",
        )
        gen_model_cls = FlowMatching
    elif args.generative_model == "diffusion":
        cfg = DiffusionConfig()
        gen_model_cls = GaussianDiffusion
    else:
        raise ValueError(f"Unknown generative model: {args.generative_model}")

    # -------------------------
    # W&B
    # -------------------------
    if args.use_wandb and is_rank0(global_rank):
        try:
            if args.wandb_offline:
                os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity if args.wandb_entity else None,
                config=vars(args),
                name=exp_name,
                dir=exp_log_dir,
            )
            print("✅ W&B initialized")
        except Exception as e:
            print(f"⚠️  W&B init failed: {e}")

    palette_19 = _make_fixed_palette_19()

    # -------------------------
    # trainer
    # -------------------------
    trainer = CityscapesTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        local_rank=local_rank,
        global_rank=global_rank,
        config=cfg,
        gen_model=gen_model_cls,
        args=args,
        exp_log_dir=exp_log_dir,
        palette_19=palette_19,
        device=device,
    )

    # Resume
    ckpt_path = determine_resume_checkpoint(args, global_rank)
    if ckpt_path:
        start_step = load_checkpoint_into_trainer(ckpt_path, trainer, global_rank)
        if start_step > 0:
            trainer.set_start_step(start_step)
            set_optimizer_lr(trainer.optimizer, args.lr, global_rank)

    # -------------------------
    # train loop
    # -------------------------
    try:
        # If DDP, ensure sampler epoch changes so each rank reshuffles across time.
        # We don't have epochs; approximate epoch from step // steps_per_epoch.
        steps_per_epoch = max(1, len(train_loader))

        while trainer.current_step < args.train_steps:
            # Set sampler epoch occasionally
            if world_size > 1 and train_sampler is not None:
                pseudo_epoch = trainer.current_step // steps_per_epoch
                train_sampler.set_epoch(pseudo_epoch)

            trainer.training_step()
            trainer.log_metrics(trainer.current_step)
            trainer.visualize(trainer.current_step)
            trainer.save_checkpoint(trainer.current_step)

        trainer.save_final_checkpoint()
        if is_rank0(global_rank):
            print("Training completed!")

    finally:
        if is_rank0(global_rank) and wandb.run is not None:
            wandb.finish()
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset_name", type=str, default="cityscapes")
    parser.add_argument("--dataset_path", type=str, default="/media/mehrnoosh/MEHRNOOSH2/citydata")
    parser.add_argument("--generative_model", type=str, default="flow_matching", choices=["flow_matching", "diffusion"])
    parser.add_argument("--image_h", type=int, default=128)
    parser.add_argument("--image_w", type=int, default=256)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--overfit_test", action="store_true")
    parser.add_argument("--overfit_samples", type=int, default=10)

    # model
    parser.add_argument("--model_dim", type=int, default=672)
    parser.add_argument("--n_dit_layers", type=int, default=12)
    parser.add_argument("--latent_patch_size", type=int, default=2)
    parser.add_argument("--n_attn_heads", type=int, default=12)
    parser.add_argument("--feed_fwd_dim", type=int, default=2688)

    # VAE
    parser.add_argument("--vae_name", type=str, default="stabilityai/sd-vae-ft-ema")

    # training
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam_b1", type=float, default=0.9)
    parser.add_argument("--adam_b2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    # EMA (FIXED ARGPARSE)
    parser.add_argument("--use_ema", action="store_true", help="Enable EMA (recommended)")

    # logging / viz / wandb
    parser.add_argument("--log_dir", type=str, default="out")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--enable_visualization", action="store_true")
    parser.add_argument("--visualize_first_step", action="store_true")
    parser.add_argument("--visualize_every", type=int, default=1000)
    parser.add_argument("--visualize_num", type=int, default=2)
    parser.add_argument("--save_visualization_images", action="store_true", default=True)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_project_name", type=str, default="Cityscapes_Latent_TinyDiT")

    # checkpoint
    parser.add_argument("--ckpt_every", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="ckpts")
    parser.add_argument("--keep_last_k", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default="")

    args = parser.parse_args()
    main(args)