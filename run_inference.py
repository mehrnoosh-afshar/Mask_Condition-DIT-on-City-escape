#!/usr/bin/env python3
"""
run_inference.py

Standalone script to run inference using a trained checkpoint.
This will load a model checkpoint and generate samples using validation masks.
"""

import argparse
import os
import sys
import torch

# Add the current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vae_dit import TinyDiTLatent, MaskCondDiTWithVAE 
from dataset.datasets import CityscapesDataset
from diffusion.Rectified_flow_matching import FlowMatchingConfig, FlowMatching
from trainer import CityscapesTrainer

from torch.utils.data import DataLoader
import numpy as np

def create_palette_19():
    """Create Cityscapes 19-class color palette."""
    palette = np.array([
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk  
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ], dtype=np.uint8)
    return palette

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained checkpoint")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Path to checkpoint file")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, 
                       default="/media/mehrnoosh/MEHRNOOSH2/citydata", 
                       help="Path to Cityscapes dataset")
    parser.add_argument("--image_h", type=int, default=128, help="Image height")
    parser.add_argument("--image_w", type=int, default=256, help="Image width")
    
    # Model arguments  
    parser.add_argument("--model_dim", type=int, default=384, help="Model dimension")
    parser.add_argument("--n_dit_layers", type=int, default=8, help="Number of DiT layers")
    parser.add_argument("--latent_patch_size", type=int, default=2, help="Patch size in latent space")
    parser.add_argument("--n_attn_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--feed_fwd_dim", type=int, default=1536, help="Feed forward dimension")
    
    # VAE arguments
    parser.add_argument("--vae_name", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="VAE model name")
    parser.add_argument("--latent_scale", type=float, default=0.18215, help="VAE latent scaling factor")
    
    # Inference arguments
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--save_dir", type=str, default="inference_results", help="Directory to save results")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model if available")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Number of sampling steps")
    
    # Flow matching arguments
    parser.add_argument("--generative_model", type=str, default="flow_matching", help="Generative model type")
    
    args = parser.parse_args()
    
    print("🚀 INFERENCE SETUP")
    print("=" * 30)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Image size: {args.image_h}x{args.image_w}")
    print(f"Samples: {args.num_samples}")
    print(f"Save to: {args.save_dir}")
    print()
    
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create validation dataset
    print("📊 Loading validation dataset...")
    val_dataset = CityscapesDataset(
        root=args.dataset_path,
        split="val",  # Use validation split for inference
        image_size=(args.image_h, args.image_w),
        num_classes=19,
        augment=False,  # No augmentation for inference
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.num_samples,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    
    print(f"  - Validation dataset: {len(val_dataset)} samples")
    
    # Create model
    print("🏗️  Creating model...")
    latent_h = args.image_h // 8
    latent_w = args.image_w // 8
    
    model = MaskCondDiTWithVAE(
        dit_latent=TinyDiTLatent(
            model_dim=args.model_dim,
            n_layers=args.n_dit_layers,
            n_heads=args.n_attn_heads,
            feed_fwd_dim=args.feed_fwd_dim,
            patch_size=args.latent_patch_size,
            latent_size=(latent_h, latent_w),
            cond_channels=19,  # Cityscapes classes
        ),
        vae_model_name=args.vae_name,
        latent_scale_factor=args.latent_scale,
        freeze_vae=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params/1e6:.2f}M")
    
    # Create generative model config and instance
    if args.generative_model == "flow_matching":
        config = FlowMatchingConfig()
        gen_model = FlowMatching
    else:
        raise ValueError(f"Unknown generative model: {args.generative_model}")
    
    # Create palette
    palette_19 = create_palette_19()
    
    # Create trainer (we only need it for inference)
    trainer = CityscapesTrainer(
        model=model,
        optimizer=None,  # Not needed for inference
        train_loader=None,  # Not needed for inference
        val_loader=val_loader,
        local_rank=0,
        global_rank=0,
        config=config,
        gen_model=gen_model,
        args=args,
        exp_log_dir="",  # Not needed for inference
        palette_19=palette_19,
        device=device,
    )
    
    # Run inference
    print("🎯 Starting inference...")
    trainer.inference(
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
    
    print("✅ Inference completed!")

if __name__ == "__main__":
    main()
