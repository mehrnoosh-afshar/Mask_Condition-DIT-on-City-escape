#!/usr/bin/env python3
"""
Inference Usage Guide

This guide shows how to use the new inference functionality.
"""

print("🎯 INFERENCE USAGE GUIDE")
print("=" * 50)
print()

print("📋 TWO WAYS TO RUN INFERENCE:")
print("-" * 30)

print("\n1️⃣  STANDALONE SCRIPT (Recommended):")
print("python run_inference.py \\")
print("    --checkpoint_path ckpts/checkpoint_step_055000.pt \\")
print("    --num_samples 8 \\")
print("    --save_dir inference_results")
print()

print("2️⃣  INTEGRATED WITH TRAINING SCRIPT:")
print("Add --inference_only flag to training script (to be implemented)")
print()

print("🔧 AVAILABLE OPTIONS:")
print("-" * 20)
print("--checkpoint_path    Path to .pt checkpoint file")
print("--num_samples        Number of samples to generate (default: 8)")
print("--save_dir          Output directory (default: inference_results)")  
print("--dataset_path      Path to Cityscapes dataset")
print("--image_h           Image height (default: 128)")
print("--image_w           Image width (default: 256)")
print("--use_ema           Use EMA model if available")
print("--sampling_steps    Number of sampling steps (default: 50)")
print()

print("📁 OUTPUT FILES:")
print("-" * 15)
print("inference_results/")
print("├── sample_00_generated.png      # Generated image")
print("├── sample_00_ground_truth.png   # Original validation image") 
print("├── sample_00_condition.png      # Condition mask (colored)")
print("├── sample_00_comparison.png     # Side-by-side comparison")
print("├── sample_01_*.png")
print("├── ...")
print("└── inference_summary.txt        # Summary with timing info")
print()

print("✨ EXAMPLE USAGE:")
print("-" * 15)
print("# Generate 16 samples from your best checkpoint")
print("python run_inference.py \\")
print("    --checkpoint_path ckpts/checkpoint_step_060000.pt \\")
print("    --num_samples 16 \\")
print("    --save_dir results_step_60k \\")
print("    --use_ema")
print()

print("# Quick test with 4 samples")  
print("python run_inference.py \\")
print("    --checkpoint_path ckpts/checkpoint_step_055000.pt \\")
print("    --num_samples 4")
print()

print("💡 TIPS:")
print("-" * 8)
print("• Use --use_ema for better quality (if EMA was enabled during training)")
print("• Higher --sampling_steps = better quality but slower")
print("• The script uses validation split for conditioning masks")
print("• Results are saved as PNG files for easy viewing")
print("• Comparison images show: condition | generated | ground_truth")
print()

print("🔍 DEBUGGING:")
print("-" * 12)
print("• Check inference_summary.txt for timing and model info")
print("• Generated images should match the semantic layout of condition masks")
print("• If results are poor, check training loss and conditioning debug output")

print()
print("🚀 Ready to run inference!")
