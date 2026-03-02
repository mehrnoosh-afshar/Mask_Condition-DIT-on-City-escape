#!/usr/bin/env python3
"""
Overfit Test Checkpoint Location Guide

This script explains where checkpoints are saved in overfit mode vs normal mode.
"""

print("📁 CHECKPOINT SAVING LOCATIONS:")
print("=" * 50)
print()

print("🔷 NORMAL TRAINING MODE:")
print("   Command: python train_city_modular_clean.py")
print("   Checkpoint dir: ckpts/")
print("   Files saved:")
print("   ├── ckpts/")
print("   │   ├── checkpoint_step_001000.pt")
print("   │   ├── checkpoint_step_002000.pt")
print("   │   └── ...")
print()

print("🔬 OVERFIT TEST MODE:")
print("   Command: python train_city_modular_clean.py --overfit_test --overfit_samples 10")
print("   Checkpoint dir: ckpts_overfit/")
print("   Files saved:")
print("   ├── ckpts_overfit/")
print("   │   ├── checkpoint_step_000010.pt")
print("   │   ├── checkpoint_step_000020.pt")
print("   │   └── ...")
print()

print("💡 WHY SEPARATE DIRECTORIES?")
print("   ✅ Keeps overfit test checkpoints separate from main training")
print("   ✅ Prevents accidentally resuming main training with overfit checkpoints")
print("   ✅ Easy to clean up overfit test files")
print("   ✅ Can run both simultaneously without conflicts")
print()

print("🧹 CLEANUP AFTER OVERFIT TEST:")
print("   rm -rf ckpts_overfit/  # Remove overfit checkpoints")
print("   rm -rf out/cityscapes_modular_*/  # Remove overfit logs")
print()

print("📊 EXAMPLE OVERFIT TEST COMMAND:")
print("python train_city_modular_clean.py \\")
print("    --overfit_test \\")
print("    --overfit_samples 10 \\")
print("    --max_steps 500 \\")
print("    --lr 1e-4 \\")
print("    --log_every 10 \\")
print("    --enable_visualization --visualize_every 50")
print()

print("✅ Your overfit checkpoints are now saved separately!")
