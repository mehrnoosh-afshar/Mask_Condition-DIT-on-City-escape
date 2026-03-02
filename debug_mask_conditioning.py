#!/usr/bin/env python3
"""
Debug mask conditioning issues in visualization vs training.

This script helps diagnose why visualization looks bad even with low loss.
Common issues:
1. Mask conditioning format differs between train/inference  
2. Mask values are incorrect (not one-hot, wrong range, etc.)
3. Model architecture issues with conditioning
4. Sampling parameters differ from training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def debug_mask_conditioning():
    print("🔍 MASK CONDITIONING DEBUGGING GUIDE")
    print("=" * 50)
    
    print("\n📋 COMMON ISSUES WITH POOR VISUALIZATION:")
    print("1. Mask conditioning format mismatch")
    print("2. Mask preprocessing differences") 
    print("3. Model not using conditioning properly")
    print("4. Sampling parameters")
    print("5. VAE encoding/decoding issues")
    
    print("\n🔍 WHAT TO CHECK IN YOUR OUTPUT:")
    print("-" * 30)
    print("TRAINING MASK:")
    print("  - Shape should be [B, 19, H, W] for Cityscapes")
    print("  - Values should be 0.0 or 1.0 (one-hot)")
    print("  - Channel sums should be 1.0 (each pixel belongs to one class)")
    print("  - Range: [0.000, 1.000]")
    
    print("\nINFERENCE MASK:")  
    print("  - Should match training exactly")
    print("  - Same shape, dtype, device, value range")
    print("  - Same preprocessing pipeline")
    
    print("\n🚨 RED FLAGS:")
    print("  ❌ Channel sums != 1.0 → Not properly one-hot encoded")
    print("  ❌ Values not in [0,1] → Wrong preprocessing")
    print("  ❌ Different shapes → Data loader mismatch")
    print("  ❌ Different dtypes → Type conversion issues")
    
    print("\n💡 DEBUGGING STEPS:")
    print("1. Check the debug output from your training")
    print("2. Compare TRAINING vs INFERENCE mask stats")
    print("3. Look at the 'cond_effect' value in loss - should be > 0.01")
    print("4. Check loss/baseline ratio - should decrease during training")
    
    print("\n🛠️  FIXES:")
    print("- Ensure same mask preprocessing in train/val datasets")
    print("- Check if mask needs resizing for latent space")
    print("- Verify mask is properly one-hot encoded")
    print("- Check model architecture accepts conditioning correctly")
    
    print("\n📊 WHAT GOOD VALUES LOOK LIKE:")
    print("TRAINING/INFERENCE MASK:")
    print("  - Shape: [B, 19, H, W]")
    print("  - Range: [0.000, 1.000]") 
    print("  - Channel sums: min=1.000, max=1.000")
    print("  - Unique values: [0.0, 1.0]")
    print("  - Non-zero channels: varies per sample")
    
    print("\nLOSS METRICS:")
    print("  - cond_effect: > 0.01 (model uses conditioning)")
    print("  - loss/baseline ratio: should decrease < 0.5")
    print("  - loss: should decrease steadily")

if __name__ == "__main__":
    debug_mask_conditioning()
