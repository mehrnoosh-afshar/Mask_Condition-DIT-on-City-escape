#!/usr/bin/env python3
"""
Analyze timestep embedding behavior in flow matching.

This script checks whether the timestep embedding becomes "almost constant" 
during flow matching training, as mentioned in the issue.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import your model components
import sys
sys.path.append('/home/mehrnoosh/Documents/workspace/tiny-models/pipelines/flow')
from models.vae_dit import TimestepEmbedder
from diffusion.Rectified_flow_matching import FlowMatchingConfig, FlowMatching

def analyze_timestep_embedding():
    """Analyze the timestep embedding behavior for flow matching"""
    
    print("🔍 ANALYZING TIMESTEP EMBEDDING FOR FLOW MATCHING")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create timestep embedder (same as in your model)
    embed_dim = 768  # typical DiT dimension
    t_embedder = TimestepEmbedder(embed_dim).to(device)
    
    # Flow matching config - uniform sampling [0,1]
    fm_config = FlowMatchingConfig(
        pred="v",
        loss_space="v", 
        t_sampling="uniform"  # This is key!
    )
    
    flow_matcher = FlowMatching(fm_config, device)
    
    print(f"📏 Timestep embedding dimension: {embed_dim}")
    print(f"⚙️  Flow matching config: {fm_config.t_sampling} sampling")
    print()
    
    # Analyze timestep distribution during training
    batch_sizes = [1, 8, 16, 32]
    num_samples = 1000
    
    all_t_values = []
    all_embeddings = []
    
    print("📊 SAMPLING TIMESTEPS AND EMBEDDINGS")
    print("-" * 40)
    
    with torch.no_grad():
        for batch_size in batch_sizes:
            print(f"Batch size: {batch_size}")
            
            batch_t_values = []
            batch_embeddings = []
            
            for _ in range(num_samples // batch_size):
                # Sample timesteps as done during training
                t = flow_matcher.sample_t(batch_size)
                
                # Get timestep embeddings
                t_emb = t_embedder(t)
                
                batch_t_values.append(t.cpu())
                batch_embeddings.append(t_emb.cpu())
            
            # Concatenate all batches
            batch_t_all = torch.cat(batch_t_values, dim=0)
            batch_emb_all = torch.cat(batch_embeddings, dim=0)
            
            all_t_values.extend(batch_t_all.numpy())
            all_embeddings.append(batch_emb_all)
            
            print(f"  t range: [{batch_t_all.min():.3f}, {batch_t_all.max():.3f}]")
            print(f"  t mean: {batch_t_all.mean():.3f} ± {batch_t_all.std():.3f}")
    
    # Combine all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_t_values = np.array(all_t_values)
    
    print(f"\n📈 OVERALL STATISTICS ({len(all_t_values)} samples)")
    print("-" * 40)
    print(f"t distribution:")
    print(f"  Mean: {all_t_values.mean():.4f}")
    print(f"  Std:  {all_t_values.std():.4f}")
    print(f"  Min:  {all_t_values.min():.4f}")
    print(f"  Max:  {all_t_values.max():.4f}")
    
    # Analyze embedding variance
    emb_mean = all_embeddings.mean(dim=0)  # [embed_dim]
    emb_std = all_embeddings.std(dim=0)    # [embed_dim]
    
    print(f"\nEmbedding statistics:")
    print(f"  Embedding shape: {all_embeddings.shape}")
    print(f"  Mean magnitude: {emb_mean.norm().item():.4f}")
    print(f"  Average std per dim: {emb_std.mean().item():.4f}")
    print(f"  Max std per dim: {emb_std.max().item():.4f}")
    print(f"  Min std per dim: {emb_std.min().item():.4f}")
    
    # Check if embeddings are "almost constant"
    relative_std = emb_std / (emb_mean.abs() + 1e-8)  # Relative variation
    high_variance_dims = (relative_std > 0.1).sum().item()
    
    print(f"\n🎯 CONSTANCY ANALYSIS")
    print("-" * 40)
    print(f"Dimensions with >10% relative variation: {high_variance_dims}/{embed_dim}")
    print(f"Percentage of varying dimensions: {100 * high_variance_dims / embed_dim:.1f}%")
    
    if high_variance_dims < embed_dim * 0.1:  # Less than 10% varying
        print("⚠️  WARNING: Timestep embedding is mostly CONSTANT!")
        print("   This suggests the model may not be learning meaningful")
        print("   temporal dynamics for flow matching.")
    else:
        print("✅ Timestep embedding shows reasonable variation.")
    
    # Compare with specific timestep values
    print(f"\n🔍 SPECIFIC TIMESTEP ANALYSIS")
    print("-" * 40)
    
    test_timesteps = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=device)
    test_embeddings = []
    
    with torch.no_grad():
        for t_val in test_timesteps:
            t_batch = t_val.repeat(1)
            emb = t_embedder(t_batch)
            test_embeddings.append(emb[0])
    
    test_embeddings = torch.stack(test_embeddings)  # [7, embed_dim]
    
    print("Timestep -> Embedding norm:")
    for i, t_val in enumerate(test_timesteps):
        emb_norm = test_embeddings[i].norm().item()
        print(f"  t={t_val:.2f} -> ||emb||={emb_norm:.4f}")
    
    # Check embedding differences
    print(f"\nEmbedding differences:")
    for i in range(1, len(test_timesteps)):
        diff = (test_embeddings[i] - test_embeddings[i-1]).norm().item()
        t_diff = test_timesteps[i] - test_timesteps[i-1]
        print(f"  Δt={t_diff:.2f} -> ||Δemb||={diff:.4f}")
    
    # Overall embedding range
    emb_range = (test_embeddings.max() - test_embeddings.min()).item()
    emb_mean_norm = test_embeddings.mean(dim=0).norm().item()
    
    print(f"\nOverall embedding analysis:")
    print(f"  Embedding range: {emb_range:.4f}")
    print(f"  Mean embedding norm: {emb_mean_norm:.4f}")
    print(f"  Range/Mean ratio: {emb_range/emb_mean_norm:.4f}")
    
    if emb_range / emb_mean_norm < 0.1:
        print("⚠️  WARNING: Embedding range is very small compared to mean!")
        print("   The timestep embedding is effectively CONSTANT.")
    
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    print("1. Flow matching uses uniform t ∈ [0,1] sampling")
    print("2. If embeddings are too constant, consider:")
    print("   • Using logit-normal t sampling instead")
    print("   • Increasing embedding dimension") 
    print("   • Adding positional encoding components")
    print("   • Checking if model architecture needs more capacity")
    
    return {
        'timesteps': all_t_values,
        'embeddings': all_embeddings,
        'test_timesteps': test_timesteps.cpu().numpy(),
        'test_embeddings': test_embeddings.cpu().numpy(),
        'high_variance_dims': high_variance_dims,
        'total_dims': embed_dim,
        'embedding_range': emb_range,
        'mean_embedding_norm': emb_mean_norm
    }

if __name__ == "__main__":
    results = analyze_timestep_embedding()
    
    # Print summary
    print(f"\n🎯 SUMMARY")
    print("=" * 60)
    
    varying_pct = 100 * results['high_variance_dims'] / results['total_dims']
    range_ratio = results['embedding_range'] / results['mean_embedding_norm']
    
    print(f"Timestep embedding analysis:")
    print(f"  Varying dimensions: {varying_pct:.1f}%")
    print(f"  Range/Mean ratio: {range_ratio:.4f}")
    
    if varying_pct < 10 or range_ratio < 0.1:
        print("  Status: ⚠️  MOSTLY CONSTANT - Potential issue!")
    else:
        print("  Status: ✅ Reasonable variation")
    
    print(f"\nThe timestep embedding behavior has been analyzed.")
    print(f"Check the output above to see if it's 'almost constant'.")
