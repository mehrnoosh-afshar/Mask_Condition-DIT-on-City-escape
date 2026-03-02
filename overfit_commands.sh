#!/bin/bash
"""
Overfit Test Commands with torchrun

These commands will run an overfit test using only a small subset of data
to verify that your model can memorize and achieve very low loss.
"""

echo "🔬 OVERFIT TEST COMMANDS"
echo "========================"
echo

echo "🚀 BASIC OVERFIT TEST (10 samples):"
echo "torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \\"
echo "    train_city_modular_clean.py \\"
echo "    --overfit_test \\"
echo "    --overfit_samples 10 \\"
echo "    --train_steps 2000 \\"
echo "    --lr 1e-4 \\"
echo "    --enable_visualization --visualize_every 50 \\"
echo "    --log_every 10"
echo

echo "🎯 FAST OVERFIT TEST (5 samples, quick):"
echo "torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 \\"
echo "    train_city_modular_clean.py \\"
echo "    --overfit_test \\"
echo "    --overfit_samples 5 \\"
echo "    --train_steps 1000 \\"
echo "    --lr 2e-4 \\"
echo "    --enable_visualization --visualize_every 25 \\"
echo "    --log_every 5"
echo

echo "🔧 DETAILED OVERFIT TEST (with gradient debugging):"
echo "torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=102 --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 \\"
echo "    train_city_modular_clean.py \\"
echo "    --overfit_test \\"
echo "    --overfit_samples 8 \\"
echo "    --train_steps 1500 \\"
echo "    --lr 1e-4 \\"
echo "    --grad_accum_steps 1 \\"
echo "    --enable_visualization --visualize_every 50 --visualize_num 4 \\"
echo "    --log_every 10 \\"
echo "    --max_grad_norm 1.0"
echo

echo "⚡ AGGRESSIVE OVERFIT TEST (higher learning rate):"
echo "torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=103 --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \\"
echo "    train_city_modular_clean.py \\"
echo "    --overfit_test \\"
echo "    --overfit_samples 10 \\"
echo "    --train_steps 1000 \\"
echo "    --lr 5e-4 \\"
echo "    --enable_visualization --visualize_every 20 \\"
echo "    --log_every 5"
echo

echo "📊 WHAT TO EXPECT:"
echo "==================="
echo "✅ Loss should decrease steadily: 2.0 → 1.0 → 0.5 → 0.1 → 0.01"
echo "✅ Generated images should match ground truth closely"
echo "✅ Checkpoints saved to: ckpts_overfit/"
echo "✅ Visualizations saved to: out/cityscapes_modular_*/visualizations/"
echo

echo "❌ TROUBLESHOOTING:"
echo "==================="
echo "• Loss not decreasing? → Try higher learning rate (5e-4, 1e-3)"
echo "• Loss exploding? → Try lower learning rate (5e-5, 1e-5)"
echo "• Generation quality poor? → Check mask conditioning debug output"
echo "• DataLoader hanging? → Check dataset path and permissions"
echo

echo "💡 COPY-PASTE READY COMMAND:"
echo "============================="
