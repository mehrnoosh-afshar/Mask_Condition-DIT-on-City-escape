conda activate cityscapes-dit-latent
python -m torch.distributed.run --nproc_per_node=2 --master_port=29400 train.py --dataset_path /home/mehrnoosh/Documents/workspace/citydata --bs 256