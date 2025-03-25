CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=4321 basicsr/train.py \
-opt options/train/EDVR/train_EDVR_L_UDC-VIT.yml --launcher pytorch
