#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python uwfqa.py \
--dataroot /TEST/IMAGES/DIR/ \
--model_dir '/TRAINED/MODEL/DIR/' \
--model_name 'best_model.pth' \
--save_dir 'SAVE/DIR/OF/ENHANCED/IMAGES/' \

CUDA_VISIBLE_DEVICES=0 python fiqa.py --save_dir original/image/path
CUDA_VISIBLE_DEVICES=0 python fiqa.py --save_dir enhanced/image/path