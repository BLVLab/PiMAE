#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main_load.py \
--dataset_name sunrgbd \
--max_epoch 1080 \
--nqueries 128 \
--base_lr 7e-4 \
--encoder_learning_rate 3.5e-4 \
--warm_lr_epochs 9 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--save_separate_checkpoint_every_epoch 20 \
--log_every 10 \
--log_metrics_every 10 \
--eval_every_epoch 25 \
--checkpoint_dir outputs/pimae_sunrgbd \
--batchsize_per_gpu 8 \
--dataset_num_workers 4 \
--enc_nlayers 6 \
--load_pretrain './pretrained/pimae.pth' \
--specific_encoder_depth 3 \
--joint_encoder_depth 3 \
--unfreeze_epoch 0 \
--ngpus 1 \
--subset 0.0 \
--no_wandb \
