# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_detr_config(cfg):
    """
    Add config for DETR.
    """
    cfg.MODEL.DETR = CN()
    cfg.MODEL.DETR.NUM_CLASSES = 19

    # For Segmentation
    cfg.MODEL.DETR.FROZEN_WEIGHTS = ''

    # LOSS
    cfg.MODEL.DETR.GIOU_WEIGHT = 2.0
    cfg.MODEL.DETR.L1_WEIGHT = 5.0
    cfg.MODEL.DETR.DEEP_SUPERVISION = True
    cfg.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1

    # TRANSFORMER
    cfg.MODEL.DETR.NHEADS = 8
    cfg.MODEL.DETR.DROPOUT = 0.1
    cfg.MODEL.DETR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DETR.ENC_LAYERS = 3
    cfg.MODEL.DETR.DEC_LAYERS = 6
    cfg.MODEL.DETR.PRE_NORM = False

    cfg.MODEL.DETR.HIDDEN_DIM = 256
    cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 100

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    
    
    single_iteration = cfg.SOLVER.IMS_PER_BATCH
    iterations_for_one_epoch = 19466 / single_iteration
    cfg.SOLVER.MAX_ITER  = int(iterations_for_one_epoch * 300)
    
    
    #pretrained data
    cfg.MODEL.PIMAE = CN()
    cfg.MODEL.PIMAE.from_pretrained = True
    cfg.MODEL.PIMAE.enc_type ="vanilla"
    # Below options are only valid for vanilla encoder
    cfg.MODEL.PIMAE.enc_nlayers=6
    cfg.MODEL.PIMAE.enc_dim=256
    cfg.MODEL.PIMAE.enc_ffn_dim=1024
    cfg.MODEL.PIMAE.enc_dropout=0.1
    cfg.MODEL.PIMAE.enc_nhead=4
    cfg.MODEL.PIMAE.enc_pos_embed=None
    cfg.MODEL.PIMAE.enc_activation="relu"
    cfg.MODEL.PIMAE.pretrain_file='/home/ubuntu/pimae/detr/pimae_detr.pt'
    cfg.MODEL.PIMAE.pretrain_file=''
    ### Other model params
    cfg.MODEL.PIMAE.preenc_npoints=2048

    ## load weight #####
    cfg.MODEL.PIMAE.load_pretrain='/home/ubuntu/pimae/detr/pimae_detr.pt'
    cfg.MODEL.PIMAE.load_pretrain=''
    cfg.MODEL.PIMAE.load_r50='/home/ubuntu/pimae/detr/d2/converted_r50.pth'
    cfg.MODEL.PIMAE.specific_encoder_depth=3
    cfg.MODEL.PIMAE.joint_encoder_depth=3
    cfg.MODEL.PIMAE.subset=0

    
    
    
