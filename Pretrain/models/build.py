from utils import registry

#
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

import sys
import third_party.models_mae as models_mae

from models.pimae import PiMAE

from tools import builder

MODELS = registry.Registry('models')

def build_model_from_cfg(cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return MODELS.build(cfg, **kwargs)

def build_multimae(config, args, logger):
    # point cloud branch
    pointmae_model = build_model_from_cfg(config.pc_model)
    # image branch
    mae_model = models_mae.build_mae_from_cfg(config.img_model, norm_pix_loss=args.norm_pix_loss, img_size=args.img_size)

    model = PiMAE(pointmae_model, mae_model, config.joint_model)

    return model









