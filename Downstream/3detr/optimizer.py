# Copyright (c) Facebook, Inc. and its affiliates.
import torch


def build_optimizer(args, model):

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n.split('.') and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "encoder" in n.split('.') and p.requires_grad],
            "lr": args.encoder_learning_rate,
        },
    ]
    # params_with_decay = []
    # params_without_decay = []

    # for name, param in model.named_parameters():
    #     if param.requires_grad is False:
    #         continue
    #     if args.filter_biases_wd and (len(param.shape) == 1 or name.endswith("bias")):
    #         params_without_decay.append(param)
    #     else:
    #         params_with_decay.append(param)

    # if args.filter_biases_wd:
    #     param_groups = [
    #         {"params": params_without_decay, "weight_decay": 0.0},
    #         {"params": params_with_decay, "weight_decay": args.weight_decay},
    #     ]
    # else:
    #     param_groups = [
    #         {"params": params_with_decay, "weight_decay": args.weight_decay},
    #     ]
    optimizer = torch.optim.AdamW(param_groups, 
                                lr=args.base_lr, 
                                weight_decay=args.weight_decay,
                                )
                                
    for param_group in optimizer.param_groups:
        print(param_group['lr'], param_group['weight_decay'])
    return optimizer
