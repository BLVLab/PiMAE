"""
Modified from 3DETR, which is based on DETR. The weights should work fine on DETR as well.
"""
import argparse
import os
from typing import Optional
import torch
from torch import Tensor, nn
from functools import partial
import copy

# model definition
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class BatchNormDim1Swap(nn.BatchNorm1d):

    def forward(self, x):
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        x = x.permute(2, 0, 1)

NORM_DICT = {
    'bn': BatchNormDim1Swap,
    'bn1d': nn.BatchNorm1d,
    'id': nn.Identity,
    'ln': nn.LayerNorm,
}

ACTIVATION_DICT = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leakyrelu': partial(nn.LeakyReLU, negative_slope=0.1)
}

WEIGHT_INIT_DICT = {
    'xavier_uniform': nn.init.xavier_uniform_,
}

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers,
                 norm=None, weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional [Tensor] = None,
                transpose_swap: Optional[bool] = False,
                return_attn_weights: Optional [bool] = False,
                ):
        attns = []
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = src
        orig_mask = mask
        if orig_mask is not None and isinstance(orig_mask, list):
            assert len(orig_mask) == len(self.layers)
        elif orig_mask is not None:
            orig_mask = [mask for _ in range(len(self.layers))]

        for idx, layer in enumerate(self.layers):
            if orig_mask is not None:
                mask = orig_mask[idx]
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)
            if return_attn_weights:
                output, attn = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, return_attn_weights=True)
                attns.append(attn)
            else:
                output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        xyz_inds = None

        if return_attn_weights:
            attns = torch.stack(attns)
            return xyz, output, xyz_inds, attns

        return xyz, output, xyz_inds

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=128,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True, norm_name="ln",
                 use_ffn=True,
                 ffn_use_bias=True):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.use_ffn = use_ffn
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.norm1 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        value = src
        src2 = self.self_attn(q, k, value=value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        if self.use_norm_fn_on_input:
            src = self.norm1(src)
        if self.use_ffn:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [Tensor] = False):

        src2 = self.norm1(src)
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=value, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        if return_attn_weights:
            return src, attn_weights
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_attn_weights: Optional [Tensor] = False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def extra_repr(self):
        st = ""
        if hasattr(self.self_attn, "dropout"):
            st += f"attn_dr={self.self_attn.dropout}"
        return st

# argparser
def make_args_parser():
    parser = argparse.ArgumentParser("3D feature extractor from PiMAE", add_help=False)

    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=6, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ##### load weight #####
    parser.add_argument("--load_pretrain", default='./pimae.pth', type=str)
    parser.add_argument("--specific_encoder_depth", default=3, type=int)
    parser.add_argument("--joint_encoder_depth", default=3, type=int) # this controls the layers to load our cross-modal shared encoder

    return parser

def loadPretrain(model,path,first_depth,second_depth, prefix='', is_distributed=False):
    # first_depth - pc specific encoder depth
    # second_depth - shared encoder depth
    model_dict=model.state_dict()
    model_weight_path = path
    if prefix != '':
        prefix += '.'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location="cpu")['base_model']
    dict ={}
    total_depth = first_depth + second_depth
    weights_toload = ["self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight", "self_attn.out_proj.bias", 
        "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]
    for i in range(0,first_depth):
        for name in weights_toload:
            if is_distributed:
                dict[f"{prefix}layers.{i}.{name}"] = f"module.pc_branch.MAE_encoder.blocks.layers.{i}.{name}"
            else:
                dict[f"{prefix}layers.{i}.{name}"] = f"pc_branch.MAE_encoder.blocks.layers.{i}.{name}"

    for i in range(first_depth,total_depth):
        j = i - first_depth
        for name in weights_toload:
            if is_distributed:
                dict[f"{prefix}layers.{i}.{name}"] = f"module.blocks.layers.{j}.{name}"
            else:
                dict[f"{prefix}layers.{i}.{name}"] = f"blocks.layers.{j}.{name}"

    for key in dict.keys():
        model_dict[key] = pre_weights[dict[key]]
    
    model.load_state_dict(model_dict)
    print(f'loaded pretrained weights from {path}')

# load pretrained weights from PiMAE
def load_pimae(model, args):
    if args.load_pretrain and args.specific_encoder_depth + args.joint_encoder_depth > 0:
        specific_encoder_depth = args.specific_encoder_depth
        joint_encoder_depth = args.joint_encoder_depth
        pretrain_path = args.load_pretrain
        try:
            loadPretrain(model, pretrain_path, specific_encoder_depth, joint_encoder_depth, prefix='', is_distributed=True)
        except:
            loadPretrain(model, pretrain_path, specific_encoder_depth, joint_encoder_depth, prefix='', is_distributed=False)

# build PiMAE encoder
def build_3D_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        d_model=args.enc_dim,
        nhead=args.enc_nhead,
        dim_feedforward=args.enc_ffn_dim,
        dropout=args.enc_dropout,
        activation=args.enc_activation,
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=args.enc_nlayers
    )
    return encoder

def main():
    parser = make_args_parser()
    args = parser.parse_args()

    encoder = build_3D_encoder(args) # build model

    load_pimae(encoder, args) # load weights

    input_token = torch.rand(4, 32, 256) # assume input in the shape of (B, M, C)
    
    input_token = input_token.transpose(0, 1) # B,M,C -> M,B,C 
    # you HAVE to transpose the token before encoding, 
    # because we use older version of nn.MultiheadAttention, 
    # which does not support batch_first function

    encoded_feature = encoder(input_token, return_attn_weights=False)[1].transpose(0, 1) # M,B,C -> B,M,C
    # remember to transpose back

    print(encoded_feature)


if __name__ == '__main__':
    main()


