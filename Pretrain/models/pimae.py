from timm.models.vision_transformer import Block
import torch.nn as nn
import torch
import random
import utils.matcher
import numpy as np
import cv2
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial

from typing import Dict, List, Optional, Union

from .multimae_utils import Block, trunc_normal_

from models.transformer import TransformerEncoderLayer as out_encoderlayer
from models.transformer import TransformerEncoder as out_encoder

from easydict import EasyDict as edict

PATCH_SIZE = 16
MAX_WIDTH = 730
MAX_HEIGHT = 530

class PiMAE(nn.Module):
    def __init__(self,
                 pc_branch,
                 img_branch, 
                 config):
        super().__init__()

        if hasattr(config, "hp_overlap_ratio"):
            self.hp_overlap_ratio = config.hp_overlap_ratio
        else:
            self.hp_overlap_ratio=0.0
        dim_tokens = config.encoder.trans_dim
        depth = config.encoder.depth
        num_heads = config.encoder.num_heads
        mlp_ratio = config.encoder.mlp_ratio
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.token_fusion = config.token_fusion
        self.pc2img = config.pc2img
        self.distill_loss = nn.MSELoss()

        # image, point cloud branches
        self.pc_branch = pc_branch
        self.img_branch = img_branch
        self.mask_ratio = config.mask_ratio
        self.modality_img_embedding = nn.Parameter(torch.zeros(1, 1, dim_tokens))
        self.modality_pc_embedding = nn.Parameter(torch.zeros(1, 1, dim_tokens))

        # CLS token init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_tokens))

        # encoder from 3detr
        if self.token_fusion:
            self.fusion_proj = nn.Linear(dim_tokens*2, dim_tokens)

        if self.pc2img:
            final_dim = 0
            if self.pc2img.rgb:
                final_dim += 3
            if self.pc2img.feat:
                final_dim += self.pc_branch.decoder_trans_dim
            
            self.increase_dim_feat = nn.Sequential(
                nn.Conv1d(self.pc_branch.decoder_trans_dim, final_dim*self.pc_branch.group_size, 1)
                )
   
        dim_feedforward = config.encoder.dim_feedforward
        dropout = config.encoder.dropout
        encoder_layer = out_encoderlayer(
            d_model=dim_tokens,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.blocks = out_encoder(encoder_layer=encoder_layer, num_layers=depth)
        self.norm = norm_layer(dim_tokens)

        # joint decoder
        if config.decoder.depth:
            self.is_joint_decoder = config.decoder.depth
            decoder_depth = config.decoder.depth
            decoder_dim = config.decoder.trans_dim
            decoder_num_heads = config.decoder.num_heads
            decoder_mlp_ratio = config.decoder.mlp_ratio

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])
            
            self.decoder_norm = norm_layer(decoder_dim)
            self.decoder_modality_img_embed = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            self.decoder_modality_pc_embed = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def generate_input_info(self, input__token_img,input_token_pc):
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        #PointCloud
        num_tokens_pc = input_token_pc.shape[1]
        d_pc = {
            'num_tokens': num_tokens_pc,
            'start_idx': i,
            'end_idx': i + num_tokens_pc,
        }
        i += num_tokens_pc
        input_info['tasks']['pc'] = d_pc
        #IMG
        num_tokens_img = input__token_img.shape[1]
        d_img = {
            'num_tokens': num_tokens_img,
            'start_idx': i,
            'end_idx': i + num_tokens_img,
        }
        i += num_tokens_img
        input_info['tasks']['img'] = d_img
        input_info['num_task_tokens'] = i
        return input_info

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(np.ceil(L * (1 - mask_ratio))) 
        
        noise = torch.rand(N, L, device=x.device)  
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep] 
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) 

        mask = torch.ones([N, L], device=x.device) 
        mask[:, :len_keep] = 0  
        mask = torch.gather(mask, dim=1, index=ids_restore) 

        return x_masked, mask, ids_restore

    def forward_tokenizer(self, pts, imgs):
        x_img = self.img_branch.patch_embed(imgs)
        # add pos embed
        x_img = x_img + self.img_branch.pos_embed  # no cls token

        neighborhood, center = self.pc_branch.group_divider(pts)
        x_pc = self.pc_branch.MAE_encoder.encoder(neighborhood)  # B, Group, C
        pos = self.pc_branch.MAE_encoder.pos_embed(center)

        x_pc = x_pc + pos
    
        return x_img, x_pc, center, neighborhood
    
    def forward_pc_decoder(self, x_vis, center, mask):
        x_vis = self.pc_branch.decoder_embed(x_vis)  # change dim

        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.pc_branch.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.pc_branch.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.pc_branch.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        full_length = x_full.shape[1]
        if self.is_joint_decoder:
            return x_full, pos_full, N

        x_rec_full = self.pc_branch.MAE_decoder(x_full, pos_full, full_length)

        x_rec_masked = x_rec_full[:, -N:, :] # only include masked points

        return x_rec_masked, x_rec_full  

    def forward_pc_loss(self, x_rec, neighborhood, center, mask, align_props=None, img_feat=None):
        """
        Calculates the point cloud reconstruction loss. If self.pc2img, project point cloud to image, interpolate features
        And calculate cross-modality reconstruction loss.
        """
        B, M, C = x_rec.shape 

        distill_loss = None
        # PC to IMG loss
        if self.pc2img:
            B, G, S, C = neighborhood.shape
            gt_points_masked = neighborhood[mask].reshape(B*M, -1, 3)  
            gt_points_masked += center[mask].unsqueeze(1) # denormalize

            gt_points_masked = gt_points_masked.reshape(B, -1, 3)
            projected_u, projected_v = self.new_align(align_props.rilt, align_props.k, align_props.scale, align_props.img_size, gt_points_masked) 

            point_extracted_feat = self.extract_feat(img_feat, projected_u, projected_v).transpose(1,2)
            feat_dim = point_extracted_feat.shape[-1]

            point_extracted_feat = point_extracted_feat.clone().detach()
            rebuild_feats = self.increase_dim_feat(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, feat_dim)
            point_extracted_feat =point_extracted_feat.reshape(B*M, S, -1)
        
            assert rebuild_feats.shape == point_extracted_feat.shape
            distill_loss = self.distill_loss(rebuild_feats, point_extracted_feat)

        rebuild_points = self.pc_branch.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3) 
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)

        pc_loss = self.pc_branch.loss_func(rebuild_points, gt_points)    

        return pc_loss, distill_loss, rebuild_points
    
    def forward_img_decoder(self, x, ids_restore):

        x = self.img_branch.decoder_embed(x)

        mask_tokens = self.img_branch.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) 
        x = x_ # no cls token

        x = x + self.img_branch.decoder_pos_embed

        if self.is_joint_decoder:  
            return x

        # Specific decoders
        for blk in self.img_branch.decoder_blocks:
            x = blk(x)
        x_feature = self.img_branch.decoder_norm(x)

        x_rec = self.img_branch.decoder_pred(x_feature)
        return x_feature, x_rec

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0) 
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1) 

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device) 

        return bool_masked_pos

    def forward_img_loss(self, imgs, pred, mask):
        """
                imgs: [N, 3, H, W]
                pred: [N, L, p*p*3]
                mask: [N, L], 0 is keep, 1 is remove,
                """
        target = self.img_branch.patchify(imgs)
        if self.img_branch.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def countPatchIndex(self, uv,img_token_dimenson):
        mask = torch.zeros(img_token_dimenson)
        for i in range(uv.shape[0]):
            tempIndex = int(uv[i][1]/16)*22+int(uv[i][0]/16)
            if(tempIndex<16*22 and tempIndex>0):
                mask[tempIndex] = 1
        return mask
    def shuffle_to_pc_mask(self, pc_bool_mask, x, mask_ratio):
        """
        Shuffles img patch indexes to pc's boolean mask.
        0 visible, 1 masked
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(np.ceil(L * (1 - mask_ratio))) 
        assert pc_bool_mask.shape[0] == N 
        pc_m_ratio = torch.sum(pc_bool_mask, dim=1) 

        noise = torch.rand(N, L, device=x.device)/2  #  [0, 0.5]
        random_pc_bool_mask = pc_bool_mask + noise
        
        if self.hp_overlap_ratio == 1.0:
            descending=False
        elif self.hp_overlap_ratio == 0.0:
            descending=True
        else: 
            raise NotImplementedError

        ids_shuffle = torch.argsort(random_pc_bool_mask, dim=1, descending=descending) 
        ids_restore = torch.argsort(ids_shuffle, dim=1) 

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # binary mask: 0 keep, 1 remove
        mask = torch.ones([N, L], device=x.device)  
        mask[:, :len_keep] = 0  
        mask = torch.gather(mask, dim=1, index=ids_restore) 

        return x_masked, mask, ids_restore
 
    def new_align(self, calib_Rtilt, calib_K, scale, img_size, points):
        xyz = torch.matmul(calib_Rtilt.transpose(2, 1), (1 / scale ** 2).unsqueeze(-1).unsqueeze(-1) * points.transpose(2, 1))
        xyz = xyz.transpose(2, 1)
        xyz[:, :, [0, 1, 2]] = xyz[:, :, [0, 2, 1]]
        xyz[:, :, 1] *= -1
        uv = torch.matmul(xyz, calib_K.transpose(2, 1)).detach()
        uv[:, :, 0] /= uv[:, :, 2]
        uv[:, :, 1] /= uv[:, :, 2]
        u, v = (uv[:, :, 0] - 1).round(), (uv[:, :, 1] - 1).round()
        # normalize
        u, v = u / MAX_WIDTH, v / MAX_HEIGHT
        return u, v
    
    def get_center_masks(self, u, v, img_size):
        B, M = u.shape
        token_nums = int((img_size[0]/PATCH_SIZE)*(img_size[1]/PATCH_SIZE))
        masks = torch.zeros((B, token_nums))
        H, W = img_size
        u, v = u*(W-1), v*(H-1)
        u, v = torch.floor(u / PATCH_SIZE), torch.floor(v / PATCH_SIZE)
        proj_patch_idx = (v * (img_size[1] // PATCH_SIZE) + u).long()
        for i in range(B):
            masks[i, proj_patch_idx[i]] = 1
        
        return masks
  
    def extract_feat(self, img_feat, u, v):
        # extract point-clouds-related features from image features
        grid = torch.stack([v, u], dim=-1) # y first
        
        if len(img_feat.shape) == 3: 
            B, L, C = img_feat.shape
            grid_size = self.img_branch.patch_embed.grid_size
            f = img_feat.reshape(B, grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2)  # B, h, w, C -> B, C, h, w
            f_dense = F.interpolate(f, size=(256, 352), mode='bilinear')  # B, C, h, w ->B, C, H, W
            grid = grid.unsqueeze(2)
            grid = 2.0 * grid - 1.0
            f_extract = F.grid_sample(f_dense, grid)  # (B, C, Groups*numbers, 1)
        else:
            B, C, H, W = img_feat.shape
            assert C == 3 and len(img_feat.shape)==4 
            grid = grid.unsqueeze(2) 
            grid = 2.0 * grid - 1.0
            f_extract = F.grid_sample(img_feat, grid) # (B, 3,  Groups*numbers, 1)
        f_ret = f_extract.squeeze(3)
        return f_ret # (B, C, Groups*numbers)
                       
    def forward_joint_decoder(self, x_img, ids_restore, x_pc, center, mask):
        # step 1: insert masked token & add positional and modality embedding
        # img
        x_img = self.forward_img_decoder(x_img, ids_restore)
        x_img = x_img + self.decoder_modality_img_embed

        # pc
        x_pc, x_pc_pos, N = self.forward_pc_decoder(x_pc, center, mask)
        x_pc = x_pc + x_pc_pos + self.decoder_modality_pc_embed

        # step 2: concate tokens along L dim
        img_sz = x_img.shape[1]
        pc_sz = x_pc.shape[1]
        decoder_inputs = [x_img, x_pc]
        decoder_inputs = torch.cat(decoder_inputs, dim=1)

        # step 3: pass through joint decoder
        for blk in self.decoder_blocks:
            decoder_inputs = blk(decoder_inputs)
        decoder_inputs = self.decoder_norm(decoder_inputs)
        
        # step 4: split tokens along L dim
        rec_img = decoder_inputs[:, :img_sz, :]
        rec_pc = decoder_inputs[:, -pc_sz:, :]

        # step 5: pass through specific decoders

            # add positional embedding before decoder
        rec_img = rec_img + self.img_branch.decoder_pos_embed

            # apply Transformer blocks
        for blk in self.img_branch.decoder_blocks:
            rec_img = blk(rec_img)
        img_feat = self.img_branch.decoder_norm(rec_img)
            # predictor projection
        rec_img = self.img_branch.decoder_pred(img_feat)
            # ready for return
        
            # transformer
        rec_pc = self.pc_branch.MAE_decoder(rec_pc, x_pc_pos, N)  # normed inside the function
            # ready for return
        
        return img_feat, rec_img, rec_pc


    def forward(self,
                pts,
                imgs, 
                rilt,
                k,
                scale,
                img_size,
                vis=False):
        # tokenizing, embedding
        img_token, pc_token, center, neighborhood = self.forward_tokenizer(pts, imgs)
    
        img_token = img_token + self.modality_img_embedding
        pc_token = pc_token + self.modality_pc_embedding
        
        if self.pc_branch.MAE_encoder.mask_type == 'rand':
           bool_masked_pos = self.pc_branch.MAE_encoder._mask_center_rand(center, noaug=False)
        else:
          bool_masked_pos = self.pc_branch.MAE_encoder._mask_center_block(center, noaug=False)
        B, L, C = pc_token.shape
        pc_vis = pc_token[~bool_masked_pos].reshape(B, -1, C)
        # projection & alignment
        if self.hp_overlap_ratio: # "hp_overlap_ratio" controls the alignment manner, where 0 is complement, 1 is uniform
            B,L_center,C_center = center.shape
            pc_mask_centers = center[bool_masked_pos].reshape(B, -1, C_center)
            pc_mask_centers_proj_u, pc_mask_centers_proj_v = self.new_align(rilt,k, scale, img_size, pc_mask_centers)
            pc_converted_mask = self.get_center_masks(pc_mask_centers_proj_u, pc_mask_centers_proj_v, img_size)
            pc_converted_mask = pc_converted_mask.to(center.device)
            overlap_ratio = (torch.sum(pc_converted_mask, dim=1) / pc_converted_mask.shape[1]) / self.mask_ratio
            #img masking
            img_vis, mask, ids_restore = self.shuffle_to_pc_mask(pc_converted_mask, img_token, self.mask_ratio)
        else:
            # random masking
            img_vis, mask, ids_restore = self.img_branch.random_masking(img_token, self.mask_ratio)
       
        img_input_size = img_vis.shape[1]
        pc_input_size = pc_vis.shape[1]
      
        # img specific encoder
        for blk in self.img_branch.blocks:
            img_vis = blk(img_vis)
        # pc specific encoder
        pc_vis = pc_vis.transpose(0, 1)  # B,M,C -> M,B,C
        ret_pc = self.pc_branch.MAE_encoder.blocks(pc_vis, return_attn_weights=True)
        pc_vis = ret_pc[1]  #returns xyz, output, xyz_inds, and we need only output
        pc_vis = pc_vis.transpose(0, 1)
        
        # concat tokens
        inputs_token = [img_vis, pc_vis]
        inputs_token = torch.cat(inputs_token, dim=1)
        # add cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(inputs_token.shape[0], -1, -1)
        inputs_token = torch.cat((cls_tokens, inputs_token), dim=1)
        # shared encoder
        inputs_token = inputs_token.transpose(0, 1) # B,M,C -> M,B,C
        ret_joint = self.blocks(inputs_token, return_attn_weights=True)
        inputs_token = ret_joint[1]  #returns xyz, output, xyz_inds, and we need only output    
        inputs_token = inputs_token.transpose(0, 1)
        
        img_vis = inputs_token[:, 1:1+img_input_size, :]  # remove cls token
        pc_vis = inputs_token[:, -pc_input_size:, :]

        if self.is_joint_decoder:
            img_feat, img_rec, pc_rec_masked = self.forward_joint_decoder(x_img=img_vis, ids_restore=ids_restore, x_pc=pc_vis, center=center, mask=bool_masked_pos)
        else:
            img_feat, img_rec = self.forward_img_decoder(x=img_vis, ids_restore=ids_restore)
            pc_rec_masked, pc_rec_full = self.forward_pc_decoder(x_vis=pc_vis, center=center, mask=bool_masked_pos)
            N = pc_rec_full.shape[1] - pc_rec_masked.shape[1]
            pc_rec_vis = pc_rec_full[:, :N, :]

        # pc2img reconstrction
        pc2img_loss = None
        rebuild_img = None
        
        # get img loss
        img_loss = self.forward_img_loss(imgs=imgs, pred=img_rec, mask=mask)

        # determine target feature
        if not self.pc2img:
            img_feat = None
        elif self.pc2img.rgb:
            img_feat = imgs
        elif self.pc2img.feat:
            img_feat = img_feat

        align_props = edict(dict(rilt=rilt,k=k,scale=scale,img_size=img_size))
        
        # get pc loss
        pc_loss, pc2img_loss, rebuild_points = self.forward_pc_loss(x_rec=pc_rec_masked, neighborhood=neighborhood, center=center, 
                mask=bool_masked_pos,align_props=align_props, img_feat=img_feat)
        
        if vis and self.pc2img:
            rebuild_pc = rebuild_points[..., :3]
            B, M, C = pc_rec_masked.shape
            vis_points = neighborhood[~bool_masked_pos].reshape(B * (self.pc_branch.num_group - M), -1, 3) #B*V, 32, 3
            mask_points = neighborhood[bool_masked_pos].reshape(B * M, -1, 3) 
            full_mask = mask_points + center[bool_masked_pos].unsqueeze(1) 
            full_vis = vis_points + center[~bool_masked_pos].unsqueeze(1) 
            full_rebuild = rebuild_pc + center[bool_masked_pos].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            full_center = torch.cat([center[bool_masked_pos], center[~bool_masked_pos]], dim=0)
            ret3 = full_mask.reshape(-1, 3).unsqueeze(0)  # full masked points
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)  # visible points
            ret1 = full.reshape(-1, 3).unsqueeze(0)  # reconstructed points

            return (img_rec, mask, ids_restore), (ret1, ret2, ret3, full_center), rebuild_img

        if vis and not self.pc2img:
            B, M, C = pc_rec_masked.shape
            vis_points = neighborhood[~bool_masked_pos].reshape(B * (self.pc_branch.num_group - M), -1, 3)
            mask_points = neighborhood[bool_masked_pos].reshape(B * M, -1, 3) 
            full_mask = mask_points + center[bool_masked_pos].unsqueeze(1) 
            full_vis = vis_points + center[~bool_masked_pos].unsqueeze(1)
            full_rebuild = rebuild_points + center[bool_masked_pos].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            full_center = torch.cat([center[bool_masked_pos], center[~bool_masked_pos]], dim=0)
            ret3 = full_mask.reshape(-1, 3).unsqueeze(0)  # full masked points
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)  # visible points
            ret1 = full.reshape(-1, 3).unsqueeze(0)  # reconstructed points

            return (img_rec, mask, ids_restore), (ret1, ret2, ret3, full_center), rebuild_img
            
        return pc_loss, img_loss, pc2img_loss



