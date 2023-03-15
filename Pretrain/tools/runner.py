import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from PIL import Image
from utils.logger import *

import cv2
import numpy as np

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from datasets import build_dataset

from models.build import build_multimae
from tools.mae_visualize import run_one_image, show_image
from sklearn.svm import LinearSVC

import open3d

def save_pc(pc,path,name,if_color=True,if_six=False):
    pcd = open3d.geometry.PointCloud()
    if not if_color:
        pcd.points = open3d.utility.Vector3dVector(pc[:, 0:3])
        pcd.colors = open3d.utility.Vector3dVector(pc[:, 3:6])
    else:
        if if_six:
            pcd.points = open3d.utility.Vector3dVector(pc[:, 0:3])
        else:
            pcd.points = open3d.utility.Vector3dVector(pc.reshape(-1,3))
    open3d.io.write_point_cloud(str(path)+"/"+name+".ply",pcd)

def flip_axis_to_camera_np(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = pc.copy()
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    # _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    datasets, dataset_config = build_dataset(config)
    dataloaders = {}
    if args.test:
        dataset_splits = ["test"]
    else:
        dataset_splits = ["train", "test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = True
            bs = config.dataset.train.others.bs
        else:
            shuffle = False
            bs = config.dataset.test.others.bs
        if args.distributed:
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=bs,
            num_workers=int(args.num_workers),
            worker_init_fn=misc.worker_init_fn,
        )
        dataloaders[split + "_sampler"] = sampler
    '''
    DONE
    '''
    test_dataloader = dataloaders['test']
    # extra_train_dataloader = dataloaders['class_train']
    # extra_test_dataloader = dataloaders['class_test']

    # base_model = build_unimae(config, args, logger)
    base_model = build_multimae(config, args, logger)
    # base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)

# visualization
def test(base_model, test_dataloader, args, config, logger = None):

    # img_resize = transforms.Resize(args.img_size)
    
    base_model.eval()  # set model to eval mode
    target = './vis'
   
    exp_name = args.exp_name
    vis_root = f'./visualization/{exp_name}'
    zip_file = f'{vis_root}/{exp_name}.zip'

    img_resize = transforms.Resize(args.img_size)

    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            print(idx)
            a, b = 30, -45
            npoints = config.npoints
            dataset_name = config.dataset.test._base_.NAME

            scan_name = data['scan_name'][0]
            # print(scan_name)
            # if scan_name != '005247':
            #     continue
            
            if dataset_name == 'ShapeNet':
                # points = data.cuda()
                points = data['point_clouds'].cuda()
                # points = data['point_clouds'][0].cuda()
                points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            final_image = []
            data_path = f'{vis_root}/{scan_name}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            pc_color = data['colored_pointclouds'][0].squeeze().detach().cpu().numpy()
            print(pc_color.shape)
            pc_color=flip_axis_to_camera_np(pc_color)

            save_pc(pc_color,data_path,"color",if_color=False)


            # load img input
            imgs = data['full_img']
            imgs = img_resize(imgs)
            if args.use_gpu:
                imgs = imgs.cuda().float()
            calib_Rtilt = data['calib_Rtilt'].cuda()
            calib_K = data['calib_K'].cuda()
            scale = data['scale'].cuda()
            patch_size = base_model.img_branch.patch_embed.patch_size
            grid_size = base_model.img_branch.patch_embed.grid_size
            
            # get prediction for image and point cloud
            img_ret, pc_ret, pc2img = base_model(points, imgs, calib_Rtilt, calib_K, scale, args.img_size, vis=True)
           
            
            img_pred, img_mask, ids_restore = img_ret
            dense_points, vis_points, mask_points, centers = pc_ret
            run_one_image(imgs, img_pred, img_mask, args.img_size, patch_size, grid_size, data_path)
        
            mask_points = mask_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'mask.txt'), mask_points, delimiter=';')

            points = points.squeeze().detach().cpu().numpy()
            print(points.shape)
            np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')
            points = misc.get_ptcloud_img(points,a,b)
            final_image.append(points[150:650,150:675,:])

            vis_points = vis_points[:].reshape(-1, 3).unsqueeze(0)  #ger certain pc token, for attention map
            vis_points = vis_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            vis_points = misc.get_ptcloud_img(vis_points,a,b)

            final_image.append(vis_points[150:650,150:675,:])

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points,a,b)
            final_image.append(dense_points[150:650,150:675,:])

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)
            # break

        return
