# Copyright (c) Facebook, Inc. and its affiliates.


""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points
import cv2
from PIL import Image
from torchvision import transforms
import utils.sunrgbd_utils as sunrgbd_utils

import utils_detr.pc_util as pc_util
from utils_detr.random_cuboid import RandomCuboid
from utils_detr.pc_util import shift_scale_points, scale_points
from configparser import ConfigParser
from utils_detr.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
)


MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
DATA_ROOT = "/home/lyh/Anton/data/sunrbgd/"  ## Replace with path to dataset
DATA_PATH_V1 = os.path.join(DATA_ROOT,"sunrgbd_pc_bbox_votes_50k_v1")
DATA_PATH_V2 = "" ## Not used in this codebase.

MAX_HEIGHT = 530
MAX_WIDTH = 730
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class SunrgbdDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 10
        self.num_angle_bin = 12
        self.max_num_obj = 64
        self.type2class = {
            "bed": 0,
            "table": 1,
            "sofa": 2,
            "chair": 3,
            "toilet": 4,
            "desk": 5,
            "dresser": 6,
            "night_stand": 7,
            "bookshelf": 8,
            "bathtub": 9,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.type2onehotclass = {
            "bed": 0,
            "table": 1,
            "sofa": 2,
            "chair": 3,
            "toilet": 4,
            "desk": 5,
            "dresser": 6,
            "night_stand": 7,
            "bookshelf": 8,
            "bathtub": 9,
        }

class SunrgbdDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        num_points=20000,
        use_color=False,
        use_height=False,
        use_v1=True,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
        img_size=(256, 352),
    ):
        assert num_points <= 50000
        assert split_set in ["train", "val", "trainval"]
        self.dataset_config = dataset_config
        self.use_v1 = use_v1

        if root_dir is None:
            root_dir = DATA_PATH_V1 if use_v1 else DATA_PATH_V2

        self.data_path = root_dir + "_%s" % (split_set)

        if split_set in ["train", "val"]:
            self.scan_names = sorted(
                list(
                    set([os.path.basename(x)[0:6] for x in os.listdir(self.data_path)])
                )
            )
            
            if split_set == "val":
            	self.scan_names = self.scan_names[0:len(self.scan_names):10]
            
        elif split_set in ["trainval"]:
            # combine names from both
            sub_splits = ["train", "val"]
            all_paths = []
            for sub_split in sub_splits:
                data_path = self.data_path.replace("trainval", sub_split)
                basenames = sorted(
                    list(set([os.path.basename(x)[0:6] for x in os.listdir(data_path)]))
                )
                basenames = [os.path.join(data_path, x) for x in basenames]
                all_paths.extend(basenames)
            all_paths.sort()
            self.scan_names = all_paths

        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points,
            aspect=0.75,
            min_crop=0.75,
            max_crop=1.0,
        )
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 64

        # img augmentation
        self.img_size = img_size
        # aug
        self.img_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # no-aug
        self.img_nomalization = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # normal resize
        self.img_resize = transforms.Resize(self.img_size)

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        if scan_name.startswith("/"):
            scan_path = scan_name
        else:
            scan_path = os.path.join(self.data_path, scan_name)
        point_cloud = np.load(scan_path + "_pc.npz")["pc"]  # Nx6
        bboxes = np.load(scan_path + "_bbox.npy")  # K,8
        
        calib_dir = os.path.join(DATA_ROOT, "calib")
       
        calib_lines = [line for line in open(os.path.join(calib_dir, scan_name + '.txt')).readlines()]
        calib_Rtilt = np.reshape(np.array([float(x) for x in calib_lines[0].rstrip().split(' ')]), (3, 3), 'F')
        calib_K = np.reshape(np.array([float(x) for x in calib_lines[1].rstrip().split(' ')]), (3, 3), 'F')
        # Read image
        img_path = os.path.join(DATA_ROOT,"image")
        full_img = sunrgbd_utils.load_image(os.path.join(img_path, scan_name + '.jpg'))
        fx, fy = MAX_WIDTH / full_img.shape[1], MAX_HEIGHT / full_img.shape[0]
        full_img = cv2.resize(full_img, None, fx=fx, fy=fy)
        full_img_height, full_img_width = full_img.shape[0], full_img.shape[1]

        colored_point_cloud = point_cloud[:, 0:6]

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        scale_ratio = 1.
        if self.augment:
            flip_flag = np.random.random() > 0.5
            if flip_flag:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                # bboxes[:, 0] = -1 * bboxes[:, 0]
                # bboxes[:, 6] = np.pi - bboxes[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = sunrgbd_utils.rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            # bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            # bboxes[:, 6] -= rot_angle

            
            R_inverse = np.copy(np.transpose(rot_mat))
            if flip_flag:
                R_inverse[0, :] *= -1
            # Update Rtilt according to the augmentation
            # R_inverse (3x3) * point (3x1) transforms an augmented depth point
            # to original point in upright_depth coordinates
            calib_Rtilt = np.dot(np.transpose(R_inverse), calib_Rtilt)

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            calib_Rtilt = np.dot(np.array([[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]]), calib_Rtilt)
            scale_ratio_expand = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio_expand
    
            if self.use_height:
                point_cloud[:, -1] *= scale_ratio

       
        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)

        ret_dict = {}
        ret_dict['colored_pointclouds'] = colored_point_cloud
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['scan_name'] = scan_name
        ret_dict['scale'] = np.array(scale_ratio).astype(np.float32)
        ret_dict['calib_Rtilt'] = calib_Rtilt.astype(np.float32)
        ret_dict['calib_K'] = calib_K.astype(np.float32)
        # full_img.shape: [530, 730, 3] -> [3, 530, 730]
        ret_dict['full_img'] = np.transpose(full_img.astype(np.float32), (2, 0, 1))
        mean = np.array(MEAN, dtype=np.float32)[:, np.newaxis, np.newaxis]
        std = np.array(STD, dtype=np.float32)[:, np.newaxis, np.newaxis]
        ret_dict['full_img'] = (ret_dict['full_img'] / 255. - mean) / std
        # ret_dict['full_img_width'] = np.array(full_img_width).astype(np.int64)
        # ret_dict['full_img_height'] = np.array(full_img_height).astype(np.int64)
        # ret_dict['target_box2d'] = target_bboxes2d.astype(np.float32)

        
        # ret_dict = {}
        # ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        # ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        # ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        # ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
        #     np.float32
        # )
        # target_bboxes_semcls = np.zeros((self.max_num_obj))
        # target_bboxes_semcls[0 : bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9
        # ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        # ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        # ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        # ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        # ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        # ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        # ret_dict["gt_angle_class_label"] = angle_classes
        # ret_dict["gt_angle_residual_label"] = angle_residuals
        # ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        # ret_dict["point_cloud_dims_max"] = point_cloud_dims_max
        # ret_dict["image"] = img
        # # ret_dict["mask"] = mask
        # ret_dict['target_bbox'] = target_bboxes.astype(np.float32)
        # ret_dict['corners'] = corners.astype(np.float32)
        # ret_dict["Rilt"] = Rilt
        # ret_dict['K'] = K
        # ret_dict["image_id"] = np.str(calib_id)
        # ret_dict["image_size"] = img_size


        # ret_dict["bboxes_2d"] = bboxes_2d
        # ret_dict["bboxes_2d_label"] = bboxes_2d_label.astype(np.int64)
        # ret_dict["bbox_num"] = bbox_num
        return ret_dict


class ClassificationDataset(Dataset):
    '''croped point clouds with class label from sunrbgd validation set'''
    def __init__(self, mode):
        self.root_dir = os.path.join(Classification_PATH, mode)
        self.label_map = ['desk', 'chair', 'table', 'bookshelf', 'sofa',
                          'toilet', 'bed', 'bathtub', 'dresser', 'night_stand']
        label_list = os.listdir(self.root_dir)
        self.pc_paths = []
        for cls in label_list:
            pc_list = os.listdir(os.path.join(self.root_dir, cls))
            for pc_path in pc_list:
                self.pc_paths.append(os.path.join(self.root_dir, cls, pc_path))

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, index):
        path = self.pc_paths[index]
        cls = path.split('/')[-2]
        label = self.label_map.index(cls)
        points = np.loadtxt(path, dtype=float)
        points = points.astype(np.float32)

        return points, label

