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

from models.build import build_unimae, build_multimae
from tools.mae_visualize import run_one_image, show_image
from sklearn.svm import LinearSVC


import zipfile
def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()

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

    '''Testing classification dataset'''
    # classify(base_model, extra_train_dataloader, extra_test_dataloader, args, config, logger=logger)
    '''DONE'''
    test(base_model, test_dataloader, args, config, logger=logger)

def classify(base_model, extra_train_dataloader, test_dataloader, args, config, logger = None):
    '''
    add parameters
    '''

    def evaluate_svm(train_features, train_labels, test_features, test_labels):
        clf = LinearSVC()
        clf.fit(train_features, train_labels)
        pred = clf.predict(test_features)
        pred = pred.reshape(-1, 1)
        return np.sum(test_labels == pred) * 1. / pred.shape[0]


    print_log(f"[Classification]", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.npoints
    # npoints = 1024
    total_cnt = 0
    padding_cnt = 0
    single_cnt = 0
    with torch.no_grad():
        if extra_train_dataloader:
            for idx, data in enumerate(extra_train_dataloader):
                points, label = data
                points = points.cuda()
                label = label.cuda()
                total_cnt += 1

                if len(points.shape) < 3:
                    # 只有一个点
                    continue

                if points.shape[1] < npoints and len(points.shape) > 2:
                    '''pad 0'''
                    # pad = torch.zeros((points.shape[0], npoints - points.shape[1], points.shape[2])).cuda()
                    # points = torch.cat((points, pad), axis=1)
                    '''DONE'''
                    '''pad repeat'''
                    repeat_times = int(np.ceil(npoints / points.shape[1]))
                    points = points.repeat((1, repeat_times, 1))
                    '''DONE'''
                    padding_cnt += 1
                    if padding_cnt % 100 == 0:
                        print(f'小于{npoints}的物体已发现有{padding_cnt}个，目前已预测{total_cnt}个物体')

                '''
                img_model
                img_feature
                '''
                try:
                    points = misc.fps(points, npoints)
                    # assert points.size(1) == npoints
                    feature = base_model.pc_model(points, noaug=True)
                    feature = feature.view(-1)
                    target = label.view(-1)
                except:
                    print('error found')
                    print(idx)
                    total_cnt -= 1
                    continue

                train_features.append(feature.detach())
                train_label.append(target.detach())

        if test_dataloader:
            for idx, data in enumerate(test_dataloader):

                points, label = data
                points = points.cuda()
                label = label.cuda()
                total_cnt += 1

                if len(points.shape) < 3:
                    # 只有一个点
                    continue

                if points.shape[1] < npoints and len(points.shape) > 2:
                    '''pad 0'''
                    # pad = torch.zeros((points.shape[0], npoints - points.shape[1], points.shape[2])).cuda()
                    # points = torch.cat((points, pad), axis=1)
                    '''DONE'''
                    '''pad repeat'''
                    repeat_times = int(np.ceil(npoints / points.shape[1]))
                    points = points.repeat((1, repeat_times, 1))
                    '''DONE'''
                    padding_cnt += 1
                    if padding_cnt % 100 == 0:
                        print(f'小于{npoints}的物体已发现有{padding_cnt}个，目前已预测{total_cnt}个物体')

                try:
                    points = misc.fps(points, npoints)
                    # assert points.size(1) == npoints
                    feature = base_model.pc_model(points, noaug=True)
                    feature = feature.view(-1)
                    target = label.view(-1)
                except:
                    print('error found')
                    print(idx)
                    total_cnt -= 1
                    continue

                test_features.append(feature.detach())
                test_label.append(target.detach())

        train_features = torch.cat(train_features, dim=0).reshape(-1, 1)
        train_label = torch.cat(train_label, dim=0).reshape(-1, 1)
        test_features = torch.cat(test_features, dim=0).reshape(-1, 1)
        test_label = torch.cat(test_label, dim=0).reshape(-1, 1)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Classification]  acc = %.4f' % svm_acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    # if val_writer is not None:
    #     val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)

# visualization
def test(base_model, test_dataloader, args, config, logger = None):

    # img_resize = transforms.Resize(args.img_size)
    
    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156", #plane
        "04379243",  #table
        "03790512", #motorbike
        "03948459", #pistol
        "03642806", #laptop
        "03467517",     #guitar
        "03261776", #earphone
        "03001627", #chair
        "02958343", #car
        "04090263", #rifle
        "03759954", # microphone
    ]

    exp_name = args.exp_name
    vis_root = f'./visualization/{exp_name}'
    zip_file = f'{vis_root}/{exp_name}.zip'

    img_resize = transforms.Resize(args.img_size)

    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            # print(idx)
            # stop = 25
            # if idx < stop:
            #     continue
            # if idx > stop:
            #     break
            # if not 30 < idx < 35:
            #     continue
            a, b = 30, -45
            npoints = config.npoints
            dataset_name = config.dataset.test._base_.NAME

            scan_name = data['scan_name'][0]
            if not scan_name in ['5284', '5221', '5681', '5965', "5197"]:
                continue
            print(scan_name)
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

            # dense_points, vis_points = base_model(points, vis=True)
            # dense_points, vis_points, centers = base_model(points, vis=True)
            # dense_points, vis_points, mask_points, centers = base_model.pc_model(points, vis=True)

            
            final_image = []
            data_path = f'{vis_root}/{scan_name}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            '''load img input'''
            imgs = data['full_img']
            imgs = img_resize(imgs)

            if args.use_gpu:
                imgs = imgs.cuda().float()


            calib_Rtilt = data['calib_Rtilt'].cuda()
            calib_K = data['calib_K'].cuda()
            scale = data['scale'].cuda()
            

            patch_size = base_model.img_branch.patch_embed.patch_size
            grid_size = base_model.img_branch.patch_embed.grid_size
            '''DONE'''
            '''get prediction for image and point cloud'''
            img_ret, pc_ret, pc2img, (attn_pc, attn_joint) = base_model(points, imgs, calib_Rtilt, calib_K, scale, args.img_size, vis=True)
            '''DONE'''
            #Image.fromarray(img_convert).save(os.path.join(data_path, "./pc_masked.jpg"))
            np.save(os.path.join(data_path, 'attention_pc.npy'), attn_pc.detach().cpu().numpy())
            np.save(os.path.join(data_path, 'attention_joint.npy'), attn_joint.detach().cpu().numpy())

            # img_pred, img_mask, ids_restore = img_ret
            # np.save(os.path.join(data_path, 'img_mask.npy'), img_mask.detach().cpu().numpy())
            # np.save(os.path.join(data_path, 'ids_restore.npy'), ids_restore.detach().cpu().numpy())
            dense_points, vis_points, mask_points, centers = pc_ret
            # run_one_image(imgs, img_pred, img_mask, args.img_size, patch_size, grid_size, os.path.join(data_path, 'mae_vis.png'))
            # if pc2img is not None:
                # run_one_image(imgs, pc2img, img_mask, args.img_size,  patch_size, grid_size, os.path.join(data_path, 'pc2img_vis.png'))
            # save masked points
            # mask_points = mask_points.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path, 'mask.txt'), mask_points, delimiter=';')

            # points = points.squeeze().detach().cpu().numpy()
            # print(points.shape)
            # np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')
            # points = misc.get_ptcloud_img(points,a,b)
            # final_image.append(points[150:650,150:675,:])

            # centers = centers.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
            # centers = misc.get_ptcloud_img(centers)
            # final_image.append(centers)
            vis_points = vis_points[:].reshape(-1, 3).unsqueeze(0)  #ger certain pc token, for attention map
        
            vis_points = vis_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            np.save(os.path.join(data_path, 'vis.npy'), vis_points)
            vis_points = misc.get_ptcloud_img(vis_points,a,b)

            final_image.append(vis_points[150:650,150:675,:])

            # dense_points = dense_points.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            # dense_points = misc.get_ptcloud_img(dense_points,a,b)
            # final_image.append(dense_points[150:650,150:675,:])

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)
            # break

        # zipDir(dirpath=vis_root, outFullName=zip_file)
        return
