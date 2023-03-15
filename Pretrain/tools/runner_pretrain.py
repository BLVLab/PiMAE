import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils

# sunrgbd
from torch.utils.data import DataLoader, DistributedSampler
from datasets import build_dataset

from models.build import build_multimae

pc_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

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


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
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
            pin_memory=False,
        )
        dataloaders[split + "_sampler"] = sampler

    train_sampler = dataloaders['train_sampler']
    train_dataloader = dataloaders['train']
    
    base_model = build_multimae(config, args, logger)
    
    if args.use_gpu:
        base_model.to(args.local_rank)

    img_resize = transforms.Resize(args.img_size)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])
        img_losses = AverageMeter(['Loss'])
        pc_losses = AverageMeter(['Loss'])
        pc2img_losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, data in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            npoints = config.npoints
            dataset_name = config.dataset.train._base_.NAME

            if dataset_name == 'ShapeNet':
                points = data['point_clouds'].cuda()
                points = misc.fps(points, npoints)
            elif dataset_name == 'ModelNet':
                points = data['point_clouds'][0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            
            imgs = data['full_img']
            imgs = img_resize(imgs)
           
            if args.use_gpu:
                imgs.to(args.local_rank, non_blocking=True)

            calib_Rtilt = data['calib_Rtilt']
            calib_K = data['calib_K']
            scale = data['scale']
    
            pc_loss, img_loss, pc2img_loss = base_model(points, imgs, calib_Rtilt, calib_K, scale, args.img_size)

            if pc2img_loss is not None:
                loss = pc_loss + img_loss + pc2img_loss
            else:
                if img_loss is not None and pc_loss is not None:
                    loss = pc_loss + img_loss
                else:
                    if pc_loss is not None:
                        loss = pc_loss
                    if img_loss is not None:
                        loss = img_loss

            try:
                loss.backward()
            except:     
                loss = loss.mean()
                loss.backward()

                img_loss = img_loss.mean() 
                pc_loss = pc_loss.mean()
                if pc2img_loss is not None:
                    pc2img_loss = pc2img_loss.mean()
                

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
            #only2d
            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()])
                if img_loss is not None:
                    img_losses.update([img_loss.item()])
                else:
                    img_losses.update([0.0])
                if pc_loss is not None:
                    pc_losses.update([pc_loss.item()])
                else:
                    pc_losses.update([0.0])
                if pc2img_loss is not None:
                    pc2img_losses.update([pc2img_loss.item()])
            else:
                losses.update([loss.item()])
                if img_loss is not None:
                    img_losses.update([img_loss.item()])
                else:
                    img_losses.update([0.0])
                if pc_loss is not None:
                    pc_losses.update([pc_loss.item()])
                else:
                    pc_losses.update([0.0])
                if pc2img_loss is not None:
                    pc2img_losses.update([pc2img_loss.item()])

            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                if pc_loss is not None:
                    train_writer.add_scalar('Loss/Batch/PC_Loss', pc_loss.item(), n_itr)
                if pc2img_loss is not None:
                    train_writer.add_scalar('Loss/Batch/PC2IMG_Loss', pc2img_loss.item(), n_itr)
                if img_loss is not None:
                    train_writer.add_scalar('Loss/Batch/IMG_Loss', img_loss.item(), n_itr)
               
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                if pc2img_loss is not None:
                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s PC_Losses = %s PC2IMG_Losses = %s IMG_Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], ['%.4f' % l for l in pc_losses.val()], ['%.4f' % l for l in pc2img_losses.val()], ['%.4f' % l for l in img_losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
                else:
                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s PC_Losses = %s IMG_Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], ['%.4f' % l for l in pc_losses.val()], ['%.4f' % l for l in img_losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/PC_Loss_1', pc_losses.avg(0), epoch)
            if pc2img_loss is not None:
                train_writer.add_scalar('Loss/Epoch/PC2IMG_Loss_1', pc2img_losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/IMG_Loss_1', img_losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s PC_Losses = %s IMG_Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], ['%.4f' % l for l in pc_losses.val()], ['%.4f' % l for l in img_losses.val()],
             optimizer.param_groups[0]['lr']), logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def test_net():
    pass