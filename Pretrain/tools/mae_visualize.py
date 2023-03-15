import sys
import os

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import cv2

from torchvision import transforms, utils


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title='', path=None):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    img_denorm = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    plt.imshow(img_denorm)
    plt.title(title, fontsize=16)
    plt.axis('off')
    if path:
        img_denorm = img_denorm.numpy()
        # img = Image.fromarray(img_denorm)
        # img.save(f'{path}/{title}.png')
        # utils.save_image(img_denorm, f'{path}/{title}.png')
        cv2.imwrite(f'{path}/{title}.png', img_denorm)
    return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


# def unpatchify(x, patch_size):
#     """
#     x: (N, L, patch_size**2 *3)
#     imgs: (N, 3, H, W)
#     """
#     p = patch_size[0]
#     h = w = int(x.shape[1] ** .5)
#     assert h * w == x.shape[1]

#     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
#     x = torch.einsum('nhwpqc->nchpwq', x)
#     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
#     return imgs

def unpatchify(x, patch_size, grid_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size[0]
    h, w = grid_size
    # h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs


def run_one_image(img, y, mask, img_size, patch_size, grid_size, save_path):
    img_resize = transforms.Resize(img_size)
    # input shape (1, 3, 530, 730)
    img = img.squeeze()
    # save
    img = img_resize(img)  # (3, 244, 244)

    x = img.unsqueeze(dim=0)

    # x = torch.tensor(img)
    # make it a batch-like
    # x = x.unsqueeze(dim=0)
    # x = torch.einsum('nhwc->nchw', x)
    # run MAE
    # loss, y, mask = model(x.cuda().float(), mask_ratio=0.75)

    y = unpatchify(y, patch_size, grid_size)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()


    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size[0] * patch_size[1] * 3)  # (N, H*W, p*p*3)
    mask = unpatchify(mask, patch_size, grid_size)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)
    x = x.detach().cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 33]

    plt.subplot(2, 2, 1)
    show_image(x[0], "original")

    plt.subplot(2, 2, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(2, 2, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(2, 2, 4)
    show_image(im_paste[0], "reconstruction+visible")

    plt.savefig(save_path+'/pimae_img_vis.png')
    plt.show()