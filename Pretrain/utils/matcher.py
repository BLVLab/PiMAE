import numpy as np
import cv2
import os
import scipy.io as sio # to load .mat files for depth points


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[:,2] *= -1
    return pc2

def project_upright_depth_to_camera(pc,Rtilt):
    ''' project point cloud from depth coord to camera coordinate
        Input: (N,3) Output: (N,3)
    '''
    # Project upright depth to depth coordinate
    pc2 = np.dot(np.transpose(Rtilt), np.transpose(pc[:,0:3])) # (3,n)
    return flip_axis_to_camera(np.transpose(pc2))

def project_upright_depth_to_image(pc,Rtilt,K):
    ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
    pc2 = project_upright_depth_to_camera(pc,Rtilt)
    uv = np.dot(pc2, np.transpose(K)) # (n,3)
    uv[:,0] /= uv[:,2]
    uv[:,1] /= uv[:,2]
    return uv[:,0:2], pc2[:,2]


def project_upright_depth_to_image_torch(pc,Rtilt,K):
    ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
    pc2 = project_upright_depth_to_camera(pc,Rtilt)
    uv = np.dot(pc2, np.permute(K)) # (n,3)
    uv[:,0] /= uv[:,2]
    uv[:,1] /= uv[:,2]
    return uv[:,0:2], pc2[:,2]