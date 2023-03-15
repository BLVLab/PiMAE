#!/bin/bash

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

cd third_party/pointnet2 && python setup.py install
cd ../../

pip install matplotlib \
    opencv-python \
    plyfile \
    'trimesh>=2.35.39,<2.35.40' \
    'networkx>=2.2,<2.3' \
    scipy \
    tensorboardX \
    tensorboard \
    wandb \

conda install cython
cd utils && python cython_compile.py build_ext --inplace
cd ../
