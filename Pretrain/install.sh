#!/bin/bash

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

conda install cython
cd utils_detr && python cython_compile.py build_ext --inplace
cd ../

# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../../
cd ./extensions/emd
python setup.py install --user
cd ../../

# pointnet2
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade http://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
