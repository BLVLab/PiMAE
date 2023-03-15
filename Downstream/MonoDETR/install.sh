#!/bin/bash
conda create -n monodetr python=3.8
conda activate monodetr

conda install pytorch torchvision cudatoolkit

pip install -r requirements.txt

cd lib/models/monodetr/ops/
bash make.sh

cd ../../../..

mkdir logs