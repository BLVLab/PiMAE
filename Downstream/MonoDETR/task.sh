#!/bin/bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

cd lib/models/monodetr/ops/
bash make.sh

cd ../../../..

bash train.sh configs/monodetr_pimae.yaml > logs/loadpimae.log