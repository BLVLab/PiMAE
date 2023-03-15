# Downstream finetuning codes

## Instructions

### 3DETR
Follow [3DETR](https://github.com/facebookresearch/3detr) codebase to prepare the training data (SUNRGBD & ScanNetV2).

Install required dependencies by
```
cd Downstream/3detr

sh install.sh
```
Run the training code (you can specify training configure in the script)
```
sh run.sh
```
<!-- 
### GroupFree-3D
Follow [GroupFree-3D](https://github.com/zeliu98/Group-Free-3D) code base to prepare training data (SUN RGB-D and ScanNetV2) as well as required dependencies.

Run the training code by
```
``` -->

### DETR
Follow [DETR](detr/README.md) to prepare data and required dependencies. Then train it by 
```
cd Downstream/detr/d2

python train_net.py --config configs/detr_256_6_6_torchvision.yaml --num-gpus 8
```

### MonoDETR
Follow [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR) codebase to prepare the training data (KITTI).
Install required dependencies by
```
cd Downstream/MonoDETR

sh install.sh
```
Run the code for training and testing.(remember to check <code>monodetr.yaml</code> where we specify path to pimae weights.)
```
bash train.sh configs/monodetr.yaml > logs/monodetr.log # training

bash test.sh configs/monodetr.yaml # testing
```
