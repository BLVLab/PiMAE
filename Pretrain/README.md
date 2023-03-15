# Pretraining Code for PiMAE

## Easy-Usage

We have provided a easy tutorial to use PiMAE's pre-trained 3D extractor. You can easily modify the code to fit in your model. 

Get our pretrained models from [here](https://drive.google.com/file/d/1sJWa_j71zA1-XELE8J5fhl__bKYplbCs/view?usp=sharing) and place it as <code>./Pretrain/pimae.pth</code>.

Install minimum required dependencies then simply run the tutorial code by: 
```
pip install torch torchvision

python Pretrain/tutorial_load.py
```

### Install
First, clone this repository into your local machine.
```
git clone https://github.com/BLVLab/PiMAE.git
```
Next, install required dependencies.
```
cd Pretrain

sh intall.sh
```

### Data Preparation
We follow the [VoteNet](https://github.com/facebookresearch/votenet) to preprocess our data. The instructions for preparing SUN RGB-D are [here](https://github.com/facebookresearch/votenet/tree/main/sunrgbd).
Remember to Edit the dataset paths in <code>Pretrain/datasets/sunrgbd.py</code>.

### Training

```
python main.py --config cfgs/pretrain.yaml --exp_name pimae
```

### Visualization
To get reconstruction visualization like this.

<img src="../Assets/visulization.png" width="60%">

```
python main_vis.py \
	--test \
	--ckpts ./experiments/pretrain/cfgs/pimae/ckpt-last.pth \
	--config ./experiments/pretrain/cfgs/pimae/config.yaml \
	--exp_name vis_pimae \
```
