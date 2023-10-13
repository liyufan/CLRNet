<div align="center">
  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clrnet-cross-layer-refinement-network-for/lane-detection-on-culane)](https://paperswithcode.com/sota/lane-detection-on-culane?p=clrnet-cross-layer-refinement-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clrnet-cross-layer-refinement-network-for/lane-detection-on-llamas)](https://paperswithcode.com/sota/lane-detection-on-llamas?p=clrnet-cross-layer-refinement-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clrnet-cross-layer-refinement-network-for/lane-detection-on-tusimple)](https://paperswithcode.com/sota/lane-detection-on-tusimple?p=clrnet-cross-layer-refinement-network-for)


</div>


<div align="center">

# CLRNet: Cross Layer Refinement Network for Lane Detection

</div>



Pytorch implementation of the paper "[CLRNet: Cross Layer Refinement Network for Lane Detection](https://arxiv.org/abs/2203.10350)" (CVPR2022 Acceptance).

## Introduction
![Arch](.github/arch.png)
- CLRNet exploits more contextual information to detect lanes while leveraging local detailed lane features to improve localization accuracy. 
- CLRNet achieves SOTA result on CULane, Tusimple, and LLAMAS datasets.

## Installation

### Prerequisites
Only test on Ubuntu18.04 and 20.04 with:
- Python >= 3.8 (tested with Python3.8)
- PyTorch >= 1.6 (tested with Pytorch1.6)
- CUDA (tested with cuda10.2)
- Other dependencies described in `requirements.txt`

### Clone this repository
Clone this code to your workspace. 
We call this directory as `$CLRNET_ROOT`
```Shell
git clone https://github.com/Turoad/clrnet
git submodule update --init --recursive
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n clrnet python=3 -y
conda activate clrnet
```

### Fix imgaug
`imgaug` has not been updated for a long time and is not compatible `numpy>=1.20.0`. We need to fix it.
```Shell
git clone https://github.com/aleju/imgaug
cd imgaug
```
Rename all `np.bool` to `np.bool_` and all `np.complex` to `np.complex_` in this repository, then install it.
```Shell
pip install .
```

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia

# Or you can install via pip
pip install torch torchvision

# Install build tools
pip install ninja pytest-runner

# Install python packages
pip install -r requirements.txt
python setup.py build develop
```

### Use docker to run CLRNet (recommended).
We recommend to use docker for deployment. If you have not installed docker, see https://docs.docker.com/. 

You can pull our pre-build docker image for convenience:
```
docker pull turoad/clrnet:torch1.13-tensorrt8.5
```
Another pre-build docker image is `turoad/clrnet:torch1.8-tensorrt7.2`.

a. You can start the docker with:
```
./docker_scripts/docker_start.sh
```
**Note**, make sure your docker tag is modified in `docker_scripts/env.sh`.

b. You can enter into the docker with:
```
./docker_scripts/docker_into.sh
```
This will create a new bash session in the container. 

c. Build for CLRNet:
```
cd $CLRNET_ROOT
python setup.py build develop
```

### Data preparation

#### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```


#### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x3
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file
```
If you want to train with your own data, remember to modify the following files:
```
configs/clrnet/clr_resnetxx_tusimple.py
# modify '3626' in 'total_iter' to your own training data size
# modify 'test_json_file' to your own test json file

clrnet/datasets/tusimple.py
# modify 'SPLIT_FILES'

tools/generate_seg_tusimple.py
# modify 'TRAIN_SET', 'VAL_SET', 'TEST_SET'
```

For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# this will generate seg_label directory
```

#### LLAMAS
Dowload [LLAMAS](https://unsupervised-llamas.com/llamas/). Then extract them to `$LLAMASROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $LLAMASROOT data/llamas
```

Unzip both files (`color_images.zip` and `labels.zip`) into the same directory (e.g., `data/llamas/`), which will be the dataset's root. For LLAMAS, you should have structure like this:
```
$LLAMASROOT/color_images/train # data folders
$LLAMASROOT/color_images/test # data folders
$LLAMASROOT/color_images/valid # data folders
$LLAMASROOT/labels/train # labels folders
$LLAMASROOT/labels/valid # labels folders
```


## Getting Started

### Training
For training, run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/clrnet/clr_resnet18_culane.py --gpus 0
```

### Validation
For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/clrnet/clr_dla34_culane.py --validate --load_from culane_dla34.pth --gpus 0
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.

For example, run
```Shell
python main.py configs/clrnet/clr_dla34_culane.py --validate --load_from culane_dla34.pth --gpus 0 --view
```

### Inference
See `tools/detect.py` for detailed information.
```
python tools/detect.py --help

usage: detect.py [-h] [--img IMG] [--show] [--savedir SAVEDIR]
                 [--load_from LOAD_FROM]
                 config

positional arguments:
  config                The path of config file

optional arguments:
  -h, --help            show this help message and exit
  --img IMG             The path of the img (img file or img_folder), for
                        example: 'data/*.png' (the quotes must be passed)
  --show                Whether to show the image
  --savedir SAVEDIR     The root of save directory
  --load_from LOAD_FROM
                        The path of model
```
To run inference on example images in `./images` and save the visualization images in `vis` folder:
```
python tools/detect.py configs/clrnet/clr_resnet18_tusimple.py --img images\
          --load_from tusimple_r18.pth --savedir ./vis
```
Note that if you want to use wildcards, remember to wrap the `--img` parameter in quotes, e.g. `--img 'images/*.png'` or `--img "images/*.png"` or you will get an error like this:
```
detect.py: error: unrecognized arguments: images/2.jpg
```

## Results
![F1 vs. Latency for SOTA methods on the lane detection](.github/latency_f1score.png)

[assets]: https://github.com/turoad/CLRNet/releases

### CULane

|   Backbone  |  mF1 | F1@50  | F1@75 |
| :---  |  :---:   |   :---:    | :---:|
| [ResNet-18][assets]     | 55.23  |  79.58   | 62.21 |
| [ResNet-34][assets]     | 55.14  |  79.73   | 62.11 |
| [ResNet-101][assets]     | 55.55| 80.13   | 62.96 |
| [DLA-34][assets]     | 55.64|  80.47   | 62.78 |



### TuSimple
|   Backbone   |      F1   | Acc |      FDR     |      FNR   |
|    :---       |          ---:          |       ---:       |       ---:       |      ---:       |
| [ResNet-18][assets]     |    97.89    |   96.84  |    2.28  |  1.92      | 
| [ResNet-34][assets]       |   97.82              |    96.87          |   2.27          |    2.08      | 
| [ResNet-101][assets]      |   97.62|   96.83  |   2.37   |  2.38  |



### LLAMAS
|   Backbone    |  <center>  valid <br><center> &nbsp; mF1 &nbsp; &nbsp;  &nbsp;F1@50 &nbsp; F1@75     | <center>  test <br> F1@50 |
|  :---:  |    :---:    |        :---:|
| [ResNet-18][assets] |  <center> 70.83  &nbsp; &nbsp; 96.93 &nbsp; &nbsp; 85.23 | 96.00 |
| [DLA-34][assets]     |  <center> 71.57 &nbsp; &nbsp;  97.06  &nbsp; &nbsp; 85.43  |   96.12 |

“F1@50” refers to the official metric, i.e., F1 score when IoU threshold is 0.5 between the gt and prediction. "F1@75" is the F1 score when IoU threshold is 0.75.

## Citation

If our paper and code are beneficial to your work, please consider citing:
```
@InProceedings{Zheng_2022_CVPR,
    author    = {Zheng, Tu and Huang, Yifei and Liu, Yang and Tang, Wenjian and Yang, Zheng and Cai, Deng and He, Xiaofei},
    title     = {CLRNet: Cross Layer Refinement Network for Lane Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {898-907}
}
```

## Acknowledgement
<!--ts-->
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [Turoad/lanedet](https://github.com/Turoad/lanedet)
* [ZJULearning/resa](https://github.com/ZJULearning/resa)
* [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
<!--te-->