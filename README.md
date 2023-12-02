## Installation guide
https://medium.com/@harunijaz/a-step-by-step-guide-to-installing-cuda-with-pytorch-in-conda-on-windows-verifying-via-console-9ba4cd5ccbef

## opencv mmpose or openpose

pytorch implementation of [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) including **Body and Hand Pose Estimation**, and the pytorch model is directly converted from [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) caffemodel by [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch). You could implement face keypoint detection in the same way if you are interested in. Pay attention to that the face keypoint detector was trained using the procedure described in [Simon et al. 2017] for hands.

openpose detects hand by the result of body pose estimation, please refer to the code of [handDetector.cpp](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp).
we have not used openpose in this demo

mmpose is used in the mixer example
opencv is used in the drums example

### Getting Started

#### Install Requirements

Create a python environement, eg:

    conda create -n pytorch-test python=3.11
    conda activate pytorch-test

Install pytorch 

    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

Install mmpose

    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.1"
    mim install "mmdet>=3.1.0"
    mim install "mmpose>=1.1.0"

    mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .

Install other requirements with pip

    pip install -r requirements.txt

#### Download the Models

* [dropbox](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0)
* [baiduyun](https://pan.baidu.com/s/1IlkvuSi0ocNckwbnUe7j-g)
* [google drive](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG?usp=sharing)

`*.pth` files are pytorch model, you could also download caffemodel file if you want to use caffe as backend.

Download the pytorch models and put them in a directory named `model` in the project root directory

#### Run the Mixer Demo
You can mix 2 songs moving your wrists up and down

Run:

    python vmc-01-mixer.py

to run a demo with a feed from your webcam

#### Run the Drums Demo
You can play drums with 1 red and 1 ball objects

Run:

    python vmc-02-drum.py

to run a demo with a feed from your webcam
