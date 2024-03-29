# Dynamic-Backward-Attention-Transformer
This is the official repository of the paper "Enhancing Material Features Using Dynamic Backward Attention on Cross-Resolution Patches".

To avoid leaking information, the code is hosted on anonymous.4open.science, which does not support clone nor download the whole project. Please view the code online. The main code is at torchtools/models/dpglt/dpglt_single_branch.py

If you want to try the code, please read the following guidance.

## Environment

To install the dependencies, please refer to the conda env file (modify the name and prefix first).
```
conda env create -f environment.yml
```

Or, if you prefer using docker, you can build the environment with the Dockerfile in this project:

```
sudo DOCKER_BUILDKIT=1 docker build -t dbat:1.0.0 -f Dockerfile . 
```

## Local Material Dataset
Please download the original dataset from https://vision.ist.i.kyoto-u.ac.jp/codeanddata/localmatdb/, into the folder datasets/localmatdb. Then you can zip the folder localmatdb since our dataloader assumes the images are zipped.

## Pre-trained DBAT checkpoint
Please download the pre-trained [checkpoints](https://drive.google.com/file/d/1ov6ol7A4NU8chlT3oEwx-V51gbOU7GGD/view?usp=sharing) into the folder "checkpoints"
```
mkdir -p checkpoints/dpglt_mode95/accuracy
```


## Train DBAT
To train our DBAT, you can use the code below:
```
python train_sota.py --data-root "./datasets" --batch-size 4 --tag dpglt --gpus 1 --num-nodes 1 --epochs 200 --mode 95 --seed 42
```
To test the trained model, you can specify the checkpoint path with the --test option
```
python train_sota.py --data-root "./datasets" --batch-size 4 --tag dpglt --gpus 1 --num-nodes 1 --epochs 200 --mode 95 --seed 42 --test accuracy/epoch\=126-valid_acc_epoch\=0.87.ckpt
```
