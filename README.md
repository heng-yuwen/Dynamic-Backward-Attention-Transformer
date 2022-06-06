# Dynamic-Backward-Attention-Transformer
This is the official repository of the paper "Enhancing Material Features Using Dynamic Backward Attention on Cross-Resolution Patches".

To install the dependencies, please refer to the conda env file.
```
conda env create -f environment.yml
```

Or, if you prefer using docker, please pull our prepared image:

```
docker pull 123mutouren/cv:1.0.0
```

## Local Material Dataset
Please download the original dataset from https://vision.ist.i.kyoto-u.ac.jp/codeanddata/localmatdb/, into the folder dataset/localmatdb. Then you can zip the folder localmatdb since our dataloader assumes the images are zipped.


