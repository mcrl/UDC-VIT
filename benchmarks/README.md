# Benchmarks

This repository provides benchmarks for various UDC restoration models.

## Downloading Checkpoints

Download the checkpoint files for each model from [here](https://drive.google.com/file/d/1SDPfRpd3dZh9NXqLHURxq3s8OXm_BCmT/view?usp=sharing).
Then, place them inside the `checkpoints` directory, maintaining the following structure:


    benchmarks/
    ├── DDRNet/
    ├── DISCNet/
    ├── EDVR/
    ├── ESTRNN/
    ├── FastDVDNet
    ├── UDC-UNet
    └── checkpoints
        ├── DDRNet/
        │   └── 400000_G.pth
        ├── DISCNet/
        │   └── net_g_980000.pth
        ├── EDVR/
        │   └── net_g_600000.pth
        ├── ESTRNN/
        │   └── model_best.pth.tar
        ├── FastDVDNet/
        │   └── ckpt_e400.pth
        └── UDC-UNet/
            └── net_g_latest.pth

## Model-Specific Instructions

For detailed instructions on each model, please refer to the corresponding `README.md` file inside each model’s directory.

