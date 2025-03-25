# UDC-VIT benchmark: ESTRNN

## Modifications

+ Dataset
  + We use the UDC-VIT dataset, comprising a training set of 510 scenes and 91,800 frames, a validation set of 69 scenes and 12,420 frames, and a test set of 68 scenes and 12,240 frames.

+ Model
  + We adhere to the original model in its original form.


## Dependencies
+ GPU: NVIDIA GeForce RTX 3090 (24GB)
+ Python=3.6.10
+ ROCM=3.10 (for AMD GPU users)
+ PyTorch=1.10.2
+ Detailed information can be found in the environment.yml file.


## Installation

```bash
git clone https://github.com/mcrl/UDC-VIT
cd UDC-VIT/benchmarks/ESTRNN
conda env create --file environment.yml
```

## How to

### Data Preparation
+ Locate the dataset within your file system.
+ Ensure that you specify the dataset path in the para/parameter.py.

### Train
```bash
python main.py --lr 1e-4 --batch_size 4 --num_gpus 4 --trainer_mode ddp
```

### Test
```bash
python main.py --test_only --test_checkpoint ../checkpoints/ESTRNN/model_best.pth.tar
```


## Acknowledgement and License
This work is licensed under MIT license. The code is mainly modified from [Original Work](https://github.com/zzh-tech/ESTRNN/tree/master). Refer to the original repositories for more details.
