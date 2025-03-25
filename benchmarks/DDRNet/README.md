# UDC-VIT benchmark: DDRNet

## Modifications

+ Dataset
  + We use the UDC-VIT dataset, comprising a training set of 510 scenes and 91,800 frames, a validation set of 69 scenes and 12,420 frames, and a test set of 68 scenes and 12,240 frames.

+ Model
  + We adhere to the original model in its original form.

+ Test Options
  + When employing the default configuration for the tile parameter in test.py as specified by the authors, each frame is divided into patches sized $3 \times 256 \times 256$, with 50 frames being input simultaneously. However, patch-wise inference introduces the borderline between patches. To address this, we conduct inference at full resolution ($3 \times 1,060 \times 1,900$) with ten frames at a time.


## Dependencies
+ GPU: Tesla V100-PCIE-32GB
+ Python=3.8.19
+ CUDA = 12.4 
+ ROCM=5.7 (for AMD GPU users)
+ PyTorch=2.2.1
+ Detailed information can be found in the environment.yml file.


## Installation

```bash
git clone https://github.com/mcrl/UDC-VIT
cd UDC-VIT/benchmarks/DDRNet
conda env create --file environment.yml
```

## How to

### Data Preparation
+ Locate the dataset within your file system.
+ Ensure that you specify the dataset path in the json file below.
```bash
./options/DDRNet/train_DDRNet_npy.json
```

### Train
```bash
torchrun --nproc_per_node=4 --master_port=23333 train.py --opt ./options/DDRNet/train_DDRNet.json --dist True
```

### Test
```bash
python test.py --save_result
```


## Acknowledgement and License
This work is licensed under MIT license. The code is mainly modified from [Original Work](https://github.com/ChengxuLiu/DDRNet). Refer to the original repositories for more details.
