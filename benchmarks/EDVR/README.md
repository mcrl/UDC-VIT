# UDC-VIT benchmark: EDVR

## Modifications

+ Dataset
  + We use the UDC-VIT dataset, comprising a training set of 510 scenes and 91,800 frames, a validation set of 69 scenes and 12,420 frames, and a test set of 68 scenes and 12,240 frames.

+ Model
  + We adhere to the original model in its original form.

+ Training Options
  + To address out-of-memory issues with EDVR, which boasts 23.6 M parameters, we reduce the patch size from 256 to 192.

+ Test Options
 + To address out-of-memory issues with EDVR, we divide each frame into two patches of size $3 \times 1,060 \times 1,060$ each and merge them afterward.


## Dependencies
+ GPU: NVIDIA GeForce RTX 3090 (24GB)
+ Python=3.7.16
+ CUDA = 11.7
+ ROCM=5.7 (for AMD GPU users)
+ PyTorch=1.13.1
+ Detailed information can be found in the environment.yml file.


## Installation

```bash
git clone https://github.com/mcrl/UDC-VIT
cd UDC-VIT/benchmarks/EDVR
conda env create --file environment.yml
python setup.py develop
```

## How to

### Data Preparation
+ Locate the dataset within your file system.
+ Ensure that you specify the dataset path in the train.sh and test.sh scripts.

### Train
```bash
sh train.sh
```

### Test
```bash
torchrun --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_L_UDC-VIT_patch_1.yml --launcher pytorch # for patch 1
torchrun --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_L_UDC-VIT_patch_2.yml --launcher pytorch # for patch 2
python merge_patches.py # merge patches
```


## Acknowledgement and License
This work is licensed under Apache 2.0 license. The code is mainly modified from [Original Work](https://github.com/xinntao/EDVR). Refer to the original repositories for more details.
