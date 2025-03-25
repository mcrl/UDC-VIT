# UDC-VIT benchmark: FastDVDnet

## Modifications

+ Dataset
  + We use the UDC-VIT dataset, comprising a training set of 510 scenes and 91,800 frames, a validation set of 69 scenes and 12,420 frames, and a test set of 68 scenes and 12,240 frames.
  + Instead of using the NVIDIA's Data Loading Library (DALI), we employ the PyTorch DataLoader tailored to the UDC-VIT dataset in `npy` format.

+ Model
  + We adhere to the original model in its original form.

+ Training Options
  + To accommodate FHD resolution and multiple degradations in the UDC-VIT dataset, we increase the patch size from 64 to 256.
  + We extend the training duration of FastDVDNet to 400 epochs, compared to the original 95, to ensure the model reaches full saturation.
  + We set the noise level to zero.


## Dependencies
+ GPU: NVIDIA GeForce RTX 3090 (24GB)
+ python=3.6.10
+ CUDA=11.0
+ pytorch=1.7.1
+ opencv=3.4.2
+ Detailed information can be found in the requirements.yml file.


## Installation

```bash
git clone https://github.com/mcrl/UDC-VIT
cd UDC-VIT/benchmarks/FastDVDNet
conda env create --file environment.yml
```

## How to

### Data Preparation
+ Locate the dataset within your file system.
+ Ensure that you specify the dataset path in the train.sh and test.sh scripts.

### Training
```bash
sh train.sh
```

### Test
```bash
sh test.sh
```


## Acknowledgement and License
This work is licensed under MIT License. The code is mainly modified from [Original Work](https://github.com/m-tassano/fastdvdnet). Refer to the original repositories for more details.
