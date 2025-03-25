# UDC-VIT benchmark: DISCNet

## Modifications

+ Dataset
  + We use the UDC-VIT dataset, comprising a training set of 510 scenes and 91,800 frames, a validation set of 69 scenes and 12,420 frames, and a test set of 68 scenes and 12,240 frames.
  + We modify the PyTorch DataLoader to use normalization with `255` instead of Reinhard tone mapping.
  + The DataLoader (PairedImgPSFNpyDataset) randomly selects one frame per video from the UDC-VIT dataset for each iteration during the training and validation phases.
  + During the test phase, the DataLoader (PairedImgPSFNpyTestDataset) processes all images from the UDC-VIT dataset.

+ Model
  + We adhere to the original model in its original form.


## Dependencies
+ GPU: NVIDIA GeForce RTX 3090 (24GB)
+ Python = 3.8.19
+ CUDA = 11.7
+ ROCM = 5.4.2 (for AMD GPU users)
+ PyTorch = 2.0.1
+ mmcv = 1.7.1


## Installation

+ Install Python and PyTorch
+ Run following command

```bash
git clone https://github.com/mcrl/UDC-VIT
cd UDC-VIT/benchmarks/DISCNet
conda env create --file environment.yml
conda activate disc_vit
bash make.sh
```

+ Install `MMCV==1.7.1` with [this repository](https://github.com/mcrl/mmcv-for-UDC-SIT)

```bash
git clone git@github.com:mcrl/mmcv-for-UDC-SIT.git
cd mmcv-for-UDC-SIT
pip install -r requirements/optional.txt
pip install -e . -v
```

> Installation with `pip`, `conda`, `mim` did not work since Aug. 2023. We ship our workaround.


## Howto

### Data Preparation

+ Locate the dataset within your file system.
+ Run to generate meta-info-files

```bash
python generate-metainfo.py \
  --train-input /path/to/train/input \ # Required
  --val-input /path/to/val/input \ # Required
  --test-input /path/to/test/input # Optional
```

You may modify `psf.npy` file location. Refer to `python generate-metainfo.py --help`.

### Reference to Train/Test Command

+ We run training in conda environment
+ If you use different setup (i.e. SLURM), refer to [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md) documentation.

### Training

+ Modify `option/train/train.yml` to specify data path.
+ The run command is as follows:

```bash
bash train.sh
```

### Test

+ Modify `option/test/test.yml` to specify data path.
+ The run command is as follows:

```bash
bash test.sh
```

## License and Acknowledgement

This work is licensed under MIT License. The codes are mainly from following repositories.
For more information, refer to [original work](https://github.com/jnjaby/DISCNet), [BasicSR](https://github.com/XPixelGroup/BasicSR) and [mmcv-for-UDC-SIT](https://github.com/mcrl/mmcv-for-UDC-SIT). Refer to the original repositories for more details.
