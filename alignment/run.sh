#!/bin/bash

# Run the command below
# MIN_DIR=0 MAX_DIR=1000 TARGET_H=1060 TARGET_W=1900 MAX_SHIFT=10 MSE=1 FFT_ABS=0 FFT_ANGLE=1 IMG_INTENSITY=255 RUN_CONVERT=1 RUN_ALIGN=1 RUN_PCK=1 ./run.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4

# Change for multinode config
set -a
: ${MIN_DIR=0}
: ${MAX_DIR=1000}
: ${TARGET_H=1060}
: ${TARGET_W=1900}
: ${MAX_SHIFT=10}
: ${MSE=1}
: ${FFT_ABS=0}
: ${FFT_ANGLE=1}
: ${IMG_INTENSITY=255}
: ${RUN_CONVERT=1}
: ${RUN_ALIGN=1}
: ${RUN_PCK=1}
set +a

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate udc

python3 main.py ${MIN_DIR} ${MAX_DIR} ${TARGET_H} ${TARGET_W} ${MAX_SHIFT} ${MSE} ${FFT_ABS} ${FFT_ANGLE} ${IMG_INTENSITY} ${RUN_CONVERT} ${RUN_ALIGN} ${RUN_PCK}
