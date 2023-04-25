#!/bin/sh
PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
SAVE_PATH_DIR=${PROJECT_ROOT}/data/hf_test

python ${PROJECT_ROOT}/hf_runtime_test.py \
    --save_dir=$SAVE_PATH_DIR 
