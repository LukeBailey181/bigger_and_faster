#!/bin/sh
PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
SUPER_MODEL=${PROJECT_ROOT}/model/KD-S1
SAVE_PATH_DIR=${PROJECT_ROOT}/data/latency_dataset

python ${PROJECT_ROOT}/inference_time_evaluation.py \
    --bert_model=$SUPER_MODEL \
    --save_dir=$SAVE_PATH_DIR 

