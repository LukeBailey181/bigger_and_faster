#!/bin/sh
PROJECT_ROOT=/Users/lukebailey/Documents/ProDocuments/projects/CS242/bigger_and_faster
LAT_DATASET_PATH=${PROJECT_ROOT}/conf_datasets/lat_quant_.tmp
PATH_TO_SAVE_MODEL=${PROJECT_ROOT}/conf_datasets/lat_predictor.pt

python ${PROJECT_ROOT}/latency_predictor.py \
    --lat_dataset_path=${LAT_DATASET_PATH} \
    --ckpt_path=${PATH_TO_SAVE_MODEL} \



