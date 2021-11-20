#!/bin/sh

SAVE_PATH_DIR="./latency_dataset"

python inference_time_evaluation.py --save_dir=$SAVE_PATH_DIR
