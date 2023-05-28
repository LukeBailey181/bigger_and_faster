#!/bin/bash

#PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
PROJECT_ROOT=/n/home00/lbailey/bigger_and_faster
#TYPE=$1
TYPE=fp
EPOCH=2

TASK_NAME="rte"   # One of {"mnli", "sst-2", "mrpc", "cola", "sts-b", "qqp", "qnli", "wnli", "rte"} 

if [ $TASK_NAME = "mnli" ]; then 
DATA_DIR="/n/holylfs05/LABS/acc_lab/Lab/glue/MNLI/processed/"
elif [ $TASK_NAME = "sst-2" ]; then 
DATA_DIR="${PROJECT_ROOT}/tmp_glue_datasets/sst2/"
elif [ $TASK_NAME = "mrpc" ]; then 
DATA_DIR="${PROJECT_ROOT}/tmp_glue_datasets/mrpc/"
elif [ $TASK_NAME = "cola" ]; then 
DATA_DIR="${PROJECT_ROOT}/tmp_glue_datasets/cola/"
elif [ $TASK_NAME = "sts-b" ]; then 
DATA_DIR="${PROJECT_ROOT}/tmp_glue_datasets/stsb/"
elif [ $TASK_NAME = "qqp" ]; then 
DATA_DIR="${PROJECT_ROOT}/tmp_glue_datasets/qqp/"
elif [ $TASK_NAME = "qnli" ]; then 
DATA_DIR="${PROJECT_ROOT}/tmp_glue_datasets/qnli/"
elif [ $TASK_NAME = "wnli" ]; then 
DATA_DIR="${PROJECT_ROOT}/tmp_glue_datasets/wnli/"
elif [ $TASK_NAME = "rte" ]; then 
DATA_DIR="${PROJECT_ROOT}/tmp_glue_datasets/rte/"
fi
echo $DATA_DIR

CKPT_PATH="${PROJECT_ROOT}/conf_datasets/lat_predictor_quant.pt"
MODEL="${PROJECT_ROOT}/model/SUPER-KD-S1/output/superbert/checkpoints/superbert_epoch_4_lr_0.0001_bsz_12_grad_accu_1_512_gpu_1/epoch_3"
CAND_FILE="${PROJECT_ROOT}/cands/eval_test_${TYPE}.cands"
OUTPUT_DIR="${PROJECT_ROOT}/output/test_${TYPE}/"

if [ "$TYPE" = "fp" ]; then
CKPT_PATH="${PROJECT_ROOT}/conf_datasets/lat_predictor.pt"
fi

# Evaluation of candidates
rm -rf $OUTPUT_DIR
python ../superbert_run_en_classifier_$TYPE.py --data_dir $DATA_DIR --model $MODEL --task_name $TASK_NAME\
 --output_dir $OUTPUT_DIR --save_model_flag 1 --num_train_epochs 2 --do_lower_case --arches_file $CAND_FILE

