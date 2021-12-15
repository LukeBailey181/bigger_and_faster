PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
TYPE=$1
EPOCH=$2

CKPT_PATH="${PROJECT_ROOT}/conf_datasets/lat_predictor_quant.pt"
MODEL="${PROJECT_ROOT}/model/SUPER-KD-S1/output/superbert/checkpoints/superbert_epoch_4_lr_0.0001_bsz_12_grad_accu_1_512_gpu_1/epoch_3"
CAND_FILE="${PROJECT_ROOT}/cands/eval_test_${TYPE}.cands"
DATA_DIR="/n/acc_lab/glue/MNLI"
OUPUT_DIR="${PROJECT_ROOT}/output/test_${TYPE}/"


if [ "$TYPE" = "fp" ]; then
CKPT_PATH="${PROJECT_ROOT}/conf_datasets/lat_predictor.pt"
fi

# Evaluation of candidates
rm -rf $OUPUT_DIR
python ../superbert_run_en_classifier_$TYPE.py --data_dir $DATA_DIR \
    --model $MODEL --task_name "mnli" --output_dir $OUPUT_DIR \
    --num_train_epochs $EPOCH --do_lower_case \
    --arches_file $CAND_FILE

