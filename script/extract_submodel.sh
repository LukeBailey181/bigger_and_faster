PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
TYPE=$1

MODEL="${PROJECT_ROOT}/output/test_${TYPE}/"
CAND_FILE="${PROJECT_ROOT}/cands/eval_test_${TYPE}.cands"
OUPUT_DIR="${PROJECT_ROOT}/output/test_${TYPE}_sub/"
kd_flag=True


# Evaluation of candidates
rm -rf $OUPUT_DIR
python ../submodel_extractor.py \
    --model $MODEL --arch $CAND_FILE\
    --output $OUPUT_DIR

