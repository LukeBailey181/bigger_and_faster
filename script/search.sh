#PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
PROJECT_ROOT=/n/home00/lbailey/bigger_and_faster
OP=$1
TYPE=$2
LATENCY_CONSTRAINT=$3
DATA_DIR=$4
TASK_NAME=$5

CKPT_PATH="${PROJECT_ROOT}/conf_datasets/lat_predictor_quant.pt"
MODEL="${PROJECT_ROOT}/model/SUPER-KD-S1/output/superbert/checkpoints/superbert_epoch_4_lr_0.0001_bsz_12_grad_accu_1_512_gpu_1/epoch_3"
SEARCH_NAME="kd_${TYPE}_${LATENCY_CONSTRAINT}"
CAND_FILE="${PROJECT_ROOT}/cands/${SEARCH_NAME}.cands"
#DATA_DIR="/n/holylfs05/LABS/acc_lab/Lab/glue/MNLI/processed/"

if [ "$TYPE" = "fp" ]; then
CKPT_PATH="${PROJECT_ROOT}/conf_datasets/lat_predictor.pt"
fi

# Obtain candidates 
if [ "$OP" = "cand" ]; then
python ../searcher.py --ckpt_path $CKPT_PATH \
    --latency_constraint $LATENCY_CONSTRAINT --method Candidate --model KD \
    --candidate_file $CAND_FILE

# Random Search
elif [ "$OP" = "rand" ]; then
python ../searcher.py --ckpt_path $CKPT_PATH \
    --candidate_file $CAND_FILE --latency_constraint $LATENCY_CONSTRAINT \
    --method Random --model KD --output_file ../cands/1st_generation_$SEARCH_NAME.cands

# Fast Search
elif [ "$OP" = "fast" ]; then
python ../searcher.py --ckpt_path $CKPT_PATH \
    --candidate_file $CAND_FILE --latency_constraint $LATENCY_CONSTRAINT \
    --method Fast --model KD --output_file ../cands/1st_generation_$SEARCH_NAME.cands

# Evaluation of candidates
elif [ "$OP" = "eval" ]; then
python ../superbert_run_en_classifier_$TYPE.py --data_dir $DATA_DIR \
    --model $MODEL --task_name $TASK_NAME --output_dir ../output/$SEARCH_NAME/ \
    --do_lower_case --arches_file ../cands/1st_generation_$SEARCH_NAME.cands 

# Evolved Search
elif [ "$OP" = "evol" ]; then
python ../searcher.py --ckpt_path $CKPT_PATH  --candidate_file $CAND_FILE \
    --latency_constraint $LATENCY_CONSTRAINT --method Evolved --model KD --output_file ../cands/1st_generation_$SEARCH_NAME.evo.cands \
    --arch_perfs_file ../output/$SEARCH_NAME/subbert.results

else
    echo "Command not found"
fi
