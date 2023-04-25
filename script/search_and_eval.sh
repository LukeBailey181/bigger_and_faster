PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
TYPE=$1
LATENCY_CONSTRAINT=$2
CKPT_PATH="${PROJECT_ROOT}/conf_datasets/lat_predictor_quant.pt"
MODEL="${PROJECT_ROOT}/model/SUPER-KD-S1/output/superbert/checkpoints/superbert_epoch_4_lr_0.0001_bsz_12_grad_accu_1_512_gpu_1/epoch_3"
SEARCH_NAME="kd_${TYPE}_${LATENCY_CONSTRAINT}"
CAND_FILE="${PROJECT_ROOT}/cands/${SEARCH_NAME}.cands"
DATA_DIR="/n/holylfs05/LABS/acc_lab/Lab/glue/MNLI/processed/"

if [ "$TYPE" = "fp" ]; then
CKPT_PATH="${PROJECT_ROOT}/conf_datasets/lat_predictor.pt"
fi

for i in {1..3}
    do
        echo ">>> Starting Evaluation of EE Iteration ${i} ..."
        # Evaluation of candidates
        python ../superbert_run_en_classifier_$TYPE.py --data_dir $DATA_DIR \
            --model $MODEL --task_name "mnli" --output_dir ../output/$SEARCH_NAME/ \
            --do_lower_case --arches_file ../cands/1st_generation_$SEARCH_NAME.evo.cands

        echo ">>> Starting Search of EE Iteration ${i} ..."
        # Evolved Search
        python ../searcher.py --ckpt_path $CKPT_PATH  --candidate_file $CAND_FILE \
            --latency_constraint $LATENCY_CONSTRAINT --method Evolved --model KD --output_file ../cands/1st_generation_$SEARCH_NAME.evo.cands\
            --arch_perfs_file ../output/$SEARCH_NAME/subbert.results
    done

