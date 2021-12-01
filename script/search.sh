CKPT_PATH="../conf_datasets/lat_predictor.pt"
MODEL="/n/holyscratch01/acc_lab/Users/yhjin0509/bigger_and_faster/model/output/superbert/checkpoints/superbert_epoch_4_lr_0.0001_bsz_12_grad_accu_1_512_gpu_1/epoch_3"
CAND_FILE="../cands/kd_100x"
DATA_DIR="/n/acc_lab/glue/MNLI"
OP=$1

# Obtain candidates 
if [ "$OP" = "cand" ]; then
python ../searcher.py --ckpt_path $CKPT_PATH \
    --latency_constraint 100 --method Candidate --model KD \
    --candidate_file $CAND_FILE

# Random Search
elif [ "$OP" = "rand" ]; then
python ../searcher.py --ckpt_path $CKPT_PATH \
    --candidate_file $CAND_FILE --latency_constraint 100 \
    --method Random --model KD --output_file ../cands/1st_generation.cands

# Fast Search
elif [ "$OP" = "fast" ]; then
python ../searcher.py --ckpt_path $CKPT_PATH \
    --candidate_file $CAND_FILE --latency_constraint 100 \
    --method Fast --model KD --output_file ../cands/1st_generation.fast.cands

# Evaluation of candidates
elif [ "$OP" = "eval" ]; then
python ../superbert_run_en_classifier.py --data_dir $DATA_DIR \
    --model $MODEL --task_name "mnli" --output_dir ../output/ \
    --do_lower_case --arches_file ../cands/1st_generation.cands 

# Evolved Search
elif [ "$OP" = "evol" ]; then
python ../searcher.py --ckpt_path $CKPT_PATH  --candidate_file $CAND_FILE \
    --latency_constraint 100 --method Evolved --model KD --output_file ../cands/1st_generation.evo.cands \
    --arch_perfs_file ../output/subbert.results

else
    echo "Command not found"
fi
