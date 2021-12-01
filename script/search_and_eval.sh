CKPT_PATH="../conf_datasets/lat_predictor.pt"
MODEL="/n/holyscratch01/acc_lab/Users/yhjin0509/bigger_and_faster/model/output/superbert/checkpoints/superbert_epoch_4_lr_0.0001_bsz_12_grad_accu_1_512_gpu_1/epoch_3"
CAND_FILE="../cands/kd_100x"
DATA_DIR="/n/acc_lab/glue/MNLI"
OP=$1


for i in {1..4}
    do
        # Evaluation of candidates
        python ../superbert_run_en_classifier.py --data_dir $DATA_DIR \
            --model $MODEL --task_name "mnli" --output_dir ../output/ \
            --do_lower_case --arches_file ../cands/1st_generation.evo.cands

        # Evolved Search
        python ../searcher.py --ckpt_path $CKPT_PATH  --candidate_file $CAND_FILE \
            --latency_constraint 100 --method Evolved --model KD --output_file ../cands/1st_generation.evo.cands \
            --arch_perfs_file ../output/subbert.results
    done

