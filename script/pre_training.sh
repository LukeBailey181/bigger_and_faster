PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
GENERATED_DIR=${PROJECT_ROOT}/data/generated
OUTPUT_DIR=${PROJECT_ROOT}/model/SUPER-KD-S1
EPOCHS=2
ACC_STEPS=1
BATCH_SIZE=8
LR=1e-4
MAX_SEQ_LENGTH=512
STUDENT_MODEL=${PROJECT_ROOT}/model/KD-S1
TEACHER_MODEL=bert-base-uncased

python ${PROJECT_ROOT}/pre_training.py \
    --pregenerated_data ${GENERATED_DIR} \
    --cache_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --gradient_accumulation_steps ${ACC_STEPS} \
    --train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --student_model ${STUDENT_MODEL} \
    --teacher_model ${TEACHER_MODEL} \
    --masked_lm_prob 0 \
    --local_rank 0 \
    --do_lower_case --scratch 