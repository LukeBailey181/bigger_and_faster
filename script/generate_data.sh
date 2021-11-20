PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
RAW_DIR=${PROJECT_ROOT}/data/raw/books1/epubtxt
GENERATED_DIR=${PROJECT_ROOT}/data/generated
BASE_MODEL=bert-base-uncased

python ${PROJECT_ROOT}/generate_data.py \
    --train_corpus ${RAW_DIR} \
    --bert_model ${BASE_MODEL} \
    --output_dir ${GENERATED_DIR} \
    --do_lower_case