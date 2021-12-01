PROJECT_ROOT=/n/holylfs05/LABS/acc_lab/Users/yujichai/bigger_and_faster
MODEL_DIR=${PROJECT_ROOT}/model/SUPER-KD-S1/output/superbert/checkpoints/superbert_epoch_4_lr_0.0001_bsz_12_grad_accu_1_512_gpu_1/epoch_3

python ${PROJECT_ROOT}/post_training_quantize.py \
    --super_model ${MODEL_DIR}