ARCH_PATH=../arch_files/test.json
INFER_CNT=5     # Number of times model runs inference and averages over
DATA_TYPE=ptq    # One of fp and ptq

python ../get_config_latency.py --arch_path $ARCH_PATH \
    --infer_cnt $INFER_CNT \
    --data_type $DATA_TYPE