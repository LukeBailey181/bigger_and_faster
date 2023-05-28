#!/bin/bash
#SBATCH -p gpu  # Partition to submit to - change to gpu_test for testing
#SBATCH -t 2-00:00         # Runtime in D-HH:MM, minimum of 10 minutes - change to 0-08:00
#SBATCH -c 16              # Number of cores (-c)
#SBATCH --gres=gpu:1                # Number of cores (-c)
#SBATCH --mem=128000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./logs/job_log_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/job_log_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yuc927@g.harvard.edu
module load Anaconda3 CUDA cudnn
eval "$(conda shell.bash hook)"
conda activate baf
perl -e 'print "Job starting ...\n"'

# Define experiment:
TYPE=ptq
LATENCY_CONSTRAINT=10
TASK_NAME="rte"   # One of {"mnli", "sst-2", "mrpc", "cola", "sts-b", "qqp", "qnli", "wnli", "rte"} 
# Experiments to run
# Luke: "mnli", "sst-2", "mrpc", "cola"
# Yuji: "sts-b", "qnli", "wnli", "rte"
# If we have time do qqp
# For latency contraint use 5, 10, 15, 20

PROJECT_ROOT=/n/home00/lbailey/bigger_and_faster
DATASET_ROOT=/n/holylfs05/LABS/acc_lab/Lab/baf_shared/tmp_glue_datasets/

# Get dataset directory from task name
if [ $TASK_NAME = "mnli" ]; then 
DATA_DIR="/n/holylfs05/LABS/acc_lab/Lab/glue/MNLI/processed/"
elif [ $TASK_NAME = "sst-2" ]; then 
DATA_DIR="${DATASET_ROOT}/sst2/"
elif [ $TASK_NAME = "mrpc" ]; then 
DATA_DIR="${DATASET_ROOT}/mrpc/"
elif [ $TASK_NAME = "cola" ]; then 
DATA_DIR="${DATASET_ROOT}/cola/"
elif [ $TASK_NAME = "sts-b" ]; then 
DATA_DIR="${DATASET_ROOT}/stsb/"
elif [ $TASK_NAME = "qqp" ]; then 
DATA_DIR="${DATASET_ROOT}/qqp/"
elif [ $TASK_NAME = "qnli" ]; then 
DATA_DIR="${DATASET_ROOT}/qnli/"
elif [ $TASK_NAME = "wnli" ]; then 
DATA_DIR="${DATASET_ROOT}/wnli/"
elif [ $TASK_NAME = "rte" ]; then 
DATA_DIR="${DATASET_ROOT}/rte/"
fi
echo $DATA_DIR


echo "Running ${TYPE} @ ${LATENCY_CONSTRAINT}ms"

bash search.sh cand $TYPE $LATENCY_CONSTRAINT $DATA_DIR $TASK_NAME
echo "Initial cand gen done"

bash search.sh fast $TYPE $LATENCY_CONSTRAINT $DATA_DIR $TASK_NAME
#bash search.sh rand $TYPE $LATENCY_CONSTRAINT  - uncomment for rand initial search
echo "Initial search done"

# Search is failing
bash search.sh eval $TYPE $LATENCY_CONSTRAINT $DATA_DIR $TASK_NAME
echo "Initial eval done"

#bash search.sh evol $TYPE $LATENCY_CONSTRAINT $DATA_DIR
echo "Initialization Pass Done!"
bash search_and_eval.sh $TYPE $LATENCY_CONSTRAINT $DATA_DIR $TASK_NAME