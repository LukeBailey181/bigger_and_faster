#!/bin/bash
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 0-01:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 8              # Number of cores (-c)
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
TYPE=ptq
MODEL_TYPE_ID=1
SAVE_MODEL_FLAG=0
echo "Running ${TYPE}"
bash eval_config_test.sh $TYPE $MODEL_TYPE_ID $SAVE_MODEL_FLAG