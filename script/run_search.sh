#!/bin/bash
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 2-00:00         # Runtime in D-HH:MM, minimum of 10 minutes
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
TYPE=ptq
LATENCY_CONSTRAINT=10
echo "Running ${TYPE} @ ${LATENCY_CONSTRAINT}ms"
bash search.sh cand $TYPE $LATENCY_CONSTRAINT 
bash search.sh fast $TYPE $LATENCY_CONSTRAINT 
# bash search.sh rand $TYPE $LATENCY_CONSTRAINT 
bash search.sh eval $TYPE $LATENCY_CONSTRAINT 
bash search.sh evol $TYPE $LATENCY_CONSTRAINT 
echo "Initialization Pass Done!"
bash search_and_eval.sh $TYPE $LATENCY_CONSTRAINT