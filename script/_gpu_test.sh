#USER=yujichai
USER=lukebailey
PARTITION=gpu # Partition to submit to, gpu or gpu_test
PARTITION_TEST=gpu_test # The test partition to submit to, gpu_test
RUNTIME=0-08:00 # Runtime in D-HH:MM, minimum of 10 minutes
CPU=8 # number of CPU cores
GPU=1 # number of GPUs
MEMORY=128000 # Memory pool for all cores (see also --mem-per-cpu)
COMMAND=/bin/bash
OP=$1
PA=$2

if [ "$OP" = "view" ]; then
  echo "View job status of $USER"
  squeue -u $USER
elif [ "$OP" = "queue" ]; then
  echo "View queue status of $PA"
  showq -o -p $PA
elif [ "$OP" = "run" ]; then
  echo "Start interactive session using $PARTITION_TEST"
  srun --pty -p $PARTITION_TEST -t $RUNTIME -c $CPU --gres=gpu:$GPU --mem $MEMORY $COMMAND
else 
  echo "Command not found!"
fi