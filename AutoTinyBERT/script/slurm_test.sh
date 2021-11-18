USER=yujichai
PARTITION=gpu_test # Partition to submit to, gpu or gpu_test
RUNTIME=0-08:00 # Runtime in D-HH:MM, minimum of 10 minutes
CPU=4 # number of CPU cores
GPU=1 # number of GPUs
MEMORY=128000 # Memory pool for all cores (see also --mem-per-cpu)
COMMAND=/bin/bash
OP=$1

if [ "$OP" = "view" ]; then
  echo "View job status of $USER"
  squeue -u yujichai
elif [ "$OP" = "run" ]; then
  echo "Start interactive session using $PARTITION"
  srun --pty -p $PARTITION -t $RUNTIME -c $CPU --gres=gpu:$GPU --mem $MEMORY $COMMAND
else 
  echo "Command not found!"
fi