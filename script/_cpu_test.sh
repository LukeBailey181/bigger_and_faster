USER=yujichai
PARTITION=unrestricted # Partition to submit to, gpu or gpu_test
PARTITION_TEST=unrestricted # The test partition to submit to, gpu_test
RUNTIME=0-08:00 # Runtime in D-HH:MM, minimum of 10 minutes
CPU=4 # number of CPU cores
MEMORY=32000 # Memory pool for all cores (see also --mem-per-cpu)
COMMAND=/bin/bash
OP=$1

if [ "$OP" = "view" ]; then
  echo "View job status of $USER"
  squeue -u yujichai
elif [ "$OP" = "queue" ]; then
  echo "View queue status of $PARTITION"
  showq -o -p $PARTITION
elif [ "$OP" = "run" ]; then
  echo "Start interactive session using $PARTITION_TEST"
  srun --pty -p $PARTITION_TEST -t $RUNTIME -c $CPU --mem $MEMORY $COMMAND
else 
  echo "Command not found!"
fi