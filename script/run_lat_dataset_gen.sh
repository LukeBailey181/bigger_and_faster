#!/bin/bash
#SBATCH -p bigmem   # Partition to submit to
#SBATCH -t 0-08:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 4               # Number of cores (-c)
#SBATCH --mem=200000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./logs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yuc927@g.harvard.edu
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate baf-latency
perl -e 'print "Job starting ...\n"'
bash lat_dataset_gen.sh