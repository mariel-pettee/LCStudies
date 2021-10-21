#!/bin/bash
#SBATCH --job-name test_8
#SBATCH --qos=flex
#SBATCH --nodes=1
#####SBATCH -n 1
#SBATCH -c 8
#SBATCH --constraint=knl
#SBATCH --time=5:00
#SBATCH --output logs/test_8.log
#SBATCH --account=m3705

echo $SLURM_JOB_NUM_NODES
echo $SLURM_JOB_CPUS_PER_NODE




conda activate ml4pions
python make_graphs.py 8
