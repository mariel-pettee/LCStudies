#!/bin/bash
#SBATCH --job-name make_graphs
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --constraint=knl
#SBATCH --time=10:00
#SBATCH --array=0-1
#SBATCH --output logs/log.log

conda activate ml4pions
SAVE_DIR=/global/cfs/cdirs/m3246/mpettee/ml4pions/LCStudies/graphs

### Distribute among multiple workers 
mkdir -p logs/${SLURM_JOB_NAME}
python -u make_graphs.py ${SLURM_ARRAY_TASK_COUNT} ${SLURM_ARRAY_TASK_ID} $SAVE_DIR &> logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}.log

