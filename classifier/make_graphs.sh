#!/bin/bash
#SBATCH --job-name pi0
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=knl
#SBATCH --time=3:00:00
#SBATCH --array=0-199
#SBATCH --output logs/log.log

conda activate ml4pions

### Use these options for neutral pion samples
INPUT_DIR=/global/cfs/cdirs/m3246/mpettee/ml4pions/LCStudies/data/user.angerami.mc16_13TeV.900246.PG_singlepi0_logE0p2to2000.e8312_e7400_s3170_r12383.v01-45-gaa27bcb_OutputStream/
SAVE_DIR=/global/cfs/cdirs/m3246/mpettee/ml4pions/LCStudies/graphs/neutral_pion/
is_charged=false

### Distribute among multiple workers 
mkdir -p logs/${SLURM_JOB_NAME}
python -u make_graphs.py ${SLURM_ARRAY_TASK_COUNT} ${SLURM_ARRAY_TASK_ID} $INPUT_DIR $SAVE_DIR > logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}.log

echo "All graphs generated."


### Use these options for charged pion samples 
# INPUT_DIR=/global/cfs/cdirs/m3246/mpettee/ml4pions/LCStudies/data/user.angerami.mc16_13TeV.900247.PG_singlepion_logE0p2to2000.e8312_e7400_s3170_r12383.v01-45-gaa27bcb_OutputStream
# SAVE_DIR=/global/cfs/cdirs/m3246/mpettee/ml4pions/LCStudies/graphs/charged_pion/
# is_charged=true

# ### Distribute among multiple workers 
# mkdir -p logs/${SLURM_JOB_NAME}
# python -u make_graphs.py ${SLURM_ARRAY_TASK_COUNT} ${SLURM_ARRAY_TASK_ID} $INPUT_DIR $SAVE_DIR --is_charged > logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}.log

# echo "All graphs generated."
