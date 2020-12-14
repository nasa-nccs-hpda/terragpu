#!/bin/bash
##-------------------------------------------------------------------------
## The following script is a slurm submission script.
## Execution:
##     Slurm: sbatch slurm_rfdriver.sh
##     Bash: bash slurm_rfdriver.sh
## Author: Jordan A Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
##-------------------------------------------------------------------------
#SBATCH -N1 -Jpredictrf

## Load conda module and configuration file
module load anaconda
conda activate xrasterlib
echo "Loaded xrasterlib anaconda module."

## Environment variables to run test
outdir="/att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/classified/classified_4band2m"
model="/att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_4band_fdi_si_ndwi.pkl"
rasters="/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/*.tif"
bands="Blue Green Red NIR1"

## There is something odd in the way the ADAPT GPU cluster is allocationg
## GPU memory, so only small windows can be used. I recommend using straight 
## salloc runs. When using salloc, window size can go up to 5000 x 5000
windowsize="200 200"

## Execute desired command
srun -n1 python rfdriver.py -o $outdir -m $model -i $rasters -b $bands --window-size $windowsize --sieve -l
