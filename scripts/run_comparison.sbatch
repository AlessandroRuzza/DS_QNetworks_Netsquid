#!/bin/bash

#SBATCH --job-name=run_comparison
#SBATCH --output=run_comparison_%j.out
#SBATCH --error=run_comparison_%j.err
#SBATCH --account=p200981
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --qos=default

echo "======================================"
echo "QNetworks simulator"
echo "======================================"
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Job ID            = $SLURM_JOB_ID"
echo "M Value           = ${M:-10}"
echo "======================================"

# Load required modules
module purge
module add Apptainer
module load Python/3.12.3-GCCcore-13.3.0 

echo "Value of M set to ${M:-10}. Default = 10."
echo "Set custom value of M with 'M=X sbatch [...]'"

apptainer exec docker://python:3.12 bash -c "source .venv/bin/activate && python src/plot_comparison.py -m ${M:-10}"
