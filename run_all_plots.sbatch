#!/bin/bash
#SBATCH --job-name=run_all_plots
#SBATCH --output=run_all_plots.out
#SBATCH --account=p200981
#SBATCH --error=run_all_plots.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --qos=default

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

echo "Value of M set to ${M:-10}. Default = 10."
echo "Set custom value of M with 'M=X sbatch [...]'"

python src/plot_all.py -m ${M:-10}
