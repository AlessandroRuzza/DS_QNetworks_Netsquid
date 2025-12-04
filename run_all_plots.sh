echo "Use as M=X bash ./run_all_plots.sh"
sbatch --job-name=M${M}_run_all_plots --output=M${M}_run_all_plots_%j.out --error=M${M}_run_all_plots_%j.err scripts/run_all_plots.sbatch