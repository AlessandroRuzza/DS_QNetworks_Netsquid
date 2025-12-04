echo "Use as M=X bash run_comparison.sh"
sbatch --job-name=M${M}_run_comparison --output=M${M}_run_comparison_%j.out --error=M${M}_run_comparison_%j.err run_comparison.sbatch