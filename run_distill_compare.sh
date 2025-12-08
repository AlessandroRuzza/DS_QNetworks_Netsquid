echo "Use as M=X bash ./run_distill_compare.sh"
sbatch --job-name=M${M}_run_distill_compare --output=M${M}distill_compare_%j.out --error=M${M}distill_compare_%j.err scripts/run_distill_compare.sbatch