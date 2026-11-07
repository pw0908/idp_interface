#!/bin/bash
#SBATCH --account=burst
#SBATCH --qos=burst
#SBATCH --partition=all
#SBATCH --job-name=idp_diblock_perturb
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=4G
#SBATCH --gres=gpu:0 --gpu-bind=closest
#SBATCH --array=1-200%25
#SBATCH --output=logs/diblock_perturb-%A_%a.out
#SBATCH --error=logs/diblock_perturb-%A_%a.err

cd /nfs/zeal_nas/home_mount/pwalker/Projects/idp_interface
julia --project=. scripts/sample_diblock_perturbation.jl "$SLURM_ARRAY_TASK_ID"
