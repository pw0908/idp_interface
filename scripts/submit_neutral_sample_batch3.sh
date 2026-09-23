#!/bin/bash
#SBATCH --account=burst
#SBATCH --qos=burst
#SBATCH --partition=all
#SBATCH --job-name=idp_neutral_sample3
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=4G
#SBATCH --gres=gpu:0 --gpu-bind=closest
#SBATCH --array=401-700%15
#SBATCH --output=logs/neutral_sample-%A_%a.out
#SBATCH --error=logs/neutral_sample-%A_%a.err

cd /nfs/zeal_nas/home_mount/pwalker/Projects/idp_interface
julia --project=. scripts/sample_neutral_sequences.jl "$SLURM_ARRAY_TASK_ID"
