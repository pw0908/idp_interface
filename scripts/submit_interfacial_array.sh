#!/bin/bash
#SBATCH --account=burst
#SBATCH --qos=burst
#SBATCH --partition=all
#SBATCH --job-name=idp_interfacial_arr
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=6G
#SBATCH --gres=gpu:0 --gpu-bind=closest
#SBATCH --array=0-6
#SBATCH --output=logs/interfacial_arr-%A_%a.out
#SBATCH --error=logs/interfacial_arr-%A_%a.err

cd /nfs/zeal_nas/home_mount/pwalker/Projects/idp_interface

SEQUENCES=(block1 block2 block5 block10 asym1 asym2 asym3)
NAME=${SEQUENCES[$SLURM_ARRAY_TASK_ID]}
NPOINTS=${1:-10}

julia --project=. scripts/run_interfacial_one.jl "$NAME" "$NPOINTS"
