#!/bin/bash
#SBATCH --account=burst
#SBATCH --qos=burst
#SBATCH --partition=all
#SBATCH --job-name=idp_interfacial_pts
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=4G
#SBATCH --gres=gpu:0 --gpu-bind=closest
#SBATCH --array=1-700%40
#SBATCH --output=logs/interfacial_pts-%A_%a.out
#SBATCH --error=logs/interfacial_pts-%A_%a.err

cd /nfs/zeal_nas/home_mount/pwalker/Projects/idp_interface

SEQUENCES=(block1 block2 block5 block10 asym1 asym2 asym3)
NPOINTS=100

T=$((SLURM_ARRAY_TASK_ID - 1))
SEQ_IDX=$((T / NPOINTS))
POINT_IDX=$((T % NPOINTS + 1))
NAME=${SEQUENCES[$SEQ_IDX]}

julia --project=. scripts/run_interfacial_single_point.jl "$NAME" "$NPOINTS" "$POINT_IDX"
