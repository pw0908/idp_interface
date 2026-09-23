#!/bin/bash
#SBATCH --account=burst
#SBATCH --qos=burst
#SBATCH --partition=all
#SBATCH --job-name=idp_interfacial
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=6G
#SBATCH --gres=gpu:0 --gpu-bind=closest
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

cd /nfs/zeal_nas/home_mount/pwalker/Projects/idp_interface
julia --project=. scripts/run_interfacial.jl
