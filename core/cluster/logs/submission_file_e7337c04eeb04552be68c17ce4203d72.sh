#!/bin/bash

# Parameters
#SBATCH --constraint=v100-32gb
#SBATCH --cpus-per-task=12
#SBATCH --error=/mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%j/%j_0_log.err
#SBATCH --gpus-per-task=1
#SBATCH --job-name=eval_methods
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%j/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@90
#SBATCH --time=600
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%j/%j_%t_log.out --error /mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%j/%j_%t_log.err /mnt/home/blyo1/venvs/jupyter-gpu/bin/python -u -m submitit.core._submit /mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%j
