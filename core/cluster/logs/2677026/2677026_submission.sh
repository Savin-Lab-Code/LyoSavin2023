#!/bin/bash

# Parameters
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --error=/mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%A_%a/%A_%a_0_log.err
#SBATCH --job-name=eval_methods
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%A_%a/%A_%a_0_log.out
#SBATCH --partition=ccn
#SBATCH --signal=USR2@90
#SBATCH --time=600
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%A_%a/%A_%a_%t_log.out --error /mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%A_%a/%A_%a_%t_log.err /mnt/home/blyo1/ceph/envs/pyenv311/bin/python -u -m submitit.core._submit /mnt/ceph/users/blyo1/projects/LyoSavin2023/core/cluster/logs/%j
