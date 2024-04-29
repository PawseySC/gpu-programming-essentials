#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --account=courses0100-gpu
#SBATCH --time=00:02:00
#SBATCH --partition=gpu

module load rocm/5.2.3

# srun hipcc --offload-arch=gfx90a query.cpp -o query
srun ./query

