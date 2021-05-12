#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey0001
#SBATCH --nodes=1
#SBATCH --partition=gpuq

module load gcc/8.3.0 cuda/10.2

srun nvcc -o m matrix_transpose.cu
srun ./m