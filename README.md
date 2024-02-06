# GPU Programming Essentials

Exercises for the GPU Programming Essentials lecture for Curtin's HPC course.

The GPU programming language chosen is HIP, a C++ language extension part of the AMD ROCm software stack. Unfortunately, Fortran is not supported at this stage. All the exercises will be run on Setonix, which is equipped with AMD GPUs. If you have a NVIDIA Graphics Card on your local machine, then you will be able to run exercises by rewriting in CUDA with minor modifications (in most cases, you can simply replace the `hip` prefix with `cuda` in API calls).

# Loading and using ROCm

ROCm is available on Setonix through the `rocm/5.2.3` module. To use it, run `module load rocm/5.2.3`. The compiler we will be using is `hipcc`.

## Example batch script
```
#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=courses0100
#SBATCH --nodes=1
#SBATCH --partition=gpuq

module use /group/courses0100/software/nvhpc/modulefiles
module load nvhpc/21.9

srun nvcc -o add vector_add/gpu_vector_add.cu
srun ./add
``` 