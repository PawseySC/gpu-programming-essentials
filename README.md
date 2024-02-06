# GPU Programming Essentials

Exercises for the GPU Programming Essentials lecture for Curtin's HPC course.

The GPU programming language chosen is HIP, a C++ language extension part of the AMD ROCm software stack. Unfortunately, Fortran is not supported at this stage. All the exercises will be run on Setonix, which is equipped with AMD GPUs. If you have a NVIDIA Graphics Card on your local machine, then you will be able to run exercises by rewriting them in CUDA with minor modifications (in most cases, you can simply replace the `cuda` prefix with `hip` in API calls).

# Loading and using ROCm

ROCm is available on Setonix through the `rocm/5.2.3` module. To use it, run `module load rocm/5.2.3`. The compiler is `hipcc`, and must be invoked with the following flag: `--offload-arch=gfx90a`. Like so,

```
hipcc --offload-arch=gfx90a
```

## Example batch script
```
#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --account=courses0100
#SBATCH --nodes=1
#SBATCH --partition=gpu

module load rocm/5.2.3

srun hipcc --offload-arch=gfx90a -o add vector_add/gpu_vector_add.cpp
srun ./add
``` 