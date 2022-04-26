# GPU Programming Essentials

Exercises for the GPU Programming Essentials lecture for Curtin's HPC course.

All the exercises will be run on Topaz cluster, although you can run them on your local machine as well if you have a NVIDIA Graphics Card. 

# Loading and using CUDA

The latest NVIDIA SDK has been installed in the course's group directory. To use it, execute the following

```
module use /group/courses0100/software/nvhpc/modulefiles
module load nvhpc/21.9
```

For CUDA C/C++ code, use the `nvcc` compiler.

For CUDA Fortran code, use the `nvfortran` compiler.

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