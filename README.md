# GPU Programming Essentials

Exercises for the GPU Programming Essentials lecture for Curtin's HPC course.

All the exercises will be run on Topaz cluster, although you can run them on your local machine as well if you have a NVIDIA Graphics Card. 


## Example batch script
```
#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey0001
#SBATCH --nodes=1
#SBATCH --partition=gpuq

module load gcc/8.3.0 cuda/10.2

srun nvcc -o add vector_add/gpu_vector_add.cu
srun ./add
``` 