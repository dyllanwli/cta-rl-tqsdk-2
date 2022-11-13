#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=RL  
#SBATCH --time=24:00:00              
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=64          
#SBATCH --mem=128G                   
#SBATCH --output=RL.%j 
#SBATCH --partition=gpu              
##SBATCH --gres=gpu:a100:1           

#First Executable Line

ml Anaconda3/2021.05
source activate
conda init bash
conda activate $SCRATCH/.conda/envs/quant
# check if the correct python version is used
pip --version

ml GCC/10.2.0 
ml iccifort/2020.4.304 
ml CUDA/11.1.1 
ml OpenMPI/4.0.5 

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
