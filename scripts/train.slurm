#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=train  
#SBATCH --time=48:00:00              
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=32          
#SBATCH --mem=128G                   
#SBATCH --output=./logs/%j.train.log
#SBATCH --partition=gpu              
#SBATCH --gres=gpu:a100:1           

#First Executable Line
module purge
module load GCC/11.2.0  OpenMPI/4.1.1
# module load TensorFlow/2.7.1-CUDA-11.4.1
module load cuDNN/8.2.2.26-CUDA-11.4.1
ml Anaconda3/2021.05
source activate
conda init bash
conda activate quant
# conda activate $SCRATCH/.conda/envs/quant
nvidia-smi
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
pip --version # check if the correct python version is used

cd ../src; python mt_train.py