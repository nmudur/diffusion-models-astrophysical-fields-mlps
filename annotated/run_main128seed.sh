#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-14:00
#SBATCH -p fink_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 12000
#SBATCH -o stout/output128_try0_%j.o
#SBATCH -e sterr/error128_try0_%j.e

module load Anaconda3/2020.11
module load CUDA/10.0.130

source activate pytorch_a100
/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python main.py config/params128_blseed.yaml
