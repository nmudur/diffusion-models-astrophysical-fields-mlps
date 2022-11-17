#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-13:00
#SBATCH -p fink_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 12000
#SBATCH -o stout/output_dust_%j.o
#SBATCH -e sterr/error_dust_%j.e

module load Anaconda3/2020.11
module load CUDA/10.0.130

source activate pytorch_a100
/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python main.py config/params_dustmm.yaml
