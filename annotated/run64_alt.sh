#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-01:30
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 6000
#SBATCH -o stout/output64_alt_%j.o
#SBATCH -e sterr/error64_alt_%j.e

module load Anaconda3/2020.11
module load CUDA/10.0.130

source activate pytorch_a100
/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python main.py config/params64_alt.yaml
