#!/bin/bash

#SBATCH --job-name=gpu_env_test.local
#SBATCH --output=gpu_env_test.local.output.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Activate the environment
# nothing to do here

# Check env
echo
echo "which python"
which python


# Run script
echo
python pytorch_test.py