#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=encoding
#SBATCH --time=3-00:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40:1
#SBATCH --account=a_culpepper
#SBATCH -o slurm-%j.output
#SBATCH -e slurm-%j.error

module load miniconda3/23.9.0-0
module load cuda

source $EBROOTMINICONDA3/etc/profile.d/conda.sh

cd code
conda activate /home/uqtsolim/miniconda3/envs/dime-sigir-2024
srun ./encode_documents.sh
