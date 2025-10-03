#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --ntasks-per-core=1
#SBATCH --mem=1024G
#SBATCH --job-name=encoding
#SBATCH --time=3-00:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40:1
#SBATCH --account=a_culpepper
#SBATCH -o slurm-%j.output
#SBATCH -e slurm-%j.error

module load miniconda3/23.9.0-0
module load cuda/12.1

source $EBROOTMINICONDA3/etc/profile.d/conda.sh

conda activate /home/uqtsolim/miniconda3/envs/dime
export IR_DATASETS_HOME=/scratch/user/uqtsolim/.ir_datasets
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
srun python code/main.py -a -c msmarco-passage -e contriever -d oracle -o figs/ -i data/memmap/msmarco-passages
