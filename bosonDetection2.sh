#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --partition=gengpu
#SBATCH --ntasks-per-node=1
#SBATCH --mem=47GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --time=64:00:00 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tyler.liddell@city.ac.uk

source /opt/flight/etc/setup.sh
flight env activate gridware

module purge
module load libs/nvidia-cuda/11.2.0/bin

source ~/archive/miniconda3/etc/profile.d/conda.sh
conda activate llmTranslate

nvidia-smi

GPUS_PER_NODE=2
WORKER_CNT=2

export MASTER_PORT=8214

python -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT ~/HiggsBosonDetectionCopy/HiggsBosonDetection/higgsDetectionTrain.py
