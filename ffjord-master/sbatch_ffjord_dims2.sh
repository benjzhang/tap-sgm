#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o slurm-ffjord.out  # %j = job ID

module load miniconda
conda activate generativetmula 

python train_toy.py --data 2spirals --dims 2-2-2 --layer_type concatsquash --save experiments/ffjord/2spirals2
python train_toy.py --data 8gaussians --dims 2-2-2 --layer_type concatsquash --save experiments/ffjord/8gaussians2
python train_toy.py --data checkerboard --dims 2-2-2 --layer_type concatsquash --save experiments/ffjord/checkerboard2
python train_toy.py --data moons --dims 2-2-2 --layer_type concatsquash --save experiments/ffjord/moons2
python train_toy.py --data pinwheel --dims 2-2-2 --layer_type concatsquash --save experiments/ffjord/pinwheel2

