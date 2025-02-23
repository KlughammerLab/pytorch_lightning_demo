#!/bin/sh
#
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=4GB

#SBATCH --mail-user=user@genzentrum.lmu.de
#SBATCH --mail-type=fail
#SBATCH -o ../logs/demo%j.log
#SBATCH -e ../logs/demo%j.err
#SBATCH -J pytorch_lightning_demo
#SBATCH -t 3-00:00:00

# echo "#########################################################################"
# echo "Print the current environment (verbose)"
# env

echo "#########################################################################"
echo "Show information on nvidia device(s)"
nvidia-smi

echo "Start Training"
echo "#########################################################################"

python3 ../model/train.py 