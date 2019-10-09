#!/bin/bash
#SBATCH -J afewres3
#SBATCH -p high
#SBATCH -N 1
#SBATCH --mem=128GB

# SBATCH --exclude=node025
# SBATCH --nodelist node031

#SBATCH --workdir=/homedtic/gulloa/ValenceArousal
#SBATCH --gres=gpu:1
# SBATCH --sockets-per-node=1
# SBATCH --cores-per-socket=2
# SBATCH --threads-per-core=2


#SBATCH -o slurm.%N.%J.%u.out # STDOUT
#SBATCH -e slurm.%N.%J.%u.err # STDERR

module load Python/3.6.4-foss-2017a
#module load Tensorflow-gpu/1.5.0-foss-2017a-Python-3.6.4

#module load CUDA/8.0.61
module load  CUDA/9.0.176

module load torchvision/0.2.1-foss-2017a-Python-3.6.4
module load PyTorch/0.4.0-foss-2017a-Python-3.6.4


module load numpy/1.14.0-foss-2017a-Python-3.6.4
module load OpenCV/3.1.0-foss-2017a
module load matplotlib/1.5.1-foss-2017a-Python-3.6.4-freetype-2.7.1

module load scikit-learn/0.19.1-foss-2017a-Python-3.6.4
module load scikit-image/0.14.0-foss-2017a-Python-3.6.4

python /homedtic/gulloa/ValenceArousal/src/NNTraining.py -m 6 -sp 3
