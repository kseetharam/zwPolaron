#!/bin/bash
#SBATCH -J quench
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH -p serial_requeue
#SBATCH --mem=350
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o data_qdynamics/cart/std/quench_%A_%a
#SBATCH -e clusterdata/std/quench_%A_%a

module load Anaconda3/4.3.0-fasrc01
python datagen_qdynamics_cart.py