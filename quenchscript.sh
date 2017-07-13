#!/bin/bash
#SBATCH -J quench
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-02:00
#SBATCH -p serial_requeue
#SBATCH --mem=200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o clusterdata/std/quench_%A_%a
#SBATCH -e clusterdata/std/quench_%A_%a

module load Anaconda3/4.3.0-fasrc01
python gquench_cluster.py