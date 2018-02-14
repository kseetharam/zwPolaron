#!/bin/bash
#SBATCH -J quench
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 2-00:00
#SBATCH -p serial_requeue
#SBATCH --mem=3000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o /n/regal/demler_lab/kis/genPol_data/std/quench_%A_%a.out
#SBATCH -e /n/regal/demler_lab/kis/genPol_data/std/quench_%A_%a.err

module load Anaconda3/5.0.1-fasrc01
source activate anaclone
python xdatagen_qdynamics_cart.py