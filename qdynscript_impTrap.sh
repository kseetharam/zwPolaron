#!/bin/bash
#SBATCH -J quench
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -t 0-4:00
#SBATCH -p shared
#SBATCH --mem=6000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o /n/regal/demler_lab/kis/ZwierleinExp_data/std/quench_%A_%a.out
#SBATCH -e /n/regal/demler_lab/kis/ZwierleinExp_data/std/quench_%A_%a.err

module load Anaconda3/5.0.1-fasrc01
source activate anaclone
python datagen_impTrap_extForce.py