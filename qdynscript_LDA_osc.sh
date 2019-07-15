#!/bin/bash
#SBATCH -J quenchLDAosc
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -p shared
#SBATCH --mem=14000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o /n/scratchlfs/demler_lab/kis/ZwierleinExp_data/std/quench_LDA_osc_%A_%a.out
#SBATCH -e /n/scratchlfs/demler_lab/kis/ZwierleinExp_data/std/quench_LDA_osc_%A_%a.err

module load Anaconda3/5.0.1-fasrc01
source activate anaclone
python datagen_LDA_osc.py