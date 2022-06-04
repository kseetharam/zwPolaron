#!/bin/bash
#SBATCH -J quenchLDAosc
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -t 6-23:59
#SBATCH -p shared
#SBATCH --mem=12000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kis@mit.edu 
#SBATCH --open-mode=append
#SBATCH -o /n/holyscratch01/jaffe_lab/Everyone/kis/std/quench_LDA_osc_%A_%a.out
#SBATCH -e /n/holyscratch01/jaffe_lab/Everyone/kis/std/quench_LDA_osc_%A_%a.err

module load Anaconda3/5.0.1-fasrc02
source activate qsim_cluster
python datagen_LDA_osc_zw2021_2D.py