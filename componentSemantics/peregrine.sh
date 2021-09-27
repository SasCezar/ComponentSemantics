#!/bin/bash
#SBATCH --job-name=CS-FE
#SBATCH --mail-type=END
#SBATCH --time=1-0:00:00
#SBATCH --mail-user=c.a.sas@rug.nl
#SBATCH --output=job-%j.log
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64000


module load Anaconda3/2020.11
source activate CompSem
python feature_extraction_multi.py "/data/p302921/output/arcanOutput" "/data/p302921/repos/" "/data/p302921/projects/" code2vec 16
