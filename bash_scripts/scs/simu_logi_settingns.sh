#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=scs/logs/Logi-%x-%j.out
#SBATCH -J Sna_fixNlam
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/hujin/jin/MyResearch/HDF_infer/bash_scripts/
####SBATCH --ntasks=30

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

# cv 0 0.1 0.2 0.3
# setting: n1 n1a n2 n2a 
cv=0.1
setting=n1
singularity exec ~/jin/singularity_containers/hdf_ball.sif python -u ../python_scripts/simu_logi_settingns.py --cs $cv --setting $setting
