#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=scs/logs/HDF-%x.out
#SBATCH -J mtest2_4
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/hujin/jin/MyResearch/HDF_infer/bash_scripts/
####SBATCH --ntasks=30

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"
singularity exec ~/jin/singularity_containers/hdf_ball.sif python -u ../python_scripts/gen_data4sinica_psd.py --cs $1 --setting $2

case $2 in 
    *1*)
        matlab -batch "addpath('/data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/sinica_code/'); simu_settingcmp([1], $1, '$2')"
    ;;
    *2*)
        matlab -batch "addpath('/data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/sinica_code/'); simu_settingcmp([1, 2], $1, '$2')"
    ;;
    *3*)
        matlab -batch "addpath('/data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/sinica_code/'); simu_settingcmp([1, 2, 3], $1, '$2')"
    ;;
esac
        


