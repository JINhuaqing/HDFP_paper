#!/bin/bash
#### The job script, run it as qsub xxx.sh 

#### the shell language when run via the job scheduler [IMPORTANT]
#$ -S /bin/bash
#### job should run in the current working directory
###$ -cwd
##### set job working directory
#$ -wd  /wynton/home/rajlab/hjin/MyResearch/HDF/bash_scripts/
#### Specify job name
#$ -N mStest_020
#### Output file
#$ -o wynton/logs/matlab_$JOB_NAME_$JOB_ID.out
#### Error file
#$ -e wynton/logs/matlab_$JOB_NAME_$JOB_ID.err
#### memory per core
#$ -l mem_free=2G
#### number of cores 
#$ -pe smp 40
#### Maximum run time 
#$ -l h_rt=48:00:00
#### job requires up to 2 GB local space
#$ -l scratch=2G
#### Specify queue
###  gpu.q for using gpu
###  if not gpu.q, do not need to specify it
###$ -q gpu.q 
#### The GPU memory required, in MiB
### #$ -l gpu_mem=12000M


echo "RUNNING"

singularity exec ~/MyResearch/hdf_orthbasis.sif python -u ../python_scripts/gen_matlab_data_settingtest.py --cs 0.2

module load matlab
matlab -batch "run('/wynton/home/rajlab/hjin/MyResearch/HDF/matlab_scripts/sinica_code/simu_settingtest.m')"

#### End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
