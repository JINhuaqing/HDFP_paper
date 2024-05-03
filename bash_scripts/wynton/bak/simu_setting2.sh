#!/bin/bash
#### The job script, run it as qsub xxx.sh 

#### the shell language when run via the job scheduler [IMPORTANT]
#$ -S /bin/bash
#### job should run in the current working directory
###$ -cwd
##### set job working directory
#$ -wd  /wynton/home/rajlab/hjin/MyResearch/HDF/bash_scripts/
#### Specify job name
#$ -N test1_6 #!!!
#### Output file
#$ -o wynton/logs/S2_$JOB_NAME_$JOB_ID.out
#### Error file
#$ -e wynton/logs/S2_$JOB_NAME_$JOB_ID.err
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

echo "Starting running"

singularity exec ~/MyResearch/hdf_snsfix.sif python -u ../python_scripts/simu_cmp2sinica_samebetaX_setting2.py --cs 0.6 #!!!

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
