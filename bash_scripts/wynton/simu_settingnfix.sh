#!/bin/bash
#### The job script, run it as qsub xxx.sh 

#### the shell language when run via the job scheduler [IMPORTANT]
#$ -S /bin/bash
#### job should run in the current working directory
###$ -cwd
##### set job working directory
#$ -wd  /wynton/home/rajlab/hjin/MyResearch/HDF/bash_scripts/
#### Specify job name
#$ -N Sn_fixNlam
#### Output file
#$ -o wynton/logs/Linear-$JOB_NAME_$JOB_ID.out
#### Error file
#$ -e wynton/logs/Linear-$JOB_NAME_$JOB_ID.err
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

for setting in n1 n1b n2 n2b n3 n3b; do
    for cv in 0.00 0.20 0.40; do
    singularity exec ~/MyResearch/hdf_orthbasis.sif python -u ../python_scripts/simu_settingns_fixNlam.py --cs $cv --setting $setting
    done
done

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
