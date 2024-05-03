#for n in 6400 12800 ; do
for n in 200 400 800 1600 3200 ; do
     job_name=conv${n}
     sbatch --job-name=$job_name simu_conv.sh $n
done
