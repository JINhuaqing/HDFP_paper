#for N in  8 ; do
for N in 4 6 8 10 12 14; do
    job_name=sighb_${N}
    sbatch --job-name=$job_name real_data_nlinear_sig_half_brain.sh $N
    #job_name=nlinear_N${N}
    #sbatch --job-name=$job_name real_data_nlinear.sh $N
done
