for cv in 0.0 0.1 0.2 0.4; do
    for setting in n0s1 n0s1a n0s1b n0s1e n0s2 n0s2a n0s2b n0s2e ; do 
        job_name=S${setting}_c${cv}
        sbatch --job-name=$job_name simu_settingn0ss_batch.sh $cv $setting
    done
done
