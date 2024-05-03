for cv in 0.0 0.1 0.2 0.4; do
    for setting in n2 n2a n2b n2e; do 
        job_name=S${setting}_c${cv}_newinte
        sbatch --job-name=$job_name simu_settingns_batch.sh $cv $setting
    done
done
