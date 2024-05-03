for cv in 0.0 0.1 0.2 0.4; do
    for setting in n2a ; do 
        job_name=S${setting}_c${cv}
        sbatch --job-name=$job_name simu_logi_settingns_batch.sh $cv $setting
    done
done
