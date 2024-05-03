for cv in 0.2; do
    for setting in nm1 nm1a nm2 nm2a ; do 
        for start in 0 50 100 150; do
            job_name=S${setting}_c${cv}_${start}_logi
            qsub -N $job_name simu_logi_settingnms_batch.sh $cv $setting $start
        done
    done
done

