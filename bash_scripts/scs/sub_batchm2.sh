for cv in 0.0 0.1 0.2 0.4; do
    for setting in cmpn0s1  cmpn0s2 cmpn0s1b cmpn0s2b ; do 
        job_name=mS${setting}_c${cv}
        sbatch --job-name=$job_name msimu_settingcmpn0ss.sh $cv $setting
    done
done
