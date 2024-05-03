for cv in 0.4; do
    for setting in cmpn0s2b ; do 
    #for setting in cmpn0s1  cmpn0s2 cmpn0s1b cmpn0s2b ; do 
    #for setting in cmpn1  cmpn2 cmpn1b cmpn2b ; do 
        job_name=S${setting}_c${cv}
        sbatch --job-name=$job_name simu_settingcmpn0ss_fixNlam.sh $cv $setting
    done
done
