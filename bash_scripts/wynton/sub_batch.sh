for n in 400 800 1600 3200; do
    job_name=conv${n}_fixNlam
    qsub -N $job_name simu_conv1.sh $n
done

