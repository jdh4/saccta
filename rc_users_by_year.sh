#!/bin/bash

# This script computes the number of unique active users on the RC clusters. Numbers
# are calculated for (1) the large clusters and (2) the large clusters plus adroit.
# One should find the number of clusters for given year and then check to see
# if there are valid jobs associated with each cluster.
# sacct -L -a -X -n -S 2017-01-01 -E 2017-12-31 -o cluster | sort | uniq
# One must include the jobid for cpu-hours and gpu-hours to drop the duplicate
# entries that arise from jobs that span two time windows.

declare -A clusters
clusters[2017]="della,perseus,tiger,tiger2,tukey"
clusters[2018]="della,eddy,perseus,tiger,tiger2,tukey"
clusters[2019]="della,eddy,perseus,tiger2,traverse"
clusters[2020]="della,eddy,perseus,tiger2,traverse"
clusters[2021]="della,eddy,perseus,stellar,tiger2,traverse"
clusters[2022]="della,stellar,tiger2,traverse"
clusters[2023]="della,stellar,tiger2,traverse"

for YEAR in {2017..2023}
do
    ssh adroit "sacct -M adroit -a -X -n -S ${YEAR}-01-01 -E ${YEAR}-12-31 -o user | sort | uniq > adroit_users_${YEAR}"
    scp -q adroit:adroit_users_${YEAR} .

    large_users=large_cluster_users_${YEAR}
    sacct -M ${clusters[${YEAR}]} -a -X -n -S ${YEAR}-01-01 -E ${YEAR}-03-31 -o user | sort | uniq >  ${large_users}
    sacct -M ${clusters[${YEAR}]} -a -X -n -S ${YEAR}-04-01 -E ${YEAR}-06-30 -o user | sort | uniq >> ${large_users}
    sacct -M ${clusters[${YEAR}]} -a -X -n -S ${YEAR}-07-01 -E ${YEAR}-09-30 -o user | sort | uniq >> ${large_users}
    sacct -M ${clusters[${YEAR}]} -a -X -n -S ${YEAR}-10-01 -E ${YEAR}-12-31 -o user | sort | uniq >> ${large_users}

    cpu_hours=cpu_hours_${YEAR}
    sacct -M ${clusters[${YEAR}]} -a -X -n -P -S ${YEAR}-01-01 -E ${YEAR}-03-31 -o jobid,cputimeraw >  ${cpu_hours}
    sacct -M ${clusters[${YEAR}]} -a -X -n -P -S ${YEAR}-04-01 -E ${YEAR}-06-30 -o jobid,cputimeraw >> ${cpu_hours}
    sacct -M ${clusters[${YEAR}]} -a -X -n -P -S ${YEAR}-07-01 -E ${YEAR}-09-30 -o jobid,cputimeraw >> ${cpu_hours}
    sacct -M ${clusters[${YEAR}]} -a -X -n -P -S ${YEAR}-10-01 -E ${YEAR}-12-31 -o jobid,cputimeraw >> ${cpu_hours}

    gpu_hours=gpu_hours_${YEAR}
    sacct -M ${clusters[${YEAR}]} -a -X -n -P -S ${YEAR}-01-01 -E ${YEAR}-03-31 -o elapsedraw,alloctres,jobid | grep gres/gpu=[1-9] >  ${gpu_hours}
    sacct -M ${clusters[${YEAR}]} -a -X -n -P -S ${YEAR}-04-01 -E ${YEAR}-06-30 -o elapsedraw,alloctres,jobid | grep gres/gpu=[1-9] >> ${gpu_hours}
    sacct -M ${clusters[${YEAR}]} -a -X -n -P -S ${YEAR}-07-01 -E ${YEAR}-09-30 -o elapsedraw,alloctres,jobid | grep gres/gpu=[1-9] >> ${gpu_hours}
    sacct -M ${clusters[${YEAR}]} -a -X -n -P -S ${YEAR}-10-01 -E ${YEAR}-12-31 -o elapsedraw,alloctres,jobid | grep gres/gpu=[1-9] >> ${gpu_hours}

    printf ${YEAR}" "
    cat ${large_users} | sort | uniq | wc -l | tr "\n" ' '
    cat ${large_users} adroit_users_${YEAR} | sort | uniq | wc -l | tr "\n" ' '
    cat ${cpu_hours} | sort | uniq | awk -F'|' '{sum += $2} END {print int(sum/3600)}' | tr "\n" ' '
    cat ${gpu_hours} | sort | uniq | sed -E "s/\|.*gpu=/,/" | awk -F"," '{sum += $1*$2} END {print int(sum/3600)}' | tr "\n" ' '
    printf ${clusters[${YEAR}]}
    echo ""
done
