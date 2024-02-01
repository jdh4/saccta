#!/bin/bash

# This script computes the number of unique active users on the RC clusters. Numbers
# are calculated for (1) the large clusters and (2) the large clusters plus adroit.
# One should find the number of clusters for given year and then check to see
# if there are valid jobs associated with each cluster.
# sacct -L -a -X -n -S 2017-01-01 -E 2017-12-31 -o cluster | sort | uniq

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

    printf ${YEAR}" "
    cat ${large_users} | sort | uniq | wc -l | tr "\n" " "
    cat ${large_users} adroit_users_${YEAR} | sort | uniq | wc -l | tr "\n" " "
    printf ${clusters[${YEAR}]}
    echo ""
done
