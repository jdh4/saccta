#!/bin/bash

CHECK_PARTITIONS=1

if [ ${CHECK_PARTITIONS} -gt 0 ]; then
    for YEAR in {2017..2023}
    do
        sacct -M della -a -X -n -S ${YEAR}-01-01T00:00:00 -E ${YEAR}-04-30T23:59:59 -o partition%20 >  partitions
        sacct -M della -a -X -n -S ${YEAR}-05-01T00:00:00 -E ${YEAR}-08-31T23:59:59 -o partition%20 >> partitions
        sacct -M della -a -X -n -S ${YEAR}-09-01T00:00:00 -E ${YEAR}-12-31T23:59:59 -o partition%20 >> partitions
        printf ${YEAR}
        cat partitions | sort | uniq | tr '\n' ' ' | tr -s ' '
        echo ""
    done
fi

#2017 all
#2018 all
#2019 all
#2020 all callan cpu datascience malik physics
#2021 all callan cpu datascience donia gpu gpu-ee malik orfeus physics physics,cpu
#2022 callan cpu cpu,physics cryoem cryoem,motion datascience donia gpu gpu-ee gputest malik motion orfeus physics physics,cpu
#2023 callan cli cpu cpu,physics cryoem cryoem,motion datascience donia gpu gpu-ee gputest malik mig motion orfeus physics physics,cpu pli

partitions="all,cpu"
for YEAR in {2017..2023}
do
    sacct -M della -r ${partitions} -a -X -n -S ${YEAR}-01-01T00:00:00 -E ${YEAR}-04-30T23:59:59 -o user >  users
    sacct -M della -r ${partitions} -a -X -n -S ${YEAR}-05-01T00:00:00 -E ${YEAR}-08-31T23:59:59 -o user >> users
    sacct -M della -r ${partitions} -a -X -n -S ${YEAR}-09-01T00:00:00 -E ${YEAR}-12-31T23:59:59 -o user >> users
    printf ${YEAR}" "
    cat users | sort | uniq | wc -l
    echo ""
done

for YEAR in {2017..2023}
do
    sacct -M della -r ${partitions} -a -X -n -P -S ${YEAR}-01-01T00:00:00 -E ${YEAR}-04-30T23:59:59 -o jobid,cputimeraw >  cputime
    sacct -M della -r ${partitions} -a -X -n -P -S ${YEAR}-05-01T00:00:00 -E ${YEAR}-08-31T23:59:59 -o jobid,cputimeraw >> cputime
    sacct -M della -r ${partitions} -a -X -n -P -S ${YEAR}-09-01T00:00:00 -E ${YEAR}-12-31T23:59:59 -o jobid,cputimeraw >> cputime
    printf ${YEAR}" "
    cat cputime | sort | uniq | awk -F'|' '{sum += $2} END {print int(sum/3600)}'
    echo ""
done
