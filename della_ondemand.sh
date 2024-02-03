#!/bin/bash

for YEAR in {2020..2023}
do
  echo ${YEAR}
  sacct -M della -a -X -P -n -S ${YEAR}-01-01T00:00:00 -E ${YEAR}-06-30T23:59:59 -o jobid,user,jobname%100 --partition cpu,gpu,datascience,mig,gputest,physics > chunk1.csv
  sacct -M della -a -X -P -n -S ${YEAR}-07-01T00:00:00 -E ${YEAR}-12-31T23:59:59 -o jobid,user,jobname%100 --partition cpu,gpu,datascience,mig,gputest,physics > chunk2.csv
  cat chunk1.csv chunk2.csv | sort | uniq > della.csv

  cat della.csv | cut -d"|" -f2 | sort | uniq | wc -l
  cat della.csv | grep sys/dashboard | cut -d"|" -f2 | sort | uniq | wc -l
done
