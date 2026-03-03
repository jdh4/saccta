#!/bin/bash

export SLURM_TIME_FORMAT="%s"

YEAR=2025
partitions=""
pp=${partitions:+"-r"}
fields="jobid,user,cluster,account,partition,cputimeraw,elapsedraw,alloctres,start,submit,eligible,qos,state,jobname"
cluster=della
OUT=${cluster}.data.${YEAR}
flags="-a -X -n -P"

sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-01-01T00:00:00 -E ${YEAR}-01-31T23:59:59 -o ${fields} >  ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-02-01T00:00:00 -E ${YEAR}-03-01T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-03-02T00:00:00 -E ${YEAR}-03-31T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-04-01T00:00:00 -E ${YEAR}-04-30T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-05-01T00:00:00 -E ${YEAR}-05-31T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-06-01T00:00:00 -E ${YEAR}-06-30T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-07-01T00:00:00 -E ${YEAR}-07-31T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-08-01T00:00:00 -E ${YEAR}-08-31T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-09-01T00:00:00 -E ${YEAR}-09-30T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-10-01T00:00:00 -E ${YEAR}-10-31T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-11-01T00:00:00 -E ${YEAR}-11-30T23:59:59 -o ${fields} >> ${OUT}.raw
sacct -M ${cluster} ${pp} ${partitions} ${flags} -S ${YEAR}-12-01T00:00:00 -E ${YEAR}-12-31T23:59:59 -o ${fields} >> ${OUT}.raw

echo "Truncating job names that use the | character"
wc ${OUT}.raw
cut -d'|' -f1-14 ${OUT}.raw > ${OUT}.jobname
wc ${OUT}.jobname

echo "Removing duplicate rows"
cat ${OUT}.jobname | sort | uniq > ${OUT}.uniq
wc ${OUT}.uniq

echo "Adding line with column names"
sed -i '1ijobid|user|cluster|account|partition|cputimeraw|elapsedraw|alloctres|start|submit|eligible|qos|state|jobname' ${OUT}.uniq
head ${OUT}.uniq
