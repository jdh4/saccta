# saccta

Notes on January 14, 2024:

After copying the cluster_utilization_20XX.tex file, comment out the latex `input` commands by running this in vim:

```
%s/\\input/%\\input
```

Then uncomment for Adroit and build the document.

Need to escape underscores in `adroit/adroit_ondemand.tex`:

```
sys/dashboard/sys/rstudio\_server-generic/nodes
```

Need to escape underscores in `adroit/adroit_state.tex` like "NODE_FAIL".

## cryoem

Run saccta.py from della-gpu with partition set to cryoem. Include tiger2 for old data.

## Della OnDemand

```
ssh della
export SLURM_TIME_FORMAT="%s"
sacct -a -X -P -n -S 2023-01-01T00:00:00 -E 2023-06-30T23:59:59 -o jobid,user,account,partition,cputimeraw%25,elapsedraw%50,alloctres%75,start,eligible,qos,state,jobname  --partition cpu,gpu,datascience,mig,gputest,physics > chunk1.csv
sacct -a -X -P -n -S 2023-07-01T00:00:00 -E 2023-12-31T23:59:59 -o jobid,user,account,partition,cputimeraw%25,elapsedraw%50,alloctres%75,start,eligible,qos,state,jobname  --partition cpu,gpu,datascience,mig,gputest,physics > chunk2.csv
cat chunk1.csv chunk2.csv | grep billing > della_cpu.csv
# set fname to "della_cpu.csv"
python saccta.py
```

## Number of Sponsors in 2024

```
jdh4@tiger3:~$ sacct -X -P -n -S 2024-01-01T00:00:00 -E now -a -o user,cluster | sort | uniq > netids_2024.tiger3
jdh4@tiger3:~$ scp netids_2024.tiger3 tigergpu:

$ sacct -M tiger2,traverse,stellar -X -P -n -S 2024-01-01T00:00:00 -E now -a -o user,cluster | sort | uniq > netids_2024.not_della
$ sacct -M della -X -P -n -S 2024-01-01T00:00:00 -E 2024-05-01T00:00:00 -a -o user,cluster | sort | uniq > netids_2024.della1; sacct -M della -X -P -n -S 2024-05-01T00:00:00 -E 2024-10-01T00:00:00 -a -o user,cluster | sort | uniq > netids_2024.della2; sacct -M della -X -P -n -S 2024-10-01T00:00:00 -E now -a -o user,cluster | sort | uniq > netids_2024.della3
$ cat netids_2024.not_della netids_2024.della1 netids_2024.della2 netids_2024.della3 netids_2024.tiger3 | sort | uniq > netids.2024
```

## Other

Adroit:

```
[jdh4@adroit5 ~]$ sacct -a -X -n -S 2021-07-01 -E 2021-12-31 -o user | sort | uniq | wc -l
466
[jdh4@adroit5 ~]$ sacct -a -X -n -S 2021-07-01 -E 2021-12-31 -o user | sort | uniq > adroit.txt
```

Large clusters:

```
[jdh4@della8 ~]$ sacct -L -a -X -n -S 2021-07-01 -E 2021-12-31 -o user | sort | uniq | wc -l
993
[jdh4@della8 ~]$ sacct -L -a -X -n -S 2021-07-01 -E 2021-12-31 -o user | sort | uniq > large_clusters.txt
```

Combination:

```
[jdh4@della8 ~] $ cat adroit.txt large_clusters.txt | sort | uniq | wc -l
1337
```

The comprehensive report in PDF cited 1253. 463 were found on Adroit in that document.

To calculate the number of GPU-hours:

```
$ sacct -M traverse -a -X -P -n -S 2022-01-01T00:00:00 -E 2022-12-31T23:59:59 -o elapsedraw,alloctres,state | grep gres/gpu=[1-9] | grep -v RUNNING | sed -E "s/\|.*gpu=/,/" | awk -F"," '{sum += $1*$2} END {print int(sum/3600)}'
```

More explicitly:

```bash
#!/bin/bash
SACCT="sacct -M traverse -a -X -P -n -S 2022-01-01T00:00:00 -E 2022-12-31T23:59:59"
jobs_total=0
gpu_seconds_total=0
echo "gpus, gpu-seconds, jobs"
for gpus in $($SACCT -o alloctres | grep "gres/gpu=" | cut -d"," -f3 | sort | uniq | sed 's/[^0-9]*//g' | sort -n)
do
    jobs=$($SACCT -o alloctres | grep "gres/gpu=$gpus," | wc -l)
    jobs_total=$((jobs_total + jobs))
    run_seconds=$($SACCT -o elapsedraw,alloctres | grep "gres/gpu=$gpus," | cut -d"|" -f1 | awk '{sum += $1} END {print sum}')
    gpu_seconds=$((gpus * run_seconds))
    gpu_seconds_total=$((gpu_seconds_total + gpu_seconds))
    echo $gpus, $gpu_seconds, $jobs
done
echo "GPU-seconds="$gpu_seconds_total
echo "GPU-hours="$((gpu_seconds_total/3600))
echo "Jobs="$jobs_total
```

## One-liner

```
$ DAYS=90; GPUS=316; PARTITION=gpu; sacct -M della -a -X -P -n -S $(date -d"${DAYS} days ago" +%F) -E now -o elapsedraw,alloctres,state --partition=${PARTITION} | grep gres/gpu=[1-9] | sed -E "s/\|.*gpu=/,/" | awk -v gpus=${GPUS} -v days=${DAYS} -F"," '{sum += $1*$2} END {print "GPU-hours used = "int(100*sum/3600/24/gpus/days)"%"}'
GPU-hours used = 92%
```

## Other

Encountered when including jobname on della (cpu).

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b in position 127525840: invalid start byte
```
