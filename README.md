# saccta

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
