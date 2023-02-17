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
#!/bin/bash

gpuseconds_total=0
for gpus in $(sacct -M traverse -a -X -P -n -S 2022-01-01T00:00:00 -E 2022-12-31T23:59:59 -o alloctres | grep gres/gpu= | cut -d"," -f3 | sort | uniq | sed 's/[^0-9]*//g')
do
    gpuseconds=$(sacct -M traverse -a -X -P -n -S 2022-01-01T00:00:00 -E 2022-12-31T23:59:59 -o elapsedraw,alloctres | grep gres/gpu=$gpus | cut -d"|" -f1 | awk '{sum += $1} END {print sum}')
    gpuseconds_total=$((gpuseconds_total + gpus * gpuseconds))
    echo $gpus, $gpuseconds, $gpuseconds_total
done
echo "GPU-hours="$((gpuseconds_total/3600))
```
