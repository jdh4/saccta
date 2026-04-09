"""This script can be used to calculate the number of GPU-hours lost to jobs
   with idle GPUs."""


import pandas as pd
import glob
import os
import subprocess

#For PEARC26 the last six months was examined:

#sacct -M della -r pli-c,pli,pli-lc,pli-p -a -X -P -n -S 2025-09-28T00:00:00 -E now \
#-o user,alloctres,state | grep gres/gpu=[1-9] | grep -v RUNNING | cut -d"|" -f1 | sort | uniq | wc -l
#151

#sacct -M della -r pli-c,pli,pli-lc,pli-p -a -X -P -n -S 2025-09-28T00:00:00 -E now -o elapsedraw,alloctres,state \
#| grep gres/gpu=[1-9] | grep -v RUNNING | sed -E "s/\|.*gpu=/,/" | awk -F"," '{sum += $1*$2} END {print int(sum/3600)}'
#1302136

"""
From running this code for PLI:

Partition
pli-c     2018
pli        228
pli-lc      85
pli-p       37
Name: count, dtype: int64
79
allocated gpus x limit = 164228
unused gpus x limit = 147430
unused gpus x (limit - hours) = 119925
lost hours = 29224

percent 0% hours = (0.5 * 164228) / 1302136 = 6.3%
# the factor of 0.5 assumes only half of run time limit

percent of users = 79/151=52%

overall usage = 1302136 / (30 * 6 * 24 * 336) = 89%

lost gpu-hours even with auto cancel = 29224 / 1302136 = 2.2%
"""

def get_timelimit_from_sacct(jobid: str) -> float:
    """Get the run time limit (wall time) in units of hours for a given
       jobid."""
    cmd = f"sacct -j {jobid} -X -n -o timelimitraw"
    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            encoding="utf8",
                            check=True,
                            text=True,
                            shell=True)
    return float(result.stdout.strip()) / 60


if __name__ == "__main__":

    # path to your folder
    path = './cancel_zero_gpu_jobs'

    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    df["Email-Sent"]= pd.to_datetime(df["Email-Sent"], format='%m/%d/%Y %H:%M:%S')
    df = df[df.Partition.str.contains("pli")]
    df = df[df["Email-Sent"] >= "2025/09/28"]

    assert get_timelimit_from_sacct("5707330") == 2.0

    df["limit-hrs"] = df.JobID.apply(get_timelimit_from_sacct)
    df["zero-hrs-max"]   = df["GPUs-Allocated"] * df["limit-hrs"]
    df["zero-hrs"]       = df["GPUs-Unused"] * df["limit-hrs"]
    df["zero-hrs-saved"] = df["GPUs-Unused"] * (df["limit-hrs"] - df["Hours"])
    df["lost-hours"]     = df["GPUs-Allocated"] * df["Hours"]

    gp = df.groupby("User").agg({"lost-hours": "sum", "User": "count"})
    gp.rename(columns={"User": "Jobs", "lost-hours": "GPU-hours-at-0%"}, inplace=True)
    gp.reset_index(inplace=True)
    gp = gp.sort_values("GPU-hours-at-0%", ascending=False)
    print(gp["GPU-hours-at-0%"].sum())
    gp["GPU-hours-at-0%"] = gp["GPU-hours-at-0%"].apply(round)
    gp.reset_index(inplace=True, drop=True)
    gp.index += 1
    print(gp.to_string())

    print(df["limit-hrs"].value_counts())
    print(f"Loaded {len(all_files)} files into a single DataFrame.")
    print(df.info())
    print(df.head(3).T)
    print(df.sort_values(by="Email-Sent").head(25))
    print(df["Email-Sent"].min())
    print(df.Partition.value_counts())
    print(df.User.unique().size)
    print("allocated gpus x limit =", round(df["zero-hrs-max"].sum()))
    print("unused gpus x limit =", round(df["zero-hrs"].sum()))
    print("unused gpus x (limit - hours) =", round(df["zero-hrs-saved"].sum()))
    print("lost hours =", round(df["lost-hours"].sum()))
