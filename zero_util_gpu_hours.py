import pandas as pd
from efficiency import num_gpus_with_zero_util
from efficiency import get_stats_dict

def gpus_per_job(tres: str) -> int:
    """Return the number of GPUs used."""
    # billing=8,cpu=4,mem=16G,node=1
    # billing=112,cpu=112,gres/gpu=16,mem=33600M,node=4
    if "gres/gpu=" in tres:
        for part in tres.split(","):
            if "gres/gpu=" in part:
                gpus = int(part.split("=")[-1])
                assert gpus > 0
                return gpus
        raise Exception(f"GPU count not extracted for {tres}")
    else:
        return 0

# sacct -M della -a -X -P -S 2023-08-31T00:00:00 -o user,account,elapsedraw,alloctres,admincomment --partition=gpu > raw.csv
sacct = pd.read_csv("raw.csv", sep="|")
sacct.columns = "JobID|User|Account|ElapsedRaw|AllocTRES|AdminComment".lower().split("|")

sacct = sacct[pd.notnull(sacct.alloctres) &
              (sacct.alloctres != "") &
              (sacct.elapsedraw > 0) &
              (sacct.admincomment != {}) &
              sacct.alloctres.str.contains("gres/gpu")]

print(sacct.describe())
print(sacct.info())
sacct["gpus"] = sacct.alloctres.apply(gpus_per_job)
print(sacct.gpus.value_counts())
sacct["admincomment"] = sacct["admincomment"].apply(get_stats_dict)
sacct = sacct[sacct.admincomment != {}]
sacct.info()
sacct["GPUs-Unused"] = sacct.admincomment.apply(num_gpus_with_zero_util)
sacct = sacct.drop(columns=["admincomment"])
print(sacct.tail(5))
sacct["zero-gpu-seconds"] = sacct["elapsedraw"] * sacct["GPUs-Unused"]
sacct["zero-gpu-hours"] = sacct["zero-gpu-seconds"] / 3600
sacct["gpu-seconds"] = sacct["elapsedraw"] * sacct["gpus"]
print(sacct["zero-gpu-seconds"].sum() / 3600)
print(sacct["gpu-seconds"].sum() / 3600)

print(sacct[(sacct["user"] == "mw0425") & (sacct["GPUs-Unused"] > 0)][["user", "jobid", "zero-gpu-hours"]].sort_values("zero-gpu-hours").to_string())

gp = sacct.groupby("user").agg({"zero-gpu-seconds":"sum"})
gp["zero-gpu-seconds"] = gp["zero-gpu-seconds"] / 3600
print(gp.sort_values("zero-gpu-seconds", ascending=False).head(10).to_string())
