import re
import subprocess
import pandas as pd


def get_data_from_sacct(clusters: str,
                        start_date: str,
                        end_date: str,
                        partitions: str,
                        fields: str) -> pd.DataFrame:
    """Return a dataframe of the sacct output."""
    cmd = f"sacct -M {clusters} -a -X -P -n -S {start_date} -E {end_date} {partitions} -o {fields}"
    output = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            shell=True,
                            timeout=100,
                            text=True,
                            check=True)
    rows = [row.split("|") for row in output.stdout.split()]
    df = pd.DataFrame(rows)
    df.columns = fields.split(",")
    return df


def gpus_per_job(tres: str) -> int:
    """Return the number of allocated GPUs."""
    gpus = re.findall(r"gres/gpu=\d+", tres)
    return int(gpus[0].replace("gres/gpu=", "")) if gpus else 0


if __name__ == "__main__":

    start_date = "2025-05-01"
    end_date = "now"
    #partitions = "-r all,bigmem,cimes,gpu,pppl,pu,serial"
    partitions = ""

    clusters = "stellar"
    fields = "user,alloctres,elapsedraw,cputimeraw"
    df = get_data_from_sacct(clusters, start_date, end_date, partitions, fields)

    # clean dataframe
    df = df[pd.notna(df.elapsedraw)]
    df = df[df.elapsedraw.str.isnumeric()]
    df.elapsedraw = df.elapsedraw.astype("int64")
    df = df[df.elapsedraw > 0]
    df.cputimeraw = df.cputimeraw.astype("int64")

    df["cpu-hours"] = df["cputimeraw"] / 3600
    df["gpus"] = df.alloctres.apply(gpus_per_job)
    df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')
    df["gpu-hours"] = df["gpu-seconds"] / 3600

    by_user = df.groupby("user").agg({"cpu-hours":"sum", "gpu-hours":"sum"})

    for xpu in ("cpu", "gpu"):
        by_user["proportion(%)"] = 100 * by_user[f"{xpu}-hours"] / by_user[f"{xpu}-hours"].sum()
        by_user["proportion(%)"] = by_user["proportion(%)"].apply(round)
        pro = by_user[by_user["proportion(%)"] > 0].copy()
        pro[f"{xpu}-hours"] = pro[f"{xpu}-hours"].apply(round)
        pro.sort_values(f"{xpu}-hours", ascending=False, inplace=True)
        print(pro[[f"{xpu}-hours", "proportion(%)"]].to_string())
