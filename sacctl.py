#!/usr/licensed/anaconda3/2021.5/bin/python

import os
import sys
import subprocess
from socket import gethostname
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(s, addressee, sender="halverson@princeton.edu"):
  msg = MIMEMultipart('alternative')
  msg['Subject'] = "Slurm job alerts"
  msg['From'] = sender
  msg['To'] = addressee
  text = "None"
  html = f'<html><head></head><body><font face="Courier New, Courier, monospace"><pre>{s}</pre></font></body></html>'
  part1 = MIMEText(text, 'plain'); msg.attach(part1) 
  part2 = MIMEText(html, 'html');  msg.attach(part2)
  s = smtplib.SMTP('localhost')
  s.sendmail(sender, addressee, msg.as_string())
  s.quit()
  return None

def get_raw_dataframe(start_date):
  fname = f"out-{start_date.replace(':','-')}.csv"
  if not os.path.exists(fname):
  #if 1:
    fields = "jobid,user,cluster,account,partition,cputimeraw,elapsedraw,timelimitraw,nnodes,ncpus,alloctres,eligible,start,qos,state"
    cmd = f"sacct -L -a -X -P -n -S {start_date} -E now -o {fields}"
    print("\nCalling sacct (which may require several seconds) ... ", end="", flush=True)
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=600, text=True, check=True)
    print("done.", flush=True)
    lines = output.stdout.split('\n')
    if lines != [] and lines[-1] == "": lines = lines[:-1]
    df = pd.DataFrame([line.split("|") for line in lines])
    df.columns = fields.split(",")
    df.rename(columns={"user":"netid", "cputimeraw":"cpu-seconds", "nnodes":"nodes", "ncpus":"cores", "timelimitraw":"limit-minutes"}, inplace=True)
    df = df[pd.notnull(df.alloctres) & (~df.cluster.isin(["tukey", "perseus"]))]
    df.to_csv(fname, index=False)
  df = pd.read_csv(fname, low_memory=False)
  df = df[pd.notnull(df.alloctres)]
  return df

def gpus_per_job(tres):
  # billing=8,cpu=4,mem=16G,node=1
  # billing=112,cpu=112,gres/gpu=16,mem=33600M,node=4
  if "gres/gpu=" in tres:
    for part in tres.split(","):
      if "gres/gpu=" in part:
        gpus = int(part.split("=")[-1])
        assert gpus > 0
    return gpus
  else:
    return 0

def is_gpu_job(tres):
  return 1 if "gres/gpu=" in tres and not "gres/gpu=0" in tres else 0


if __name__ == "__main__":

  email = True

  num_days_ago = 7
  start_date = datetime.now() - timedelta(num_days_ago)

  # pandas display settings
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 1000)

  # convert Slurm timestamps to seconds
  os.environ['SLURM_TIME_FORMAT'] = "%s"

  df = get_raw_dataframe(f"{start_date.strftime('%Y-%m-%d')}T00:00:00")
  if 0:
    df.info()
    print(df.describe())
  # filters
  # cleaning
  #df.state = df.state.apply(lambda x: "CANCELLED" if "CANCEL" in x else x)
  df.cluster = df.cluster.str.replace("tiger2", "tiger")

  if 0:
    print("\nTotal NaNs:", df.isnull().sum().sum(), "\n")
    print("\nPartitions:", np.sort(df.partition.unique()))
    print("Accounts:", np.sort(df.account.unique()))
    print("QOS:", np.sort(df.qos.unique()))
    print("Number of unique users:", df.netid.unique().size)
    print("Clusters:", df.cluster.unique())
    clusters = df.cluster.unique()

  seconds_per_minute = 60
  seconds_per_hour = 3600

  # add derived fields
  df["gpus"] = df.alloctres.apply(gpus_per_job)
  df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')
  df["gpu-job"] = df.alloctres.apply(is_gpu_job)
  df["cpu-only-seconds"] = df.apply(lambda row: 0 if row["gpus"] else row["cpu-seconds"], axis="columns")
  df["elapsed-hours"] = df.elapsedraw.apply(lambda x: round(x / seconds_per_hour, 1))
  df["start-date"] = df.start.apply(lambda x: x if x == "Unknown" else datetime.fromtimestamp(int(x)).date().strftime("%-m/%-d"))
  df["cpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * seconds_per_minute - row["elapsedraw"]) * row["cores"] / seconds_per_hour), axis="columns")
  df["gpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * seconds_per_minute - row["elapsedraw"]) * row["gpus"]  / seconds_per_hour), axis="columns")
  df["cpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * seconds_per_minute * row["cores"] / seconds_per_hour), axis="columns")
  df["gpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * seconds_per_minute * row["gpus"]  / seconds_per_hour), axis="columns")
  df["cpu-hours"] = df["cpu-seconds"] / seconds_per_hour
  df["gpu-hours"] = df["gpu-seconds"] / seconds_per_hour

  cls = (("della", ("cpu", "datascience", "physics"), "cpu"), \
         ("della", ("gpu",), "gpu"), \
         ("stellar", ("all", "pppl", "pu", "serial"), "cpu"), \
         ("stellar", ("bigmem", "cimes"), "cpu"), \
         ("tiger", ("cpu", "ext", "serial"), "cpu"), \
         ("tiger", ("gpu",), "gpu"), \
         ("traverse", ("all",), "gpu"))

  fmt = "%a %b %-d"
  s = f"{start_date.strftime(fmt)} -- {datetime.now().strftime(fmt)}\n\n"
  s += f"Total users: {df.netid.unique().size}\n"
  s += f"Total jobs:  {df.shape[0]}\n\n"
  s += "========= Unused allocated CPU-Hours (of completed and 2+ hour jobs) =========\n"
  for cluster, partitions, xpu in cls:
    s += f"{cluster.upper()}\n"
    d = {f"{xpu}-waste-hours":np.sum, f"{xpu}-alloc-hours":np.sum, f"{xpu}-hours":np.sum, "netid":np.size, "partition":lambda series: ",".join(sorted(set(series)))}
    wh = df[(df.cluster == cluster) & \
            (df.state == "COMPLETED") & \
            (df["elapsed-hours"] > 2) & \
            (df.partition.isin(partitions))].groupby("netid").agg(d).rename(columns={"netid":"jobs"})
    wh = wh.sort_values(by=f"{xpu}-hours", ascending=False).reset_index(drop=False)
    wh["rank"] = wh.index + 1
    wh = wh.sort_values(by=f"{xpu}-waste-hours", ascending=False)
    wh = wh[:5]
    wh[f"{xpu}-hours"] = wh[f"{xpu}-hours"].apply(round).astype("int64")
    wh["ratio(%)"] = 100 * wh[f"{xpu}-hours"] / wh[f"{xpu}-alloc-hours"]
    wh["ratio(%)"] = wh[f"ratio(%)"].apply(round).astype("int64")
    wh = wh[["netid",f"{xpu}-waste-hours", f"{xpu}-alloc-hours", f"{xpu}-hours", "ratio(%)", "rank", "jobs", "partition"]]
    s += wh.to_string(index=False, justify="center")
    s += "\n\n\n"

  s += "\n\n========= Multinode jobs with 1 core per node: =========\n"
  tmp = df[(df["elapsed-hours"] > 2) & (df.nodes > 1) & (df.nodes >= df.cores)][["jobid", "netid", "cluster", "nodes", "cores", "gpus", "elapsed-hours", "start-date"]]
  s += tmp.to_string(index=False, justify="center")

  s += "\n\n========= Multinode jobs with 1 GPU per node: =========\n"
  tmp = df[(df["elapsed-hours"] > 2) & (df.nodes > 1) & (df.gpus > 1) & (df.nodes >= df.gpus)][["jobid", "netid", "cluster", "nodes", "cores", "gpus", "elapsed-hours", "start-date"]]
  s += tmp.to_string(index=False, justify="center")

  if email:
    send_email(s, "halverson@princeton.edu")
  else:
    print(s)
