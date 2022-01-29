#!/usr/licensed/anaconda3/2021.5/bin/python

# github url

import os
import time
import subprocess
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

def raw_dataframe_from_sacct(flags, start_date, fields, renamings=[], numeric_fields=[], use_cache=False):
  fname = f"cache_sacct_{start_date.strftime('%Y%m%d')}.csv"
  if use_cache and os.path.exists(fname):
    print("\nUsing cache file.\n", flush=True)
    rw = pd.read_csv(fname, low_memory=False)
  else:
    cmd = f"sacct {flags} -S {start_date.strftime('%Y-%m-%d')}T00:00:00 -E now -o {fields}"
    print("\nCalling sacct (which may require several seconds) ... ", end="", flush=True)
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=60, text=True, check=True)
    print("done.", flush=True)
    lines = output.stdout.split('\n')
    if lines != [] and lines[-1] == "": lines = lines[:-1]
    rw = pd.DataFrame([line.split("|") for line in lines])
    rw.columns = fields.split(",")
    rw.rename(columns=renamings, inplace=True)
    rw[numeric_fields] = rw[numeric_fields].apply(pd.to_numeric)
    if use_cache: rw.to_csv(fname, index=False)
  return rw

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

def add_new_and_derived_fields(df):
  # new and derived fields
  df["gpus"] = df.alloctres.apply(gpus_per_job)
  df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')
  df["gpu-job"] = df.alloctres.apply(is_gpu_job)
  df["cpu-only-seconds"] = df.apply(lambda row: 0 if row["gpus"] else row["cpu-seconds"], axis="columns")
  df["elapsed-hours"] = df.elapsedraw.apply(lambda x: round(x / seconds_per_hour, 1))
  df["start-date"] = df.start.apply(lambda x: "Unk." if x == "Unknown" else datetime.fromtimestamp(int(x)).date().strftime("%-m/%-d"))
  df["cpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * seconds_per_minute - row["elapsedraw"]) * row["cores"] / seconds_per_hour), axis="columns")
  df["gpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * seconds_per_minute - row["elapsedraw"]) * row["gpus"]  / seconds_per_hour), axis="columns")
  df["cpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * seconds_per_minute * row["cores"] / seconds_per_hour), axis="columns")
  df["gpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * seconds_per_minute * row["gpus"]  / seconds_per_hour), axis="columns")
  df["cpu-hours"] = df["cpu-seconds"] / seconds_per_hour
  df["gpu-hours"] = df["gpu-seconds"] / seconds_per_hour
  return df

def unused_allocated_hours_of_completed(df):
  wh = df[(df.cluster == cluster) & \
          (df.state == "COMPLETED") & \
          (df["elapsed-hours"] >= 2) & \
          (df.partition.isin(partitions))].copy()
  wh["ratio"] = 100 * wh[f"{xpu}-hours"] / wh[f"{xpu}-alloc-hours"]
  d = {f"{xpu}-waste-hours":np.sum, f"{xpu}-alloc-hours":np.sum, f"{xpu}-hours":np.sum, "netid":np.size, \
       "partition":lambda series: ",".join(sorted(set(series))), "ratio":"median"}
  wh = wh.groupby("netid").agg(d).rename(columns={"netid":"jobs", "ratio":"median(%)"})
  wh = wh.sort_values(by=f"{xpu}-hours", ascending=False).reset_index(drop=False)
  wh["rank"] = wh.index + 1
  wh = wh.sort_values(by=f"{xpu}-waste-hours", ascending=False).reset_index(drop=False)
  wh = wh[:5]
  wh.index += 1
  wh[f"{xpu}-hours"] = wh[f"{xpu}-hours"].apply(round).astype("int64")
  wh["ratio(%)"] = 100 * wh[f"{xpu}-hours"] / wh[f"{xpu}-alloc-hours"]
  wh["ratio(%)"] = wh["ratio(%)"].apply(round).astype("int64")
  wh["median(%)"] = wh["median(%)"].apply(round).astype("int64")
  wh = wh[["netid", f"{xpu}-waste-hours", f"{xpu}-hours", f"{xpu}-alloc-hours", "ratio(%)", "median(%)", "rank", "jobs", "partition"]]
  return wh.rename(columns={f"{xpu}-waste-hours":"unused", f"{xpu}-hours":"used", f"{xpu}-alloc-hours":"total"})

def excessive_queue_times(raw):
  # sacct does not return queued jobs with a NODELIST(REASON) of (Dependency) or (JobHeldUser)
  # below we use submit instead of eligible to compute the queued time
  q = raw[raw.state == "PENDING"].copy()
  q["q-days"] = round((time.time() - q["submit"]) / seconds_per_hour / hours_per_day)
  q["q-days"] = q["q-days"].astype("int64")
  cols = ["jobid", "netid", "cluster", "cores", "qos", "partition", "q-days"]
  q = q[cols].groupby("netid").apply(lambda d: d.iloc[d["q-days"].argmax()]).sort_values("q-days", ascending=False)[:10]
  return q

def multinode_with_one_core_per_node(df):
  cols = ["jobid", "netid", "cluster", "nodes", "cores", "gpus", "elapsed-hours", "start-date", "start"]
  m = df[(df["elapsed-hours"] > 2) & (df.nodes > 1) & (df.cores / df.nodes < 14)][cols]
  #m = df[(df["elapsed-hours"] > 2) & (df.nodes > 1) & (df.nodes >= df.cores)][cols]
  m = m.sort_values("netid", ascending=True).drop(columns=["start"])
  return m

def multinode_with_one_gpu_per_node(df):
  cols = ["jobid", "netid", "cluster", "nodes", "gpus", "elapsed-hours", "start-date", "start"]
  cond1 = (df["elapsed-hours"] > 2) & (df.nodes > 1) & (df.gpus > 0) & (df.nodes >= df.gpus)
  cond2 = (df["elapsed-hours"] > 2) & (df.nodes > 1) & (df.gpus > 0) & (df.cluster.isin(("tiger", "traverse"))) & (df.gpus < 4 * df.nodes)
  m = df[cond1|cond2][cols]
  m = m.sort_values("netid", ascending=True).drop(columns=["start"])
  return m

def jobs_with_the_most_gpus(df):
  """Top 10 users with the highest number of GPUs in a job. Only one job per user is shown."""
  cols = ["jobid", "netid", "cluster", "gpus", "nodes", "elapsed-hours", "start-date", "start"]
  g = df[cols].groupby("netid").apply(lambda d: d.iloc[d["gpus"].argmax()])
  g = g.sort_values("gpus", ascending=False)[:10].sort_values("start", ascending=False).drop(columns=["start"])
  return g


if __name__ == "__main__":

  email = False
  num_days_ago = 7

  # pandas display settings
  pd.set_option("display.max_rows", None)
  pd.set_option("display.max_columns", None)
  pd.set_option("display.width", 1000)

  # convert Slurm timestamps to seconds
  os.environ["SLURM_TIME_FORMAT"] = "%s"

  # conversion factors
  seconds_per_minute = 60
  seconds_per_hour = 3600
  hours_per_day = 24

  flags = "-L -a -X -P -n"
  start_date = datetime.now() - timedelta(num_days_ago)
  fields = "jobid,user,cluster,account,partition,cputimeraw,elapsedraw,timelimitraw,nnodes,ncpus,alloctres,submit,eligible,start,qos,state"
  renamings = {"user":"netid", "cputimeraw":"cpu-seconds", "nnodes":"nodes", "ncpus":"cores", "timelimitraw":"limit-minutes"}
  numeric_fields = ["cpu-seconds", "elapsedraw", "limit-minutes", "nodes", "cores", "submit", "eligible"]
  raw = raw_dataframe_from_sacct(flags, start_date, fields, renamings, numeric_fields, use_cache=bool(not email))

  raw = raw[~raw.cluster.isin(("tukey", "perseus"))]
  raw.cluster = raw.cluster.str.replace("tiger2", "tiger")
  raw.partition = raw.partition.str.replace("datascience", "datasci")
  raw.state = raw.state.apply(lambda x: "CANCELLED" if "CANCEL" in x else x)

  df = raw.copy()
  df = df[pd.notnull(df.alloctres)]
  df.start = df.start.astype("int64")
  df = add_new_and_derived_fields(df)

  if not email:
    df.info()
    print(df.describe())
    print("\nTotal NaNs:", df.isnull().sum().sum(), "\n")

  # header
  fmt = "%a %b %-d"
  s = f"{start_date.strftime(fmt)} -- {datetime.now().strftime(fmt)}\n\n"
  s += f"Total users: {raw.netid.unique().size}\n"
  s += f"Total jobs:  {raw.shape[0]}\n\n"

  #####################################
  #### used allocated cpu/gpu hours ###
  #####################################
  cls = (("della", "Della (CPU)", ("cpu", "datascience", "physics"), "cpu"), \
         ("della", "Della (GPU)", ("gpu",), "gpu"), \
         ("stellar", "Stellar (AMD)", ("bigmem", "cimes"), "cpu"), \
         ("stellar", "Stellar (Intel)", ("all", "pppl", "pu", "serial"), "cpu"), \
         ("tiger", "TigerCPU", ("cpu", "ext", "serial"), "cpu"), \
         ("tiger", "TigerGPU", ("gpu",), "gpu"), \
         ("traverse", "Traverse (GPU)", ("all",), "gpu"))
  s += "========= Unused allocated CPU/GPU-Hours (of COMPLETED 2+ hour jobs) =========\n"
  for cluster, name, partitions, xpu in cls:
    s += f"{name}\n"
    s += unused_allocated_hours_of_completed(df).to_string(index=True, justify="center")
    s += "\n\n"

  ######################
  #### fragmentation ###
  ######################
  s += "\n== Multinode CPU jobs with < 14 cores per node (all jobs, 2+ hours) ==\n"
  s += multinode_with_one_core_per_node(df).to_string(index=False, justify="center")
  s += "\n\n\n== Multinode GPU jobs with < max GPUs per node (all jobs, 2+ hours) ==\n"
  s += multinode_with_one_gpu_per_node(df).to_string(index=False, justify="center")

  #######################
  #### large gpu jobs ###
  #######################
  s += "\n\n\n=========== Jobs with the most GPUs (1 job per user) ===========\n"
  s += jobs_with_the_most_gpus(df).to_string(index=False, justify="center")

  ##############################
  #### excessive queue times ###
  ##############################
  s += "\n\n\n================== Excessive queue times (1 job per user) ==================\n"
  s += excessive_queue_times(raw).to_string(index=False, justify="center")

  if email:
    send_email(s, "halverson@princeton.edu")
  else:
    print(s)
