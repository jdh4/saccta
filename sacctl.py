#!/usr/licensed/anaconda3/2021.5/bin/python

# version control
# https://github.com/jdh4/saccta/blob/main/sacctl.py

import argparse
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

# conversion factors
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24

# job states
states = {
  'BF'  :'BOOT_FAIL',
  'CLD' :'CANCELLED',
  'COM' :'COMPLETED',
  'DL'  :'DEADLINE',
  'F'   :'FAILED',
  'NF'  :'NODE_FAIL',
  'OOM' :'OUT_OF_MEMORY',
  'PD'  :'PENDING',
  'PR'  :'PREEMPTED',
  'R'   :'RUNNING',
  'RQ'  :'REQUEUED',
  'RS'  :'RESIZING',
  'RV'  :'REVOKED',
  'S'   :'SUSPENDED',
  'TO'  :'TIMEOUT'
  }
JOBSTATES = dict(zip(states.values(), states.keys()))

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
    if use_cache: print("\nCalling sacct (which may require several seconds) ... ", end="", flush=True)
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=120, text=True, check=True)
    if use_cache: print("done.", flush=True)
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
  df["elapsed-hours"] = df.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR))
  df["start-date"] = df.start.apply(lambda x: x if x == "Unknown" else datetime.fromtimestamp(int(x)).strftime("%a %-m/%-d"))
  df["cpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * SECONDS_PER_MINUTE - row["elapsedraw"]) * row["cores"] / SECONDS_PER_HOUR), axis="columns")
  df["gpu-waste-hours"] = df.apply(lambda row: round((row["limit-minutes"] * SECONDS_PER_MINUTE - row["elapsedraw"]) * row["gpus"]  / SECONDS_PER_HOUR), axis="columns")
  df["cpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * SECONDS_PER_MINUTE * row["cores"] / SECONDS_PER_HOUR), axis="columns")
  df["gpu-alloc-hours"] = df.apply(lambda row: round(row["limit-minutes"] * SECONDS_PER_MINUTE * row["gpus"]  / SECONDS_PER_HOUR), axis="columns")
  df["cpu-hours"] = df["cpu-seconds"] / SECONDS_PER_HOUR
  df["gpu-hours"] = df["gpu-seconds"] / SECONDS_PER_HOUR
  return df

def unused_allocated_hours_of_completed(df, cluster, name, partitions, xpu):
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

def multinode_cpu_fragmentation(df):
  cols = ["jobid", "netid", "cluster", "nodes", "cores", "gpus", "state", "partition", "elapsed-hours", "start-date", "start"]
  m = df[(df["elapsed-hours"] >= 2) & (df.nodes > 1) & (df.cores / df.nodes < 14)][cols]
  m = m.sort_values(["netid", "start"], ascending=[True, False]).drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  m.state = m.state.apply(lambda x: JOBSTATES[x])
  return m

def multinode_gpu_fragmentation(df):
  cols = ["jobid", "netid", "cluster", "nodes", "gpus", "state", "partition", "elapsed-hours", "start-date", "start"]
  cond1 = (df["elapsed-hours"] >= 2) & (df.nodes > 1) & (df.gpus > 0) & (df.nodes == df.gpus)
  cond2 = (df["elapsed-hours"] >= 2) & (df.nodes > 1) & (df.gpus > 0) & (df.cluster.isin(["tiger", "traverse"])) & (df.gpus < 4 * df.nodes)
  m = df[cond1 | cond2][cols]
  m.state = m.state.apply(lambda x: JOBSTATES[x])
  m = m.sort_values(["netid", "start"], ascending=[True, False]).drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  return m

def jobs_with_the_most_gpus(df):
  """Top 10 users with the highest number of GPUs in a job. Only one job per user is shown."""
  cols = ["jobid", "netid", "cluster", "gpus", "nodes", "state", "partition", "elapsed-hours", "start-date", "start"]
  g = df[cols].groupby("netid").apply(lambda d: d.iloc[d["gpus"].argmax()])
  g = g.sort_values("gpus", ascending=False)[:10].drop(columns=["start"]).rename(columns={"elapsed-hours":"hours"})
  g.state = g.state.apply(lambda x: JOBSTATES[x])
  return g

def longest_queue_times(raw):
  # below we use submit instead of eligible to compute the queue time
  q = raw[raw.state == "PENDING"].copy()
  q["q-days"] = round((time.time() - q["submit"]) / SECONDS_PER_HOUR / HOURS_PER_DAY)
  q["q-days"] = q["q-days"].astype("int64")
  cols = ["jobid", "netid", "cluster", "cores", "qos", "partition", "q-days"]
  q = q[cols].groupby("netid").apply(lambda d: d.iloc[d["q-days"].argmax()]).sort_values("q-days", ascending=False)[:10]
  return q


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Slurm job alerts')
  parser.add_argument('--days', type=int, default=14, metavar='N',
                      help='Create report over N previous days from now (default: 14)')
  parser.add_argument('--email', action='store_true', default=False,
                      help='Send output via email')
  args = parser.parse_args()

  # pandas display settings
  pd.set_option("display.max_rows", None)
  pd.set_option("display.max_columns", None)
  pd.set_option("display.width", 1000)

  # convert Slurm timestamps to seconds
  os.environ["SLURM_TIME_FORMAT"] = "%s"

  flags = "-L -a -X -P -n"
  start_date = datetime.now() - timedelta(days=args.days)
  fields = "jobid,user,cluster,account,partition,cputimeraw,elapsedraw,timelimitraw,nnodes,ncpus,alloctres,submit,eligible,start,qos,state"
  renamings = {"user":"netid", "cputimeraw":"cpu-seconds", "nnodes":"nodes", "ncpus":"cores", "timelimitraw":"limit-minutes"}
  numeric_fields = ["cpu-seconds", "elapsedraw", "limit-minutes", "nodes", "cores", "submit", "eligible"]
  raw = raw_dataframe_from_sacct(flags, start_date, fields, renamings, numeric_fields, use_cache=not args.email)

  raw = raw[~raw.cluster.isin(["tukey", "perseus"])]
  raw.cluster = raw.cluster.str.replace("tiger2", "tiger")
  raw.partition = raw.partition.str.replace("datascience", "datasci")
  raw.partition = raw.partition.str.replace("physics", "phys")
  raw.state = raw.state.apply(lambda x: "CANCELLED" if "CANCEL" in x else x)

  # df excludes pending jobs
  df = raw.copy()
  df = df[pd.notnull(df.alloctres) & (df.alloctres != "")]
  df.start = df.start.astype("int64")
  df = add_new_and_derived_fields(df)

  if not args.email:
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
    u = unused_allocated_hours_of_completed(df, cluster, name, partitions, xpu)
    s += u.to_string(index=True, justify="center")
    s += "\n\n"

  ####### consider jobs in the last 5 days only #######
  df = df[df.start >= time.time() - 5 * HOURS_PER_DAY * SECONDS_PER_HOUR]

  ######################
  #### fragmentation ###
  ######################
  s += "\n===== Multinode CPU jobs with < 14 cores per node (all jobs, 2+ hours) =====\n"
  s += multinode_cpu_fragmentation(df).to_string(index=False, justify="center")
  s += "\n\n\n===== Multinode GPU jobs with fragmentation (all jobs, 2+ hours) =====\n"
  s += multinode_gpu_fragmentation(df).to_string(index=False, justify="center")

  #######################
  #### large gpu jobs ###
  #######################
  s += "\n\n\n============== Jobs with the most GPUs (1 job per user) ==============\n"
  s += jobs_with_the_most_gpus(df).to_string(index=False, justify="center")

  ##############################
  #### longest queue times ###
  ##############################
  s += "\n\n\n====== Longest queue times of currently PENDING jobs (1 job per user) ======\n"
  s += longest_queue_times(raw).to_string(index=False, justify="center")

  send_email(s, "halverson@princeton.edu") if args.email else print(s)
