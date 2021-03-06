# this script cannot be ran blindly for sponsor, position or dept

# on initial pass, run over all partitions as a check for new partitions

# run on della-gpu for della GPU data
# run on della8 for della CPU data
# run on tigergpu for tigergpu data
# run on tigercpu for tigercpu data
# run on stellar-amd for amd
# run on stellar-intel for intel

import os
import sys
import subprocess
from socket import gethostname
from datetime import datetime
import re
import numpy as np
import pandas as pd
import dossier  # wget https://raw.githubusercontent.com/jdh4/tigergpu_visualization/master/dossier.py

start_date = "2021-07-01T00:00:00"
end_date   = "2021-12-31T23:59:59"

# generate latex files
latex = True

# run test case
test_case = False

slurmacct_states = False  # use Nielsen convention (True) or not (False)
if slurmacct_states:
  states = "-s ca,cd,f,to,pr,oom"
else:
  states = ""

###################################
# user inputs have been set above
###################################

# pandas display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# convert Slurm timestamps to seconds
os.environ['SLURM_TIME_FORMAT'] = "%s"

def get_host():
  host = gethostname().lower().split('.')[0]
  return "".join([c for c in host if not c.isdigit()])

host = get_host()
# checkgpu is only available on three login nodes
checkgpu_hosts = ("della-gpu", "tigergpu", "traverse")
ondemand_hosts = ("adroit", "della", "stellar-amd")

if host == "tigergpu":
  partition = "--partition gpu"
  qos_test = ["gpu-test"]
elif host == "tigercpu":
  partition = "--partition cpu,ext,serial"
  qos_test = ["tiger-test"]
elif host == "traverse":
  partition = ""
  qos_test = ["trav-test"]
elif host == "adroit":
  partition = ""
  qos_test = ["test", "gpu-test"]
elif host == "della":
  partition = "--partition cpu,datascience,physics"
  #partition = "--partition cpu,datascience,gpu,physics"  # for other versus ondemand
  qos_test = ["test"]
elif host == "della-gpu":
  partition = "--partition gpu"
  qos_test = []
elif host == "stellar-intel":
  partition = "--partition all,pppl,pu,serial"
  qos_test = ["stellar-debug"]
elif host == "stellar-amd":
  partition = "--partition bigmem,cimes"
  #partition = "--partition gpu"  # for gpu jobs
  #partition = ""  # for other versus ondemand jobs
  qos_test = ["stellar-debug"]
else:
  print("Host not recognized. Exiting.")
  sys.exit()


# date range and host
start_range = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S").strftime("%B %-d, %Y")
end_range   = datetime.strptime(end_date,   "%Y-%m-%dT%H:%M:%S").strftime("%B %-d, %Y")
date_range = f"{start_range} to {end_range}"
cluster_names = {"adroit":"Adroit", "della":"Della (CPU)", "della-gpu":"Della (GPU)",
                 "stellar-amd":"Stellar (AMD)", "stellar-intel":"Stellar (Intel)",
                 "tigercpu":"TigerCPU", "tigergpu":"TigerGPU", "traverse":"Traverse"}
caption_host = cluster_names[host]

if test_case:
  host = "test"
  caption_host = "test"
  partition = ""
  qos_test = []

if partition:
  print(f"\nRunning on {host} with {partition}\n")
else:
  print(f"\nRunning on {host} over all partitions\n")

fname = f"{host}_sacct_{start_date}_{end_date}.csv".replace(":","-")
cols = ["jobid", "netid", "account", "partition", "cpu-seconds", "elapsedraw", "alloctres", "start", "eligible", "qos", "state", "jobname"]
if test_case:
  # create test input
  jobs = []
  jobs.append(("42",   "jdh4", "cses", "gpu", "100000",  "10000", "billing=112,cpu=10,gres/gpu=2,mem=400M,node=1", "36000", "15000", "test-gpu", "COMPLETED", "jobname"))
  jobs.append(("43",   "jdh4", "cses", "gpu", "200000",  "20000", "billing=112,cpu=10,gres/gpu=2,mem=400M,node=1", "37000", "16500", "test-gpu",    "FAILED", "sys/dashboard/sys/jupyter"))
  jobs.append(("44",   "jdh4", "cses", "cpu", "222222", "222222", "billing=112,cpu=1,mem=4000M,node=1",            "42000", "11000", "test-cpu", "COMPLETED", "jobname"))
  jobs.append(("66",   "curt", "cses", "cpu", "999999", "999999", "billing=112,cpu=1,mem=4000M,node=1",            "33000",  "2200", "test-cpu",    "FAILED", "jobname"))
  jobs.append(("79", "yz8614",   "cs", "cpu", "600000", "600000", "billing=112,cpu=1,mem=4000M,node=1",            "92000", "11000", "test-cpu", "COMPLETED", "jobname"))
  jobs.append(("80", "yz8614",   "cs", "gpu", "111100",  "11110", "billing=112,cpu=10,gres/gpu=1,mem=40G,node=1",  "64000", "32000", "test-gpu", "NODE_FAIL", "jobname"))
  df_test = pd.DataFrame(jobs)
  df_test.columns = cols
  print(df_test)
  df_test.to_csv(fname, index=False)
elif not os.path.exists(fname):
  print("\nCalling sacct (which may require several seconds) ... ", end="", flush=True)
  # cputimeraw has units of cpu-seconds (it is equal to cores multiplied by elapsed)
  # pending jobs have cputimeraw of 0
  fmt = "jobid,user,account,partition,cputimeraw%25,elapsedraw%50,alloctres%75,start,eligible,qos,state,jobname%50"
  # jobname must be last in next line to deal with "|" characters in a few lines
  cmd = f"sacct -a -X -P -n -S {start_date} -E {end_date} -o {fmt} {states} {partition}"
  output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=600, text=True, check=True)
  lines = output.stdout.split('\n')
  if lines != [] and lines[-1] == "": lines = lines[:-1]

  # deal with "|" characters in jobname with [:len(cols)] in next line
  df = pd.DataFrame([line.split("|")[:len(cols)] for line in lines])
  df.columns = cols
  df.to_csv(fname, index=False)
  print("done.", flush=True)
else:
  print("\nUsing sacct cache file.\n")

df = pd.read_csv(fname, low_memory=False)
df.info()
print("\nTotal NaNs:", df.isnull().sum().sum(), "\n")

df = df[~(pd.isna(df.alloctres) & (df["cpu-seconds"] == 0))]
print("\nRemoved rows where alloctres == NULL and cpu-seconds == 0")

df = df[~(df.state == "RUNNING")]
print("Removed rows where state == RUNNING", "\n")

df.account = df.account.str.replace("wws", "spia", regex=False)

# next line serves as check of data type and may cause error
# if encounter error then change jobname to jobid in sacct call above (but then no ondemand data)
df["start"] = df["start"].astype("int64")

# a small number of jobs have "Unknown" as eligible with non-null alloctres and state "COMPLETED"
# 7460614,mcmuniz,cbe,gpu,0,0,"billing=7,cpu=1,gres/gpu=1,mem=4G,node=1",1633939938,Unknown
# 7538517_711,mklatt,chem,cpu,128000,800,"billing=160,cpu=160,mem=625G,node=4",1636383770,Unknown
ans = df[(df.eligible == "Unknown") & pd.notna(df.alloctres)].shape[0]
print(f"Number of jobs with eligible == 'Unknown' and alloctres not NULL: {ans} ({round(100 * ans / df.shape[0])}%)\n")
df["eligible"] = df.apply(lambda row: row["eligible"] if row["eligible"] != "Unknown" else row["start"], axis='columns')
df["eligible"] = df["eligible"].astype("int64")
print("\nTotal NaNs:", df.isnull().sum().sum(), "\n")
df.info()
print("")
print(df.describe().astype('int64'))

if host == "della-gpu":
  print("\nraw QOSes:", np.sort(df.qos.unique()))
  before = df.shape[0]
  df = df[df.qos == "della-gpu"]
  print(f"Only considering jobs in the della-gpu QOS (excluded {before - df.shape[0]} jobs).\n")

if host == "stellar-intel":
  print("\nraw accounts:", np.sort(df.account.unique()))
  before = df.shape[0]
  df = df[~df.account.str.contains('cimes')]
  print(f"\nRemoving jobs with cimes* account (excluded {before - df.shape[0]} jobs).\n")

print("\nPartitions:", np.sort(df.partition.unique()))
print("Accounts:", np.sort(df.account.unique()))
print("QOS:", np.sort(df.qos.unique()))
print("Number of unique users:", df.netid.unique().size)

# write out unique netids
with open(f"{host}_netids.txt", "w") as f:
  f.write("\n".join(sorted(df.netid.unique().tolist())))



#########################
#  Q U E U E    T I M E
#########################

def queue_time(start, eligible):
  q_seconds = start - eligible
  return q_seconds if q_seconds >= 0 else 0

ans = df[df.start < df.eligible].shape[0]
print(f"\nNumber of jobs with start < eligible: {ans} ({round(100 * ans / df.shape[0])}%)\n")
df["q-seconds"] = df.apply(lambda row: queue_time(row["start"], row["eligible"]), axis='columns')

ans = df[df["cpu-seconds"] == 0].shape[0]
print(f"Number of jobs with cpu-seconds = 0: {ans} ({round(100 * ans / df.shape[0])}%)")

# production jobs
prod = df[~df.qos.isin(qos_test)].copy()
ans = prod[prod["cpu-seconds"] == 0].shape[0]
print(f"Number of production (non-test) jobs with cpu-seconds = 0: {ans} ({round(100 * ans / prod.shape[0])}%)")
seconds_per_hour = 3600
ans = prod[prod["q-seconds"] <= seconds_per_hour].shape[0]
print(f"Number of production (non-test) jobs that started within 1 hour: {ans} ({round(100 * ans / prod.shape[0])}%)")
ans = prod[prod["q-seconds"] <= 24 * seconds_per_hour].shape[0]
print(f"Number of production (non-test) jobs that started within 24 hours: {ans} ({round(100 * ans / prod.shape[0])}%)\n\n")
del prod




############
#  G P U S
############

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

df["gpus"] = df.alloctres.apply(gpus_per_job)
df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')
df["gpu-job"] = df.alloctres.apply(is_gpu_job)
df["cpu-only-seconds"] = df.apply(lambda row: 0 if row["gpus"] else row["cpu-seconds"], axis="columns")

# convert seconds to hours
df["cpu-only-hours"] = df["cpu-only-seconds"] / seconds_per_hour
df["cpu-hours"] = df["cpu-seconds"] / seconds_per_hour
df["gpu-hours"] = df["gpu-seconds"] / seconds_per_hour
df["q-hours"]   = df["q-seconds"]   / seconds_per_hour




# helper function
def add_proportion_in_parenthesis(dframe, column_name, replace=False):
  assert column_name in dframe.columns
  dframe["proportion"] = 100 * dframe[column_name] / dframe[column_name].sum()
  dframe["proportion"] = dframe["proportion"].apply(lambda x: round(x, 1))
  name = column_name if replace else f"{column_name}-cmb"
  dframe[name] = dframe.apply(lambda row: f"{round(row[column_name])} ({row['proportion']}%)", axis='columns')
  dframe = dframe.drop(columns=["proportion"])
  return dframe

# helper function
def pad_multicolumn(fname, column_names):
  with open(fname, "r") as f:
    lines = f.readlines()
  for column_name in column_names:
    thres = 0
    # find column index
    for i, line in enumerate(lines):
      if column_name in line and " \\" in line:
        num_columns = len(line.split("&"))
        cols = [c.replace("\\", "").strip() for c in line.split("&")]
        idx = cols.index(column_name)
        cols[idx] = "multicolumn{1}{c}{" + column_name + "}"
        lines[i] = " & ".join(cols).replace("multicolumn", "\\multicolumn") + " \\\\\n"
      if " \\" in line and " & " in line and not column_name in line:
        parts = line.split("&")
        value, percent = parts[idx].replace("\\\\\n", "").replace("\\", "").strip().split()
        if len(percent) > thres: thres = len(percent)
    # second pass
    for i, line in enumerate(lines):
      if " \\" in line and " & " in line and not column_name in line:
        parts = line.replace("\\\\\n", "").replace("\\\\ \n", "").strip().split("&")
        value, percent = parts[idx].replace("\\", "").strip().split()
        parts[idx] = value + "\hphantom{" + "0" * (thres - len(percent) + 1) + "}" + percent.replace("%", "\%")
        lines[i] = " & ".join(parts) + r" \\ " + "\n"
  with open(fname, "w") as f:
    f.write("".join(lines))




###########################################################
# Q O S,  S T A T E,  P A R T I T I O N,  O N D E M A N D
###########################################################

# Q O S
field  = "gpu-hours" if host in checkgpu_hosts else "cpu-hours"
field2 = "GPU-Hours" if host in checkgpu_hosts else "CPU-Hours"
q = df[["netid", "qos", field, "q-hours"]].copy()
q = q.groupby("qos").agg({"qos":np.size, field:np.sum, "q-hours":[np.sum, np.mean, 'median'], "netid":lambda series: len(set(series))})
q.columns = ["Number of Jobs", field2, "Q-Hours", "Mean Q-Hours per Job", "Median Q-Hours per Job", "Number of Users"]
q["QOS"] = q.index
q["Mean Q-Hours per Job"]   = q["Mean Q-Hours per Job"].apply(lambda x: round(x, 1))
q["Median Q-Hours per Job"] = q["Median Q-Hours per Job"].apply(lambda x: round(x, 1))
q = q.sort_values(field2, ascending=False)
q = add_proportion_in_parenthesis(q, "Number of Jobs", replace=True)
q = add_proportion_in_parenthesis(q, field2, replace=True)
q = add_proportion_in_parenthesis(q, "Q-Hours", replace=True)
q = q[["QOS", field2, "Number of Users", "Number of Jobs", "Q-Hours", "Mean Q-Hours per Job", "Median Q-Hours per Job"]].reset_index(drop=True)
print(q, end="\n\n")
if latex:
  base = f"{host}_qos"
  q.to_csv(f"{base}.csv")
  fname = f"{base}.tex"
  caption = (f"Breakdown of jobs by QOS on {caption_host} from {date_range}.", f"{caption_host} -- Utilization by QOS")
  q.to_latex(fname, index=False, caption=caption, column_format="rrcrrcc", label=f"{host}_qos")
  pad_multicolumn(fname, ["Number of Jobs", "Q-Hours", field2])

# next line helps understand skew in queue times
# added fillna as a fix on 4-29-2022 (useful when std is NaN)
print(df[["qos", "q-hours"]].groupby("qos").describe().apply(round).fillna(0).astype('int64'))
print("\n")

# S T A T E
df.state = df.state.apply(lambda x: "CANCELLED" if "CANCEL" in x else x)
s = df[["netid", "state"]].copy()
s = s.groupby("state").agg({"state":['first', np.size], "netid":lambda series: len(set(series))}).reset_index(drop=True)
#s = s.reset_index(drop=True)
s.columns = ["State", "Number of Jobs", "Number of Users"]
s = s.sort_values("Number of Jobs", ascending=False)
s = add_proportion_in_parenthesis(s, "Number of Jobs", replace=True)
print(s, end="\n\n")
if latex:
  base = f"{host}_state"
  s.to_csv(f"{base}.csv")
  fname = f"{base}.tex"
  caption = (f"Breakdown of jobs by state on {caption_host} from {date_range}.",  f"{caption_host} -- Utilization by Job State")
  s.to_latex(fname, index=False, caption=caption, column_format="rrc", label=f"{host}_state")
  pad_multicolumn(fname, ["Number of Jobs"])

# P A R T I T I O N
field  = "gpu-hours" if host in checkgpu_hosts else "cpu-hours"
field2 = "GPU-Hours" if host in checkgpu_hosts else "CPU-Hours"
q = df[["netid", "partition", field, "q-hours"]].copy()
d = {"partition":np.size, field:np.sum, "q-hours":[np.sum, 'median'], "netid":lambda series: len(set(series))}
q = q.groupby("partition").agg(d)
q.columns = ["Number of Jobs", field2, "Q-Hours", "Median Q-Hours per Job", "Number of Users"]
q["Partition"] = q.index
q["Median Q-Hours per Job"] = q["Median Q-Hours per Job"].apply(lambda x: round(x, 1))
q = q.sort_values(field2, ascending=False)
q = add_proportion_in_parenthesis(q, "Number of Jobs", replace=True)
q = add_proportion_in_parenthesis(q, field2, replace=True)
q = add_proportion_in_parenthesis(q, "Q-Hours", replace=True)
q = q[["Partition", field2, "Number of Users", "Number of Jobs", "Q-Hours"]].reset_index(drop=True)
print(q, end="\n\n")
if latex:
  base = f"{host}_partition"
  q.to_csv(f"{base}.csv")
  fname = f"{base}.tex"
  caption = (f"Breakdown of jobs by partition on {caption_host} from {date_range}.",  f"{caption_host} -- Utilization by Partition")
  q.to_latex(fname, index=False, caption=caption, column_format="rrcrrc", label=f"{host}_partition")
  pad_multicolumn(fname, ["Number of Jobs", "Q-Hours", field2])

# O N D E M A N D
if (host in ondemand_hosts) and ("gpu" in df.partition.unique().tolist()):

  ans = df[df.jobname.str.contains("sys/dashboard", regex=False)].netid.unique().size
  print(f"Number of OnDemand users: {ans} ({round(100 * ans / df.netid.unique().size)}%)")

  df.jobname = df.jobname.apply(lambda x: x if "sys/dashboard" in x else "other")
  ondemand = df[["netid", "jobname", "cpu-hours", "gpu-hours", "q-hours", "gpu-job"]].copy()
  d = {"jobname":np.size, "gpu-job":np.sum, "cpu-hours":np.sum, "gpu-hours":np.sum, "q-hours":np.sum, "netid":lambda series: len(set(series))}
  ondemand = ondemand.groupby("jobname").agg(d)
  ondemand["Job Name"] = ondemand.index
  ondemand.columns = ["Total Jobs", "GPU Jobs", "CPU-Hours", "GPU-Hours", "Q-Hours", "Number of Users", "Job Name"]
  ondemand = ondemand.sort_values(by="CPU-Hours", ascending=False)
  ondemand.reset_index(drop=True, inplace=True)
  ondemand.index += 1
  ondemand = add_proportion_in_parenthesis(ondemand, "Total Jobs", replace=True)
  ondemand = add_proportion_in_parenthesis(ondemand, "CPU-Hours", replace=True)
  ondemand = add_proportion_in_parenthesis(ondemand, "GPU-Hours", replace=True)
  ondemand = add_proportion_in_parenthesis(ondemand, "GPU Jobs", replace=True)
  ondemand = add_proportion_in_parenthesis(ondemand, "Q-Hours", replace=True)
  ondemand = ondemand[["Job Name", "CPU-Hours", "Total Jobs", "Number of Users", "GPU-Hours", "GPU Jobs", "Q-Hours"]]
  print(ondemand, end="\n\n")
  if latex:
    base = f"{host}_ondemand"
    ondemand.to_csv(f"{base}.csv")
    fname = f"{base}.tex"
    chost = "Della (CPU and GPU)" if caption_host == "Della (CPU)" else caption_host
    chost = "Stellar (AMD and Intel)" if caption_host == "Stellar (AMD)" else caption_host
    caption = (f"Other versus OnDemand jobs on {chost} from {date_range}.",  f"{chost} -- Other versus OnDemand")
    ondemand.to_latex(fname, index=True, caption=caption, column_format="rlrrcrrr", label=f"{host}_ondemand")
    pad_multicolumn(fname, ["CPU-Hours", "GPU-Hours", "Q-Hours", "Total Jobs", "GPU Jobs"])
else:
  if host in ondemand_hosts:
    print("\nWARNING: Omitting OnDemand since gpu not in partitions.", end="\n\n")




# T O T A L S
print("Total CPU-Hours:", round(df["cpu-hours"].sum()))
print("Total CPU-Only-Hours:", round(df["cpu-only-hours"].sum()))
print("Total GPU-Hours:", round(df["gpu-hours"].sum()))
print("Total Q-Hours:",   round(df["q-hours"].sum()))
print("Total Jobs:", df.shape[0], "\n")

#sys.exit()



########################################################
#  N E T I D - S L U R M    A C C O U N T    P A I R S
########################################################
#df["cpu-eff"] = df.apply(lambda row: row["avecpu"] / row["elapsedraw"] if row["elapsedraw"] > 0 else -1, axis='columns')

def uniq_list(x):
  return ",".join(sorted(set(x)))

df["netid-account"] = df.apply(lambda row: row["netid"] + "@" + row["account"], axis='columns')
# needed when groupby account
pairs_account = df.groupby("netid-account").agg({"netid":"first", "account":"first", "netid-account":np.size, \
                                                 "partition":uniq_list, "cpu-hours":np.sum, \
                                                 "gpu-hours":np.sum, "gpu-job":np.sum, \
                                                 "q-hours":np.sum})
# needed for all other calculations
pairs = df.groupby("netid").agg({"netid":np.size, "account":uniq_list, \
                                         "partition":uniq_list, "cpu-hours":np.sum, \
                                         "gpu-hours":np.sum, "gpu-job":np.sum, \
                                         "q-hours":np.sum})

pairs.rename(columns={"netid":"Total Jobs"}, inplace=True)
pairs["netid"] = pairs.index

pairs["CPU Jobs"] = pairs.apply(lambda row: row["Total Jobs"] - row["gpu-job"], axis='columns')
pairs = pairs.sort_values(by="cpu-hours", ascending=False).reset_index(drop=True)
pairs.partition = pairs.partition.str.replace("datascience", "datasci", regex=False)
print(pairs)

# how many users have more than one Slurm account
accts = pairs_account["netid"].value_counts()
ans = accts[accts > 1].size
if ans > 0:
  print("\n\n")
  print(accts[accts > 1].sort_values(ascending=False))
print(f"\nThere are {ans} users with more than one Slurm account.\n\n")

#sys.exit()



####################
#  C H E C K G P U
####################

def call_checkgpu(start_date, end_date):
  year, month, day = start_date.split("T")[0].split("-")
  start_date_fmt = "/".join([month, day, year])
  year, month, day = end_date.split("T")[0].split("-")
  end_date_fmt = "/".join([month, day, year])

  cmd = f"/home/jdh4/bin/checkgpu -b {start_date_fmt} -e {end_date_fmt} -i"
  output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=600, text=True, check=True)
  lines = output.stdout.split('\n')
  if lines != [] and lines[-1] == "": lines = lines[:-1]
  cg = pd.DataFrame([line.split("|") for line in lines])
  cg.columns = cg.iloc[0]
  cg.drop(cg.index[0], inplace=True)
  return cg

if host in checkgpu_hosts:
  fname = f"{host}_checkgpu_{start_date}_{end_date}.csv".replace(":","-")
  if not os.path.exists(fname):
    print("Calling checkgpu (which may require several seconds) ... ", end="", flush=True)
    cg = call_checkgpu(start_date, end_date)
    cg.to_csv(fname, index=False)
    print("done.", flush=True)
  cg = pd.read_csv(fname)
  pairs = pairs.merge(cg, how="outer", left_on="netid", right_on="USER")

  # check for ghost users (may be explained by running near downtime or short jobs)
  # checkgpu will give the same value for users with multiple slurm accounts
  ans = pairs[pairs["MEAN(%)"].isna()].shape[0]
  print(f"\nNumber of users where MEAN(%) is NULL: {ans} of {pairs.netid.unique().size}")
  if ans:
    print(pairs[pairs["MEAN(%)"].isna()][["netid", "gpu-hours", "gpu-job", "MEAN(%)"]])
    print("")

  pairs.drop(columns=["USER", "STD(%)", "DEPT", "SPONSOR", "PROPORTION(%)", "POSITION"], inplace=True)
  pairs["MEAN(%)"] = pairs["MEAN(%)"].apply(lambda x: str(round(x)) if pd.notna(x) else "--")

#sys.exit()



####################################################
#  P O S I T I O N,    N A M E    A N D    D E P T
####################################################

fname = f"{host}_ldap_{start_date}_{end_date}.csv".replace(":","-")
if not os.path.exists(fname):
  print("Calling ldapsearch on each user (which may require several seconds) ... ", end="", flush=True)
  ld = pd.DataFrame(dossier.ldap_plus(pairs.netid.unique().tolist()))
  ld.columns = ld.iloc[0]
  ld = ld.drop(ld.index[0])
  ld.to_csv(fname, index=False)
  print("done.", flush=True)
ld = pd.read_csv(fname)
pairs = pairs.merge(ld, how="left", left_on="netid", right_on="NETID_TRUE")

def get_name_getent_passwd(netid):
  cmd = f"getent passwd {netid}"
  try:
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=5, text=True, check=True)
  except:
    return None
  else:
    line = output.stdout
    if line.count(":") == 6:
      # jdh4:*:150340:20121:Jonathan D. Halverson,CSES,Curtis W. Hillegas:/home/jdh4:/bin/bash
      # tshemma:x:127188:1035:Tali R. Shemma,,,:/home/tshemma:/bin/bash  # adroit
      fullname = line.split(":")[4]
      if fullname.endswith(",,,"):
        return fullname.replace(",,,", "")
      phone = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
      if fullname.count(",") == 3 and bool(phone.search(fullname)):
        # Michael A. Bino,114 Sherrerd Hall,609-258-8454,
        return fullname.split(",")[0]
      elif fullname.count(",") == 3 and fullname.endswith(",NONE,"):
        # Benjamin D Singer,Green Hall,NONE,
        return fullname.split(",")[0]
      elif fullname.count(",") == 2:
        return fullname.split(",")[0]
      else:
        print(f"WARNING: could not get name for {netid} ({fullname}).")
        return None
    else:
      print(f"NOTSIXCOLON for {netid} in get_name_getent_passwd()")
      return None

# overwrite value from dossier since a few are usually corrupted
pairs.NAME = pairs.netid.apply(get_name_getent_passwd)
null_names = pairs[pd.isna(pairs.NAME)][["netid", "NAME"]]
if not null_names.empty:
  print("\n")
  print("***Need to find name for these users:")
  print(null_names, end="\n\n")

if host in checkgpu_hosts:
  # GPU-HOURS comes from checkgpu; it can be compared to gpu-hours which comes from sacct
  pairs = pairs[["netid", "Total Jobs", "NAME", "partition", "account", "cpu-hours", \
                 "MEAN(%)", "GPU-HOURS", "gpu-hours", "gpu-job", "CPU Jobs", "q-hours", "DEPT", "POSITION"]]
else:
  pairs = pairs[["netid", "Total Jobs", "NAME", "partition", "account", "cpu-hours", \
                 "gpu-hours", "gpu-job", "CPU Jobs", "q-hours", "DEPT", "POSITION"]]

pairs.POSITION = pairs.POSITION.str.replace(' \(formerly G[0-9]\)', '', regex=True)  # RCU/DCU/Casual
pairs.POSITION = pairs.POSITION.str.replace('Alumni \(G[0-9]\)', 'Alumni', regex=True)
pairs.POSITION = pairs.POSITION.str.replace('Staff (visitor)', 'Staff', regex=False)
pairs.DEPT = pairs.DEPT.fillna("UNKNOWN")


##################################################
# M A N U E L    U P D A T E S
##################################################
if host == "tigergpu":
  #pairs.at[pairs[pairs.netid == "alvaros"].index[0], "NAME"] = "??lvaro Luna"
  #pairs.at[pairs[pairs.netid ==  "hzerze"].index[0], "NAME"] = "G??l Zerze"
  pass
if host == "tigercpu":
  #pairs.at[pairs[pairs.netid ==  "fj4172"].index[0], "NAME"] = "Farzaneh Jahanbakhshi"
  #pairs.at[pairs[pairs.netid == "yixiaoc"].index[0], "NAME"] = "Yixiao Chen"
  #pairs.at[pairs[pairs.netid ==   "alemay"].index[0], "NAME"] = "Am??lie Lemay"
  #pairs.at[pairs[pairs.netid == "ccaimapo"].index[0], "NAME"] = "Carlos Eduardo Hervias Caimapo"
  #pairs.at[pairs[pairs.netid ==   "hzerze"].index[0], "NAME"] = "G??l Zerze"
  #pairs.at[pairs[pairs.netid == "mathewm"].index[0], "NAME"] = "Mathew Syriac Madhavacheril"
  #pairs.at[pairs[pairs.netid ==  "grighi"].index[0], "NAME"] = "Giulia Righi"
  #pairs.at[pairs[pairs.netid ==  "grighi"].index[0], "POSITION"] = "G5"
  #pairs.at[pairs[pairs.netid ==    "brio"].index[0], "NAME"] = "Beatriz Gonzalez del Rio"
  #pairs.at[pairs[pairs.netid ==    "brio"].index[0], "POSITION"] = "Staff"
  pass
if host == "della":
  #pairs.DEPT = pairs.DEPT.str.replace('203 BOBST HALL', 'UNKNOWN', regex=False)
  #pairs.at[pairs[pairs.netid == "bgovil"].index[0], "DEPT"] = "UNKNOWN"
  pass
if host == "della-gpu":
  #pairs.at[pairs[pairs.netid == "yixiaoc"].index[0], "NAME"] = "Yixiao Chen"
  pass
if host == "adroit":
  #pairs.at[pairs[pairs.netid == "jw2918"].index[0], "DEPT"] = "PHYSICS"
  #pairs.at[pairs[pairs.netid == "jw2918"].index[0], "POSITION"] = "RCU"
  #pairs.at[pairs[pairs.netid ==  "pbisbal"].index[0], "NAME"] = "Prentice Bisbal"
  #pairs.at[pairs[pairs.netid == "efleisig"].index[0], "NAME"] = "Eve N. Fleisig"
  #pairs.at[pairs[pairs.netid == "danieleg"].index[0], "NAME"] = "Daniel E. Gitelman"
  pass
print(pairs)

#sys.exit()



#################
# S P O N S O R
#################

# The primary sponsor is obtained by calling:
# ldapsearch -x -H ldap://ldap1.rc.princeton.edu -b dc=rc,dc=princeton,dc=edu uid=gbwright manager
# In some cases the above produces multiple primary sponsors.
# The cluster-specific sponsor is obtained by:
# ldapsearch -x -H ldap://ldap1.rc.princeton.edu -b dc=rc,dc=princeton,dc=edu uid=gbwright description
# For users that have left, one can use R. Knight's CSV file to maybe find their sponsor
# Rules are then applied between these three sources to find the correct sponsor for each user
# Manual corrections can be made as the last step

def get_sponsors_getent_passwd(netid):
  """This method is not an official way to get the sponsor of a user. This is called only for a
     consistency check and can be ignored."""
  cmd = f"getent passwd {netid}"
  try:
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=5, text=True, check=True)
  except:
    return "--"
  else:
    line = output.stdout
    if line.count(":") == 6:
      # jdh4:*:150340:20121:Jonathan D. Halverson,CSES,Curtis W. Hillegas:/home/jdh4:/bin/bash
      sponsor = line.split(":")[4]
      if sponsor.count(",") == 2:
        return sponsor.split(",")[-1]
      else:
        print(f"WARNING: {netid} is possibly co-sponsored ({sponsor}). Added to netids_with_sponsor_trouble.")
        netids_with_sponsor_trouble.append(netid)
        return sponsor
    else:
      return f"NOTSIXCOLON for {netid} in get_sponsors_getent_passwd()"

def get_sponsors_cses_ldap(netid):
  # ldapsearch -x -H ldap://ldap1.rc.princeton.edu -b dc=rc,dc=princeton,dc=edu uid=jdh4 manager
  # manager: uid=curt,cn=users,cn=accounts,dc=rc,dc=princeton,dc=edu

  # ldapsearch -x -H ldap://ldap1.rc.princeton.edu -b dc=rc,dc=princeton,dc=edu uid=yixiaoc manager
  # manager: uid=rcar,cn=users,cn=accounts,dc=rc,dc=princeton,dc=edu
  # manager: uid=weinan,cn=users,cn=accounts,dc=rc,dc=princeton,dc=edu

  cmd = f"ldapsearch -x -H ldap://ldap1.rc.princeton.edu -b dc=rc,dc=princeton,dc=edu uid={netid} manager"
  output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=5, text=True, check=True)
  lines = output.stdout.split('\n')
  if lines != [] and lines[-1] == "": lines = lines[:-1]
  managers = []
  for line in lines:
    if "manager: " in line and "uid=" in line:
      managers.append(line.split("uid=")[1].split(",")[0])
  if len(managers) > 1:
    netids_with_sponsor_trouble.append(netid)
  if managers == []:
    print(f"No primary sponsor found for {netid} in get_sponsors_cses_ldap()")
    return None
  else:
    return ",".join(managers)
    
def get_sponsors_from_description(netid):
  # ldapsearch -x -H ldap://ldap1.rc.princeton.edu -b dc=rc,dc=princeton,dc=edu uid=npetsev description
  # description: della:pdebene=dGlja2V0IDI5NjYx,perseus:pdebene=dGlja2V0IDI5NjYx,s
  #  tellar:pdebene=bm9uZQ==,tiger:pdebene=dGlja2V0IDI5NjYx,tigress:pdebene=bm9uZQ
  #
  # ldapsearch -x -H ldap://ldap1.rc.princeton.edu -b dc=rc,dc=princeton,dc=edu uid=gbwright description
  # description: della:curt=R2FycmV0dCBpcyBhIGRldmVsb3BlciBpbiBSQw==,tiger:wtang=M
  # jc5MjQ=,tigress:curt=R2FycmV0dCBpcyBhIGRldmVsb3BlciBpbiBSQw==,traverse:curt=M
  # jg1MTg=
  #
  # ldapsearch -x -H ldap://ldap1.rc.princeton.edu -b dc=rc,dc=princeton,dc=edu uid=ksabsay
  cmd = f"ldapsearch -x -H ldap://ldap1.rc.princeton.edu -b dc=rc,dc=princeton,dc=edu uid={netid} description"
  output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=5, text=True, check=True)
  lines = output.stdout.split('\n')
  if lines != [] and lines[-1] == "": lines = lines[:-1]
  s = ""
  for line in lines:
    s += line.strip() if not line.startswith("#") else ""
  cluster = f"{host.replace('cpu', '').replace('gpu', '').replace('-gpu', '').replace('-intel', '').replace('-amd', '')}:"
  if cluster in s:
    snetid = s.split(cluster)[1].split("=")[0]
    return snetid if not "(" in snetid else snetid.split("(")[0].strip()
  else:
    return None
  
def format_sponsor(s):
  if (not s) or pd.isna(s): return s
  names = list(filter(lambda x: x not in ['Jr.', 'II', 'III', 'IV'], s.split()))
  if len(names) == 0:
    return None
  elif len(names) == 1:
    return names[0]
  else:
    return f"{names[0][0]}. {names[-1]}"

def primary_vs_clusterspecific_vs_rk(user_netid, primary, specific, rk):
  if pd.notnull(primary) and pd.notnull(specific) and (primary != specific):
    print(f"{user_netid} added to trouble since primary ({primary}) != specific ({specific})")
    netids_with_sponsor_trouble.append(user_netid)
    return specific
  elif pd.isna(primary) and pd.isna(specific) and pd.notnull(rk):
    return rk
  elif pd.isna(primary) and pd.notnull(specific) and pd.notnull(rk):
    print(f"{user_netid} added to trouble since primary is null and specific ({specific}) is not")
    netids_with_sponsor_trouble.append(user_netid)
    return specific
  else:
    return primary

if host != "adroit":
  # global list
  netids_with_sponsor_trouble = []

  # dataframe of users that left university
  fname = "users_left_university_from_robert_knight.csv"
  if os.path.exists(fname):
    rk = pd.read_csv(fname)
    rk.info()
  else:
    rk = pd.DataFrame({"Netid_":pairs.netid, "Sponsor_Name_":pairs.shape[0] * [np.nan], \
                       "Sponsor_Netid_":pairs.shape[0] * [np.nan]})
  print("\n\nFirst few lines of rk:\n")
  print(rk.head())
  pairs = pairs.merge(rk, how="left", left_on="netid", right_on="Netid_")
  
  print("\n")
  print("Running getent passwd on each user ... ", flush=True, end="")
  pairs["sponsor-getent"] = pairs.netid.apply(get_sponsors_getent_passwd)
  print("done.", flush=True)

  print("Running ldapsearch (manager) on each user ... ", flush=True, end="")
  pairs["sponsor-ldap"] = pairs.netid.apply(get_sponsors_cses_ldap)
  print("done.", flush=True)

  print("Running ldapsearch (description) on each user ... ", flush=True, end="")
  pairs["sponsor-desc"] = pairs.netid.apply(get_sponsors_from_description)
  print("done.", flush=True)

  print("Comparing the three possible values for sponsor netid ... ", flush=True, end="")
  pairs["sponsor-best"] = pairs.apply(lambda row: primary_vs_clusterspecific_vs_rk(row["netid"], \
                                      row["sponsor-ldap"], row["sponsor-desc"], row["Sponsor_Netid_"]), axis='columns')
  print("done.", flush=True, end="\n\n")

  print("\n\nUsers with no sponsor (BEFORE manual corrections):\n")
  print(pairs[pd.isna(pairs["sponsor-best"])][["netid", "NAME"]], end="\n\n")

  # M A N U A L    C O R R E C T I O N S    O N   S P O N S O R    N E T I D
  if host == "tigercpu" or host == "tigergpu":
    pairs["sponsor-getent"] = pairs["sponsor-getent"].str.replace("Yixiao Chen,Chemistry,Roberto Car,Weinan E", "Roberto Car", regex=False)
    pairs["sponsor-getent"] = pairs["sponsor-getent"].str.replace("Shirkey, Jaden D.,MolBio,Nieng Yan", "Nieng Yan", regex=False)
    pairs["sponsor-getent"] = pairs["sponsor-getent"].str.replace("Mathew Syriac Madhavacheril,Astro,Robert H. Lupton,Jo Dunkley", "Robert H. Lupton", regex=False)
    #pairs.at[pairs[pairs.netid ==  "yixiaoc"].index[0], "sponsor-best"] = "rcar"
    #pairs.at[pairs[pairs.netid ==  "mathewm"].index[0], "sponsor-best"] = "rhl"
  if host == "tigercpu":
    # mistake with jennyg -> description: tiger:USER=MzA5NDE=
    #pairs.at[pairs[pairs.netid == "jennyg"].index[0], "sponsor-best"] = "jennyg"
    #pairs.at[pairs[pairs.netid == "fj4172"].index[0], "sponsor-best"] = "E. Carter"
    pass
  if host == "tigergpu":
    #pairs.at[pairs[pairs.netid ==  "ksabsay"].index[0], "sponsor-best"] = "W. Bialek"
    pass
  if host == "traverse":
    pass
  if host == "della":
    #pairs.at[pairs[pairs.netid ==      "vbb"].index[0], "sponsor-best"] = "Vir B. Bulchandani"
    #pairs.at[pairs[pairs.netid == "ecmorale"].index[0], "sponsor-best"] = "Eduardo Morales"
    #pairs.at[pairs[pairs.netid == "eivshina"].index[0], "sponsor-best"] = "Joshua N. Winn"
    #pairs.at[pairs[pairs.netid == "pekalski"].index[0], "sponsor-best"] = "William M. Jacobs"
    pass
  if host == "della-gpu":
    pairs["sponsor-getent"] = pairs["sponsor-getent"].str.replace("Yixiao Chen,Chemistry,Roberto Car,Weinan E", "Roberto Car", regex=False)
    #pairs.at[pairs[pairs.netid ==  "yixiaoc"].index[0], "sponsor-best"] = "R. Car"
  if host == "stellar-intel":
    #pairs.at[pairs[pairs.netid == "nlinzer"].index[0], "sponsor-best"] = "Eliot Quataert"
    #pairs.at[pairs[pairs.netid ==    "ws17"].index[0], "sponsor-best"] = "M. Cohen"
    pass

  print("\n\nUsers with no sponsor (AFTER manual corrections):\n")
  print(pairs[pd.isna(pairs["sponsor-best"])][["netid", "NAME"]], end="\n\n")

  print("Running getent passwd on best sponsor ... ", flush=True, end="")
  pairs["sponsor-name"] = pairs["sponsor-best"].apply(get_name_getent_passwd)
  print("done.", flush=True, end="\n\n")
  pairs["sponsor-name"] = pairs["sponsor-name"].fillna("UNKNOWN")

  # M A N U A L    C O R R E C T I O N S    O N   S P O N S O R    N A M E
  if host == "tigercpu":
    pairs.at[pairs[pairs.netid == "meerag"].index[0], "sponsor-name"] = "Martin Helmut Wuhr"
  
  pairs["sponsor-getent"] = pairs["sponsor-getent"].apply(format_sponsor)
  pairs["sponsor-name"]   = pairs["sponsor-name"].apply(format_sponsor)

  #print(pairs[["netid", "sponsor-ldap", "sponsor-desc", "Sponsor_Netid_", "sponsor-best", "sponsor-name", "sponsor-getent", "Sponsor_Name_"]])
  print("\nUsers with sponsor trouble:")
  print(netids_with_sponsor_trouble)
  pairs["sponsor-trouble"] = pairs.netid.apply(lambda x: 1 if x in netids_with_sponsor_trouble else None)

  # sponsor-getent is untrusted since it is obtained using the netid of the user instead directly using the netid of the sponsor
  mismatch = pairs[pairs["sponsor-getent"] != pairs["sponsor-name"]][["netid", "sponsor-getent", "sponsor-name", "Sponsor_Netid_", "Sponsor_Name_"]]
  if not mismatch.empty:
    print("\n\nsponsor-getent does not match sponsor-name:")
    print(mismatch, end="\n\n")

  print(pairs, end="\n\n")
  #print(pairs[["netid", "sponsor-getent", "sponsor-ldap", "sponsor-desc", "Sponsor_Netid_", "sponsor-name", "Sponsor_Name_", "sponsor-trouble"]])

  print("***Examine the sponsor columns above for consistency. This must be done manually.***")

#sys.exit()




########################
# F O R M A T T I N G
########################

def shorten(name):
  if len(name) > 18:
    first, last = name.split()
    return f"{first[0]}. {last}"
  else:
    return name

def format_user(s):
  if not s: return s
  names = list(filter(lambda x: x not in ['Jr.', 'II', 'III', 'IV'], s.split()))
  if len(names) == 1:
    return names[0]
  else:
    return shorten(f"{names[0]} {names[-1]}")




##################################
# C O L U M N    R E N A M I N G
##################################

if host in checkgpu_hosts:
  pairs = pairs[["netid", "NAME", "POSITION", "DEPT", "sponsor-ldap", "sponsor-getent", "sponsor-desc", "sponsor-name", \
                 "sponsor-trouble", "partition", "account", "cpu-hours", "gpu-hours", "MEAN(%)", \
                 "gpu-job", "CPU Jobs", "Total Jobs", "q-hours"]]
  pairs = pairs.sort_values("gpu-hours", ascending=False).reset_index(drop=True)
elif (host == "stellar-amd" and partition == "--partition gpu"):
  pairs = pairs[["netid", "NAME", "POSITION", "DEPT", "partition", "account", \
                 "sponsor-ldap", "sponsor-getent", "sponsor-desc", "sponsor-name", "sponsor-trouble", \
                 "cpu-hours", "gpu-hours", "gpu-job", "CPU Jobs", "Total Jobs", "q-hours"]]
  pairs = pairs.sort_values("gpu-hours", ascending=False).reset_index(drop=True)
else:
  if host == "adroit":
    pairs = pairs[["netid", "NAME", "POSITION", "DEPT", "partition", "account", \
                   "cpu-hours", "gpu-hours", "gpu-job", "CPU Jobs", "Total Jobs", "q-hours"]]
  else:
    pairs = pairs[["netid", "NAME", "POSITION", "DEPT", "sponsor-getent", "sponsor-trouble", "sponsor-name", \
                   "partition", "account", \
                   "cpu-hours", "gpu-hours", "gpu-job", "CPU Jobs", "Total Jobs", "q-hours"]]
  pairs = pairs.sort_values("cpu-hours", ascending=False).reset_index(drop=True)

pairs = add_proportion_in_parenthesis(pairs, "gpu-job", replace=False)
pairs = add_proportion_in_parenthesis(pairs, "gpu-hours", replace=False)
pairs = add_proportion_in_parenthesis(pairs, "Total Jobs", replace=True)
pairs = add_proportion_in_parenthesis(pairs, "CPU Jobs", replace=True)
pairs = add_proportion_in_parenthesis(pairs, "cpu-hours", replace=False)
pairs = add_proportion_in_parenthesis(pairs, "q-hours", replace=False)

pairs = pairs.rename(columns={"sponsor-name":"Sponsor"})
pairs.NAME = pairs.NAME.apply(format_user)
pairs.index += 1
print("")
print(pairs.head(20), end="\n")
print("")
pairs.info()
print("")

#sys.exit()




########################
# U S E R    T A B L E
########################

users = pairs.rename(columns={"NAME": "Name", "netid": "NetID", "POSITION":"Position", \
                              "DEPT":"User Dept", "MEAN(%)":"UTIL(%)", \
                              "partition":"Partitions", "account":"Account", \
                              "cpu-hours-cmb":"CPU-Hours", "gpu-hours-cmb":"GPU-Hours", \
                              "gpu-job-cmb":"GPU Jobs", \
                              "q-hours-cmb":"Q-Hours"}).copy()

users["Account"] = users.Account.apply(lambda x: x.upper())
users["Account"] = users.Account.str.replace("PSYCHOLOGY", "PSYCH.", regex=False)
if host != "adroit":
  users["Sponsor"] = users.Sponsor.str.replace("Panagiotopoulos", "Panagiotop.", regex=False)
if host != "adroit":
  users["NetID"] = users.apply(lambda row: row["NetID"] if pd.isna(row["sponsor-trouble"]) else row["NetID"] + "*", axis='columns')

if host in checkgpu_hosts:
  if host == "traverse":
    users = users[["NetID", "Name", "Position", "Sponsor", "Account", "GPU-Hours", "UTIL(%)", "GPU Jobs", "Total Jobs", "Q-Hours"]]
  else:
    users = users[["NetID", "Name", "Position", "User Dept", "Sponsor", "Partitions", "Account", "GPU-Hours", "UTIL(%)", "GPU Jobs", "Q-Hours"]]
  print(users)
  if latex:
    base = f"{host}_users"
    users.to_csv(f"{base}.csv")
    fname = f"{base}.tex"
    caption = (f"GPU utilization by user on {caption_host} from {date_range}. *User with non-trivial sponsorship.",  f"{caption_host} -- Utilization by User")
    include_index = True
    cf = (users.shape[1] + include_index) * "r"
    users.to_latex(fname, index=True, caption=caption, column_format=cf, label=f"{host}_users", longtable=True)
    if host == "traverse":
      pad_multicolumn(fname, ["GPU Jobs", "GPU-Hours", "Total Jobs", "Q-Hours"])
    else:
      pad_multicolumn(fname, ["GPU Jobs", "GPU-Hours", "Q-Hours"])
else:
  if host == "adroit":
    users = users[["NetID", "Name", "Position", "User Dept", "Partitions", "Account", "CPU-Hours", "Total Jobs", "GPU-Hours", "GPU Jobs", "Q-Hours"]]
  elif (host == "stellar-amd" and partition == "--partition gpu"):
    users = users[["NetID", "Name", "Position", "User Dept", "Partitions", "Account", "GPU-Hours", "GPU Jobs", "Q-Hours"]]
  elif (host == "stellar-amd" and partition == "--partition bigmem,cimes"):
    users = users[["NetID", "Name", "Position", "Sponsor", "Partitions", "Account", "CPU-Hours", "Total Jobs", "Q-Hours"]]
  elif host == "test":
    cols = ["NetID", "Name", "Position", "User Dept", "Partitions", "Sponsor", "Account", "CPU-Hours", "Total Jobs", "GPU-Hours", "GPU Jobs", "Q-Hours"]
    users = users[cols]
    answer = []
    answer.append((  "curt",    "Curtis Hillegas", "Staff",    "CSES",     "cpu", "C. Hillegas", "CSES", "278 (44.8%)", "1 (16.7%)",   "0 (0.0%)",  "0 (0.0%)",  "9 (14.2%)"))
    answer.append(("yz8614",       "Yuxuan Zhang",    "G1",      "CS", "cpu,gpu",    "F. Heide",   "CS", "198 (31.8%)", "2 (33.3%)",  "3 (15.6%)", "1 (33.3%)", "31 (52.2%)"))
    answer.append((  "jdh4", "Jonathan Halverson", "Staff", "PICSciE", "cpu,gpu", "C. Hillegas", "CSES", "145 (23.4%)", "3 (50.0%)", "17 (84.4%)", "2 (66.7%)", "20 (33.5%)"))
    answer = pd.DataFrame(answer)
    answer.columns = cols
    answer.index += 1
    print(users)
    print(answer)
    print("\n***Success***\n\n") if users.equals(answer) else print("Fail")
  else:
    users = users[["NetID", "Name", "Position", "Sponsor", "Partitions", "Account", "CPU-Hours", "Total Jobs", "Q-Hours"]]
  print(users)
  if latex:
    base = f"{host}_users"
    users.to_csv(f"{base}.csv")
    fname = f"{base}.tex"
    if host == "adroit":
      caption = (f"CPU and GPU utilization by user on {caption_host} from {date_range}.",  f"{caption_host} -- Utilization by User")
    elif (host == "stellar-amd" and partition == "--partition gpu"):
      caption = (f"GPU utilization by user on {caption_host} from {date_range}.",  f"{caption_host} -- Utilization by User")
    else:
      caption = (f"CPU utilization by user on {caption_host} from {date_range}. *User with non-trivial sponsorship.",  f"{caption_host} -- Utilization by User")
    include_index = True
    cf = (users.shape[1] + include_index) * "r"
    label = f"{host}_users" if not (host == "stellar-amd" and partition == "--partition gpu") else f"{host}_gpu_users"
    users.to_latex(fname, index=True, caption=caption, column_format=cf, label=label, longtable=True)
    if host == "adroit":
      pad_multicolumn(fname, ["GPU-Hours", "CPU-Hours", "Q-Hours", "Total Jobs", "GPU Jobs"])
    elif (host == "stellar-amd" and partition == "--partition gpu"):
      pad_multicolumn(fname, ["GPU-Hours", "Q-Hours", "GPU Jobs"])
    else:
      pad_multicolumn(fname, ["Total Jobs", "CPU-Hours", "Q-Hours"])

#sys.exit()





#####################################################
#  G R O U P B Y 
# return to pairs dataframe, ignore users dataframe
#####################################################

if host != "adroit":
    # S P O N S O R
    d = {"cpu-hours":[np.size, np.sum], "account":uniq_list, "gpu-hours":np.sum, "q-hours":np.sum}
    by_sponsor = pairs[["Sponsor", "account", "cpu-hours", "gpu-hours", "q-hours"]].groupby("Sponsor").agg(d).copy()
    by_sponsor.columns = ["Number of Users", "CPU-Hours", "Slurm Accounts", "GPU-Hours", "Q-Hours"]
    by_sponsor["Number of Users"] = by_sponsor["Number of Users"].astype("int32")
    if host in checkgpu_hosts:
      by_sponsor = by_sponsor.sort_values("GPU-Hours", ascending=False).reset_index(drop=False)
    else:
      by_sponsor = by_sponsor.sort_values("CPU-Hours", ascending=False).reset_index(drop=False)
    by_sponsor.index += 1
    by_sponsor = add_proportion_in_parenthesis(by_sponsor, "Number of Users", replace=True)
    by_sponsor = add_proportion_in_parenthesis(by_sponsor, "CPU-Hours", replace=True)
    by_sponsor = add_proportion_in_parenthesis(by_sponsor, "GPU-Hours", replace=True)
    by_sponsor = add_proportion_in_parenthesis(by_sponsor, "Q-Hours", replace=True)
    by_sponsor["Slurm Accounts"] = by_sponsor["Slurm Accounts"].apply(lambda x: x.upper())
    by_sponsor["Slurm Accounts"] = by_sponsor["Slurm Accounts"].apply(lambda x: ",".join(sorted(set(x.split(",")))))
    if host in checkgpu_hosts:
      by_sponsor = by_sponsor[["Sponsor", "Slurm Accounts", "GPU-Hours", "Number of Users", "Q-Hours"]]
      print(by_sponsor)
      if latex:
        base = f"{host}_sponsor"
        by_sponsor.to_csv(f"{base}.csv")
        fname = f"{base}.tex"
        caption = (f"GPU utilization by sponsor on {caption_host} from {date_range}.",  f"{caption_host} -- Utilization by Sponsor")
        by_sponsor.to_latex(fname, index=True, caption=caption, column_format="rrrrrr", label=f"{host}_sponsor", longtable=True)
        pad_multicolumn(fname, ["Number of Users", "GPU-Hours", "Q-Hours"])
    else:
      by_sponsor = by_sponsor[["Sponsor", "Slurm Accounts", "CPU-Hours", "Number of Users", "Q-Hours"]]
      print(by_sponsor)
      if latex:
        base = f"{host}_sponsor"
        by_sponsor.to_csv(f"{base}.csv")
        fname = f"{base}.tex"
        caption = (f"CPU utilization by sponsor on {caption_host} from {date_range}.",  f"{caption_host} -- Utilization by Sponsor")
        by_sponsor.to_latex(fname, index=True, caption=caption, column_format="rrrrrr", label=f"{host}_sponsor", longtable=True)
        pad_multicolumn(fname, ["Number of Users", "CPU-Hours", "Q-Hours"])

# A C C O U N T
d = {"cpu-hours":[np.size, np.sum], "gpu-hours":np.sum, "q-hours":np.sum}
by_account = pairs_account[["account", "cpu-hours", "gpu-hours", "q-hours"]].groupby("account").agg(d).copy()
by_account["Slurm Account"] = by_account.index
by_account.columns = ["Number of Users", "CPU-Hours", "GPU-Hours", "Q-Hours", "Slurm Account"]
by_account["Number of Users"] = by_account["Number of Users"].astype("int32")
if host in checkgpu_hosts:
  by_account = by_account.sort_values("GPU-Hours", ascending=False).reset_index(drop=False)
else:
  by_account = by_account.sort_values("CPU-Hours", ascending=False).reset_index(drop=False)
by_account.index += 1
by_account = add_proportion_in_parenthesis(by_account, "Number of Users", replace=True)
by_account = add_proportion_in_parenthesis(by_account, "CPU-Hours", replace=True)
by_account = add_proportion_in_parenthesis(by_account, "GPU-Hours", replace=True)
by_account = add_proportion_in_parenthesis(by_account, "Q-Hours", replace=True)
by_account["Slurm Account"] = by_account["Slurm Account"].apply(lambda x: x.upper())
if host in checkgpu_hosts:
  by_account = by_account[["Slurm Account", "GPU-Hours", "Number of Users", "Q-Hours"]]
  print(by_account)
  if latex:
    base = f"{host}_account"
    by_account.to_csv(f"{base}.csv")
    fname = f"{base}.tex"
    caption = (f"GPU utilization by account on {caption_host} from {date_range}.",  f"{caption_host} -- Utilization by Slurm Account")
    by_account.to_latex(fname, index=True, caption=caption, column_format="rrrrr", label=f"{host}_account")
    pad_multicolumn(fname, ["Number of Users", "GPU-Hours", "Q-Hours"])
else:
  if host == "adroit":
    by_account = by_account[["Slurm Account", "CPU-Hours", "Number of Users", "GPU-Hours", "Q-Hours"]]
  else:
    by_account = by_account[["Slurm Account", "CPU-Hours", "Number of Users", "Q-Hours"]]
  print(by_account)
  if latex:
    base = f"{host}_account"
    by_account.to_csv(f"{base}.csv")
    fname = f"{base}.tex"
    caption = (f"CPU utilization by account on {caption_host} from {date_range}.",  f"{caption_host} -- Utilization by Slurm Account")
    include_index = True
    cf = (by_account.shape[1] + include_index) * "r"
    by_account.to_latex(fname, index=include_index, caption=caption, column_format=cf, label=f"{host}_account")
    if host == "adroit":
      pad_multicolumn(fname, ["Number of Users", "CPU-Hours", "GPU-Hours", "Q-Hours"])
    else:
      pad_multicolumn(fname, ["Number of Users", "CPU-Hours", "Q-Hours"])

# P O S I T I O N
pos = pairs.copy()
pos.POSITION = pos.POSITION.str.replace('G[0-9]', 'Graduate', regex=True)
pos.POSITION = pos.POSITION.str.replace('Alumni', 'Graduate', regex=False)
pos.POSITION = pos.POSITION.str.replace('U[0-9][0-9][0-9][0-9]', 'Undergraduate', regex=True)
pos.POSITION = pos.POSITION.str.replace('XStaff', 'Staff', regex=False)
pos.POSITION = pos.POSITION.str.replace('XDCU', 'DCU', regex=False)
pos.POSITION = pos.POSITION.str.replace('DCU|RCU|RU|XMiscAffil', 'DCU, RCU, RU, XMiscAffil', regex=True)

d = {"cpu-hours":[np.size, np.sum], "gpu-hours":np.sum, "q-hours":np.sum}
by_position = pos[["POSITION", "cpu-hours", "gpu-hours", "q-hours"]].groupby("POSITION").agg(d)
by_position["Position"] = by_position.index
by_position.columns = ["Number of Users", "CPU-Hours", "GPU-Hours", "Q-Hours", "Position"]
by_position["Number of Users"] = by_position["Number of Users"].astype("int32")
if host in checkgpu_hosts:
  by_position = by_position.sort_values("GPU-Hours", ascending=False).reset_index(drop=False)
else:
  by_position = by_position.sort_values("CPU-Hours", ascending=False).reset_index(drop=False)
by_position.index += 1
by_position = add_proportion_in_parenthesis(by_position, "Number of Users", replace=True)
by_position = add_proportion_in_parenthesis(by_position, "CPU-Hours", replace=True)
by_position = add_proportion_in_parenthesis(by_position, "GPU-Hours", replace=True)
by_position = add_proportion_in_parenthesis(by_position, "Q-Hours", replace=True)
if host in checkgpu_hosts:
  by_position = by_position[["Position", "GPU-Hours", "Number of Users", "Q-Hours"]]
  print(by_position)
  if latex:
    base = f"{host}_position"
    by_position.to_csv(f"{base}.csv")
    fname = f"{base}.tex"
    caption = (f"GPU utilization by position on {caption_host} from {date_range}.", f"{caption_host} -- Utilization by Position")
    by_position.to_latex(fname, index=True, caption=caption, column_format="rrrrr", label=f"{host}_position")
    pad_multicolumn(fname, ["GPU-Hours", "Number of Users", "Q-Hours"])
else:
  if host == "adroit":
    by_position = by_position[["Position", "CPU-Hours", "Number of Users", "GPU-Hours", "Q-Hours"]]
  else:
    by_position = by_position[["Position", "CPU-Hours", "Number of Users", "Q-Hours"]]
  print(by_position)
  if latex:
    base = f"{host}_position"
    by_position.to_csv(f"{base}.csv")
    fname = f"{base}.tex"
    caption = (f"CPU utilization by position on {caption_host} from {date_range}.", f"{caption_host} -- Utilization by Position")
    include_index = True
    cf = (by_account.shape[1] + include_index) * "r"
    by_position.to_latex(fname, index=include_index, caption=caption, column_format=cf, label=f"{host}_position")
    if host == "adroit":
      pad_multicolumn(fname, ["CPU-Hours", "Number of Users", "GPU-Hours", "Q-Hours"])
    else:
      pad_multicolumn(fname, ["CPU-Hours", "Number of Users", "Q-Hours"])
