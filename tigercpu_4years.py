import numpy  as np
import pandas as pd

import subprocess
import re

from sponsor import sponsor_full_name
from sponsor import sponsor_per_cluster
from sponsor import get_full_name_of_user

# pandas display settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

def get_missing_name_from_log(netid):
  with open("tigress_user_changes.log", "r") as f:
      data = f.readlines()
  ct = 0

  pattern = f" {netid} "
  logname = ""
  for line in data:
      if pattern in line:
          ct += 1
          if f" Added user {netid} (" in line:
              logname = line.split(f" Added user {netid} (")[-1].split(")")[0].split(" - ")[-1]
  return logname

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

#2019-06-19 11:52:33 - Added user tzajdel (130474 - Thomas J. Zajdel), group mae (30007 - MAE) with sponsor djctwo (Daniel J. Cohen).
#2019-06-19 11:52:34 - Added user tzajdel to cluster tiger
#2019-06-19 11:52:43 - Added user tzajdel to cluster tigress
#2022-08-04 10:38:10 - Removed user tzajdel (Thomas J. Zajdel), uid=130474; sponsor djctwo; main group mae; no extra groups; clusters: tiger, tigress

def get_missing_sponsor(netid):
  #breakpoint()
  with open("tigress_user_changes.log", "r") as f:
      data = f.readlines()
  ct = 0
  pattern = f" {netid} "
  sponsor = "NOTHING"
  for line in data:
      if pattern in line:
          ct += 1
          if f" Added user {netid} " in line and " with sponsor " in line:
              sponsor = line.split(" with sponsor ")[-1].split()[0]
          if f" Removed user {netid} " in line and " sponsor " in line:
              sponsor = line.split(" sponsor ")[-1].split(";")[0]
  print(netid, sponsor, ct)
  return sponsor         

def format_sponsor(s):
  if (not s) or pd.isna(s): return s
  names = list(filter(lambda x: x not in ['Jr.', 'II', 'III', 'IV'], s.split()))
  if len(names) == 0:
    return None
  elif len(names) == 1:
    return names[0]
  else:
    return f"{names[0][0]}. {names[-1]}"

def get_name_getent_passwd(netid):
  cmd = f"getent passwd {netid}"
  try:
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, timeout=5, text=True, check=True)
  except:
    print(f"getent passwd return nothing for {netid} trying logfile")
    fromlog = get_missing_name_from_log(netid)
    if fromlog != "":
      print(fromlog)
      return fromlog
    else:
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

#sp = df[["account", "netid"]].drop_duplicates().copy()
#sp["sponsor-dict"] = sp.netid.apply(lambda netid: sponsor_per_cluster(netid, verbose=True)["tiger"])
#sp.to_csv("sponsors.csv")

sp = pd.read_csv("sponsors.csv")
sp.info()
sp = sp.fillna("UNKNOWN")
sp["sponsor-dict"] = sp.apply(lambda row: row["sponsor-dict"] if row["sponsor-dict"] != "UNKNOWN" else get_missing_sponsor(row["netid"]), axis='columns')
sp.info()
for i, net in enumerate(sorted(sp[sp["sponsor-dict"] == "UNKNOWN"].netid.unique())):
  print(i+1, net)

# sacct -M tiger2 -a -X -P -n -S 2019-02-01T00:00:00 -E 2023-01-31T23:59:59 -o user,account,cputimeraw --partition=cpu,ext,serial > raw.csv
df = pd.read_csv("raw.csv", sep="|")
df = df.rename(columns={"user":"netid", "cputimeraw":"cpu-seconds"})
df.info()
print("\nTotal NaNs:", df.isnull().sum().sum(), "\n")
print(df.columns)
print(df.shape)
print(df.account.unique())

#### ACCOUNT ####
if 1:
  ac = df.groupby("account").agg({"cpu-seconds":np.sum, "netid":lambda series: len(set(series)), "account":np.size}).rename(columns={"account":"Jobs", "netid":"Number of Users"})
  ac = ac.reset_index(drop=False)
  ac = ac[["account", "cpu-seconds", "Number of Users", "Jobs"]].sort_values("cpu-seconds", ascending=False)
  ac["CPU-Hours"] = ac["cpu-seconds"].apply(lambda x: round(x/3600.0))
  ac = ac.rename(columns={"account":"Account"})
  ac["Slurm Account"] = ac.Account.apply(lambda x: x.upper())
  ac = ac[["Slurm Account", "CPU-Hours", "Number of Users"]].reset_index(drop=True)
  ac.index += 1
  print(ac)
  caption = (f"CPU utilization by Slurm account on TigerCPU from February 1, 2019 to January 31, 2023.", f"TigerCPU -- Utilization by Slurm Account")
  print("CPU-Hours=", ac["CPU-Hours"].sum())
  ac = add_proportion_in_parenthesis(ac, "Number of Users", replace=True)
  ac = add_proportion_in_parenthesis(ac, "CPU-Hours", replace=True)
  fname = "tigercpu_account_feb1_2019_jan31_2023.tex"
  ac.to_latex(fname, index=True, caption=caption, column_format="rrrr", label=f"tigercpu_account_feb1_2019_jan31_2023", longtable=False)
  pad_multicolumn(fname, ["Number of Users", "CPU-Hours"])

### SPONSOR ###
s = df[["netid", "cpu-seconds", "account"]].copy()
s = s.merge(sp, on="netid", how="left")
s = s.groupby("sponsor-dict").agg({"cpu-seconds":np.sum, "netid":lambda series: len(set(series)), "account":lambda series: ",".join(sorted(set(series)))}).rename(columns={"sponsor-dict":"Jobs", "netid":"Number of Users"})
s["CPU-Hours"] = s["cpu-seconds"].apply(lambda x: round(x/3600))
s["Slurm Accounts"] = s.account.apply(lambda x: x.upper())
s = s.reset_index(drop=False)
s = s.rename(columns={"sponsor-dict":"Sponsor"})
s = s[["Sponsor", "Slurm Accounts", "CPU-Hours", "Number of Users"]].sort_values("CPU-Hours", ascending=False)
s = s.reset_index(drop=True)
s.index += 1
s["SName"] = s.Sponsor.apply(get_name_getent_passwd)
s["SName"] = s.SName.str.replace("Panagiotopoulos", "Panagiotop.", regex=False)
s["SName"] = s.SName.apply(format_sponsor)
s = s.drop(columns=["Sponsor"])
s = s.rename(columns={"SName":"Sponsor"})
s = s[["Sponsor", "Slurm Accounts", "CPU-Hours", "Number of Users"]]
s = add_proportion_in_parenthesis(s, "Number of Users", replace=True)
s = add_proportion_in_parenthesis(s, "CPU-Hours", replace=True)
fname = "tigercpu_sponsor_feb1_2019_jan31_2023.tex"
caption = (f"CPU utilization by sponsor on TigerCPU from February 1, 2019 to January 31, 2023.", f"TigerCPU -- Utilization by Sponsor")
s.to_latex(fname, index=True, caption=caption, column_format="rrrrr", label=f"tigercpu_sponsor_feb1_2019_jan31_2023", longtable=True)
pad_multicolumn(fname, ["Number of Users", "CPU-Hours"])


if 1:
  ### users ###
  users = df.copy()
  users = users.merge(sp, on="netid", how="left")
  users = users.groupby("netid").agg({"cpu-seconds":np.sum, "sponsor-dict":min, "netid":np.size}).rename(columns={"netid":"Total Jobs", "sponsor-dict":"Sponsor"})
  users["CPU-Hours"] = users["cpu-seconds"].apply(lambda x: round(x/3600.0))
  users["SName"] = users.Sponsor.apply(get_name_getent_passwd)
  users["SName"] = users.SName.str.replace("Panagiotopoulos", "Panagiotop.", regex=False)
  users["SName"] = users.SName.apply(format_sponsor)
  users = users.reset_index(drop=False)
  users = users.sort_values(["SName", "CPU-Hours"], ascending=[True, False]).reset_index(drop=True) # added on feb 8 2023
  users.index += 1
  users = users.drop(columns=["Sponsor"])
  users = users.rename(columns={"netid":"NetID", "SName":"Sponsor"})
  users = users[["NetID", "CPU-Hours", "Total Jobs", "Sponsor"]]
  print(users)
  caption = (f"CPU utilization by user on TigerCPU from February 1, 2019 to January 31, 2023.", f"TigerCPU -- Utilization by User")
  print(users["CPU-Hours"].sum())
  users.to_latex("tigercpu_users_feb1_2019_jan31_2023.tex", index=True, caption=caption, column_format="rrrrr", label=f"tigercpu_users_feb1_2019_jan31_2023", longtable=True)
