import os
import subprocess
import pandas as pd
import dossier  # wget https://raw.githubusercontent.com/jdh4/tigergpu_visualization/master/dossier.py

columns = "jobid,user,account,partition,cputimeraw,elapsedraw,alloctres,start,eligible,qos,state"
fields = "jobid,user,account,partition,cputimeraw%25,elapsedraw%50,alloctres%75,start,eligible,qos,state"

use_cache = False

start_date = "2023-01-01"
cluster = "stellar"
fname = f"sacct_{cluster}.csv"

if not use_cache:

    cmd = f"sacct -M {cluster} -a -X -P -n -S {start_date} -o {fields}"
    output = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            shell=True,
                            timeout=100,
                            text=True,
                            check=True)

    rows = [row.split("|") for row in output.stdout.split()]
    df = pd.DataFrame(rows)
    df.columns = columns.split(",")
    df.to_csv(fname)

df = pd.read_csv(fname)
df.info()

df = df[pd.notna(df.elapsedraw)]
#df = df[pd.notna(df.elapsedraw) & df.elapsedraw.str.isnumeric()]
df.elapsedraw = df.elapsedraw.astype("int64")
df = df[df.elapsedraw > 0]

def gpus_per_job(tres: str) -> int:
    """Extract number of GPUs from alloctres."""
    if "gres/gpu=" in tres:
        for part in tres.split(","):
            if "gres/gpu=" in part:
                gpus = int(part.split("=")[-1])
        return gpus
    else:
        return 0

df = df.rename(columns={"cputimeraw":"cpu-seconds"})
df["gpus"] = df.alloctres.apply(gpus_per_job)
df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')
print(df.head(2).T)


##########################################################
### group by
##########################################################
d = {"cpu-seconds":"sum", "gpu-seconds":"sum", "user":"size"}
by_user = df.groupby("user").agg(d)
by_user = by_user.rename(columns={"user":"jobs"})
by_user.reset_index(drop=False, inplace=True)
print(by_user.head(10))


##########################################################
## dossier
##########################################################
dname = f"from_dossier_{cluster}.csv"
if os.path.isfile(dname):
    print(f"Using {dname} ...")
    ds = pd.read_csv(dname)
else:
    netids = sorted(df.user.unique().tolist())
    ds = pd.DataFrame(dossier.ldap_plus(netids))
    headers = ds.iloc[0]
    ds = pd.DataFrame(ds.values[1:], columns=headers)
    ds.to_csv(dname)
ds.info()


##########################################################
### join
##########################################################
cmb = by_user.merge(ds, how="left", left_on="user", right_on="NETID_TRUE")
print(cmb.head(2).T)
print(cmb.POSITION.value_counts().to_string())


##########################################################
### 2nd groupby
##########################################################

# ignore cases such as "DCU (formerly G5)"
cmb["affil"] = cmb.POSITION.apply(lambda p: "external" if p in ["RCU", "DCU", "RU", "XRCU", "XDCU"] else "internal")

d = {"cpu-seconds":"sum", "gpu-seconds":"sum", "affil":"size"}
ext = cmb.groupby("affil").agg(d)
ext = ext.rename(columns={"affil":"users"})
ext.reset_index(drop=False, inplace=True)

def add_proportion_in_parenthesis(dframe, column_name, replace=False):
    assert column_name in dframe.columns
    dframe["proportion"] = 100 * dframe[column_name] / dframe[column_name].sum()
    dframe["proportion"] = dframe["proportion"].apply(lambda x: round(x))
    name = column_name if replace else f"{column_name}-cmb"
    dframe[name] = dframe.apply(lambda row: f"{round(row[column_name])} ({row['proportion']}%)", axis='columns')
    dframe = dframe.drop(columns=["proportion"])
    return dframe

ext["cpu-hours"] = ext["cpu-seconds"] / 3600
ext["gpu-hours"] = ext["gpu-seconds"] / 3600

ext = add_proportion_in_parenthesis(ext, "cpu-hours", replace=True)
ext = add_proportion_in_parenthesis(ext, "gpu-hours", replace=True)
ext = add_proportion_in_parenthesis(ext, "users", replace=True)

print(ext[["affil", "cpu-hours", "gpu-hours", "users"]].to_string(index=False, col_space=20))
