import pandas as pd
import dossier
import sponsor

# sacct -M tiger3 -a -X -n -P -S 2025-01-01T00:00:00 -E now -o user,cluster,cputimeraw > tiger.csv
# cat della.csv stellar.csv tiger.csv > slurm.csv

# rename all.csv each time below
# trn -e w26 -a
# trn -e f25 -a
# trn -e r25 -a
# trn -e s25 -a
# trn -e w25 -a

jobs = pd.read_csv("slurm.csv", sep="|")
print(jobs.head())

gp = jobs.groupby(by=["user", "cluster"]).agg({"cpuseconds":"sum"}).reset_index()
print(gp.head())
print(type(gp))
print(gp.columns)

df_new = (
    gp.pivot(index="user", columns="cluster", values="cpuseconds")
      .reset_index()
)

df_new = df_new.fillna(0)
df_new["della"] = df_new["della"] / 3600
df_new["stellar"] = df_new["stellar"] / 3600
df_new["tiger3"] = df_new["tiger3"] / 3600
df_new = df_new.astype({'della': 'int64', 'stellar': 'int64', 'tiger3': 'int64'})
print(df_new.head(50))

df_list = (pd.read_csv(f) for f in ["w25.csv", "s25.csv", "r25.csv", "f25.csv", "w26.csv"])
combined_df = pd.concat(df_list, ignore_index=True)
print(combined_df.head())

w = combined_df["Net ID"].value_counts().to_frame().reset_index()
w = w.rename(columns={"count":"Workshops"})
print(w)
print(w.columns)


w = w[w.Workshops >= 3]


cmb = pd.merge(df_new, w, how="inner", left_on="user", right_on="Net ID")
cmb = cmb.sort_values(by="Workshops", ascending=False)
print(cmb.head())
#import sys; sys.exit()



# sponsor and dossier
netids = cmb["user"].tolist()
df = pd.DataFrame(dossier.ldap_plus(netids, level=1))
headers = df.iloc[0]
df = pd.DataFrame(df.values[1:], columns=headers)

import sponsor
df = sponsor.user_and_sponsor_with_dept(df, cluster="della")
print(df.to_string())

df = df[df.POSITION.str.contains("G|U", case=True, regex=True)]
df = df[df.POSITION != "DCU"]
df = df[df.POSITION != "RCU"]
print(df.to_string())

final = pd.merge(df, cmb, how="inner", left_on="NETID", right_on="user")

final = final[["NAME", "NETID", "POSITION", "DEPT", "SPONSOR_NAME", "Workshops", "della", "stellar", "tiger3"]]
final = final.sort_values(by="NETID")
final = final.reset_index(drop=True)
final.index = final.index + 1
print(final)
final.to_latex("workshops_clusters.tex")
