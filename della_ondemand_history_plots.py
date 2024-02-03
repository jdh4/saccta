import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

years = range(2020, 2024)
users = [576, 753, 936, 1262]
ood_users = [155, 213, 341, 515]

nrows = 1
ncols = 2

opts = {"mfc":"tab:blue",
        "mec":"tab:blue",
        "linestyle":'dashed',
        "linewidth":1,
        "markersize":6,
        "color":'lightgrey'}
opts2 = {"mfc":"tab:red",
         "mec":"tab:red",
         "linestyle":'dashed',
         "linewidth":1,
         "markersize":6,
         "color":'lightgrey'}

fig = plt.figure(figsize=(8, 2.5))
plt.subplot(nrows, ncols, 1)
plt.plot(years, users, 'o', label="All Users", **opts)
plt.plot(years, ood_users, 'o', label="OOD Users", **opts2)
plt.xlabel("Year")
plt.ylabel("Number of Users")
plt.legend(fontsize="8")
plt.xticks(years, map(str, years))

opts = {"mfc":"black",
        "mec":"black",
        "linestyle":'dashed',
        "linewidth":1,
        "markersize":6,
        "color":'lightgrey'}

plt.subplot(nrows, ncols, 2)
plt.plot(years, [x/y for x,y in zip(ood_users, users)], 'o', **opts)
plt.xlabel("Year")
plt.ylabel("Num. OOD Users / Num. Users", fontsize=9)
plt.xticks(years, map(str, years))

plt.tight_layout()
cluster = "della_ondemand_plot"
plt.savefig(f"{cluster}_history.png", dpi=200)
plt.savefig(f"{cluster}_history.pdf")
