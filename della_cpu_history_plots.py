import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# data generated by https://github.com/jdh4/saccta/blob/main/della_cpu_history.sh
years = range(2017, 2024)
users = [393, 419, 468, 583, 685, 803, 1056]
cpu_hours = [26450641, 29058839, 37985656, 44380310, 37801931, 29512750, 38921236]

nrows = 1
ncols = 3

opts = {"mfc":"tab:blue",
        "mec":"tab:blue",
        "linestyle":'dashed',
        "linewidth":1,
        "markersize":6,
        "color":'lightgrey'}

fig = plt.figure(figsize=(12, 2.5))
plt.subplot(nrows, ncols, 1)
plt.plot(years, users, 'o', **opts)
plt.xlabel("Year")
plt.ylabel("Number of Users")
plt.xticks(years, map(str, years))

plt.subplot(nrows, ncols, 2)
plt.plot(years, [x/1e6 for x in cpu_hours], 'o', **opts)
plt.xlabel("Year")
plt.ylabel("CPU-Hours / $10^6$")
plt.xticks(years, map(str, years))

plt.subplot(nrows, ncols, 3)
avail = 365 * 24 * (96 * 28 + 64 * 32 + 64 * 32)
plt.plot(years, [x/avail for x in cpu_hours], 'o', **opts)
plt.ylim(0, 1)
plt.xlabel("Year")
plt.ylabel("CPU-Hours / CPU-Hours Avail.")
plt.xticks(years, map(str, years))

plt.tight_layout()
cluster = "della_cpu"
plt.savefig(f"{cluster}_history.png", dpi=200)
plt.savefig(f"{cluster}_history.pdf")
