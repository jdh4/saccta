import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_from_sacct(year: str, cluster: str, action: str) -> int:
    """Return the output of the command as an integer."""
    start = f"{year}-01-01T00:00:00"
    end   = f"{year}-12-31T23:59:59"
    cmd = f"sacct -M {cluster} -a -X -P -n -S {start} -E {end} {action}"
    output = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            shell=True,
                            timeout=100,
                            text=True,
                            check=True)
    output = output.stdout
    print(cmd)
    print(output)
    return 0 if output == "" else int(output)


if __name__ == "__main__":

    cluster = "tiger2"
    if cluster == "adroit":
        all_partitions = "--partition=all,class,gpu"
        cpu_partitions = "--partition=all,class,gpu"
        gpu_partitions = "--partition=gpu"
    elif cluster == "traverse":
        all_partitions = "--partition=all"
        cpu_partitions = "--partition=all"
        gpu_partitions = "--partition=all"
    elif cluster == "tiger2":
        #all_partitions = "--partition=cpu,ext,serial"
        all_partitions = "--partition=gpu"
        cpu_partitions = "--partition=cpu,ext,serial"
        gpu_partitions = "--partition=gpu"

    years = [2019, 2020, 2021, 2022, 2023]
    users = []
    gpu_users = []
    ondemand_users = []
    cpu_hours = []
    gpu_hours = []
    ondemand_cpu_hours = []
    ondemand_gpu_hours = []
    for year in years:
        action = f"-o user {all_partitions} | sort | uniq | wc -l"
        users.append(get_from_sacct(year, cluster, action))

        action = f"-o cputimeraw {cpu_partitions}" + " | awk '{sum += $1} END {print int(sum/3600)}'"
        cpu_hours.append(get_from_sacct(year, cluster, action))

        action = f"-o elapsedraw,alloctres {gpu_partitions}" + " | grep gres/gpu=[1-9] | sed -E 's/\|.*gpu=/,/' | awk -F',' '{sum += $1*$2} END {print int(sum/3600)}'"
        gpu_hours.append(get_from_sacct(year, cluster, action))

        if cluster == "adroit":
            action = f"-o user {gpu_partitions} | sort | uniq | wc -l"
            gpu_users.append(get_from_sacct(year, cluster, action))

            action = f"-o user,jobname {all_partitions} | grep sys/dashboard | cut -d'|' -f1 | sort | uniq | wc -l"
            ondemand_users.append(get_from_sacct(year, cluster, action))

            action = f"-o cputimeraw,jobname {cpu_partitions}" + " | grep sys/dashboard | awk -F',' '{sum += $1} END {print int(sum/3600)}'"
            ondemand_cpu_hours.append(get_from_sacct(year, cluster, action))

            action = f"-o elapsedraw,alloctres,jobname {gpu_partitions}" + " | grep sys/dashboard | grep gres/gpu=[1-9] | sed -E 's/\|.*gpu=/,/' | awk -F',' '{sum += $1*$2} END {print int(sum/3600)}'"
            ondemand_gpu_hours.append(get_from_sacct(year, cluster, action))

    print(years)
    print(users)
    print(gpu_users)
    print(ondemand_users)
    print(cpu_hours)
    print(gpu_hours)
    print(ondemand_cpu_hours)
    print(ondemand_gpu_hours)

    ####################
    # write the figure #
    ####################
    if cluster == "adroit":
        nrows = 3
        ncols = 3

        fig = plt.figure(figsize=(11, 7.5))
        plt.subplot(nrows, ncols, 1)
        plt.scatter(years, users)
        plt.xlabel("Year")
        plt.ylabel("Number of Users")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 2)
        plt.scatter(years, gpu_users)
        plt.xlabel("Year")
        plt.ylabel("Number of GPU Users")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 3)
        plt.scatter(years, ondemand_users)
        plt.xlabel("Year")
        plt.ylabel("Number of OOD Users")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 4)
        plt.scatter(years, [x/1e6 for x in cpu_hours])
        plt.xlabel("Year")
        plt.ylabel("CPU-Hours / $10^6$")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 5)
        plt.scatter(years, gpu_hours)
        plt.xlabel("Year")
        plt.ylabel("GPU-Hours")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 6)
        plt.scatter(years,  [x / y for x, y in zip(ondemand_cpu_hours, cpu_hours)])
        plt.xlabel("Year")
        plt.ylabel("OOD CPU-Hours / CPU-Hours")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 7)
        plt.scatter(years,  [x / y for x, y in zip(ondemand_gpu_hours, gpu_hours)])
        plt.xlabel("Year")
        plt.ylabel("OOD GPU-Hours / GPU-Hours")
        plt.xticks(years, map(str, years))

    elif cluster == "traverse":
        nrows = 1
        ncols = 3

        opts = {"mfc":"tab:blue",
                "mec":"tab:blue",
                "linestyle":'dashed',
                "linewidth":1,
                "markersize":6,
                "color":'lightgrey'}

        fig = plt.figure(figsize=(11, 2.5))
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
        plt.plot(years, [x/1e6 for x in gpu_hours], 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("GPU-Hours / $10^6$")
        plt.xticks(years, map(str, years))

    elif cluster == "tiger2":
        nrows = 1
        ncols = 2

        opts = {"mfc":"tab:blue",
                "mec":"tab:blue",
                "linestyle":'dashed',
                "linewidth":1,
                "markersize":6,
                "color":'lightgrey'}

        fig = plt.figure(figsize=(8, 2.5))
        plt.subplot(nrows, ncols, 1)
        plt.plot(years, users, 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("Number of Users")
        plt.xticks(years, map(str, years))

        if False:
            plt.subplot(nrows, ncols, 2)
            plt.plot(years, [x/1e6 for x in cpu_hours], 'o', **opts)
            plt.xlabel("Year")
            plt.ylabel("CPU-Hours / $10^6$")
            plt.xticks(years, map(str, years))
            cluster = "tigercpu"

        if True:
            plt.subplot(nrows, ncols, 2)
            plt.plot(years, [x/1e6 for x in gpu_hours], 'o', **opts)
            plt.xlabel("Year")
            plt.ylabel("GPU-Hours / $10^6$")
            plt.xticks(years, map(str, years))
            cluster = "tigergpu"

    plt.tight_layout()
    plt.savefig(f"{cluster}_history.png", dpi=200)
    plt.savefig(f"{cluster}_history.pdf")
