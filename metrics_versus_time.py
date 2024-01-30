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

    cluster = "adroit"
    if cluster == "adroit":
        # ignoring cloud partition
        # including gpu partition in total cpu-hours (see cpu_partitions)
        all_partitions = "--partition=all,class,gpu"
        cpu_partitions = "--partition=all,class,gpu"
        gpu_partitions = "--partition=gpu"
    elif cluster == "traverse":
        all_partitions = "--partition=all"
        cpu_partitions = "--partition=all"
        gpu_partitions = "--partition=all"
    elif cluster == "tiger2":
        # must add new for 2017 and all for 2017 and 2018
        #all_partitions = "--partition=all,cpu,ext,new,serial"
        all_partitions = "--partition=gpu"
        cpu_partitions = "--partition=all,cpu,ext,new,serial"
        gpu_partitions = "--partition=gpu"
    elif cluster == "stellar":
        all_partitions = "--partition=cimes"
        #all_partitions = "--partition=all,pppl,pu,serial"
        #all_partitions = "--partition=gpu"
        #cpu_partitions = "--partition=all,pppl,pu,serial"
        cpu_partitions = "--partition=cimes"
        gpu_partitions = "--partition=gpu"

    years = range(2018, 2024)
    users = []
    gpu_users = []
    ondemand_users = []
    cpu_hours = []
    gpu_hours = []
    ondemand_cpu_hours = []
    ondemand_gpu_hours = []
    partitions = []
    
    for year in years:

        action = f"-o partition | sort | uniq"
        partitions.append(get_from_sacct(year, cluster, action))
        
        action = f"-o user {all_partitions} | sort | uniq | wc -l"
        # action = f"-o user,alloctres {all_partitions} | grep gres/gpu=[1-9] | cut -d'|' -f1 | sort | uniq | wc -l"
        users.append(get_from_sacct(year, cluster, action))

        action = f"-o cputimeraw {cpu_partitions}" + " | awk '{sum += $1} END {print int(sum/3600)}'"
        cpu_hours.append(get_from_sacct(year, cluster, action))

        action = f"-o elapsedraw,alloctres {gpu_partitions}" + " | grep gres/gpu=[1-9] | sed -E 's/\|.*gpu=/,/' | awk -F',' '{sum += $1*$2} END {print int(sum/3600)}'"
        gpu_hours.append(get_from_sacct(year, cluster, action))

        if cluster == "adroit" or cluster == "stellar":
            action = f"-o user {gpu_partitions} | sort | uniq | wc -l"
            gpu_users.append(get_from_sacct(year, cluster, action))

            action = f"-o user,jobname {all_partitions} | grep sys/dashboard | cut -d'|' -f1 | sort | uniq | wc -l"
            # action = f"-o user,alloctres,jobname {all_partitions} | grep gres/gpu=[1-9] | grep sys/dashboard | cut -d'|' -f1 | sort | uniq | wc -l"
            ondemand_users.append(get_from_sacct(year, cluster, action))

            action = f"-o cputimeraw,jobname {cpu_partitions}" + " | grep sys/dashboard | awk -F',' '{sum += $1} END {print int(sum/3600)}'"
            ondemand_cpu_hours.append(get_from_sacct(year, cluster, action))

            action = f"-o elapsedraw,alloctres,jobname {gpu_partitions}" + " | grep sys/dashboard | grep gres/gpu=[1-9] | sed -E 's/\|.*gpu=/,/' | awk -F',' '{sum += $1*$2} END {print int(sum/3600)}'"
            ondemand_gpu_hours.append(get_from_sacct(year, cluster, action))

    print(partitions)
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
    opts = {"mfc":"tab:blue",
            "mec":"tab:blue",
            "linestyle":'dashed',
            "linewidth":1,
            "markersize":6,
            "color":'lightgrey'}

    if cluster == "adroit":
        nrows = 3
        ncols = 3

        fig = plt.figure(figsize=(11, 7.5))
        plt.subplot(nrows, ncols, 1)
        plt.plot(years, users, 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("Number of Users")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 2)
        plt.plot(years, gpu_users, 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("Number of GPU Users")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 3)
        plt.plot(years, ondemand_users, 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("Number of OOD Users")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 4)
        plt.plot(years, [x/1e6 for x in cpu_hours], 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("CPU-Hours / $10^6$")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 5)
        plt.plot(years, gpu_hours, 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("GPU-Hours")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 6)
        plt.plot(years,  [x / y for x, y in zip(ondemand_cpu_hours, cpu_hours)], 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("OOD CPU-Hours / CPU-Hours")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 7)
        plt.plot(years,  [x / y for x, y in zip(ondemand_gpu_hours, gpu_hours)], 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("OOD GPU-Hours / GPU-Hours")
        plt.xticks(years, map(str, years))

    elif cluster == "traverse":
        nrows = 2
        ncols = 3

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

        plt.subplot(nrows, ncols, 4)
        avail = 365 * 24 * 46 * 4
        plt.plot(years[1:], [x/avail for x in gpu_hours[1:]], 'o', **opts)
        plt.ylim(0, 1)
        plt.xlabel("Year")
        plt.ylabel("GPU-Hours / GPU-Hours Available")
        plt.xticks(years, map(str, years))
    elif cluster == "stellar":
        nrows = 1
        ncols = 2

        fig = plt.figure(figsize=(8, 2.5))
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
        #cluster = "stellar_intel"
        cluster = "stellar_amd"

    elif cluster == "tiger2":
        nrows = 1
        ncols = 3

        fig = plt.figure(figsize=(12, 2.5))
        plt.subplot(nrows, ncols, 1)
        plt.plot(years, users, 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("Number of Users")
        plt.xticks(years, map(str, years))

        if 0:
            plt.subplot(nrows, ncols, 2)
            plt.plot(years, [x/1e6 for x in cpu_hours], 'o', **opts)
            plt.xlabel("Year")
            plt.ylabel("CPU-Hours / $10^6$")
            plt.xticks(years, map(str, years))

            plt.subplot(nrows, ncols, 3)
            avail = 408 * 40 * 365 * 24
            plt.plot(years[1:], [x/avail for x in cpu_hours[1:]], 'o', **opts)
            plt.ylim(0.5, 1)
            plt.xlabel("Year")
            plt.ylabel("CPU-Hours / CPU-Hours Avail.")
            plt.xticks(years, map(str, years))
            cluster = "tigercpu"

        if 1:
            plt.subplot(nrows, ncols, 2)
            plt.plot(years, [x/1e6 for x in gpu_hours], 'o', **opts)
            plt.xlabel("Year")
            plt.ylabel("GPU-Hours / $10^6$")
            plt.xticks(years, map(str, years))

            plt.subplot(nrows, ncols, 3)
            avail = 80 * 4 * 365 * 24
            plt.plot(years[2:-1], [x/avail for x in gpu_hours[2:-1]], 'o', **opts)
            plt.ylim(0, 1)
            plt.xlabel("Year")
            plt.ylabel("GPU-Hours / GPU-Hours Avail.")
            plt.xticks(years, map(str, years))
            cluster = "tigergpu"

    elif cluster == "della":
        nrows = 2
        ncols = 3

        fig = plt.figure(figsize=(12, 5))
        plt.subplot(nrows, ncols, 1)
        plt.plot(years, users, 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("Number of Users")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 2)
        plt.plot(years, [x/1e6 for x in gpu_hours], 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("GPU-Hours / $10^6$")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 3)
        plt.plot(years, ondemand_users, 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("Number of OOD GPU Users")
        plt.xticks(years, map(str, years))

        plt.subplot(nrows, ncols, 4)
        plt.plot(years,  [x / y for x, y in zip(ondemand_gpu_hours, gpu_hours)], 'o', **opts)
        plt.xlabel("Year")
        plt.ylabel("OOD GPU-Hours / GPU-Hours")
        plt.xticks(years, map(str, years))
        cluster = "della_gpu"

    plt.tight_layout()
    plt.savefig(f"{cluster}_history.png", dpi=200)
    plt.savefig(f"{cluster}_history.pdf")
