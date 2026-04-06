"""
nvidia_gpu_sm_occupancy_percent
nvidia_gpu_sm_util_percent
nvidia_gpu_fp16_util_percent
nvidia_gpu_fp32_util_percent
nvidia_gpu_fp64_util_percent
nvidia_gpu_duty_cycle
nvidia_gpu_memory_used_bytes
nvidia_gpu_memory_total_bytes
nvidia_gpu_any_tensor_util_percent
nvidia_gpu_pcie_rx_per_sec
nvidia_gpu_pcie_tx_per_sec
nvidia_gpu_nvlink_total_rx_per_sec
nvidia_gpu_nvlink_total_tx_per_sec
DCGM_FI_PROF_PIPE_FP64_ACTIVE
DCGM_FI_PROF_SM_OCCUPANCY
"""

import os
import re
import ast
import json
import requests
import subprocess
from typing import List, Union
import pandas as pd
from efficiency import get_stats_dict
from efficiency import gpu_efficiency


os.environ["SLURM_TIME_FORMAT"] = "%s"
PROM_SERVER = "http://vigilant2:8480/api/v1/query"


def convert_string(val):
    """Convert strings to int or float otherwise leave it a string."""
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def get_ai_metric(metric: str,
                  op: str,
                  cluster: str,
                  jobidraw: str,
                  elapsed: int,
                  end: int) -> List[Union[int, float]]:
    """Compute the operation for a given quantity and job details."""
    if op == "avg":
        s = "avg_over_time(("
    elif op == "max":
        s = "max_over_time(("
    else:
        s = ""
    q = s + metric + "{cluster=\"" + cluster + "\"} and nvidia_gpu_jobId == " \
        + jobidraw + ")[" + str(elapsed) + "s:])"
    params = {'query': q, 'time': end}
    print(params)
    response = requests.get(PROM_SERVER, params)
    results = response.json()["data"]["result"]
    metrics = []
    for result in results:
        if "value" in result:
            value = convert_string(result["value"][1])
            if isinstance(value, float):
                if value <= 1:
                    value = round(value, 6)
                else:
                    value = round(value, 1)
            metrics.append(value)
    return metrics


def get_data_from_sacct(clusters: str,
                        partition: str,
                        start_date: str,
                        end_date: str,
                        fields: str) -> pd.DataFrame:
    """Return a dataframe of the sacct output."""
    cmd = f"sacct -M {clusters} -a -X -P -n -S {start_date} -E {end_date} -o {fields}"
    if partition:
        cmd += f" -r {partition}"
    output = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            shell=True,
                            timeout=100,
                            text=True,
                            check=True)
    rows = output.stdout.strip().split('\n')
    cols = fields.split(",")
    raw = pd.DataFrame([row.split("|")[:len(cols)]
                       for row in rows if row.count("|") > len(cols) - 2])
    raw.columns = cols
    return raw


def gpus_per_job(tres: str) -> int:
    """Return the number of allocated GPUs."""
    gpus = re.findall(r"gres/gpu=\d+", tres)
    return int(gpus[0].replace("gres/gpu=", "")) if gpus else 0


def clean_dataframe(df: pd.DataFrame):
    """Clean the dataframe and set data types."""
    for col in ["elapsedraw", "start", "end"]:
        df = df[pd.notna(df[col])]
        df = df[df[col].str.isnumeric()]
        df[col] = df[col].astype("int64")
    df = df[df.elapsedraw > 0]
    df.cputimeraw = df.cputimeraw.astype("int64")
    df["gpus"] = df.alloctres.apply(gpus_per_job)
    df["gpu-seconds"] = df.apply(lambda row:
                                 row["gpus"] * row["elapsedraw"]
                                 if row["partition"] != "mig"
                                 else row["gpus"] * row["elapsedraw"] / 7,
                                 axis="columns")
    return df


if __name__ == "__main__":

    clusters = "della"
    partition = "pli-c,pli,pli-lc,pli-p"
    partition = "ailab"
    start_date = "2026-03-24T00:00:00"
    end_date   = "2026-03-26T20:00:00"
    fields = "jobid,jobidraw,user,cluster,partition,account,elapsedraw,cputimeraw,state,alloctres,start,end,nnodes,ncpus,admincomment"

    df = get_data_from_sacct(clusters, partition, start_date, end_date, fields)
    df = df[(df.state != "RUNNING") | (df.state != "PENDING")]
    df = clean_dataframe(df)
    df = df[(df.start > 1_000_000_000) & (df.end > 1_000_000_000)]
    df = df[df.start <= df.end]
    df["admincomment"] = df["admincomment"].apply(get_stats_dict)
    df = df[df["admincomment"] != {}]

    df["gpu-tuple"] = df.apply(lambda row:
                               gpu_efficiency(row["admincomment"],
                                              row["elapsedraw"],
                                              row["jobid"],
                                              row["cluster"],
                                              single=True,
                                              verbose=False),
                                              axis="columns")

    cols = ["gpu-eff", "gpu-error-code"]
    df[cols] = pd.DataFrame(df["gpu-tuple"].tolist(), index=df.index)
    df = df[df["gpu-error-code"] == 0]

    """
    df["util"] = df.apply(lambda row:
                          get_ai_metric("nvidia_gpu_duty_cycle",
                                        "avg",
                                        row["cluster"],
                                        row["jobidraw"],
                                        row["elapsedraw"],
                                        row["end"]), axis="columns")
    df["mem"] = df.apply(lambda row:
                         get_ai_metric("nvidia_gpu_memory_used_bytes",
                                       "max",
                                       row["cluster"],
                                       row["jobidraw"],
                                       row["elapsedraw"],
                                       row["end"]), axis="columns")
    """

    # Percentage of time the GPU's half-precision (FP16) arithmetic
    # pipes/cores were active over a sample period.
    df["fp16%"] = df.apply(lambda row:
                          get_ai_metric("nvidia_gpu_fp16_util_percent",
                                        "avg",
                                        row["cluster"],
                                        row["jobidraw"],
                                        row["elapsedraw"],
                                        row["end"]), axis="columns")

    df["fp32%"] = df.apply(lambda row:
                          get_ai_metric("nvidia_gpu_fp32_util_percent",
                                        "avg",
                                        row["cluster"],
                                        row["jobidraw"],
                                        row["elapsedraw"],
                                        row["end"]), axis="columns")

    df["fp64%"] = df.apply(lambda row:
                          get_ai_metric("nvidia_gpu_fp64_util_percent",
                                        "avg",
                                        row["cluster"],
                                        row["jobidraw"],
                                        row["elapsedraw"],
                                        row["end"]), axis="columns")

    # Percentage of the time your GPU's specialized AI hardware was actively
    # working during a specific measurement interval. It tracks activity
    # across all supported precision types (e.g., FP16, BF16, INT8, or TF32).
    df["tensor_cores%"] = df.apply(lambda row:
                        get_ai_metric("nvidia_gpu_any_tensor_util_percent",
                                      "avg",
                                      row["cluster"],
                                      row["jobidraw"],
                                      row["elapsedraw"],
                                      row["end"]), axis="columns")

    # Measures the average activity of the Streaming Multiprocessors (SMs) on
    # your GPU or the % of all available SMs that are currently active. An SM
    # is considered active if it has at least one warp (a bundle of 32 threads)
    # assigned to it. This metric is the ratio of cycles where SMs had active
    # warps compared to the total possible cycles, averaged across all SMs on
    # the chip.
    df["sm%"] = df.apply(lambda row:
                         get_ai_metric("nvidia_gpu_sm_util_percent",
                                       "avg",
                                       row["cluster"],
                                       row["jobidraw"],
                                       row["elapsedraw"],
                                       row["end"]), axis="columns")

    df["occupancy%"] = df.apply(lambda row:
                                get_ai_metric("nvidia_gpu_sm_occupancy_percent",
                                              "avg",
                                              row["cluster"],
                                              row["jobidraw"],
                                              row["elapsedraw"],
                                              row["end"]), axis="columns")

    # The metric nvidia_gpu_nvlink_total_rx_per_sec represents the total rate of
    # data being received by a specific GPU across all its active NVLink
    # connections, measured in bytes per second. This is a critical performance
    # counter used in high-performance computing (HPC) and deep learning
    # environments to monitor how efficiently GPUs are communicating with one
    # another. If this number is pinned at the maximum theoretical bandwidth
    # of your hardware, your workload is "communication-bound." It helps ensure
    # that your software (like NCCL for PyTorch or TensorFlow) is actually using
    # NVLink rather than falling back to the much slower PCIe bus. A sudden drop
    # in this value during a heavy workload could indicate a hardware failure or
    # a "degraded" link where one or more NVLink lanes have shut down.

    # The metric nvidia_gpu_pcie_rx_per_sec represents the rate of data being
    # received by the GPU over the PCIe (Peripheral Component Interconnect
    # Express) bus. In the context of GPU monitoring (typically using NVIDIA
    # DCGM and Prometheus), this metric tracks the throughput of "Host-to-Device"
    # (H2D) communication—essentially how fast data is moving from your
    # CPU/System RAM into the GPU’s memory.

    # The metric nvidia_gpu_pcie_tx_per_sec represents the rate of data being
    # transmitted from the GPU to the host (CPU/system memory) over the PCIe
    # bus. In simpler terms, it measures how fast the GPU is "sending" data
    # back to the rest of the computer.

    # What to do about Grace Hopper where need C2C metrics from DCGM since
    # it uses NVLink instead of PCIe between CPU and GPU?

    #df["util"] = df["util"].apply(lambda x: round(sum(x) / len(x), 1))
    df["sm%"] = df["sm%"].apply(lambda x: round(100 * sum(x) / len(x), 1))
    df["occupancy%"] = df["occupancy%"].apply(lambda x: round(100 * sum(x) / len(x), 1))
    df["fp16%"] = df["fp16%"].apply(lambda x: round(100 * sum(x) / len(x), 1))
    df["fp32%"] = df["fp32%"].apply(lambda x: round(100 * sum(x) / len(x), 1))
    df["fp64%"] = df["fp64%"].apply(lambda x: round(100 * sum(x) / len(x), 1))
    df["tensor_cores%"] = df["tensor_cores%"].apply(lambda x: round(100 * sum(x) / len(x), 1))

    print("GPU-hours", round(df["gpu-seconds"].sum() / 3600))
    print(df.info())
    print(df.head(5).T)
    print(df[["jobidraw", "gpu-eff", "sm%", "occupancy%", "fp16%", "fp32%", "fp64%", "tensor_cores%"]].to_string())

    df["gpu-eff-time"] = df["gpu-eff"] * df["elapsedraw"] * df["gpus"] / 100
    df["sm%-time"] = df["sm%"] * df["elapsedraw"] * df["gpus"] / 100
    df["occupancy%-time"] = df["occupancy%"] * df["elapsedraw"] * df["gpus"] / 100
    df["fp16%-time"] = df["fp16%"] * df["elapsedraw"] * df["gpus"] / 100
    df["fp32%-time"] = df["fp32%"] * df["elapsedraw"] * df["gpus"] / 100
    df["fp64%-time"] = df["fp64%"] * df["elapsedraw"] * df["gpus"] / 100
    df["tensor_cores%-time"] = df["tensor_cores%"] * df["elapsedraw"] * df["gpus"] / 100

    d = {"gpu-eff-time": "sum",
         "sm%-time": "sum",
         "occupancy%-time": "sum",
         "fp16%-time": "sum",
         "fp32%-time": "sum",
         "fp64%-time": "sum",
         "tensor_cores%-time": "sum",
         "gpu-seconds": "sum"}
    gp = df.groupby("user").agg(d)
    gp.reset_index(inplace=True)
    gp.rename(columns={"user":"User"}, inplace=True)
    print(gp)

    gp["GPU-Hours"] = gp["gpu-seconds"] / 3600
    gp["UTIL(%)"] = 100 * gp["gpu-eff-time"] / gp["gpu-seconds"]
    gp["SM(%)"] = 100 * gp["sm%-time"] / gp["gpu-seconds"]
    gp["OCCUPANCY(%)"] = 100 * gp["occupancy%-time"] / gp["gpu-seconds"]
    gp["FP16(%)"] = 100 * gp["fp16%-time"] / gp["gpu-seconds"]
    gp["FP32(%)"] = 100 * gp["fp32%-time"] / gp["gpu-seconds"]
    gp["FP64(%)"] = 100 * gp["fp64%-time"] / gp["gpu-seconds"]
    gp["TENSOR-CORES(%)"] = 100 * gp["tensor_cores%-time"] / gp["gpu-seconds"]

    gp["GPU-Hours"] = gp["GPU-Hours"].round(decimals=0).astype(int)
    gp["UTIL(%)"] = gp["UTIL(%)"].round(decimals=0).astype(int)
    gp["SM(%)"] = gp["SM(%)"].round(decimals=0).astype(int)
    gp["OCCUPANCY(%)"] = gp["OCCUPANCY(%)"].round(decimals=0).astype(int)
    gp["FP16(%)"] = gp["FP16(%)"].round(3)
    gp["FP32(%)"] = gp["FP32(%)"].round(1)
    gp["FP64(%)"] = gp["FP64(%)"].round(3)
    gp["TENSOR-CORES(%)"] = gp["TENSOR-CORES(%)"].round(1)
    gp["UTIL - SM"] = gp["UTIL(%)"] - gp["SM(%)"]

    gp.sort_values(by="gpu-seconds", ascending=False, inplace=True)
    print(gp[["User",
              "GPU-Hours",
              "UTIL(%)",
              "SM(%)",
              "UTIL - SM",
              "OCCUPANCY(%)",
              "FP16(%)",
              "FP32(%)",
              "FP64(%)",
              "TENSOR-CORES(%)"]])
