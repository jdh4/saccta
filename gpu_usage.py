#!/usr/licensed/anaconda3/2024.10/bin/python -uB

import os
import re
import argparse
import subprocess
import smtplib
from datetime import datetime
from datetime import timedelta
from email.message import EmailMessage
import pandas as pd


def get_time_window(num_days: int) -> tuple[str, str, int]:
    """Find the start and end dates."""
    today = datetime.now()
    start_date = today - timedelta(days=num_days)
    py_start_date = datetime(start_date.year, start_date.month, start_date.day, 8, 0, 0)
    py_end_date = datetime(today.year, today.month, today.day, 8, 0, 0)
    elapsed_seconds = (py_end_date - py_start_date).total_seconds()
    start_date = py_start_date.strftime("%Y-%m-%dT%H:%M:%S")
    end_date = py_end_date.strftime("%Y-%m-%dT%H:%M:%S")
    return start_date, end_date, elapsed_seconds


def gpus_per_job(tres: str) -> int:
    """Return the number of allocated GPUs."""
    gpus = re.findall(r"gres/gpu=\d+", tres)
    return int(gpus[0].replace("gres/gpu=", "")) if gpus else 0


def get_data_from_sacct(clusters: str,
                        start_date: str,
                        end_date: str,
                        partitions: str,
                        fields: str) -> pd.DataFrame:
    """Return a dataframe of the sacct output."""
    cmd = f"sacct -M {clusters} -a -X -P -n -S {start_date} -E {end_date} {partitions} -o {fields}"
    output = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            shell=True,
                            timeout=100,
                            text=True,
                            check=True)
    rows = [row.split("|") for row in output.stdout.split()]
    df = pd.DataFrame(rows)
    df.columns = fields.split(",")
    return df


def send_email_html(text, addressee, subject="GPU Usage", sender="halverson@princeton.edu"):
    """Send an email in HTML."""
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = addressee
    html = '<html><head></head><body>'
    html += f'<font face="Courier New, Courier, monospace"><pre>{text}</pre></font></body></html>'
    msg.set_content(html, subtype="html")
    with smtplib.SMTP('localhost') as s:
        s.send_message(msg)


def format_output(d1: str, d2: str, pct: str, N: int, G: int, url: str) -> str:
    """Prepare the results for the email message."""
    msg  = f"Start: {d1}\n"
    msg += f"  End: {d2}\n"
    msg += f" GPUs: {G}\n"
    msg += f"{url}\n\n"
    s =  f"{pct} = GPU-hours used / GPU-hours available (previous {N} days)"
    msg += "=" * len(s) + "\n"
    msg += s + "\n"
    msg += "=" * len(s)
    return msg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cryoem GPU Usage')
    parser.add_argument('-M', '--clusters', type=str, default="della",
                        help='Specify cluster(s) (e.g., --clusters=della,traverse)')
    parser.add_argument('-r', '--partition', type=str, default="cryoem",
                        help='Specify partition(s) (e.g., --partition=gpu,mig)')
    parser.add_argument('-e', '--email', type=str, default=None,
                        help='Email address of the recipient')
    parser.add_argument('-s', '--subject', type=str, default="Cryoem GPU Usage",
                        help='Subject of the email')
    parser.add_argument('--days',
                        type=int,
                        default=14,
                        metavar='N',
                        help='Start date is N previous days from today (default: 14)')
    parser.add_argument('--gpus',
                        type=int,
                        default=152,
                        metavar='N',
                        choices=range(1, 1000),
                        help='Maximum number of GPUs available (default: 152)')
    parser.add_argument('--no-correction', action='store_true', default=False,
                        help='Do not apply correction to only include usage during time window and not before')
    args = parser.parse_args()

    # convert slurm timestamps to seconds
    os.environ["SLURM_TIME_FORMAT"] = "%s"

    start_date, end_date, elapsed_seconds = get_time_window(args.days)
    partitions = f"-r {args.partition}"
    fields = "user,account,alloctres,elapsedraw,start"
    df = get_data_from_sacct(args.clusters, start_date, end_date, partitions, fields)

    # clean elapsedraw field
    df = df[pd.notna(df.elapsedraw)]
    df = df[df.elapsedraw.str.isnumeric()]
    df.elapsedraw = df.elapsedraw.astype("int64")
    df = df[df.elapsedraw > 0]
    # clean start field
    df = df[pd.notna(df.start)]
    df = df[df.start.str.isnumeric()]
    df.start = df.start.astype("int64")

    # apply correction to only include the usage during the time window and not before
    # the start of the window
    if not args.no_correction:
        start_dt = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
        df["secs-from-start"] = df["start"] - start_dt.timestamp()
        df["secs-from-start"] = df["secs-from-start"].apply(lambda x: x if x < 0 else 0)
        df["elapsedraw"] = df["elapsedraw"] + df["secs-from-start"]

    # add columns
    df["gpus"] = df.alloctres.apply(gpus_per_job)
    df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')

    max_gpu_seconds = elapsed_seconds * args.gpus
    used_over_available = df["gpu-seconds"].sum() / max_gpu_seconds
    percent_usage = f"{round(100 * used_over_available)}%"

    # prepare email message
    url = "della: /home/jdh4/bin/gpu_usage.py"
    msg = format_output(start_date, end_date, percent_usage, args.days, args.gpus, url)

    # by-user
    df["gpu-hours"] = round(df["gpu-seconds"] / 3600)
    d = {"account":lambda series: ",".join(sorted(set(series))), "gpu-hours":"sum"}
    gp = df.groupby("user").agg(d).sort_values("gpu-hours", ascending=False)
    gp.reset_index(drop=False, inplace=True)
    gp.index += 1
    # note that gpu-hours have been rounded at this point
    gp["proportion(%)"] = gp["gpu-hours"] / gp["gpu-hours"].sum()
    gp["proportion(%)"] = round(100 * gp["proportion(%)"])
    gp["proportion(%)"] = gp["proportion(%)"].astype("int32")
    gp["gpu-hours"] = gp["gpu-hours"].astype("int32")

    msg += "\n\n" + gp.to_string()
    print(msg)

    #send_email_html(msg, "halverson@princeton.edu", subject=args.subject)
    if args.email:
        send_email_html(msg, args.email, subject=args.subject)
