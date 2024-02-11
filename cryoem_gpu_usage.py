#!/usr/licensed/anaconda3/2023.9/bin/python -uB

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


def send_email_html(text, addressee, subject="Cryoem GPU Usage", sender="halverson@princeton.edu"):
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
    s =  f"GPU usage = {pct} (previous {N} days)"
    msg += "=" * len(s) + "\n"
    msg += s + "\n"
    msg += "=" * len(s)
    return msg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cryoem GPU Usage')
    parser.add_argument('--days',
                        type=int,
                        default=14,
                        metavar='N',
                        help='Start date is N previous days from today (default: 14)')
    parser.add_argument('--gpus',
                        type=int,
                        default=152,
                        metavar='G',
                        choices=range(1, 153),
                        help='Maximum number of GPUs available (default: 152)')
    args = parser.parse_args()

    # convert slurm timestamps to seconds
    os.environ["SLURM_TIME_FORMAT"] = "%s"

    clusters = "della"
    start_date, end_date, elapsed_seconds = get_time_window(args.days)
    partitions = "-r cryoem"
    fields = "alloctres,elapsedraw,start"
    df = get_data_from_sacct(clusters, start_date, end_date, partitions, fields)

    # clean elapsedraw field
    df = df[pd.notna(df.elapsedraw)]
    df = df[df.elapsedraw.str.isnumeric()]
    df.elapsedraw = df.elapsedraw.astype("int64")
    df = df[df.elapsedraw > 0]
    # clean start field
    df = df[pd.notna(df.start)]
    df = df[df.start.str.isnumeric()]
    df.start = df.start.astype("int64")

    # apply correction to only include the usage during the time window
    correction = True
    if correction:
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
    url = "tiger: /home/jdh4/bin/cryoem/cryoem_gpu_usage.py"
    msg = format_output(start_date, end_date, percent_usage, args.days, args.gpus, url)

    send_email_html(msg, "mcahn@princeton.edu")
    send_email_html(msg, "halverson@princeton.edu")
