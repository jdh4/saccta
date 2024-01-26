import subprocess
import pandas as pd
from datetime import datetime
from datetime import timedelta
import smtplib
from email.message import EmailMessage


def gpus_per_job(tres: str) -> int:
    """Extract number of GPUs from alloctres."""
    if "gres/gpu=" in tres:
        for part in tres.split(","):
            if "gres/gpu=" in part:
                gpus = int(part.split("=")[-1])
        return gpus
    else:
        return 0


def get_data_from_sacct(start_date: str, fields: str) -> pd.DataFrame:
    cmd = f"sacct -M della -a -X -P -n -S {start_date} -E now -o {fields} -r cryoem"
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


def send_email_html(s, addressee, subject="Cryoem GPU Usage", sender="halverson@princeton.edu"):
    """Send an email in HTML."""
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = addressee
    html = f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title></title></head><body><table width="600px" border="0"><tr><td align="center">{s}</td></tr></table></body></html>'
    msg.set_content(html, subtype="html")
    with smtplib.SMTP('localhost') as s:
        s.send_message(msg)


if __name__ == "__main__":

    days = 14
    start_date = datetime.today() - timedelta(days)
    start_date = start_date.strftime("%Y-%m-%d")
    start_date = start_date + "T00:00:00"

    fields = "alloctres,elapsedraw"
    df = get_data_from_sacct(start_date, fields)

    # clean data
    df = df[pd.notna(df.elapsedraw)]
    df = df[df.elapsedraw.str.isnumeric()]
    df.elapsedraw = df.elapsedraw.astype("int64")
    df = df[df.elapsedraw > 0]

    df["gpus"] = df.alloctres.apply(gpus_per_job)
    df["gpu-seconds"] = df.apply(lambda row: row["elapsedraw"] * row["gpus"], axis='columns')

    seconds_per_hour = 3600
    hours_per_day = 24
    gpus = 136

    max_gpu_seconds = days * hours_per_day * seconds_per_hour * gpus
    ratio = df["gpu-seconds"].sum() / max_gpu_seconds
    print(round(100 * ratio))

    send_email_html(str(ratio), "halverson@princeton.edu")
