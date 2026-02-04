"""Get the sponsor of a user by parsing the monthly user reports. This is
   useful since the sponsor of a user may change over time. The report
   files are unstructured data so must be wary of changes to the reports
   since code assumes that certain rules.

   Use Python 3.10+

   self.multiple stores tuples of the form (user, cluster) indicating that
   'user' had multiple sponsors on 'cluster' in the given time window.
"""


import os
import glob
import warnings
import functools
from datetime import datetime
from typing import Optional


class SponsorFromMonthlyReports:

    def __init__(self,
                 path_to_reports: str,
                 clusters: list[str],
                 start_date: str,
                 end_date: str) -> None:
        self.path_to_reports = path_to_reports
        self.clusters = clusters
        self.start_date = start_date
        self.end_date = end_date
        self.sorted_files = []
 
    def build_index(self) -> None:
        """Given a set of monthly user report files, find the files that were
           correspond to the specified time window."""
        full_path = os.path.join(self.path_to_reports, "users.log.*")
        files = glob.glob(full_path)
        sorted_files = sorted(files, key=lambda f: int(f.rsplit('.', 1)[-1]))
        format_string = "%Y-%m-%dT%H:%M:%S"
        start_secs = datetime.strptime(self.start_date, format_string).timestamp()
        end_secs = datetime.strptime(self.end_date, format_string).timestamp()
        assert start_secs < end_secs, "start_secs > end_secs"
        for sorted_file in sorted_files:
            secs = int(sorted_file.split(".")[-1])
            if secs >= start_secs and secs <= end_secs:
                self.sorted_files.append(sorted_file)
 
    def build_map(self) -> None:
        """The dictionary u2s maps the sponsor for a given user and cluster. A
           set of users is maintained for those that have multiple sponsors
           per cluster due to a sponsor change (self.multiple)."""
        self.u2s = {}
        self.multiple = set()
        for sorted_file in self.sorted_files:
            with open(sorted_file, "r") as fp:
                lines = fp.readlines()
            skip = True
            for line in lines:
                if line.startswith("User: "):
                    skip = False
                if not skip:
                    if line.startswith("User: "):
                        user = line.split()[-1]
                        if user not in self.u2s:
                            self.u2s[user] = {}
                    for cluster in self.clusters:
                        if cluster in line:
                            sponsor = line.split()[-1]
                            if cluster not in self.u2s[user]:
                                self.u2s[user][cluster] = sponsor
                            elif self.u2s[user][cluster] != sponsor:
                                self.multiple.add((user, cluster))
        
    @staticmethod
    @functools.cache
    def sponsor_from_single_file(user: str,
                                 cluster: str,
                                 filename: str) -> Optional[str]:
        """Find the sponsor of a user for a given cluster in a single report
           file."""
        with open(filename, "r") as fp:
            lines = fp.readlines()
        skip = True
        for line in lines:
            if line.startswith(f"User: {user}"):
                skip = False
                continue
            if not skip:
                if cluster in line:
                    sponsor = line.split()[-1]
                    return sponsor
                if "Definitions" in line:
                    return None
        return None


    def get_sponsor(self,
                    user: str,
                    cluster: str,
                    secs_since_epoch: int,
                    days: int=16) -> str:
        """We call sponsor_from_single_file() to see if that user appears in
           report."""
        if (user, cluster) in self.multiple:
            distance = 1e32
            index = -1
            for i, curr_file in enumerate(self.sorted_files):
                secs = int(curr_file.split(".")[-1])
                d = abs(secs - secs_since_epoch)
                sponsor = self.sponsor_from_single_file(user, cluster, curr_file)
                if d < distance and sponsor is not None:
                    index = i
                    distance = d
            secs_per_day = 24 * 60 * 60
            if distance > days * secs_per_day:
                far = round(distance / secs_per_day)
                msg = f"Closest file not close for {user} on {cluster} ({far} days)"
                warnings.warn(msg, RuntimeWarning)
            best_file = self.sorted_files[index]
            sponsor = self.sponsor_from_single_file(user, cluster, best_file)
            if index == -1 or sponsor is None:
                raise ValueError(f"{user} {cluster}: index is -1 or sponsor is None")
            return sponsor
        else:
            return self.u2s[user][cluster]

    def __str__(self) -> str:
        msg = "\nFiles\n"
        for sorted_file in s.sorted_files:
            secs = int(sorted_file.split(".")[-1])
            msg += f"{sorted_file}, {datetime.fromtimestamp(secs)}\n"
        msg += "\nUsers with a sponsor change\n"
        for user, cluster in self.multiple:
            msg += f"{user}, {cluster}\n"
        msg += "\nMultiple sponsors but one per cluster\n"
        for user, sponsors in self.u2s.items():
            if len(set(sponsors.values())) > 1:
                msg += f"{user}, {sponsors}\n"
        return msg


if __name__ == "__main__":

    start_date = "2025-01-01T00:00:00"
    end_date   = "2025-12-31T23:59:59"
    reports_path = "/projects/CSES/jdh4/monthly_reports/users/"
    # tiger and tiger3 appear in the report files but tiger works for both
    clusters = ["della", "stellar", "tiger"]
    s = SponsorFromMonthlyReports(reports_path, clusters, start_date, end_date)
    s.build_index()
    s.build_map()
    print(s.get_sponsor("jdh4", "della", 1741328000))
    print(s.u2s["jdh4"])
    print(s)
