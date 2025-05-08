"""Convert the raw ATLASv1 audit logs into a dataframe usable by the ATLASv2
graph parser.

This uses an ad-hoc line-by-line parser, as the audit log is still highly
structured even if it is in human-readable text instead of a more compact
format."""

import datetime
import io
import logging
import os
import re
from typing import Dict, TextIO, Tuple

import pandas as pd
import tqdm

import config

log = logging.getLogger(__name__)

class LogStream:
    """Wrapper for the raw text audit log that tracks a lookahead, as this is
    commonly needed for parsing."""
    def __init__(self, f: TextIO):
        self.f = f
        self.lookahead = None
        self.__next__()

    def __next__(self) -> str:
        if isinstance(self.lookahead, StopIteration):
            raise self.lookahead

        self.cur = self.lookahead
        try:
            self.lookahead = next(self.f)
        except StopIteration as e:
            self.lookahead = e
        return self.cur

    def __iter__(self) -> str:
        return self

class ParseState:
    # this always starts a new log event
    event_start_re = re.compile("^Audit (Success|Failure)")
    tabs_re = re.compile("\t+")
    single_tab_re = re.compile("^\t[^\t]")

    def __init__(self):
        self.ctr = 0

    def _parse_event_data(self, line: str, event: Dict):
        """Parse the event-specific data in a log event. This only extracts
        data needed for the provenance graph, which is a very small subset of
        all the possible data:

            Process ID
            Process Name
            New Process ID
            New Process Name
            Object Type
            Object Name
            Access Mask
            Application Name
            Direction
            Destination Address
            Destination Port

        Ideally, we also want the IP Address of a network share, but event
        5140 doesn't actually show up at all in the ATLASv1 logs.
        """
        # ignore non-single-indented data
        if not self.single_tab_re.match(line):
            return

        # lstrip to remove leading tabs; don't rstrip yet because there may be
        # empty values, but this would destroy the tab separator
        line = line.lstrip()
        try:
            # maxsplit=1 to avoid mangling v if it contains tabs
            k, v = self.tabs_re.split(line, maxsplit=1)
            k = k.rstrip(":").rstrip()
            v = v.strip()
        except ValueError:
            return # not a useful field

        extract_as = {
            "Process ID": "ProcessId",
            "Process Name": "ProcessName",
            "New Process ID": "NewProcessId",
            "New Process Name": "NewProcessName",
            "Creator Process ID": "ProcessId",
            "Object Type": "ObjectType",
            "Object Name": "ObjectName",
            "Application Name": "Application",
            "Destination Address": "DestAddress",
            "Destination Port": "DestPort",
        }

        if k in extract_as:
            event["Event Data Fields"][extract_as[k]] = v
            return

        # for some reason Access Mask often has a trailing quote
        if k == "Access Mask":
            event["Event Data Fields"]["AccessMask"] = v.rstrip('"')
            return

        # translate Inbound and Outbound to their corresponding constants
        if k == "Direction":
            if v == "Inbound":
                v = "%%14592"
            elif v == "Outbound":
                v = "%%14593"
            else:
                raise ValueError("Unknown direction")
            event["Event Data Fields"]["Direction"] = v
            return

    @staticmethod
    def _to_iso8601(time: str) -> str:
        """Convert the timestamp in "month/day/year hour:min:second AM/PM"
        to an ISO 8601 string."""
        fmt = "%m/%d/%Y %H:%M:%S %p"
        return datetime.datetime.strptime(time, fmt).isoformat()

    def parse_event(self, stream: LogStream) -> Tuple[int, Dict]:
        """Parse one event from the log line stream, or raise StopIteration if
        there are no events left. New audit events are demarcated by either
        "Audit Success" or "Audit Failure", so use those as landmarks.

        Return the number of characters read and the event. (The number of
        characters is not necessarily the number of bytes, so the progress bar
        will be slightly inaccurate. However, there doesn't seem to be any
        good way to resolve this aside from reading in binary mode, which
        comes with its own issues.)"""
        event = {}
        line = next(stream)
        n_bytes = len(line)
        assert self.event_start_re.match(line)
        try:
            _, time, _, event_id, *_ = line.split("\t")
        except ValueError:
            # malformed log event (e.g., see h1_M3, line 570822)
            # skip until reaching the next event
            m = self.event_start_re.match(stream.lookahead)
            while not m:
                line = next(stream)
                n_bytes += len(line)
                if isinstance(stream.lookahead, StopIteration):
                    break
                m = self.event_start_re.match(stream.lookahead)

            n_next, event = self.parse_event(stream)
            return n_bytes + n_next, event

        # some of the XML metadata such as version, log PID, etc., is missing
        event["Event ID"] = event_id

        # system time resolution is unforunately very lacking
        event["System Time"] = self._to_iso8601(time)

        # make up a record ID; this will actually be backwards since the IDs
        # should be descending, but it ultimately doesn't matter for UUIDs
        event["Event Record ID"] = self.ctr
        self.ctr += 1

        # nested dict format to match ATLASv2
        event["Event Data Fields"] = {}

        # add misc data until reaching next event
        m = self.event_start_re.match(stream.lookahead)
        while not m:
            line = next(stream)
            n_bytes += len(line)
            self._parse_event_data(line, event)
            if isinstance(stream.lookahead, StopIteration):
                break
            m = self.event_start_re.match(stream.lookahead)

        return n_bytes, event

    def parse_file(self, f: TextIO) -> pd.DataFrame:
        stream = LogStream(f)
        header = next(stream) # skip the header
        events = []

        size = os.fstat(f.fileno()).st_size - len(header)

        try:
            with tqdm.tqdm(total=size, unit="B", unit_scale=True) as pbar:
                while True:
                    n, e = self.parse_event(stream)
                    events.append(e)
                    pbar.update(n)
        except StopIteration:
            return pd.DataFrame.from_records(events)
        except BaseException:
            log.error("Error at line: %s", stream.cur)
            raise

def parse_audit_log(path: str, name: str) -> None:
    log.info("processing %s...", name)
    with open(path, "r") as f:
        state = ParseState()
        df = state.parse_file(f)

    out_path = f"{config.atlasv1.log_dataframes}/{name}.json.xz"
    df.to_json(path_or_buf=out_path)

def parse_atlasv1(base: str, name: str) -> None:
    # S. => single host (h1 only)
    # M. => both hosts h1 and h2
    if name.startswith("S"):
        out_name = f"h1-{name.lower()}"
        log = f"{base}/{name}/logs/security_events.txt"
        parse_audit_log(log, out_name)
    elif name.startswith("M"):
        for host in ("h1", "h2"):
            out_name = f"{host}-{name.lower()}"
            log = f"{base}/{name}/{host}/logs/security_events.txt"
            parse_audit_log(log, out_name)
    else:
        raise ValueError("Invalid name")

def main(argv):
    base = argv[1]
    for arg in argv[2:]:
        parse_atlasv1(base, arg)
