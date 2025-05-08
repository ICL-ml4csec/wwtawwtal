"""Count the number of raw ATLASv1 security events.

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

    def __init__(self):
        pass

    def parse_event(self, stream: LogStream) -> int:
        """Parse one event from the log line stream, or raise StopIteration if
        there are no events left. New audit events are demarcated by either
        "Audit Success" or "Audit Failure", so use those as landmarks.

        Return the number of characters read. (The number of characters is not
        necessarily the number of bytes, so the progress bar will be slightly
        inaccurate. However, there doesn't seem to be any good way to resolve
        this aside from reading in binary mode, which comes with its own
        issues.)"""
        event = {}
        line = next(stream)
        n_bytes = len(line)
        assert self.event_start_re.match(line)

        # read until the next event
        m = self.event_start_re.match(stream.lookahead)
        while not m:
            line = next(stream)
            n_bytes += len(line)
            if isinstance(stream.lookahead, StopIteration):
                break
            m = self.event_start_re.match(stream.lookahead)

        return n_bytes

    def parse_file(self, f: TextIO) -> pd.DataFrame:
        stream = LogStream(f)
        header = next(stream) # skip the header

        size = os.fstat(f.fileno()).st_size - len(header)

        n_events = 0
        try:
            with tqdm.tqdm(total=size, unit="B", unit_scale=True) as pbar:
                while True:
                    n = self.parse_event(stream)
                    n_events += 1
                    pbar.update(n)
        except StopIteration:
            return n_events
        except BaseException:
            log.error("Error at line: %s", stream.cur)
            raise

def parse_audit_log(path: str, name: str) -> int:
    log.info("counting events from %s...", name)
    with open(path, "r") as f:
        state = ParseState()
        return state.parse_file(f)

def parse_atlasv1(base: str, name: str) -> int:
    # S. => single host (h1 only)
    # M. => both hosts h1 and h2
    if name.startswith("S"):
        out_name = f"h1-{name.lower()}"
        log = f"{base}/{name}/logs/security_events.txt"
        return parse_audit_log(log, out_name)
    elif name.startswith("M"):
        n_events = 0
        for host in ("h1", "h2"):
            out_name = f"{host}-{name.lower()}"
            log = f"{base}/{name}/{host}/logs/security_events.txt"
            n_events += parse_audit_log(log, out_name)
        return n_events
    else:
        raise ValueError("Invalid name")

def main(argv):
    n_events = 0

    base = argv[1]
    for arg in argv[2:]:
        n_events += parse_atlasv1(base, arg)

    log.info("ATLASv1: total events = %s", n_events)
