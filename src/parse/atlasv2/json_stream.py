"""Collect all of the ATLASv2 JSON logs into a single stream per host."""

import abc
import glob
import json
import logging
import os
from typing import NamedTuple

import pandas as pd
import tqdm
import util

log = logging.getLogger(__name__)

class AbstractParseState(abc.ABC):
    """Interface for JSON stream consumers."""
    def __init__(self, host: str):
        self.host = host

    @abc.abstractmethod
    def parse_record(self, record: NamedTuple) -> None:
        """Parse one JSON record (i.e., a row of the table)."""
        pass

    def set_current_logfile(self, path: str) -> None:
        """Called on each individual log file with the full log path."""
        pass

def parse_json(path, state: AbstractParseState) -> int:
    """Parse a JSON file at `path`."""
    log.info("reading %s...", path)
    df = pd.read_json(path)
    # rename columns to be NamedTuple-friendly
    df.rename(columns=lambda s: s.replace(" ", ""), inplace=True)
    # convert timestamps to ns
    df["SystemTime"] = df["SystemTime"].apply(lambda s:
            util.as_ns(util.from_iso8601(s)))
    # sort the timestamps to be ascending
    df.sort_values(by="SystemTime")
    for row in tqdm.tqdm(df.itertuples(),
                         total=df.shape[0], unit="events"):
        state.parse_record(row)
    return os.path.getsize(path)

def parse_json_set(base_path: str, host: str, state: AbstractParseState):
    files = glob.glob(f"{base_path}/*-{host}-*.json")
    files.sort()

    total_bytes = 0
    for path in files:
        state.set_current_logfile(path)
        total_bytes += parse_json(path, state)

    log.info("raw log size: %s", util.si_binary(total_bytes))
