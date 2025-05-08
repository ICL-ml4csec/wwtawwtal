"""Reads the ecar outputs of OpTC for arbitrary parsing. See the schema for
details.

This can only produce graphs for one host at a time. Technically, because each
log file contains 25 different hosts, we could do this 25x more efficiently if
you wanted all 25 graphs by constructing all graphs simultaneously. I mostly
didn't support this for simplicity, although memory constraints would be a
concern in that scenario as well.

Note that the OpTC data doesn't actually have all 1000 hosts; certain 25-host
ranges are missing. Within each log, each host is named
`SysClientNNNN.systemia.com`."""

import abc
import gzip
import json
import logging
import lzma
import pathlib
from typing import Dict, List, Optional, TextIO

import tqdm

import util

log = logging.getLogger(__name__)

class AbstractParseState(abc.ABC):
    """Interface for JSON stream consumers."""
    def __init__(self, host_nbr: int):
        self.host_nbr = host_nbr
        self.hostname = f"SysClient{host_nbr:04d}.systemia.com"

        # determine the 25-day interval in which `host_nbr` falls
        # note that in the modular ring, intervals start at 1 and end at 0 (!)
        interval = (host_nbr - 1) // 25
        lo = (interval * 25) + 1
        hi = (interval + 1) * 25
        self.logdir = f"AIA-{lo}-{hi}"

    @abc.abstractmethod
    def parse_record(self, record: Dict) -> None:
        """Parse one record of a JSON dataset."""
        pass

    def check_host(self, record: Dict) -> bool:
        """Return True iff `record["hostname"]` corresponds to
        `self.hostname`."""
        return record["hostname"] == self.hostname

def parse_line(line: str) -> Optional[Dict]:
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        log.exception(e)
        return None

def parse_file(path: pathlib.Path, state: AbstractParseState) -> None:
    """Parse an individual json.gz log file."""
    with open(path, "rb") as gzf:
        with gzip.open(gzf, "rt") as f:
            with tqdm.tqdm(total=path.stat().st_size, desc=str(path),
                           unit="B", unit_scale=True) as pbar:
                for line in iter(f.readline, ""):
                    data = parse_line(line)
                    if data is not None and state.check_host(data):
                        state.parse_record(data)
                    pbar.update(gzf.tell() - pbar.n)

def parse_day(path: pathlib.Path, state: AbstractParseState) -> None:
    """Given the path to a particular day (e.g., ecar/benign/18-19Sep19),
    determine the path containing `state`'s host, and parse if it exists."""
    path /= state.logdir
    if not path.is_dir():
        #log.warning("No directory for host %s in %s", state.host_nbr, path)
        return
    # natural sort works, because the files are either timestamp-ordered or
    # named 'last', but 'l' comes after any number in the default
    files = sorted(path.glob("*.json.gz"))
    for path in files:
        parse_file(path, state)

def parse_host(state: AbstractParseState,
               basedir: str):
    """Parse the graph for a specific host (which is specified by `state`)."""
    # NOTE: we skip the "short" logs because they are missing data
    basedir = pathlib.Path(basedir)
    paths = sorted(basedir.glob("benign/*"))
    paths.extend(sorted(basedir.glob("evaluation/*")))
    for path in paths:
        parse_day(path, state)

def parse_file_batch(path: pathlib.Path, states: dict) -> None:
    """Parse an individual json.gz log file."""
    with open(path, "rb") as gzf:
        with gzip.open(gzf, "rt") as f:
            with tqdm.tqdm(total=path.stat().st_size, desc=str(path),
                           unit="B", unit_scale=True) as pbar:
                for line in iter(f.readline, ""):
                    data = parse_line(line)
                    if data is not None:
                        host = int(data["hostname"][9:13])
                        states[host].parse_record(data)
                    pbar.update(gzf.tell() - pbar.n)

def parse_day_batch(path: pathlib.Path, logdir: str, states: dict) -> None:
    """Given the path to a particular day (e.g., ecar/benign/18-19Sep19),
    parse the data for the batch's hosts."""
    path /= logdir
    if not path.is_dir():
        #log.warning("No directory for host %s in %s", state.host_nbr, path)
        return
    # natural sort works, because the files are either timestamp-ordered or
    # named 'last', but 'l' comes after any number in the default
    files = sorted(path.glob("*.json.gz"))
    for path in files:
        parse_file_batch(path, states)

def parse_batch(states: Dict[int, AbstractParseState], logdir: str,
                basedir: str = "<PATH_TO_OPTC>"):
    """Parse the graph for a range of hosts that are all contained within one
    specific log directory, i.e., 25-host range."""
    basedir = pathlib.Path(basedir)
    paths = sorted(basedir.glob("benign/*"))
    paths.extend(sorted(basedir.glob("evaluation/*")))
    for path in paths:
        parse_day_batch(path, logdir, states)
