"""Reads the JSON outputs of DARPA-TC3 for arbitrary parsing. See the schema
for details."""

import abc
import glob
import json
import logging
import lzma
import tarfile
from typing import Dict, List, Optional, TextIO

import tqdm

import util

log = logging.getLogger(__name__)
avro_ns = "com.bbn.tc.schema.avro.cdm18"

class AbstractParseState(abc.ABC):
    """Interface for JSON stream consumers. Subclasses should run these
    superclass methods if overriden, as they handle details such as the
    FiveDirections dataset missing namespaces."""
    def __init__(self, set_name: str):
        self.set_name = set_name
        # for first FiveDirections archive, namespaces are missing. Start with
        # empty ns, then switch by using `next_archive`.
        if set_name == "fivedirections":
            self.ns = ""
        else:
            self.ns = f"{avro_ns}."

    @abc.abstractmethod
    def parse_record(self, record: Dict) -> None:
        """Parse one record of a JSON dataset."""
        pass

    def next_archive(self) -> None:
        """Called between opening multiple tarballs of one dataset. e.g.,
        <dataset>.tar.json -> next_archive -> <dataset>-2.tar.json -> ..."""
        if self.set_name == "fivedirections":
            self.ns = f"{avro_ns}."

def parse_line(line: str) -> Optional[Dict]:
    try:
        # for some reason, FiveDirections has trailing commas
        line = line.rstrip(b",\n")
        return json.loads(line)
    except json.JSONDecodeError as e:
        log.exception(e)
        return None

def parse_json(f: TextIO, size: int, state: AbstractParseState) -> None:
    """Parse one JSON file. `fn` takes one JSON dict and any extra `args` and
    `kwargs` passed into this function."""
    with tqdm.tqdm(total=size, unit="B", unit_scale=True) as pbar:
        for line in f:
            data = parse_line(line)
            if data is not None:
                state.parse_record(data)
            pbar.update(f.tell() - pbar.n)

def parse_json_tar(path: str, state: AbstractParseState) -> int:
    """Parse a JSON data tarball. These can contain multiple files for larger
    datasets (e.g., theia-1r). Return the number of bytes read."""
    total_bytes = 0
    with tarfile.open(path, "r") as tar:
        for info in tar:
            # use the order in the tarball. This seems to be chronological,
            # but even if it's not, it shouldn't affect the graph's
            # correctness (although the node dict order may be odd).
            total_bytes += info.size
            with tar.extractfile(info) as f:
                parse_json(f, info.size, state)
    return total_bytes

def parse_dataset(tar_dir: str, state: AbstractParseState) -> None:
    """Parse an entire DARPA TC dataset. `tar_dir` should be the directory
    containing `<dataset>.json.tar.gz` archives. `typ` controls whether this
    outputs an entire provenance graph, or just the process tree.

    The DARPA TC datasets all experienced various problems during operation,
    and were thus restarted at least once. Each restart is a new tarball with
    an incremented number.

    Since each dataset is supposed to represent a single host, in reality, the
    log should be concatenated. Therefore, this concatenates all the logs
    together."""
    suffix = ".json.tar.gz"
    files = glob.glob(f"{tar_dir}/*{suffix}")
    files.sort(key=lambda s: s[:-len(suffix)])
    log.info("reading these files: %s", files)

    total_bytes = 0
    for f in files:
        total_bytes += parse_json_tar(f, state)
        state.next_archive()

    log.info("raw log size: %s", util.si_binary(total_bytes))
