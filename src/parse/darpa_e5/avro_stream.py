"""Read avro binary files as a stream of records. This is currently somewhat
rudimentary, and doesn't handle things like sorting the files very well."""

import abc
import glob
import logging
import lzma
import os
from typing import Dict

import fastavro
import tqdm

import config
import util

log = logging.getLogger(__name__)

class AbstractParseState(abc.ABC):
    def __init__(self, set_name: str):
        self.set_name = set_name

    @abc.abstractmethod
    def parse_record(self, record: Dict) -> None:
        pass

def parse_avro_xz(path: str, state: AbstractParseState) -> None:
    """Parse the xz-compressed avro file at `path`."""
    with lzma.open(path, "rb") as f:
        # don't bother using fd.tell(), it reports the compressed size. Just
        # use `xz -l` on all the files!
        #fd = os.fdopen(f.fileno(), closefd=False)
        for record in fastavro.reader(f, handle_unicode_errors="replace"):
            state.parse_record(record)
        #return fd.tell()

def parse_dataset(data_dir: str, state: AbstractParseState) -> None:
    """Parse an entire E5 dataset. `data_dir` should contain all the
    xz-compressed avro binary files."""
    files = glob.glob(f"{data_dir}/*.bin*.xz")
    # don't bother sorting for now. natsort is 99% of the way there, but has
    # the non-numbered file last when it should be first

    log.info("reading avro logs from %s...", data_dir)
    pbar = tqdm.tqdm(files)
    for path in pbar:
        pbar.set_description(path)
        parse_avro_xz(path, state)

    log.info("finished.")
