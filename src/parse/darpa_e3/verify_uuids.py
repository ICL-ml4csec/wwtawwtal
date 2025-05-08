"""Check if any duplicate UUIDs exist in a set."""

from collections import defaultdict
import logging
from typing import Dict, List

import config
from parse.darpa_e3 import json_stream
import util

log = logging.getLogger(__name__)

class ParseState(json_stream.AbstractParseState):
    def __init__(self, set_name: str):
        super().__init__(set_name)
        self.uuids = {}
        self.duplicate_pairs = defaultdict(lambda: 0)

    def parse_record(self, record: Dict) -> None:
        try:
            for k, v in record["datum"].items():
                uuid = v["uuid"]
                if uuid in self.uuids:
                    pair = tuple(sorted((k, self.uuids[uuid])))
                    self.duplicate_pairs[(k, self.uuids[uuid])] += 1
                else:
                    self.uuids[uuid] = k
        except KeyError:
            pass

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, *_) -> None:
        for (t1, t2), n in self.duplicate_pairs.items():
            log.warning("%s duplicates of types %s and %s", n, t1, t2)

def verify_uuids(tar_dir: str) -> None:
    set_name = util.tc_set_name(tar_dir)

    with ParseState(set_name) as state:
        json_stream.parse_dataset(tar_dir, state)

def main(argv: List[str]) -> None:
    for path in argv[1:]:
        verify_uuids(path)
