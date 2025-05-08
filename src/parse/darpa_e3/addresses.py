"""Collect all IP addresses in a dataset."""

from collections import defaultdict
import sys
from typing import Dict, List

import config
from parse.darpa_e3 import json_stream
import util

class ParseState(json_stream.AbstractParseState):
    def __init__(self, set_name: str):
        super().__init__(set_name)
        self.addresses = {}

    def parse_record(self, record: Dict) -> None:
        try:
            v = record["datum"][f"{self.ns}NetFlowObject"]
            addr = f"{v['remoteAddress']}:{v['remotePort']}"
            if addr not in self.addresses:
                self.addresses[addr] = 0
            self.addresses[addr] += 1
        except KeyError:
            pass

def write_addresses(tar_dir: str) -> None:
    set_name = util.tc_set_name(tar_dir)
    addresses_path = f"{config.e3.addresses}/{set_name}.txt"

    state = ParseState(set_name)
    json_stream.parse_dataset(tar_dir, state)

    with open(addresses_path, "w") as f:
        for addr, count in state.addresses.items():
            addr, port = addr.split(":")
            f.write(f"{addr} {port} {count}\n")

def main(argv: List[str]) -> None:
    for path in argv[1:]:
        write_addresses(path)
