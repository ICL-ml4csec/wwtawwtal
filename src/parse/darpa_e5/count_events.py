"""Count the total number of events in a particular dataset."""

import logging
from typing import Dict

import config
import util
from parse.darpa_e5 import avro_stream

log = logging.getLogger(__name__)

class EventCounter(avro_stream.AbstractParseState):
    def __init__(self, set_name: str):
        super().__init__(set_name)
        self.n_events = 0

    def parse_record(self, record: Dict):
        try:
            if record["type"] == "RECORD_EVENT":
                self.n_events += 1
        except:
            log.error("BAD RECORD: %s", record)

def count_dataset(data_dir: str, set_name: str):
    state = EventCounter(set_name)
    avro_stream.parse_dataset(data_dir, state)
    log.info("Total number of events: %s", state.n_events)

def main(argv):
    base = "<PATH_TO_DARPA_E5>"
    for set_name in ["cadets", "theia", "fivedirections", "trace"]:
        data_dir = f"{base}/{set_name}"
        count_dataset(data_dir, set_name)
