import logging
import lzma

import config
from parse.atlasv2.events import ParseState
from parse.atlasv1 import json_stream
import util

log = logging.getLogger(__name__)

def write_events(host: str) -> None:
    events_path = f"{config.atlasv1.events}/{host}.xz"

    if util.skip_overwrite(events_path):
        return

    log.info("reading events of %s...", host)
    with lzma.open(events_path, "wb") as f:
        with ParseState(host, f) as state:
            json_stream.parse_json_set(host, state)
            log.info("finished %s; total events: %d.", host, state.writer.n)

def main(argv) -> None:
    for arg in argv[1:]:
        write_events(arg)
