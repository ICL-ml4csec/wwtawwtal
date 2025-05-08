import glob
import logging

import config
import parse.atlasv2.json_stream as v2
import util

log = logging.getLogger(__name__)

def parse_json_set(host: str, state: v2.AbstractParseState):
    files = glob.glob(f"{config.atlasv1.log_dataframes}/{host}-*.json.xz")
    files.sort()

    total_bytes = 0
    for path in files:
        state.set_current_logfile(path)
        total_bytes += v2.parse_json(path, state)

    log.info("raw log size: %s", util.si_binary(total_bytes))
