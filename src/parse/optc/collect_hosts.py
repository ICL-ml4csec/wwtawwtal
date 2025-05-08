import gzip
import json
import pathlib
from typing import Dict, Optional, Set

import tqdm

def parse_line(line: str) -> Optional[Dict]:
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        log.exception(e)
        return None

def parse_file(path: pathlib.Path, hosts: Set[str]):
    with open(path, "rb") as gzf:
        with gzip.open(gzf, "rt") as f:
            with tqdm.tqdm(total=path.stat().st_size, desc=str(path),
                           unit="B", unit_scale=True, leave=False) as pbar:
                for line in iter(f.readline, ""):
                    data = parse_line(line)
                    if data is not None:
                        hosts.add(data["hostname"])
                    pbar.update(gzf.tell() - pbar.n)

def parse_all(basedir: str):
    basedir = pathlib.Path(basedir)
    hosts = set()
    for path in tqdm.tqdm(sorted(basedir.glob("**/*.json.gz"))):
        parse_file(path, hosts)
    with open("optc_hosts.txt", "wt") as f:
        for host in sorted(hosts):
            f.write(host)
            f.write("\n")

def main(argv):
    basedir = "<PATH_TO_OPTC>"
    parse_all(basedir)
