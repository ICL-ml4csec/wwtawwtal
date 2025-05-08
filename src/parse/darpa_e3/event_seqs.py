"""Templatize the TC CDM data as many log parsers would. Automatic
templatization with tools such as Drain often results in imperfect templates;
for example, running Spell on the `logfmt` logs results in most logs being
given the same template because it didn't split each syscall type into a key.
Furthermore, Spell often overfits on common parameters (some UUIDs end up
within templates), which may or may not make sense.

Rather than explicitly outputting the templates, we only need sequences of log
events here. They are split up into time windows to match DeepLog's
expectations, but are also divided into a benign file and attack file. The
files also have timestamps included per window in case separation of different
attacks or behaviors is needed.

For templates, we give sockets, files, and registry keys their own event IDs,
and every syscall their own event ID. There is no further division based on
dataflow."""

import enum
import logging
import lzma
from typing import Dict, List, TextIO

import config
from parse import events
from parse.darpa_e3 import json_stream
from parse.darpa_e3.label_events import in_attack_period
import util

log = logging.getLogger(__name__)

_start = max(events.Event).value + 2
# inherit existing Events (syscalls), but start from 1 instead of 0
EventTemplate = enum.IntEnum("EventTemplates",
    [(e.name, e.value + 1) for e in events.Event] +
    [("SOCKET", _start),
     ("FILE", _start + 1),
     ("REGISTRY", _start + 2),
    ])

# only do Events, NetFlowObject (sockets), FileObject, RegistryKeyObject
#
# Note that only Events have timestamps. The others just steal the timestamp
# of the last seen Event.
class ParseState(json_stream.AbstractParseState):
    def __init__(self, set_name: str, attack_f: TextIO, benign_f: TextIO):
        super().__init__(set_name)
        self.benign_f = benign_f
        self.attack_f = attack_f
        # to handle this more accurately, defer non-events to after the next
        # Event (which they actually correlate to). For example, socket
        # objects appear before the first sendto, which doesn't make sense.
        self.deferred = []

        self.time = 1522540800000000000 # April 1, 2018, GMT (failsafe)
        self.period_start = self.time
        self.period_end = self.period_start + config.event_seq_period
        self.attack_seq = []
        self.benign_seq = []

    def add_event(self, template: int) -> None:
        """If an event is within the current period, add it to the sequence.
        Otherwise, write the sequence and start a new one with the event."""
        assert self.time >= self.period_start # shouldn't ever go backwards
        def _add(template):
            if in_attack_period(self.time, self.set_name):
                self.attack_seq.append(template)
            else:
                self.benign_seq.append(template)

        if self.time <= self.period_end:
            _add(template)
            return

        # start a new period
        self.period_start = self.time // config.event_seq_period * config.event_seq_period
        self.period_end = self.period_start + config.event_seq_period
        self.flush_sequence()
        _add(template)

    def flush_sequence(self) -> None:
        """Write the current sequence (if any) and start a new, empty one."""
        def _write(f, seq):
            if seq:
                f.write(f"{self.period_start} ")
                f.write(" ".join(map(str, seq)) + "\n")

        _write(self.attack_f, self.attack_seq)
        self.attack_seq = []
        _write(self.benign_f, self.benign_seq)
        self.benign_seq = []

    def add_deferred_records(self) -> None:
        """Add any deferred Object records."""
        for template in self.deferred:
            self.add_event(template)
        self.deferred = []

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, *_) -> None:
        self.add_deferred_records()
        self.flush_sequence()

    def parse_record(self, record: Dict) -> None:
        for typ, d in record["datum"].items():
            if typ == f"{self.ns}NetFlowObject":
                self.deferred.append(EventTemplate.SOCKET.value)
            elif typ == f"{self.ns}FileObject":
                self.deferred.append(EventTemplate.FILE.value)
            elif typ == f"{self.ns}RegistryKeyObject":
                self.deferred.append(EventTemplate.REGISTRY.value)
            elif typ == f"{self.ns}Event":
                time = d["timestampNanos"]
                # there are occasionally times that are 0, or that seem to go
                # "backwards"; ignore them (assume same as current timestamp)
                # this actually happens with very high frequency for
                # FiveDirections (usually around a delta <= 10s)
                if time >= self.time:
                    self.time = time
                #else:
                #    log.warning("ignoring backwards time; t=%d s, prev=%d s, delta=%d s",
                #                time / 1e9, self.time / 1e9, (self.time - time) / 1e9)
                prefix = "EVENT_"
                try:
                    template = EventTemplate[d["type"][len(prefix):]]
                except KeyError:
                    template = EventTemplate.OTHER

                self.add_event(template.value)

                # add the deferred object records, if any
                self.add_deferred_records()
            else:
                continue

            return # successful match, no need to keep looping

def write_event_seqs(tar_dir: str) -> None:
    set_name = util.tc_set_name(tar_dir)
    base_path = f"{config.e3.event_seqs}/{set_name}"
    attack_path = f"{base_path}_attack.txt.xz"
    benign_path = f"{base_path}_benign.txt.xz"

    if set_name == "clearscope":
        log.warning("ClearScope timestamps are bugged; giving up...")
        return

    if util.skip_overwrite(attack_path) or util.skip_overwrite(benign_path):
        return

    log.info("writing event ID sequences for %s...", set_name)
    with lzma.open(attack_path, "wt") as attack_f:
        with lzma.open(benign_path, "wt") as benign_f:
            with ParseState(set_name, attack_f, benign_f) as state:
                json_stream.parse_dataset(tar_dir, state)
    log.info("finished %s", set_name)

def main(argv: List[str]) -> None:
    for path in argv[1:]:
        write_event_seqs(path)
