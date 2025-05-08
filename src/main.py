"""Run scripts from here. It's very important to pickle data from here, as
otherwise, the class references within the pickle may be wrong if `__main__`
defines the classes.

Note that it is possible to dispatch other module `main`s from here. To do so,
pass the module's name (as it would be imported) as the first argument, and
remaining arguments for that `main` afterwards."""

import importlib
import logging
import lzma
import pickle
import sys
from typing import List

import config
import util

log = logging.getLogger(__name__)

class FixRefUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "cluster":
            module = "neighborhood"
        return super().find_class(module, name)

def fix_pickle_refs(pickle_path: str) -> None:
    log.info("reading %s to fix...", pickle_path)
    with lzma.open(pickle_path, "rb") as f:
        unpickler = FixRefUnpickler(f)
        data = unpickler.load()

    log.info("writing fixed %s...", pickle_path)
    util.pickle_write(data, pickle_path)

def dispatch_module_main(argv: List[str]) -> None:
    """Use `argv[1]` to lookup a module and call its `main()` with the
    remaining arguments."""
    if len(argv) <= 1:
        sys.exit(f"usage: {argv[0]} MODULE_NAME ARGS...")

    m = importlib.import_module(argv[1])
    m.main(argv[1:])

def enable_postmortem():
    def excepthook(typ, value, tb):
        if hasattr(sys, "ps1") or not sys.stderr.isatty():
            # interactive mode, use default handler
            sys.__excepthook__(typ, value, tb)
        else:
            import traceback, pdb
            traceback.print_exception(typ, value, tb)
            print()
            pdb.post_mortem(tb)
    sys.excepthook = excepthook

def main(argv: List[str]) -> int:
    rc = dispatch_module_main(argv)
    #for path in argv[1:]:
    #    fix_pickle_refs(path)
    return 0 if rc is None else rc

if __name__ == "__main__":
    # Turn on post-mortem debugging (would be nice to have a usercustomize,
    # but it seems to be disabled, not sure why)
    enable_postmortem()
    sys.exit(main(sys.argv))
