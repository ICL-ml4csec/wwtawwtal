"""Try to sanitize process names, especially for Windows. Notable artifacts:

1. Remove path names entirely, as they tend to be noisy.
2. Use a consistent case (e.g., lower()), as cases tend to vary and Windows
   is case-insensitive. This could technically cause problems on *NIX, but in
   reality, case-sensitivity shouldn't be important.
3. Some paths are lacking separators (e.g., backslash). These seem to only be
   System32, so look for that particular pattern."""

import logging
import re

log = logging.getLogger(__name__)

def sanitize_name(name: str, sep: str) -> str:
    name = name.lower()
    if sep == "\\" and sep not in name:
        # seems to be a pathless Windows name
        # this won't work for a few specific cases, e.g.,
        # C:UsersadminAppDataLocalMicrosoftOneDriveOneDriveStandaloneUpdater.exe
        # First, try system32
        l = name.rsplit("system32", maxsplit=1)
        if len(l) > 1:
            return l[-1]

        # didn't contain system32; try looking for a UUID
        uuid_pat = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        l = re.split(uuid_pat, name)
        if len(l) > 1:
            return l[-1]

        # didn't contain a UUID; split on number
        # this can definitely fail (name might contain a number, or path might
        # not contain a number)
        l = re.split(r"[0-9]", name)
        if len(l) > 1:
            return l[-1]

        #log.warning("%s: no heuristics worked...", name)
        return name

    return name.rsplit(sep, maxsplit=1)[-1]
