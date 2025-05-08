"""Generate templates of names within a provenance graph. This allows
neighborhoods to group semantically equivalent behaviors (e.g.,
reading/writing to cache or temp files)."""

import logging
from typing import List, Optional

import config
import util
import parse.graph

log = logging.getLogger(__name__)

class Templatizer:
    path_threshold = 1

    def split(self, s: str) -> List[str]:
        """Split `s` according to `self.path_separator`. This removes empty
        components in the split and the final component. e.g.,
        "foo//bar/baz" and "foo/bar/baz" will both result in ["foo", "bar"];
        the former will not be ["foo", "", "bar", "baz"]."""
        return [x for x in s.split(self.path_separator)[:-1] if x]

    def __init__(self, names: List[str], path_separator: str = "/"):
        """Generate a templatizer based on `names` that can later match names
        to a particular template."""
        self.templates = []
        self.path_separator = path_separator
        log.info("generating templates...")
        # ignore case in templates
        names = [s.lower() for s in names]
        # sort names by number of separators, length, then alphabetically, ascending
        names.sort(key=lambda s: (s.count(path_separator), len(s), s))
        while names: # loop backwards (i.e., longest to shortest)
            template = names.pop()
            # handle these case-by-case instead of creating a template
            if template.startswith(parse.graph.unknown_prefix):
                continue

            # skip if this already belongs to a template
            if self.get_template(template) is not None:
                continue

            template_dirs = self.split(template)
            if len(template_dirs) == 0: # skip if not a path
                continue

            make_template = False
            diffs = set()
            buf = []
            for name in names:
                buf.append(name)
                # only worry about paths
                name_dirs = self.split(name)
                if len(name_dirs) == 0:
                    buf.pop()
                    continue
                if len(template_dirs) != len(name_dirs):
                    continue

                add = True
                cur_diffs = []
                for k, (t_dir, n_dir) in enumerate(zip(template_dirs, name_dirs)):
                    if t_dir != n_dir:
                        cur_diffs.append(k)
                        if len(cur_diffs) > self.path_threshold:
                            add = False
                            break

                if add:
                    buf.pop()
                    diffs.update(cur_diffs)
                    make_template = True
            names = buf

            if not make_template:
                continue

            name = template
            template = [None if i in diffs else d
                        for (i, d) in enumerate(template_dirs)]
            template_name = self.path_separator.join(
                    map(lambda s: "*" if s is None else s,
                        template)).rstrip()
            self.templates.append((template, template_name))
        log.info("finished generating %s templates", len(self.templates))

    def get_template(self, name: str) -> Optional[str]:
        """Return the template for `name`, or None if `name` doesn't fit into
        a known template."""
        if name.startswith(parse.graph.unknown_prefix):
            return name[:5]

        for (t, t_name) in self.templates:
            match = True
            s = self.split(name)
            if len(s) == len(t):
                for (t_dir, n_dir) in zip(t, s):
                    if t_dir is not None and t_dir != n_dir:
                        match = False
                        break
                if match:
                    return t_name

        return None

def main(argv):
    sep = "\\"
    set_name = "fivedirections"

    name_path = f"{config.e3.names}/{set_name}_prov_graph.txt"
    with open(name_path, "rt") as f:
        templatizer = Templatizer(f.readlines(), sep)

    out_path = f"{set_name}_templates.txt"
    with open(out_path, "wt") as f:
        for t in templatizer.templates:
            f.write(t[1])
            f.write("\n")
