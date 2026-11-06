"""Physical density-run labels and source-record grouping conventions."""

import re
import numpy as np


def recording_group(name):
    return re.sub(r"_sub\d+.*$", "", str(name))


def physical_layers(profile):
    edges = np.r_[0, np.flatnonzero(np.diff(profile)) + 1, len(profile)]
    return [
        (int(a), int(b), float(profile[a]))
        for a, b in zip(edges[:-1], edges[1:])
        if profile[a] > 0.05
    ]
