# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2025 Wolfram Tuschewitzki, Jelto Lange (modifications and additions)
# SPDX-FileCopyrightText: 2012-2023 Michael L. Waskom (portions derived from seaborn)
#
# SPDX-License-Identifier: MIT AND BSD-3-Clause
"""
Beeswarm overlays for the split violin figures of plots.py.

seaborn's swarmplot packs points symmetrically around the centre of a band.
OneSidedBeeswarm only uses one side, so the swarm sits on its violin half.

Written for seaborn 0.13.2 (pinned in src/environment.yaml), re-check the two
overridden methods for other versions.


Third-party attribution
-----------------------
`OneSidedBeeswarm` subclasses `seaborn.categorical.Beeswarm`, and its
`position_candidates` and `add_gutters` methods are modified versions of
the corresponding seaborn methods (seaborn 0.13.2). seaborn is
Copyright (c) 2012-2023, Michael L. Waskom, all rights reserved, and is
licensed under the BSD 3-Clause "New" or "Revised" License. The full
license text is kept in `LICENSES/seaborn-BSD-3-Clause.md`;
Upstream source: https://github.com/mwaskom/seaborn
"""

import matplotlib.colors as mcolors
import numpy as np
from seaborn.categorical import Beeswarm

# Shared marker style --> similar dot appearance in every figure
SIZE_RANGE = (0.9, 3.4)  # marker size when scaled by a nodal value
SIZE_CLIP_PERCENTILES = (5, 95)
CONSTANT_SIZE = 2.2  # marker size where every point counts the same
EDGE_COLOR = "white"
EDGE_WIDTH = 0.15
GAP_PT = 0.4  # clearance from the split line, on top of one marker radius
WIDTH_FRAC = 0.95  # swarm band as a fraction of the violin half band
VIOLIN_ALPHA = 0.75  # violins are lightened so the dots read on top


def sizes_from_values(
    values, size_range=SIZE_RANGE, clip_percentiles=SIZE_CLIP_PERCENTILES
):
    """
    Map nodal values onto marker sizes, log scaled and clipped at percentiles.

    One mapping for the whole figure, so a dot of a given size always
    means the same value across all series and cases.
    """
    log_values = np.log10(np.asarray(values, dtype=float))
    low, high = np.nanpercentile(log_values, clip_percentiles)
    fraction = np.clip((log_values - low) / (high - low), 0, 1)
    return size_range[0] + fraction * (size_range[1] - size_range[0])


def darken(color, factor=0.55):
    """
    Darker shade of `color`, for dots sitting on a violin of the same hue.
    """
    return tuple(channel * factor for channel in mcolors.to_rgb(color))


def violin_half_band(positions, violin_width=1.0):
    """
    Half width of one dodged split violin, in data units.

    seaborn takes the native width from the smallest gap between the
    plotted positions; the half violin then spans width * native_width /
    2.
    """
    unique = np.unique(np.asarray(positions, dtype=float))
    native_width = np.min(np.diff(unique)) if unique.size > 1 else 1.0
    return violin_width * native_width / 2


class OneSidedBeeswarm(Beeswarm):
    """
    Beeswarm that keeps every point on one side of the split line.
    """

    def __init__(self, data_sign, gap_pt=GAP_PT, **kwargs):
        super().__init__(**kwargs)
        self.data_sign = data_sign
        self.display_sign = data_sign
        self.gap_pt = gap_pt

    def __call__(self, points, center):
        ax = points.axes
        # the scenario axis is inverted in the cost variants, so the direction
        # the swarm has to grow in display space is read off the transform
        low, high = (self._to_display(ax, value) for value in (0, 1))
        self.display_sign = self.data_sign * (1 if high >= low else -1)
        super().__call__(points, center + self._clearance(points, ax))

    def _to_display(self, ax, value):
        point = (0, value) if self.orient == "y" else (value, 0)
        return ax.transData.transform(point)[1 if self.orient == "y" else 0]

    def _clearance(self, points, ax):
        """
        Offset that keeps markers clear of the split line, in data units.
        """
        edge = np.max(points.get_linewidth())
        radius_pt = (np.sqrt(points.get_sizes()).max() + edge) / 2
        pixels = (radius_pt + self.gap_pt) * ax.figure.dpi / 72
        inverted = ax.transData.inverted()
        offset = (0, pixels) if self.orient == "y" else (pixels, 0)
        index = 1 if self.orient == "y" else 0
        span = abs(
            inverted.transform(offset)[index] - inverted.transform((0, 0))[index]
        )
        return self.data_sign * span

    def position_candidates(self, xyr_i, neighbors):
        # modified from seaborn.categorical.Beeswarm.position_candidates
        # (BSD-3-Clause, see module docstring): only the candidate on the
        # requested side is kept, seaborn alternates left and right
        candidates = [xyr_i]  # the split line itself
        _, y_i, r_i = xyr_i
        for x_j, y_j, r_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max((r_i + r_j) ** 2 - dy**2, 0)) * 1.05
            candidates.append((x_j + self.display_sign * dx, y_i, r_i))
        return np.array(candidates)

    def add_gutters(self, points, center, trans_fwd, trans_inv):
        # modified from seaborn.categorical.Beeswarm.add_gutters
        # (BSD-3-Clause, see module docstring): the band runs from the split
        # line to one side instead of half a width around the centre
        edges = sorted(
            [trans_fwd(center), trans_fwd(center) + self.data_sign * self.width]
        )
        low, high = (trans_inv(edge) for edge in edges)
        # points left on the split line come back a float ulp off
        tol = abs(high - low) * 1e-9
        points[points < low - tol] = low
        points[points > high + tol] = high
        return points


def add_beeswarm(
    ax,
    center,
    values,
    colors,
    size,
    width,
    data_sign=None,
    orient="x",
    edge_color=EDGE_COLOR,
    edge_width=EDGE_WIDTH,
    gap_pt=GAP_PT,
    zorder=3,
):
    """
    Scatter `values` at `center` and beeswarm them at every draw.

    `orient` names the categorical axis as in seaborn. Without `data_sign` the
    swarm is symmetric around `center`, with -1 / +1 it stays on one side.
    """
    values = np.asarray(values, dtype=float)
    centers = np.full(len(values), float(center))
    x_data, y_data = (values, centers) if orient == "y" else (centers, values)

    points = ax.scatter(
        x_data,
        y_data,
        s=np.asarray(size, dtype=float) ** 2,
        c=colors,
        edgecolors=edge_color,
        linewidths=edge_width,
        zorder=zorder,
    )
    if data_sign is None:
        beeswarm = Beeswarm(width=width, orient=orient, warn_thresh=1.0)
    else:
        beeswarm = OneSidedBeeswarm(
            data_sign, gap_pt=gap_pt, width=width, orient=orient, warn_thresh=1.0
        )

    # Beeswarm works in display space, so it runs at every draw
    def draw(self, renderer, *, center=float(center)):
        beeswarm(self, center)
        super(self.__class__, self).draw(renderer)

    points.draw = draw.__get__(points)
    return points
