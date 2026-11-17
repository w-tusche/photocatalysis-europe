# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2025 Wolfram Tuschewitzki, Jelto Lange
#
# SPDX-License-Identifier: MIT
import copy
import os

import cartopy.crs as ccrs
import cmcrameri
import geopandas as gpd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from matplotlib.font_manager import FontProperties

from cases import REPO_ROOT, find_network
from plots import COLORS_YAML

# ---------------------------------------------------------------------------
# Scenarios this figure is built from.
# The base sweep: 10 % STH, UGHS enabled, photocatalysis CAPEX in EUR/kW.
# 950 is the PC-0 case, 660 the PC-50 case.
SCENARIO_TEMPLATE = "150_lv1.25_I_H_2045_PC_{capex}Euro"
BASE_PC_CAPEX = (660, 700, 750, 800, 850, 900, 925, 950)
BASE_SCENARIOS = {
    capex: SCENARIO_TEMPLATE.format(capex=capex) for capex in BASE_PC_CAPEX
}
PC_50_SCENARIO = BASE_SCENARIOS[660]
# CAPEX token of the PC-50 case, used to key the `el_<capex>` / `pc_<capex>`
# columns of the joined capacity table (see `dominant_technology`).
PC_50_CAPEX = 660


def variation_scenario(capex, variation):
    """
    Name of a variation run of the base scenario at `capex`, e.g.
    `..._660Euro_noUGHS`.
    """
    if not variation.startswith("_"):
        variation = "_" + variation
    return BASE_SCENARIOS[capex] + variation


# The same sweep without underground hydrogen storage; only these six exist.
NO_UGHS_PC_CAPEX = (660, 700, 750, 800, 850, 900)
NO_UGHS_SCENARIOS = {
    capex: variation_scenario(capex, "noUGHS") for capex in NO_UGHS_PC_CAPEX
}

# Cost point whose electrolysis capacities rank the nodes shown on the
# electrolysis side of the figure: the PC-0 case, the counterpart of the PC-50
# photocatalysis set.
EL_RANKING_CAPEX = 950

# Photocatalysis nodes left out because their bars would be invisible
EXCLUDED_PC_NODES = ()

BUS_SIZES_DIR = os.path.join(REPO_ROOT, "data", "bus_sizes")
TOTAL_SUMMARY_FILE = os.path.join(REPO_ROOT, "results", "total_summary.csv")
REGIONS_FILE = os.path.join(REPO_ROOT, "data", "regions_onshore_elec_s_150.geojson")
SALT_CAVERN_FILE = os.path.join(REPO_ROOT, "data", "salt_cavern_potentials_s_150.csv")


def create_bus_df_for_selected_buses(bus_sizes_list, selected_buses=None):
    bus_sizes_list_ = copy.deepcopy(bus_sizes_list)
    df = pd.DataFrame()
    for data_series in bus_sizes_list_:
        series = data_series[1]
        if selected_buses:
            series = series.loc[list(selected_buses)]
        # Rename series to capex val
        series.name = data_series[0]
        # Get rid of multi index and only keep the location
        series.index = [x[0] for x in series.index]
        df = pd.concat([df, series], axis=1)
    return df


def order_regions_north_to_south(regions, names):
    """
    Order region names by the latitude of their centroid.
    """
    return list(regions.loc[list(names)].centroid.y.sort_values().index)


def extract_buses_for_technology(technology, case):
    """
    Read the per-node capacities of one technology for a named cost sweep.

    Args:
        technology (str): carrier to extract
        case (str): "ughs" for the base sweep (BASE_SCENARIOS), "aghs" for the
            same sweep without underground hydrogen storage (NO_UGHS_SCENARIOS)

    Returns:
        list: (pc_capex, buses pd.Series) per scenario, ascending in CAPEX
    """
    if technology not in [
        "photocatalysis",
        "H2 Electrolysis",
        "H2 Fuel Cell",
        "H2 turbine",
    ]:
        raise ValueError(
            "Select valid case ['photocatalysis', 'H2 Electrolysis', 'H2 Fuel Cell	', 'H2 turbine']"
        )

    if case == "ughs":
        scenarios = BASE_SCENARIOS
    elif case == "aghs":
        scenarios = NO_UGHS_SCENARIOS
    else:
        raise ValueError("Select valid case (ughs, aghs)")

    # Store (pc_capex, buses pd.Series)
    bus_sizes_list = []
    for capex, scenario in sorted(scenarios.items()):
        file_path = os.path.join(BUS_SIZES_DIR, f"bus_sizes_{scenario}.csv")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Missing bus size export for scenario {scenario!r}: {file_path}"
            )
        bus_sizes = pd.read_csv(file_path, index_col=[0, 1])
        bus_sizes = pd.Series(bus_sizes["p_nom_opt"], index=bus_sizes.index)
        # Extract the bus data
        buses = bus_sizes[(bus_sizes.index.get_level_values(1) == technology)]
        bus_sizes_list.append((capex, buses))
    return bus_sizes_list


# Function to plot capacity data
def plot_capacity_data_both(
    ax_pc, ax_el, df, colors_pc, colors_el, max_limits, label_map
):
    """
    Args:
        ax (ax): ax on which the wind plot should be shown
        df (df): df which contains the data
        colors (list): colors for bars
    """
    bar_width = 0.7
    index = range(len(df))
    y_offset = np.zeros(len(df))

    # Add all bars for photocatalysis for each of the index values
    # Iterate through years in reverse order
    for i, capex in enumerate(str(c) for c in BASE_PC_CAPEX):
        pc_vals = df[f"pc_{capex}"]

        # Plot Wind capacity bars with proper offsets to stack them behind each other
        ax_pc.barh(
            index,
            pc_vals,
            bar_width,
            left=y_offset,
            color=colors_pc[i],
            edgecolor="black",
            linewidth=0.3,
            label=str(label_map[capex]),
        )

    # Flag to add legend entry only once
    legend_entry = False
    # Add max limits for photocatalysis (max installation at certain destination)
    # for every node that has a photocatalysis bar
    pc_columns = [f"pc_{capex}" for capex in BASE_PC_CAPEX]
    for idx, node in enumerate(df.index):
        if df.loc[node, pc_columns].notna().any() and node in max_limits.index:
            # To add legend entry only once
            if not legend_entry:
                ax_pc.plot(
                    [max_limits[node], max_limits[node]],
                    [idx - bar_width / 1.8, idx + bar_width / 1.8],
                    color="red",
                    linewidth=2,
                    label="Max. space",
                )
                legend_entry = True
            else:
                ax_pc.plot(
                    [max_limits[node], max_limits[node]],
                    [idx - bar_width / 1.8, idx + bar_width / 1.8],
                    color="red",
                    linewidth=2,
                )

    # Add all bars for electrolysis for each of the index values
    for i, capex in enumerate(str(c) for c in reversed(BASE_PC_CAPEX)):
        el_vals = df[f"el_{capex}"]

        # Plot Wind capacity bars with proper offsets to stack them behind each other
        ax_el.barh(
            index,
            el_vals,
            bar_width,
            left=y_offset,
            color=colors_el[i],
            edgecolor="black",
            linewidth=0.3,
            label=str(label_map[capex]),
        )

    # Adjusting labels and ticks for subplot
    for ax in [ax_el, ax_pc]:
        ax.set_yticks(index)
        ax.set_yticklabels(df.index)
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_axisbelow(True)
        ax.grid(alpha=0.5)


def add_stacking_legend(
    ax,
    colors_pc,
    colors_el,
    y_label_adjustment=0,
    title_adjustment=0,
    font_weight=None,
    fontsize=None,
    title_fontsize=None,
    x_margin=0.2,
):
    """
    x_margin: free space left and right of the rectangles, in their length
    units; 0 if a frame around the axes pads them.
    """
    # Initial dimensions and increments
    init_heigth = 0.4
    init_length = 1
    heigth_increment = 0.13
    length_increment = 0.3

    # Number of rectangles: one per capex point actually plotted, so this
    # stays correct if BASE_PC_CAPEX ever has a different length than 8.
    num_rectangles = len(colors_pc)

    # Create rectangles from largest to smallest so that smaller ones appear on top
    for direction in range(2):
        if direction == 0:
            dir = 1
            colors_to_use = colors_pc[::-1]
            text_label = "Photocatalysis"
            ha = "right"
        else:
            dir = -1
            colors_to_use = colors_el[::-1]
            text_label = "Electrolysis"
            ha = "left"

        for idx in range(num_rectangles - 1, -1, -1):
            # Calculate current width and length
            current_heigth = init_heigth + idx * heigth_increment
            current_length = (init_length + idx * length_increment) * dir

            # Create a rectangle patch filled with color
            # not clipped, the outlines on the axes edges stay whole
            rect = patches.Rectangle(
                (0.1 * dir, 0),
                current_length,
                current_heigth,
                edgecolor="black",
                facecolor=colors_to_use[idx],
                clip_on=False,
            )

            # Add the rectangle to the axes
            ax.add_patch(rect)

        ax.text(
            (init_length + (num_rectangles - 1) * length_increment) * dir,
            1.4 + y_label_adjustment,
            text_label,
            fontsize=fontsize,
            horizontalalignment=ha,
            verticalalignment="center",
            color="black",
            weight=font_weight,
        )

    title = ax.set_title(
        "Bar stacking visualization",
        fontdict=dict(weight=font_weight),
        pad=-5 + title_adjustment,
    )
    # None keeps the axes title size
    if title_fontsize is not None:
        title.set_fontsize(title_fontsize)

    # # Set limits for x and y axis to accommodate all rectangles
    reach = 0.1 + init_length + (num_rectangles - 1) * length_increment
    ax.set_xlim(-reach - x_margin, reach + x_margin)
    ax.set_ylim(0, 1.5)
    # ax_overlay_stacked.set_ylim(0, 1.4)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Turn of the spines around the axis
    ax.spines[:].set_visible(False)
    ax.patch.set_edgecolor("black")

    return ax


def prepare_bar_plot_data():
    # Only the photocatalysis potentials (`p_nom_max`) are read from this
    # network; they are the same in every run of the base sweep.
    n = pypsa.Network(find_network(BASE_SCENARIOS[950]))

    regions = gpd.read_file(REGIONS_FILE).set_index("name")

    bus_sizes_list_pc = extract_buses_for_technology("photocatalysis", "ughs")
    bus_sizes_list_el = extract_buses_for_technology("H2 Electrolysis", "ughs")

    bus_size_factor_to_GW = 1e2

    total_summary_df = pd.read_csv(TOTAL_SUMMARY_FILE, index_col=0)
    claimed_regions_set = set(
        total_summary_df.loc[PC_50_SCENARIO, "pc larger 1MW"].split(";")
    ) - set(EXCLUDED_PC_NODES)

    df_pc = (
        create_bus_df_for_selected_buses(bus_sizes_list_pc, claimed_regions_set)
        * bus_size_factor_to_GW
    )
    order_north_to_south = (
        regions.loc[list(claimed_regions_set)]
        .centroid.y.sort_values(ascending=True)
        .index
    )

    # df_pc_reordered = df_pc.loc[order_north_to_south]

    proj = ccrs.EqualEarth()

    regions = gpd.read_file(REGIONS_FILE).set_index("name")
    regions = regions.to_crs(proj.proj4_init)
    # The map colours exactly the nodes that have a bar
    pc_regions = regions.loc[
        order_regions_north_to_south(regions, claimed_regions_set)
    ].copy()

    # Find best electrolysis regions
    df_el = create_bus_df_for_selected_buses(bus_sizes_list_el)
    best_el_regions = list(
        df_el.sort_values(EL_RANKING_CAPEX, ascending=False)[
            : len(claimed_regions_set)
        ].index
    )
    el_region_choice = order_regions_north_to_south(
        regions, set(best_el_regions).union(claimed_regions_set)
    )
    el_regions = regions.loc[best_el_regions].copy()

    # # Select best electrolysis regions
    df_el = (
        create_bus_df_for_selected_buses(bus_sizes_list_el, el_region_choice)
        * bus_size_factor_to_GW
    )
    order_north_to_south = (
        regions.loc[el_region_choice].centroid.y.sort_values(ascending=True).index
    )

    df_el_reordered = df_el.loc[order_north_to_south]

    # Prepare data for the bar plot by joining the dataframes of electrolysis and photocatalysis
    joined_df = df_el.add_prefix("el_").join(df_pc.add_prefix("pc_"), how="outer")
    order_north_to_south = (
        regions.loc[joined_df.index].centroid.y.sort_values(ascending=True).index
    )
    joined_df_reordered = joined_df.loc[order_north_to_south]
    joined_df_reordered.fillna(0, inplace=True)
    # Replace all values below 0.01 with np.nan to ensure to have no bar stumps
    joined_df_reordered.mask(joined_df_reordered <= 0.01, np.nan, inplace=True)

    # Creat colors maps
    colors_pc_bat = plt.get_cmap("cmc.batlow_r")
    colors_pc = colors_pc_bat(np.linspace(0, 1, len(df_el_reordered.columns)))
    colors_el_bat = plt.get_cmap("cmc.batlow")
    colors_el = colors_el_bat(np.linspace(0, 1, len(df_el_reordered.columns)))

    # Create annualized photocatalysis mapping depending on the pc_capex stored
    # in the index; keys are strings because the bar columns are "pc_<capex>"
    capex_annualized_cost_map = {
        str(capex): round(
            float(total_summary_df.loc[scenario, "pc total annual cost"]), 1
        )
        for capex, scenario in BASE_SCENARIOS.items()
    }

    # Get maximum installable photocatalysis
    df_p_nom_max = n.generators[
        n.generators.carrier.isin(["photocatalysis"])
    ].p_nom_max.copy()
    df_p_nom_max.rename(
        index=lambda x: x.replace(" photocatalysis", ""), level=0, inplace=True
    )
    df_p_nom_max = df_p_nom_max / 1e3

    return (
        capex_annualized_cost_map,
        df_p_nom_max,
        joined_df_reordered,
        colors_pc,
        colors_el,
        regions,
        pc_regions,
        el_regions,
        proj,
    )


MAP_REGION_ALPHA = 0.7
BOTH_HATCH = "////"
# Thin white outline around each glyph of the node labels
MAP_LABEL_OUTLINE = [patheffects.withStroke(linewidth=0.75, foreground="white")]


def dominant_technology(joined_df, pc50_capex):
    """
    Per node, which of the two technologies is larger in the PC-50 case
    ("photocatalysis" or "H2 Electrolysis"), for the two colors of the inset
    map.
    """
    el_col, pc_col = f"el_{pc50_capex}", f"pc_{pc50_capex}"
    missing = [c for c in (el_col, pc_col) if c not in joined_df.columns]
    if missing:
        raise KeyError(
            f"{missing} not in the joined capacity table; its columns are "
            f"{sorted(joined_df.columns)}. The PC-50 capex token must match "
            "the one the bus-size files were read under."
        )

    el = joined_df[el_col].fillna(0)
    pc = joined_df[pc_col].fillna(0)
    return pd.Series(
        np.where(pc > el, "photocatalysis", "H2 Electrolysis"), index=joined_df.index
    )


def draw_inset_map_regions(
    ax,
    regions,
    pc_regions,
    el_regions,
    tech_colors,
    dominant,
):
    """
    Draw the inset map's node sets on `ax` and return legend handles.

    As photocatalysis and the electrolysis top nodes overlap, there is
    hatching to represent the overlap.
    """
    shown = set(pc_regions.index) | set(el_regions.index)
    unassigned = sorted(shown - set(dominant.index))
    if unassigned:
        # Silently dropping such a node would leave it grey on the map while it
        # still carries a bar in the panel next to it.
        raise KeyError(
            f"{unassigned} carry a bar but have no capacity row to rank them by"
        )

    # Base plot: every node, so the ones carrying neither stay visible
    regions.plot(ax=ax, linewidth=1, color="lightgrey", edgecolor="white", alpha=0.4)

    both = set(pc_regions.index) & set(el_regions.index)
    single = {
        "photocatalysis": set(pc_regions.index) - both,
        "H2 Electrolysis": set(el_regions.index) - both,
    }

    handles = []
    for carrier, label in (
        ("H2 Electrolysis", f"Top {len(el_regions)} electrolysis nodes at PC-0"),
        (
            "photocatalysis",
            f"Top {len(pc_regions)} photocatalysis nodes at PC-50",
        ),
    ):
        names = order_regions_north_to_south(regions, single[carrier])
        color = mpl.colors.to_rgba(tech_colors[carrier], MAP_REGION_ALPHA)
        if names:
            regions.loc[names].plot(
                ax=ax,
                linewidths=1,
                color=color,
                edgecolor="white",
            )
        handles.append(patches.Patch(facecolor=color, label=label))

    face = mpl.colors.to_rgba(tech_colors["photocatalysis"], MAP_REGION_ALPHA)
    hatch = mpl.colors.to_rgba(tech_colors["H2 Electrolysis"], MAP_REGION_ALPHA)
    if both:
        names = order_regions_north_to_south(regions, both)
        regions.loc[names].plot(
            ax=ax, facecolor=face, edgecolor=hatch, hatch=BOTH_HATCH, linewidth=0
        )
        regions.loc[names].boundary.plot(ax=ax, color="white", linewidth=1)
    handles.append(
        patches.Patch(
            facecolor=face, edgecolor=hatch, hatch=BOTH_HATCH, label="Overlap"
        )
    )
    return handles


ANNUALIZED_COST_LEGEND_TITLE = (
    "Annualized\nphotocatalysis\ncost in\n€ a$^{-1}$ kW$^{-1}$"
)
MARKET_SHARE_LEGEND_TITLE = "Photocatalysis\nmarket share\nin " r"$\%$"

# Frames of the inset map and the stacking legend: over the bar panels, under
# the insets themselves (zorder 40 and 50)
INSET_FRAME_ZORDER = 30


def add_legend_frame(fig, bbox, fontsize):
    """
    Frame `bbox` (display units) the way a legend with `fontsize` (pt) frames
    itself: white, rcParams legend.edgecolor and legend.framealpha, corners
    rounded by 0.2 font sizes.
    """
    box = bbox.transformed(fig.transFigure.inverted())
    radius = 0.2 * fontsize * fig.dpi / 72
    fig.add_artist(
        patches.FancyBboxPatch(
            box.p0,
            box.width,
            box.height,
            # the corners are rounded in figure fractions, the aspect makes
            # them circular
            boxstyle=patches.BoxStyle(
                "round", pad=0, rounding_size=radius / fig.bbox.width
            ),
            mutation_aspect=fig.bbox.width / fig.bbox.height,
            transform=fig.transFigure,
            facecolor="white",
            edgecolor=mpl.rcParams["legend.edgecolor"],
            alpha=mpl.rcParams["legend.framealpha"],
            zorder=INSET_FRAME_ZORDER,
        )
    )


def place_framed_insets(fig, ax_pc, legend_pc, ax_map, ax_stacking):
    """
    Lay out the empty top of the photocatalysis panel and frame the insets.

    The map frame spans the panel; below it, tops aligned, sit the
    stacking legend frame and the photocatalysis legend, and the
    stacking frame reaches down to the highest bar under it. Spaced like
    the legend: its borderaxespad between frames, legend and panel
    edges, its borderpad between a frame and its content (the map with
    its tick labels, the stacking legend with its title).

    ax_map must be anchored at the top. Positions are stored in figure
    fractions, so the figure must not be resized afterwards.
    """
    renderer = fig.canvas.get_renderer()
    fontsize = legend_pc.prop.get_size_in_points()
    gap = mpl.rcParams["legend.borderaxespad"] * fontsize * fig.dpi / 72
    pad = mpl.rcParams["legend.borderpad"] * fontsize * fig.dpi / 72
    panel = ax_pc.get_window_extent(renderer)
    to_figure = fig.transFigure.inverted()

    def fit(ax, frame):
        """
        Fit `ax` with its tick labels and title into `frame` minus pad.
        """
        tight = ax.get_tightbbox(renderer)
        window = ax.get_window_extent(renderer)
        box = mpl.transforms.Bbox.from_extents(
            frame.x0 + pad + (window.x0 - tight.x0),
            frame.y0 + pad + (window.y0 - tight.y0),
            frame.x1 - pad - (tight.x1 - window.x1),
            frame.y1 - pad - (tight.y1 - window.y1),
        )
        ax.set_position(box.transformed(to_figure))

    left, right, top = panel.x0 + gap, panel.x1 - gap, panel.y1 - gap
    # open to the panel bottom, so only the width limits the map
    fit(ax_map, mpl.transforms.Bbox.from_extents(left, panel.y0, right, top))
    map_frame = mpl.transforms.Bbox.from_extents(
        left, ax_map.get_tightbbox(renderer).y0 - pad, right, top
    )

    top = map_frame.y0 - gap
    legend_pc.set_bbox_to_anchor(
        to_figure.transform((right, top)), transform=fig.transFigure
    )
    stacking_right = legend_pc.get_window_extent(renderer).x0 - gap
    extents = [
        artist.get_window_extent(renderer) for artist in (*ax_pc.patches, *ax_pc.lines)
    ]
    # bars of missing capacities have NaN widths
    bar_top = max(
        extent.y1
        for extent in extents
        if np.isfinite(extent.extents).all()
        and extent.x1 > left
        and extent.x0 < stacking_right
    )
    stacking_frame = mpl.transforms.Bbox.from_extents(
        left, bar_top + gap, stacking_right, top
    )
    fit(ax_stacking, stacking_frame)

    for frame in (map_frame, stacking_frame):
        add_legend_frame(fig, frame, fontsize)


def mark_ughs_nodes(ax_el, nodes, style, color):
    """
    Mark the nodes with a salt cavern potential (UGHS) in the bar panels.

    nodes: the bar rows, bottom to top (the index of the joined capacity table).
    style:
        "label": color their tick labels
        "column": colored tick labels plus a marker column with a "UGHS" header
    """
    if style not in ("label", "column"):
        raise ValueError(f"style must be 'label' or 'column', not {style!r}")

    salt_cavern_regions = pd.read_csv(SALT_CAVERN_FILE, index_col=0)
    ughs = {node for node in nodes if node in salt_cavern_regions.index}

    for label in ax_el.get_yticklabels():
        if label.get_text() in ughs:
            label.set_color(color)

    if style == "column":
        trans = mpl.transforms.blended_transform_factory(
            ax_el.transAxes, ax_el.transData
        )
        x = UGHS_COLUMN_X
        for row, node in enumerate(nodes):
            has = node in ughs
            ax_el.plot(
                x,
                row,
                marker="o",
                ms=6,
                transform=trans,
                clip_on=False,
                color=color if has else "none",
                markeredgecolor=color if has else "grey",
            )
        ax_el.text(
            x,
            len(nodes) - 0.5,
            "UGHS",
            transform=trans,
            ha="center",
            va="bottom",
        )


# Position of the "column" UGHS markers, in axes coordinates of the
# electrolysis panel
UGHS_COLUMN_X = -0.24


def plot_vertical(
    capex_label_map,
    df_p_nom_max,
    joined_df_reordered,
    colors_pc,
    colors_el,
    regions,
    pc_regions,
    el_regions,
    proj,
    tech_colors,
    h2_storage_color,
    xmax_el=60,
    xmax_pc=95,
    legend_title=ANNUALIZED_COST_LEGEND_TITLE,
    tick_step=None,
    pc50_capex=PC_50_CAPEX,
    stacking_label_adjustment=0.2,
    stacking_title_adjustment=21,
    ughs_style="label",
    legend_scale=1.0,
    figsize=(10, 15),
):
    """
    capex_label_map : dict
        str(capex) -> legend label for that scenario. Either the annualized
        photocatalysis cost or the resulting photocatalysis market share
    xmax_el, xmax_pc : float
        Upper limits of the electrolysis / photocatalysis capacity axes, in GW.
        Also set the width ratio of the two panels
    legend_title : str
        Title of both bar legends, describing what `capex_label_map` holds.
    tick_step : float, optional
        Capacity tick spacing in GW for both panels, None leaves it to matplotlib
    stacking_label_adjustment, stacking_title_adjustment : float
        passed to add_stacking_legend()
    ughs_style :
        see mark_ughs_nodes()
    legend_scale : float
        Font sizes of the two bar legends and of the stacking legend's texts,
        relative to the rcParams legend sizes
    figsize : tuple
        in inches; the insets are laid out for it, see place_framed_insets()

    The inset map and the stacking legend are placed in the empty top of the
    photocatalysis panel by place_framed_insets().
    """
    # Set up figure with two parts
    fig = plt.figure(figsize=figsize)
    width_ratio = [xmax_el, xmax_pc]  # Set split manually to correspond ax limits
    gs = fig.add_gridspec(1, 2, width_ratios=width_ratio, wspace=0.03)
    (ax1, ax2) = gs.subplots(sharey="row")

    plot_capacity_data_both(
        ax2,
        ax1,
        joined_df_reordered,
        colors_pc,
        colors_el,
        df_p_nom_max,
        capex_label_map,
    )

    # The rcParams sizes, resolved as a legend resolves them (a None title
    # size falls back to font.size), then scaled
    legend_fontsize = (
        legend_scale
        * FontProperties(size=mpl.rcParams["legend.fontsize"]).get_size_in_points()
    )
    legend_title_fontsize = (
        legend_scale
        * FontProperties(
            size=mpl.rcParams["legend.title_fontsize"]
        ).get_size_in_points()
    )
    legend_kwargs = dict(
        title=legend_title,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )

    #  Configure legend positions (and reverse the legend order to have bar on top, on top of the legend), labels and limits
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc="lower left", **legend_kwargs)
    handles, labels = ax2.get_legend_handles_labels()
    # anchored by place_framed_insets()
    legend_pc = ax2.legend(
        handles[::-1], labels[::-1], loc="upper right", borderaxespad=0, **legend_kwargs
    )
    ax1.set_xlabel("Installed electrolysis capacity (GW)")
    ax2.set_xlabel("Installed photocatalysis capacity (GW)")
    ax1.set_ylim(-0.5, len(joined_df_reordered.index) - 0.5)
    ax1.set_xlim(0, xmax_el)
    ax1.invert_xaxis()
    ax2.set_xlim(0, xmax_pc)
    if tick_step is not None:
        for ax in (ax1, ax2):
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_step))
    ax1.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax2.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Mark the nodes where salt cavern storage can be built
    mark_ughs_nodes(ax1, list(joined_df_reordered.index), ughs_style, h2_storage_color)

    #### Add Overlay axis to show bar overlay legend
    # Placed by place_framed_insets(), its frame is the background

    ax = fig.add_axes(ax2.get_position().bounds, zorder=40)
    ax.patch.set_visible(False)

    add_stacking_legend(
        ax,
        y_label_adjustment=stacking_label_adjustment,
        title_adjustment=stacking_title_adjustment,
        colors_pc=colors_pc,
        colors_el=colors_el,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        x_margin=0,
    )

    #### Add legend image / map (if boxplot visuals are changed the image needs to be adjusted/recreated)
    # Placed by place_framed_insets()

    ax_overlay = fig.add_axes(
        ax2.get_position().bounds, anchor="N", projection=proj, zorder=50
    )
    ax_overlay.patch.set_alpha(1.0)
    ax_overlay.set_facecolor("white")

    map_font_size = 7

    map_handles = draw_inset_map_regions(
        ax_overlay,
        regions,
        pc_regions,
        el_regions,
        tech_colors,
        dominant_technology(joined_df_reordered, pc50_capex),
    )

    # Define annotation parameters for each zone
    node_annotations_pc = {
        "FR1 11": (620000, 5330000),
        "FR1 1": (354949, 5361033),
        "ES1 2": (180937, 5161688),
        "ES1 4": (-330000, 4605000),
        "IT1 11": (1354146, 4643111),
        "IT1 0": (1131597, 4662965),
    }

    # Common annotation parameters
    common_params = {
        "fontsize": map_font_size,
        "color": "black",
        "ha": "center",
        "va": "center",
        "path_effects": MAP_LABEL_OUTLINE,
    }

    # Annotate each node with its bus name (for pc_regions)
    for idx, row in pc_regions.iterrows():
        centroid = row["geometry"].centroid
        # print(idx, centroid.x, centroid.y)
        xytext = node_annotations_pc.get(idx, (centroid.x, centroid.y))
        ax_overlay.annotate(
            str(idx), xy=(centroid.x, centroid.y), xytext=xytext, **common_params
        )

    node_annotations_el = {
        "DK2 0": (895544, 6511730),
        "DK1 0": (709430, 6588491),
        "DE1 5": (685047, 6275656),
        "NL1 0": (342397, 6268247),
        "NL1 4": (554864, 6202336),
    }

    # Annotate each node with its bus name (for el_regions)
    for idx, row in el_regions.iterrows():
        # Nodes in both sets are already labeled by the photocatalysis loop
        if idx in pc_regions.index:
            continue
        centroid = row["geometry"].centroid
        # print(idx, centroid.x, centroid.y)
        xytext = node_annotations_el.get(idx, (centroid.x, centroid.y))
        ax_overlay.annotate(
            str(idx), xy=(centroid.x, centroid.y), xytext=xytext, **common_params
        )

    # ax.get_extent()
    ax_overlay.set_extent([-9.5, 26.5, 34.7, 62])

    gl = ax_overlay.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)

    # Control which labels should be drawn
    gl.left_labels = True
    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = True

    # Setting the property for spacing or clipping
    gl.xlabel_style = {"size": map_font_size, "color": "k", "clip_on": True}
    gl.ylabel_style = {"size": map_font_size, "color": "k", "clip_on": True}

    ax_overlay.legend(
        handles=map_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=map_font_size,
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.3,
        labelspacing=0.25,
        framealpha=0.9,
    )

    place_framed_insets(fig, ax2, legend_pc, ax_overlay, ax)

    fig.show()

    # fig.savefig("img/capacity_per_cost_bar.svg", bbox_inches="tight", format="svg")
    return fig


def plot_capacity_data_both_bars_vertical(
    ax_pc, ax_el, df, colors_pc, colors_el, max_limits, label_map
):
    """
    Args:
        ax (ax): ax on which the wind plot should be shown
        df (df): df which contains the data
        colors (list): colors for bars
    """
    bar_width = 0.7
    index = range(len(df))
    y_offset = np.zeros(len(df))

    # Add all bars for photocatalysis for each of the index values
    # Iterate through years in reverse order
    for i, capex in enumerate(str(c) for c in BASE_PC_CAPEX):
        pc_vals = df[f"pc_{capex}"]

        # Plot Wind capacity bars with proper offsets to stack them behind each other
        ax_pc.bar(
            index,
            pc_vals,
            bar_width,
            bottom=y_offset,
            color=colors_pc[i],
            edgecolor="black",
            linewidth=0.3,
            label=str(label_map[capex]),
        )

    # Flag to add legend entry only once
    legend_entry = False
    # Add max limits for photocatalysis (only if it actually carries a PC bar)
    pc_columns = [f"pc_{capex}" for capex in BASE_PC_CAPEX]
    for idx, node in enumerate(df.index):
        if df.loc[node, pc_columns].notna().any() and node in max_limits.index:
            # To add legend entry only once
            if not legend_entry:
                ax_pc.plot(
                    [idx - bar_width / 1.8, idx + bar_width / 1.8],
                    [max_limits[node], max_limits[node]],
                    color="red",
                    linewidth=2,
                    label="Max. space",
                )
                legend_entry = True
            else:
                ax_pc.plot(
                    [idx - bar_width / 1.8, idx + bar_width / 1.8],
                    [max_limits[node], max_limits[node]],
                    color="red",
                    linewidth=2,
                )

    # Add all bars for electrolysis for each of the index values
    for i, capex in enumerate(str(c) for c in reversed(BASE_PC_CAPEX)):
        el_vals = df[f"el_{capex}"]

        # Plot Wind capacity bars with proper offsets to stack them behind each other
        ax_el.bar(
            index,
            el_vals,
            bar_width,
            bottom=y_offset,
            color=colors_el[i],
            edgecolor="black",
            linewidth=0.3,
            label=str(label_map[capex]),
        )

    # Adjusting labels and ticks for subplot
    for ax in [ax_el, ax_pc]:
        ax.set_xticks(index)
        ax.set_xticklabels(df.index)
        ax.tick_params(axis="x", which="both", length=0, rotation=90)
        ax.set_axisbelow(True)
        ax.grid(True, color="gainsboro")


def plot_horizontal(
    capex_label_map,
    df_p_nom_max,
    joined_df_reordered,
    colors_pc,
    colors_el,
    regions,
    pc_regions,
    el_regions,
    proj,
    tech_colors,
    h2_storage_color,
    xmax_el=60,
    xmax_pc=95,
    legend_title=ANNUALIZED_COST_LEGEND_TITLE,
    tick_step=None,
    pc50_capex=PC_50_CAPEX,
):
    # Changes introduced to get vertical bars
    # "barh()" to "bar()" and in there "left" to "bottom"
    # Change the tick label adjustment to have x instead of y in everywhere (in for loop)
    # Change figure layout to wide format e.g.: figsize=(10,5)
    # Change axis layout in gridspec to 2 rows 1 col + width_ratios to "height_ratios", wspace to hspace + gs.subplots(sharex="col")
    # Change legend position, switch ax1 and ax2 (in input and every where else)
    # Change positions. Etc. etc.

    height_ratio = [xmax_pc, xmax_el]
    # Set up figure with two parts
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratio, hspace=0.03)
    (ax1, ax2) = gs.subplots(sharex="col")

    plot_capacity_data_both_bars_vertical(
        ax1,
        ax2,
        joined_df_reordered,
        colors_pc,
        colors_el,
        df_p_nom_max,
        capex_label_map,
    )

    #  Configure legend positions (and reverse the legend order to have bar on top, on top of the legend), labels and limits
    handles, labels = ax1.get_legend_handles_labels()
    l1 = ax1.legend(handles[::-1], labels[::-1], loc="upper right")
    # l1 = ax1.legend(loc="lower left")
    handles, labels = ax2.get_legend_handles_labels()
    l2 = ax2.legend(handles[::-1], labels[::-1], loc="lower left")
    l1.set_title(legend_title)
    l2.set_title(legend_title)
    ax1.set_ylabel("Installed photocatalysis capacity (GW)")
    ax2.set_ylabel("Installed electrolysis capacity (GW)")
    ax2.set_xlim(-0.5, len(joined_df_reordered.index) - 0.5)
    ax1.set_ylim(0, xmax_pc)
    ax2.set_ylim(0, xmax_el)
    ax2.invert_yaxis()
    if tick_step is not None:
        for ax in (ax1, ax2):
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_step))
    ax1.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax2.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Highlight nodes where salt cavern storage can be build
    salt_cavern_regions = pd.read_csv(SALT_CAVERN_FILE, index_col=0)
    # All could be used to further filter for a minimum capacity.
    salt_cavern_regions["all"] = (
        salt_cavern_regions[["nearshore", "offshore", "onshore"]].fillna(0).sum(axis=1)
    )
    highlighted_categories = [
        node for node in joined_df_reordered.index if node in salt_cavern_regions.index
    ]
    for label in ax2.get_xticklabels():
        if label.get_text() in highlighted_categories:
            label.set_color("blue")

    #### Add Overlay axis to show bar overlay legend

    x0, y0, width, height = (
        0.56,
        0.42,
        0.3,
        0.07,
    )  # Define position and size of the overlay ax
    # Add an overlay axes for the map

    ax = fig.add_axes([x0, y0, width, height], anchor="SE", zorder=40)

    ax.patch.set_alpha(1.0)
    ax.set_facecolor("white")

    add_stacking_legend(
        ax,
        y_label_adjustment=0.15,
        title_adjustment=10,
        colors_pc=colors_pc,
        colors_el=colors_el,
    )

    #### Add legend image / map (if boxplot visuals are changed the image needs to be adjusted/recreated)

    # Positioning of the image in relative coordinates (x0, y0, width, height)
    # x0, y0 lower-left corner of the image box; width, height size of the image box
    x0, y0, width, height = (
        0.465,
        0.535,
        0.33,
        0.33,
    )  # Define position and size of the overlay ax
    # Add an overlay axes for the map

    ax_overlay = fig.add_axes(
        [x0, y0, width, height], anchor="SE", projection=proj, zorder=50
    )
    ax_overlay.patch.set_alpha(1.0)
    ax_overlay.set_facecolor("white")

    map_font_size = 7

    map_handles = draw_inset_map_regions(
        ax_overlay,
        regions,
        pc_regions,
        el_regions,
        tech_colors,
        dominant_technology(joined_df_reordered, pc50_capex),
    )

    # Define annotation parameters for each zone
    node_annotations_pc = {
        "FR1 11": (620000, 5330000),
        "FR1 1": (354949, 5361033),
        "ES1 2": (180937, 5161688),
        "ES1 4": (-330000, 4605000),
        "IT1 11": (1354146, 4643111),
        "IT1 0": (1131597, 4662965),
    }

    # Common annotation parameters
    common_params = {
        "fontsize": map_font_size,
        "color": "black",
        "ha": "center",
        "va": "center",
        "path_effects": MAP_LABEL_OUTLINE,
    }

    # Annotate each node with its bus name (for pc_regions)
    for idx, row in pc_regions.iterrows():
        centroid = row["geometry"].centroid
        # print(idx, centroid.x, centroid.y)
        xytext = node_annotations_pc.get(idx, (centroid.x, centroid.y))
        ax_overlay.annotate(
            str(idx), xy=(centroid.x, centroid.y), xytext=xytext, **common_params
        )

    node_annotations_el = {
        "DK2 0": (895544, 6511730),
        "DK1 0": (709430, 6588491),
        "DE1 5": (685047, 6275656),
        "NL1 0": (342397, 6268247),
        "NL1 4": (554864, 6202336),
    }

    # Annotate each node with its bus name (for el_regions)
    for idx, row in el_regions.iterrows():
        # Nodes in both sets are already labeled by the photocatalysis loop
        if idx in pc_regions.index:
            continue
        centroid = row["geometry"].centroid
        # print(idx, centroid.x, centroid.y)
        xytext = node_annotations_el.get(idx, (centroid.x, centroid.y))
        ax_overlay.annotate(
            str(idx), xy=(centroid.x, centroid.y), xytext=xytext, **common_params
        )

    # ax.get_extent()
    ax_overlay.set_extent([-9.5, 26.5, 34.7, 62])

    gl = ax_overlay.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)

    # Control which labels should be drawn
    gl.left_labels = True
    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = True

    # Setting the property for spacing or clipping
    gl.xlabel_style = {"size": map_font_size, "color": "k", "clip_on": True}
    gl.ylabel_style = {"size": map_font_size, "color": "k", "clip_on": True}

    ax_overlay.legend(
        handles=map_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=map_font_size,
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.3,
        labelspacing=0.25,
        framealpha=0.9,
    )

    fig.show()
    # fig.savefig("img/capacity_per_cost_bar_horizontal.svg", bbox_inches="tight", format="svg")


def main():
    (
        capex_annualized_cost_map,
        df_p_nom_max,
        joined_df_reordered,
        colors_pc,
        colors_el,
        regions,
        pc_regions,
        el_regions,
        proj,
    ) = prepare_bar_plot_data()

    with open(COLORS_YAML) as stream:
        colors = yaml.safe_load(stream)
    tech_colors = colors["tech_colors"]
    h2_storage_color = colors["others"]["h2_storage"]

    plot_vertical(
        capex_annualized_cost_map,
        df_p_nom_max,
        joined_df_reordered,
        colors_pc,
        colors_el,
        regions,
        pc_regions,
        el_regions,
        proj,
        tech_colors,
        h2_storage_color,
    )

    plot_horizontal(
        capex_annualized_cost_map,
        df_p_nom_max,
        joined_df_reordered,
        colors_pc,
        colors_el,
        regions,
        pc_regions,
        el_regions,
        proj,
        tech_colors,
        h2_storage_color,
    )


if __name__ == "__main__":
    main()
