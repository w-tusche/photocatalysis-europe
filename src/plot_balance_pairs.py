# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: 2025 Wolfram Tuschewitzki, Jelto Lange
#
# SPDX-License-Identifier: MIT
"""
Paired PC-0 / PC-50 energy balance figures for the "Energy balances" note.

PC-0 on the left, PC-50 on the right, one shared legend below. The maps and time
series follow PyPSA-Eur's ``plot_balance_map.py`` and
``plot_balance_timeseries.py``; ``add_missing_carriers`` and
``sanitize_carriers`` are taken from PyPSA-Eur's ``scripts/add_electricity.py``,
``rename_techs`` from its ``scripts/_helpers.py``
(https://github.com/pypsa/pypsa-eur).

Needs the balance environment (PyPSA 1.0, see ``balances_python()`` in
``cases.py``), energy_balance() takes a list for ``groupby`` there.

``python src/plot_balance_pairs.py extract``
    reads both networks once and caches the balances, prices, colors and a
    static copy of the network in ``data/balance_pairs/<scenario>/``
``python src/plot_balance_pairs.py plot [--only AC gas ...]``
    draws the figures from that cache into ``img/`` (``--outdir``) as
    ``.png`` (400 dpi) and ``.svg``, or in the formats of the environment
    variable PC_FIGURE_FORMATS (``cases.figure_formats()``)

Run from the repository root.
"""

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from matplotlib.patches import Patch
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import (
    FIGURE_FOLDER,
    PC0,
    PC0_LABEL,
    PC50,
    PC50_LABEL,
    figure_formats,
    find_network,
)

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1

REPO = Path(__file__).resolve().parent.parent
REGIONS_FILE = REPO / "data" / "regions_onshore_elec_s_150.geojson"
PLOTTING_CONFIG = REPO / "src" / "plotting.default.yaml"
CACHE_ROOT = REPO / "data" / "balance_pairs"

# panel title -> scenario folder, left panel first
CASES = {PC0_LABEL: PC0, PC50_LABEL: PC50}

# the four balance maps of the annex, in the order they appear there
MAP_CARRIERS = ["AC", "H2", "urban central heat", "urban decentral heat"]

# the eight balance time series of the annex; values are the bus carriers that
# are summed into one panel (mirrors `balance_timeseries` in the plotting config)
TS_GROUPS = {
    "electricity": ["AC", "low voltage"],
    "H2": ["H2"],
    "heat": [
        "urban central heat",
        "urban decentral heat",
        "rural heat",
        "residential urban decentral heat",
        "residential rural heat",
        "services urban decentral heat",
        "services rural heat",
    ],
    "gas": ["gas"],
    "oil": ["oil"],
    "methanol": ["methanol"],
    "co2": ["co2"],
    "co2 stored": ["co2 stored"],
}

MAX_LEGEND_COLUMNS = 8  # upper bound; how many actually fit is computed below
FIGURE_WIDTH = 11.0  # inch, both panels plus margins
LEGEND_FONTSIZE = 11
LEGEND_ROW_HEIGHT = 0.19 * LEGEND_FONTSIZE / 7  # inch per legend row at LEGEND_FONTSIZE
LEGEND_TITLE_HEIGHT = (
    0.26 * LEGEND_FONTSIZE / 7
)  # inch for the "Supply" / "Consumption" title


def slug(name):
    return name.replace(" ", "_")


def load_plotting_config():
    with open(PLOTTING_CONFIG, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


# --------------------------------------------------------------------------- #
# carrier helpers from PyPSA-Eur (add_electricity.py, _helpers.py)
# --------------------------------------------------------------------------- #


def add_missing_carriers(n, carriers):
    """
    Function to add missing carriers to the network without raising errors.
    """
    missing_carriers = set(carriers) - set(n.carriers.index)
    if len(missing_carriers) > 0:
        n.add("Carrier", missing_carriers)


def sanitize_carriers(n, config):
    """
    Sanitize the carrier information in a PyPSA Network object.

    The function ensures that all unique carrier names are present in the network's
    carriers attribute, and adds nice names and colors for each carrier according
    to the provided configuration dictionary.

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA Network object that represents an electrical power system.
    config : dict
        A dictionary containing configuration information, specifically the
        "plotting" key with "nice_names" and "tech_colors" keys for carriers.

    Returns
    -------
    None
        The function modifies the 'n' PyPSA Network object in-place, updating the
        carriers attribute with nice names and colors.

    Warnings
    --------
    Raises a warning if any carrier's "tech_colors" are not defined in the config dictionary.
    """

    for c in n.iterate_components():
        if "carrier" in c.df:
            add_missing_carriers(n, c.df.carrier)

    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series())
    )
    n.carriers["nice_name"] = n.carriers.nice_name.where(
        n.carriers.nice_name != "", nice_names
    )

    tech_colors = config["plotting"]["tech_colors"]
    colors = pd.Series(tech_colors).reindex(carrier_i)
    # try to fill missing colors with tech_colors after renaming
    missing_colors_i = colors[colors.isna()].index
    colors[missing_colors_i] = missing_colors_i.map(rename_techs).map(tech_colors)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        print(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = n.carriers.color.where(n.carriers.color != "", colors)


def rename_techs(label: str) -> str:
    """
    Rename technology labels for better readability.

    Removes some prefixes and renames if certain conditions defined in function body are met.

    Parameters
    ----------
    label: str
        Technology label to be renamed

    Returns
    -------
    str
        Renamed label
    """
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "land transport fuel cell",
        "land transport oil": "land transport oil",
        "oil shipping": "shipping oil",
        # "CC": "CC"
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "offwind-float": "offshore wind (Float)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "NH3": "ammonia",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new
    return label


# --------------------------------------------------------------------------- #
# stage 1: extract
# --------------------------------------------------------------------------- #


def extract_case(scenario, config):
    """
    Read one network and cache everything the paired figures need.
    """
    cache = CACHE_ROOT / scenario
    cache.mkdir(parents=True, exist_ok=True)
    plotting = config["plotting"]

    path = find_network(scenario)
    print(f"reading {path} ...", flush=True)
    n = pypsa.Network(path)
    sanitize_carriers(n, config)
    pypsa.set_option("params.statistics.round", 3)
    pypsa.set_option("params.statistics.drop_zero", True)
    pypsa.set_option("params.statistics.nice_names", False)

    # PyPSA-Eur default colors, colors.yaml only covers a few carriers
    n.carriers.update({"color": plotting["tech_colors"]})
    n.carriers["color"] = n.carriers.color.replace("", "grey").fillna("grey")
    n.carriers.color.to_csv(cache / "colors.csv")

    # bus locations, as in PyPSA-Eur's plot_balance_map.py
    eu_location = plotting["eu_node_location"]
    n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]
    n.buses["location"] = n.buses["location"].replace("", "EU").fillna("EU")
    n.buses["x"] = n.buses.location.map(n.buses.x)
    n.buses["y"] = n.buses.location.map(n.buses.y)

    for carrier in MAP_CARRIERS:
        print(f"  map data: {carrier}", flush=True)
        conversion = plotting["balance_map"][carrier]["unit_conversion"]

        # nodal energy balance; positive = supply into the bus
        eb = n.statistics.energy_balance(
            bus_carrier=carrier, groupby=["bus", "carrier"]
        )

        # drop the transmission carriers, whose balance is only losses
        transmission_carriers = get_transmission_carriers(
            n, bus_carrier=carrier
        ).rename({"name": "carrier"})
        components = transmission_carriers.unique("component")
        drop_carriers = transmission_carriers.unique("carrier")
        drop_carriers = drop_carriers[
            drop_carriers.isin(eb.index.get_level_values("carrier"))
        ]
        eb.loc[components] = eb.loc[components].drop(
            index=drop_carriers, level="carrier"
        )
        eb = eb.dropna()

        bus_sizes = eb.groupby(level=["bus", "carrier"]).sum().div(conversion)
        bus_sizes.sort_values(ascending=False).to_csv(
            cache / f"bus_sizes_{slug(carrier)}.csv"
        )

        # branch widths and flows of that bus carrier
        flow = n.statistics.transmission(groupby=False, bus_carrier=carrier).div(
            conversion
        )
        if not flow.empty:
            reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
            flow_reversed = flow[reversed_mask].rename(
                lambda x: x.replace("-reversed", "")
            )
            flow = flow[~reversed_mask].subtract(flow_reversed, fill_value=0)
        flow.to_csv(cache / f"flow_{slug(carrier)}.csv")

        # snapshot-weighted average marginal price per region
        buses = n.buses.query("carrier in @carrier").index
        weights = n.snapshot_weightings.generators
        prices = weights @ n.buses_t.marginal_price[buses] / weights.sum()
        price = prices.rename(n.buses.location).groupby(level="name").mean()
        price.to_csv(cache / f"price_{slug(carrier)}.csv")

    print("  balance time series ...", flush=True)
    balance = n.statistics.energy_balance(aggregate_time=False, nice_names=False)
    for group, carriers in TS_GROUPS.items():
        mask = balance.index.get_level_values("bus_carrier").isin(carriers)
        df = balance[mask].groupby("carrier").sum().div(1e3).T  # GW (kt/h for CO2)
        if df.empty:
            print(f"    no balance for group '{group}', skipped", flush=True)
            continue
        # the thresholds for the "other" grouping are taken at native
        # resolution, the figures are drawn from the daily means
        stats = pd.DataFrame({"max": df.abs().max(), "mean": df.abs().mean()})
        stats.to_csv(cache / f"ts_stats_{slug(group)}.csv")
        daily = df.resample("1h").ffill().resample("D").mean()
        daily.to_csv(cache / f"ts_{slug(group)}.csv")

    # a copy without the time series is all that n.plot needs and loads in
    # seconds instead of minutes
    print("  writing network_static.nc ...", flush=True)
    n.set_snapshots(n.snapshots[:1])
    static = cache / "network_static.nc"
    if static.exists():
        static.unlink()
    n.export_to_netcdf(static)


# --------------------------------------------------------------------------- #
# shared legend helpers
# --------------------------------------------------------------------------- #


def legend_columns(labels, max_columns=MAX_LEGEND_COLUMNS):
    """
    Columns and rows that spread `labels` evenly over the figure width.

    With ``mode="expand"`` every column is as wide as the longest label.
    """
    labels = list(labels)
    if not labels:
        return 0, 0
    # 0.55 em average character width at LEGEND_FONTSIZE, plus patch and padding
    longest = max(len(str(label)) for label in labels)
    entry_width = 0.40 + longest * 0.55 * LEGEND_FONTSIZE / 72
    fitting = max(1, min(max_columns, int(FIGURE_WIDTH // entry_width)))
    rows = int(np.ceil(len(labels) / fitting))
    columns = int(np.ceil(len(labels) / rows))
    return columns, rows


def spread_patch_legend(ax, colors, title=None):
    """
    Full-width, multi-column patch legend inside a dedicated blank axis.
    """
    columns, _ = legend_columns(colors.index)
    handles = [Patch(facecolor=c, edgecolor="none") for c in colors]
    ax.legend(
        handles=handles,
        labels=list(colors.index),
        ncol=max(columns, 1),
        loc="upper left",
        bbox_to_anchor=(0, 0, 1, 1),
        mode="expand",
        frameon=False,
        alignment="left",
        title=title,
        title_fontproperties={"weight": "bold"},
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.2,
        handleheight=1.0,
        borderaxespad=0,
    )


def extent_aspect(boundaries, crs):
    """
    Height/width of the map extent in projected coordinates.

    Used to set the row height before drawing, since cartopy widens the
    data limits to fill the cell.
    """
    lon = np.linspace(boundaries[0], boundaries[1], 50)
    lat = np.linspace(boundaries[2], boundaries[3], 50)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    points = crs.transform_points(
        ccrs.PlateCarree(), grid_lon.ravel(), grid_lat.ravel()
    )
    x_span = points[:, 0].max() - points[:, 0].min()
    y_span = points[:, 1].max() - points[:, 1].min()
    return y_span / x_span


def blank_axis(fig, spec):
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    return ax


def save(fig, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in figure_formats(["png", "svg"]):
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, **({"dpi": 400} if fmt == "png" else {}))
        print(f"wrote {path}", flush=True)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# stage 2a: balance maps
# --------------------------------------------------------------------------- #


def read_map_cache(scenario, carrier):
    cache = CACHE_ROOT / scenario
    bus_sizes = pd.read_csv(
        cache / f"bus_sizes_{slug(carrier)}.csv", index_col=[0, 1]
    ).squeeze("columns")
    flow = pd.read_csv(cache / f"flow_{slug(carrier)}.csv", index_col=[0, 1]).squeeze(
        "columns"
    )
    price = pd.read_csv(cache / f"price_{slug(carrier)}.csv", index_col=0).squeeze(
        "columns"
    )
    colors = pd.read_csv(cache / "colors.csv", index_col=0).squeeze("columns")
    return dict(bus_sizes=bus_sizes, flow=flow, price=price, colors=colors)


def split_supply_consumption(bus_sizes_per_case):
    """
    Classify every carrier of both cases as supply or consumption.

    A carrier on both sides goes to the side with the larger total over
    both cases, so the panels share one legend.
    """
    combined = pd.concat(bus_sizes_per_case)
    grouped = combined.groupby(level="carrier")
    positive = grouped.apply(lambda s: s[s > 0].sum())
    negative = grouped.apply(lambda s: -s[s < 0].sum())
    order = (positive + negative).sort_values(ascending=False).index
    supply = [c for c in order if positive[c] >= negative[c] and positive[c] > 0]
    consumption = [c for c in order if c not in supply and negative[c] > 0]
    return supply, consumption


def price_limits(price, map_config):
    """
    Vmin / vmax of the region color scale for one or both cases.
    """
    shift = round(abs(price.mean()) / 20, 0) if price.size == 1 else 0
    vmin = price.min() - shift if map_config["vmin"] is None else map_config["vmin"]
    vmax = price.max() + shift if map_config["vmax"] is None else map_config["vmax"]
    return vmin, vmax


def plot_map_pair(carrier, outdir, config, shared_price_scale=True):
    plotting = config["plotting"]
    map_config = plotting["balance_map"][carrier]
    boundaries = plotting["map"]["boundaries"]

    data = {case: read_map_cache(scenario, carrier) for case, scenario in CASES.items()}
    networks = {
        case: pypsa.Network(CACHE_ROOT / scenario / "network_static.nc")
        for case, scenario in CASES.items()
    }
    regions = gpd.read_file(REGIONS_FILE).set_index("name")

    # one price scale over both cases, or one per case
    if shared_price_scale:
        limits = {
            case: price_limits(
                pd.concat([d["price"] for d in data.values()]), map_config
            )
            for case in CASES
        }
    else:
        limits = {
            case: price_limits(d["price"], map_config) for case, d in data.items()
        }

    supply, consumption = split_supply_consumption(
        [d["bus_sizes"] for d in data.values()]
    )
    colors = pd.concat([d["colors"] for d in data.values()])
    colors = colors[~colors.index.duplicated()]
    _, supply_rows = legend_columns(supply)
    _, consumption_rows = legend_columns(consumption)

    crs = ccrs.EqualEarth()
    colorbar_height = 0.30
    supply_height = LEGEND_TITLE_HEIGHT + LEGEND_ROW_HEIGHT * supply_rows
    consumption_height = LEGEND_TITLE_HEIGHT + LEGEND_ROW_HEIGHT * consumption_rows
    # panel width minus the layout padding, times the aspect of the extent,
    # plus room for the panel title
    map_height = (FIGURE_WIDTH / 2 - 0.15) * extent_aspect(boundaries, crs) + 0.32
    fig = plt.figure(
        figsize=(
            FIGURE_WIDTH,
            map_height + colorbar_height + supply_height + consumption_height,
        ),
        layout="constrained",
    )
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[map_height, colorbar_height, supply_height, consumption_height],
    )

    axes = {}
    for column, case in enumerate(CASES):
        ax = fig.add_subplot(gs[0, column], projection=crs)
        axes[case] = ax
        d = data[case]
        n = networks[case]
        bus_sizes = d["bus_sizes"].sort_values(ascending=False)
        bus_colors = (
            bus_sizes.index.get_level_values("carrier").unique().to_series().map(colors)
        )
        flow = d["flow"]
        fallback = pd.Series(dtype=float)
        line_widths = flow.get("Line", fallback).abs()
        link_widths = flow.get("Link", fallback).abs()
        line_flow = flow.get("Line")
        link_flow = flow.get("Link")
        transformer_flow = flow.get("Transformer")

        n.plot(
            bus_sizes=bus_sizes * map_config["bus_factor"],
            bus_colors=bus_colors,
            bus_split_circles=True,
            line_widths=line_widths * map_config["branch_factor"],
            link_widths=link_widths * map_config["branch_factor"],
            line_flow=(
                line_flow * map_config["flow_factor"] if line_flow is not None else None
            ),
            link_flow=(
                link_flow * map_config["flow_factor"] if link_flow is not None else None
            ),
            transformer_flow=(
                transformer_flow * map_config["flow_factor"]
                if transformer_flow is not None
                else None
            ),
            ax=ax,
            margin=0.2,
            geomap_colors={"border": "darkgrey", "coastline": "darkgrey"},
            geomap=True,
            boundaries=boundaries,
        )

        # pin the extent to the row height from `extent_aspect`
        ax.set_extent(boundaries, crs=ccrs.PlateCarree())

        price = d["price"]
        if price.size == 1:
            regions_case = regions.assign(price=price.values[0])
        else:
            regions_case = regions.assign(price=price.reindex(regions.index).fillna(0))
        vmin, vmax = limits[case]
        regions_case.to_crs(crs.proj4_init).plot(
            ax=ax,
            column="price",
            cmap=map_config["cmap"],
            vmin=vmin,
            vmax=vmax,
            edgecolor="None",
            linewidth=0,
        )
        ax.set_title(case, fontweight="bold")

    # colorbar(s) for the regional prices, one for a shared scale
    label = f"Average marginal price [{map_config['region_unit']}]"
    bar_cases = [list(CASES)[0]] if shared_price_scale else list(CASES)
    for column, case in enumerate(bar_cases):
        vmin, vmax = limits[case]
        cax = fig.add_subplot(gs[1, :] if shared_price_scale else gs[1, column])
        sm = plt.cm.ScalarMappable(
            cmap=map_config["cmap"], norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        cbr = fig.colorbar(sm, cax=cax, label=label, orientation="horizontal")
        cbr.outline.set_edgecolor("None")

    # size legends into the left map, as in the single-case figures
    left_ax = axes[list(CASES)[0]]
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
        "fontsize": LEGEND_FONTSIZE,
    }
    unit = map_config["unit"]
    if map_config["bus_sizes"] is not None:
        add_legend_semicircles(
            left_ax,
            [
                s * map_config["bus_factor"] * SEMICIRCLE_CORRECTION_FACTOR
                for s in map_config["bus_sizes"]
            ],
            [f"{s} {unit}" for s in map_config["bus_sizes"]],
            patch_kw={"color": "#666"},
            legend_kw={"bbox_to_anchor": (0, 1), **legend_kwargs},
        )
    if map_config["branch_sizes"] is not None:
        add_legend_lines(
            left_ax,
            [s * map_config["branch_factor"] for s in map_config["branch_sizes"]],
            [f"{s} {unit}" for s in map_config["branch_sizes"]],
            patch_kw={"color": "#666"},
            legend_kw={"bbox_to_anchor": (0.28, 1), **legend_kwargs},
        )

    # one shared carrier legend below both maps
    spread_patch_legend(blank_axis(fig, gs[2, :]), colors[supply], title="Supply")
    spread_patch_legend(
        blank_axis(fig, gs[3, :]), colors[consumption], title="Consumption"
    )

    suffix = "" if shared_price_scale else "_own_price_scale"
    save(fig, outdir, f"{slug(carrier).lower()}_map_pc0_vs_pc50{suffix}")


# --------------------------------------------------------------------------- #
# stage 2b: balance time series
# --------------------------------------------------------------------------- #


def plot_stacked_area_steplike(ax, df, colors):
    """
    Stacked area chart with step-like transitions (as in PyPSA-Eur's
    plot_balance_timeseries.py).
    """
    df_cum = df.cumsum(axis=1)
    previous = np.zeros_like(df_cum.iloc[:, 0].values)
    for col in df_cum.columns:
        ax.fill_between(
            df_cum.index,
            previous,
            df_cum[col],
            step="pre",
            linewidth=0,
            color=colors.get(col, "grey"),
            label=col,
        )
        previous = df_cum[col].values


def prepare_timeseries(scenario, group, config):
    cache = CACHE_ROOT / scenario
    df = pd.read_csv(cache / f"ts_{slug(group)}.csv", index_col=0, parse_dates=True)
    stats = pd.read_csv(cache / f"ts_stats_{slug(group)}.csv", index_col=0)
    ts_config = config["plotting"]["balance_timeseries"]
    below = stats.index[
        (stats["max"] < ts_config["max_threshold"])
        & (stats["mean"] < ts_config["mean_threshold"])
    ]
    rename = {tech: "other" for tech in below}
    if rename:
        df = df.T.groupby(df.columns.map(lambda a: rename.get(a, a))).sum().T
    return df


def plot_timeseries_pair(group, outdir, config):
    frames = {
        case: prepare_timeseries(scenario, group, config)
        for case, scenario in CASES.items()
    }
    colors = pd.read_csv(
        CACHE_ROOT / list(CASES.values())[0] / "colors.csv", index_col=0
    ).squeeze("columns")
    colors = colors.to_dict()
    colors["other"] = "grey"

    # identical column set and stacking order in both panels
    columns = pd.Index(sorted(set().union(*[set(f.columns) for f in frames.values()])))
    frames = {
        case: f.reindex(columns=columns).fillna(0.0) for case, f in frames.items()
    }
    combined = pd.concat(frames.values())
    order = (combined / combined.max()).var().sort_values().index
    frames = {case: f.loc[:, order] for case, f in frames.items()}

    ylim = 0.0
    for f in frames.values():
        ylim = max(
            ylim,
            f.where(f > 0).fillna(0.0).sum(axis=1).max(),
            -f.where(f < 0).fillna(0.0).sum(axis=1).min(),
        )
    ylim = np.ceil(ylim / 50) * 50

    # carriers actually drawn, largest first, for the shared legend
    magnitude = combined.abs().mean().sort_values(ascending=False)
    legend_carriers = [c for c in magnitude.index if combined[c].abs().max() > 0]
    _, rows = legend_columns(legend_carriers)

    panel_height = 4.4
    legend_height = LEGEND_ROW_HEIGHT * rows + 0.1
    fig = plt.figure(
        figsize=(FIGURE_WIDTH, panel_height + legend_height), layout="constrained"
    )
    gs = fig.add_gridspec(2, 2, height_ratios=[panel_height, legend_height])

    axes = []
    for column, (case, df) in enumerate(frames.items()):
        ax = fig.add_subplot(gs[0, column], sharey=axes[0] if axes else None)
        axes.append(ax)
        plot_stacked_area_steplike(ax, df.where(df > 0).fillna(0.0), colors)
        plot_stacked_area_steplike(ax, df.where(df < 0).fillna(0.0), colors)
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(-ylim, ylim)
        ax.grid(axis="y")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%e\n%b"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%e"))
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )
        ax.tick_params(axis="x", which="minor", labelcolor="grey")
        ax.set_title(case, fontweight="bold")
        if column == 0:
            unit = "kt/h" if "co2" in group.lower() else "GW"
            ax.set_ylabel(f"{group} balance [{unit}]")
        else:
            ax.tick_params(labelleft=False)

    spread_patch_legend(
        blank_axis(fig, gs[1, :]),
        pd.Series({c: colors.get(c, "grey") for c in legend_carriers}),
    )

    save(fig, outdir, f"{slug(group).lower()}_time_balance")


# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["extract", "plot"])
    parser.add_argument(
        "--outdir",
        default=FIGURE_FOLDER,
        help="where the figures are written (plot only)",
    )
    parser.add_argument(
        "--price-scale",
        default="shared",
        choices=["shared", "per-case"],
        help="region colors on one scale over both cases (default) or on the "
        "range of each case, which shows the pattern within a case but hides "
        "the difference between them (plot only, maps only)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="restrict to these map carriers / time series groups",
    )
    args = parser.parse_args()

    config = load_plotting_config()

    if args.stage == "extract":
        for scenario in CASES.values():
            extract_case(scenario, config)
        return

    outdir = Path(args.outdir)
    carriers = [c for c in MAP_CARRIERS if args.only is None or c in args.only]
    groups = [g for g in TS_GROUPS if args.only is None or g in args.only]
    for carrier in carriers:
        plot_map_pair(
            carrier,
            outdir,
            config,
            shared_price_scale=args.price_scale == "shared",
        )
    for group in groups:
        plot_timeseries_pair(group, outdir, config)


if __name__ == "__main__":
    main()
