# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2025 Wolfram Tuschewitzki, Jelto Lange
#
# SPDX-License-Identifier: MIT
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde
from shapely.geometry import box

from create_basic_summary import get_base_index, load_df_without_unit_row

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(REPO_ROOT, "data", "generation_data")
BUS_LOCATIONS_PATH = os.path.join(REPO_ROOT, "data", "bus_locations.csv")
REGIONS_GEOJSON_PATH = os.path.join(
    REPO_ROOT, "data", "regions_onshore_elec_s_150.geojson"
)
SUMMARY_PATH = os.path.join(REPO_ROOT, "results", "total_summary.csv")
COLOR_DEFINITION_PATH = os.path.join(REPO_ROOT, "src", "colors.yaml")
# MWh of hydrogen per repeated row, seaborn's violinplot has no weights.
# Nodes below SCALING/2 (2 GWh/a) get no row.
SCALING = 4e3

# The figure leaves out the 950 (PC-0) and 925 EUR/kW cost points
SKIP_KEYWORDS = [
    "6pct",
    "3pct",
    "925",
    "950",
    "_net_zero",
    "_vopt",
    "_CCS",
    "_wind",
    "_transport",
    "_elheat0",
]


# --- Helper functions ---
def filter_filename(filename, skip_keywords=None):
    """
    Returns False if filename is not a generation file
    (generation_<scenario>.csv) or contains any keyword in skip_keywords
    (defaults to SKIP_KEYWORDS), True otherwise.

    Logs skipped filenames.
    """
    if not (filename.startswith("generation_") and filename.endswith(".csv")):
        return False
    keywords = SKIP_KEYWORDS if skip_keywords is None else skip_keywords
    if any(keyword in filename for keyword in keywords):
        print(f"File '{filename}' skipped (contains keywords: {', '.join(keywords)})")
        return False
    return True


def categorize_filename(filename):
    """
    Categorize filename as 'no UGHS' or 'UGHS' based on substring 'noUGHS'.
    """
    return "no UGHS" if "noUGHS" in filename else "UGHS"


def load_bus_locations(direction):
    """
    Load and transform bus locations depending on the direction:
    - vertical: keep latitude only (drop x, rename y -> latitude)
    - horizontal: keep longitude only (drop y, rename x -> longitude)
    """
    bus_locations = pd.read_csv(BUS_LOCATIONS_PATH, index_col=0)

    if direction == "vertical":
        bus_locations = bus_locations.drop(columns="x").rename(
            columns={"y": "latitude"}
        )
    elif direction == "horizontal":
        bus_locations = bus_locations.drop(columns="y").rename(
            columns={"x": "longitude"}
        )
    else:
        raise ValueError(
            f"Invalid direction '{direction}', expected 'vertical' or 'horizontal'."
        )

    return bus_locations


def extract_pc_capex(filename):
    """
    Extract PC capex numerical value from filename using regex.

    Args:
        filename (str): filename to extract from.

    Returns:
        str: extracted PC capex value.

    Raises:
        ValueError: if pattern not found.
    """
    match = re.search(r"\d+(?=Euro)", filename)
    if not match:
        raise ValueError(f"Could not extract PC capex from filename: {filename}")
    return match.group()


def repeat_rows_by_round_column(df, round_col):
    """
    Repeat rows of DataFrame according to the rounded integer values in
    round_col.

    Args:
        df (pd.DataFrame): input dataframe.
        round_col (str): column name whose rounded values determine repetition count.

    Returns:
        pd.DataFrame: dataframe with rows repeated.
    """
    repeated_df = df.loc[df.index.repeat(df[round_col].round())].reset_index(drop=True)
    return repeated_df


def process_files(filtered_files, scaling, direction, case="split"):
    """
    Process filtered files and extract demand and generation DataFrames.

    Args:
        filtered_files (list[str]): list of filenames to process.
        scaling (float): value to scale energy values.
        direction (str): 'vertical' or 'horizontal'.
        case (str): 'total' or 'split' case for generation processing.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (demand_df, generation_df)
    """
    bus_locations = load_bus_locations(direction)

    demand_dfs = []
    generation_dfs = []

    for filename in filtered_files:
        category = categorize_filename(filename)
        print(f"Processing {filename} with category {category}")
        pc_capex = extract_pc_capex(filename)

        # Load base dataframe once
        df_base = pd.read_csv(os.path.join(DATA_DIR, filename), index_col=0)

        # --- Demand Data ---
        df_demand = pd.concat([bus_locations, df_base["total_h2_demand_MWh"]], axis=1)
        df_demand["total_h2_demand"] = df_demand["total_h2_demand_MWh"] / scaling
        df_demand = df_demand.drop(columns=["total_h2_demand_MWh"])

        repeated_demand = repeat_rows_by_round_column(df_demand, "total_h2_demand")
        repeated_demand["PC capex"] = pc_capex
        repeated_demand["case"] = category

        demand_dfs.append(repeated_demand)

        # --- Generation Data ---
        if case == "total":
            df_gen = pd.concat(
                [bus_locations, df_base["h2_generation_total_MWh"]], axis=1
            )
            df_gen["h2_generation_total"] = df_gen["h2_generation_total_MWh"] / scaling
            df_gen = df_gen.drop(columns=["h2_generation_total_MWh"])

            repeated_gen = repeat_rows_by_round_column(df_gen, "h2_generation_total")
            repeated_gen["PC capex"] = pc_capex
            repeated_gen["case"] = category

            generation_dfs.append(repeated_gen)

        elif case == "split":
            gen_technos = ["Electrolysis", "Photocatalysis"]

            for techno in gen_technos:
                col_name = f"h2_generation_{techno}_MWh"
                df_gen = pd.concat([bus_locations, df_base[col_name]], axis=1)
                scaled_col = "h2_generation_scaled"
                df_gen[scaled_col] = df_gen[col_name] / scaling
                df_gen = df_gen.drop(columns=[col_name])

                repeated_gen = repeat_rows_by_round_column(df_gen, scaled_col)
                repeated_gen["PC capex"] = pc_capex
                repeated_gen["case"] = techno
                repeated_gen["ughs_state"] = category

                generation_dfs.append(repeated_gen)

        else:
            raise ValueError(f"Invalid case '{case}', expected 'total' or 'split'.")

    # Concatenate all individual frames into final DataFrames
    demand_df = pd.concat(demand_dfs, axis=0).reset_index(drop=True)
    generation_df = pd.concat(generation_dfs, axis=0).reset_index(drop=True)

    return demand_df, generation_df


def load_capex_annualized_cost_map(total_summary_df):
    """
    Map the capex token of a scenario folder name to its annualized
    photocatalysis cost.

    Args:
        total_summary_df (pd.DataFrame): summary DataFrame.

    Returns:
        dict: capex token (e.g. "660") -> annualized cost.
    """
    return {
        re.findall(r"\d+(?=Euro)", idx)[0]: round(float(val), 1)
        for idx, val in total_summary_df.loc[
            [
                i
                for i in list(total_summary_df.index)
                # drops scenarios without "<n>Euro" in the name
                if "pct_efficiency" not in i
                and "_noUGHS" not in i
                and "Unit" not in i
                and re.findall(r"\d+(?=Euro)", i)
            ],
            :,
        ]["pc total annual cost"].items()
    }


def prepare_swarm_data(
    capex_annualized_cost_map, direction="vertical", skip_keywords=None
):
    """
    One row per (scenario, node, series) for the beeswarm overlay, with the
    number of rows the node has in the violin. Nodes without a row are dropped.

    Args:
        capex_annualized_cost_map (dict): capex token -> annualized cost.
        direction (str): 'vertical' or 'horizontal'.
        skip_keywords (list[str], optional): passed on to filter_filename,
            defaults to SKIP_KEYWORDS.

    Returns:
        pd.DataFrame: columns Bus, latitude/longitude, value_MWh, n_rows,
        "PC capex", series.
    """
    series_columns = {
        "Electrolysis": "h2_generation_Electrolysis_MWh",
        "Photocatalysis": "h2_generation_Photocatalysis_MWh",
        "Demand": "total_h2_demand_MWh",
    }
    coordinate = "latitude" if direction == "vertical" else "longitude"
    bus_locations = load_bus_locations(direction)

    frames = []
    for filename in [
        f for f in os.listdir(DATA_DIR) if filter_filename(f, skip_keywords)
    ]:
        if categorize_filename(filename) != "UGHS":
            continue
        pc_capex = extract_pc_capex(filename)
        df_base = pd.read_csv(os.path.join(DATA_DIR, filename), index_col=0)

        for label, column in series_columns.items():
            values = df_base[column]
            frames.append(
                pd.DataFrame(
                    {
                        coordinate: bus_locations[coordinate],
                        "value_MWh": values,
                        "n_rows": (values / SCALING).round(),
                        "PC capex": float(capex_annualized_cost_map[pc_capex]),
                        "series": label,
                    }
                )
            )

    swarm_df = pd.concat(frames).reset_index(names="Bus")
    return swarm_df[swarm_df["n_rows"] >= 1].reset_index(drop=True)


def volume_profiles(swarm_df, grid, bandwidth, coordinate="latitude"):
    """
    Hydrogen volume per degree of `coordinate` for every (scenario, series).

    Weighted Gaussian KDE (scipy's gaussian_kde, weights = volume of each
    node) with a fixed kernel standard deviation of `bandwidth` degrees,
    multiplied by the total volume of the series. The result integrates to
    that volume instead of 1, so all series share one unit (TWh/a per degree)
    and can be added up for the stacked violins.

    Args:
        swarm_df (pd.DataFrame): one row per node, as prepare_swarm_data()
            returns it (needs `coordinate`, "n_rows", "PC capex", "series").
        grid (np.ndarray): coordinate values to evaluate the density on.
        bandwidth (float): kernel bandwidth in degrees, common to all series.
        coordinate (str): "latitude" or "longitude".

    Returns:
        dict[tuple[float, str], np.ndarray]: (scenario, series) -> TWh per
        degree on `grid`.
    """
    twh_per_row = SCALING / 1e6
    profiles = {}
    for (scenario, series), group in swarm_df.groupby(["PC capex", "series"]):
        coordinates = group[coordinate].to_numpy()
        weights = group["n_rows"].to_numpy() * twh_per_row
        # scipy's bw_method is a factor on the weighted standard deviation of
        # the data; dividing by that std fixes the kernel std at `bandwidth`
        # degrees for every series (np.cov as gaussian_kde computes it)
        weighted_std = np.sqrt(np.cov(coordinates, aweights=weights))
        kde = gaussian_kde(
            coordinates, weights=weights, bw_method=bandwidth / weighted_std
        )
        # gaussian_kde integrates to 1; times the total volume gives TWh/a per degree
        profiles[(scenario, series)] = kde(grid) * weights.sum()
    return profiles


def get_density_data(
    orientation,
    bus_locations_path=BUS_LOCATIONS_PATH,
    geojson_path=REGIONS_GEOJSON_PATH,
    step=0.25,
    coord_limits=None,
    fixed_limits=None,
):
    """
    Compute node_density and expanded_area_density dataframes for a given
    orientation.

    Args:
        orientation (str): "vertical" or "horizontal"
        bus_locations_path (str): path to bus_locations.csv
        geojson_path (str): path to regions geojson file
        step (float): binning step size in degrees
        coord_limits (tuple[float, float], optional): min/max of variable axis coordinate
        fixed_limits (tuple[float, float], optional): min/max of fixed axis coordinate

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (node_density_df, expanded_area_density_df)
    """
    assert orientation in (
        "vertical",
        "horizontal",
    ), "orientation must be 'vertical' or 'horizontal'"

    # Defaults if no limits are provided
    if coord_limits is None:
        coord_limits = (35, 70) if orientation == "vertical" else (-11, 32)
    if fixed_limits is None:
        fixed_limits = (-11, 32) if orientation == "vertical" else (35, 70)

    # Load bus locations
    node_density = pd.read_csv(bus_locations_path, index_col=0)

    # Map orientation to column mappings and binning axis
    config = {
        "vertical": {"var_source": "y", "var_target": "Latitude", "fixed_col": "x"},
        "horizontal": {"var_source": "x", "var_target": "longitude", "fixed_col": "y"},
    }
    conf = config[orientation]

    # Rename node_density columns accordingly
    node_density = node_density.rename(columns={conf["var_source"]: conf["var_target"]})
    node_density[conf["fixed_col"]] = " "
    node_density["hue"] = "Node density"

    # Load GeoJSON and prepare projected geodataframe
    gdf = gpd.read_file(geojson_path)
    cart_gdf = gdf.copy()
    cart_gdf.set_index("name", inplace=True)
    cart_gdf = cart_gdf.to_crs(3035)

    # Prepare bins along variable axis
    coord_min, coord_max = coord_limits
    fixed_min, fixed_max = fixed_limits

    bin_edges = np.arange(coord_min, coord_max + step, step)

    bin_geodfs = {}
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if orientation == "vertical":
            # bbox: lon_min, lat_min, lon_max, lat_max
            bbox = box(fixed_min, start, fixed_max, end)
        else:
            # bbox: lon_min, lat_min, lon_max, lat_max
            bbox = box(start, fixed_min, end, fixed_max)
        geobox = gpd.GeoDataFrame({"geometry": [bbox]}, crs="epsg:4326").to_crs(
            cart_gdf.crs
        )
        bin_geodfs[(start, end)] = geobox

    # Calculate the overlap area for each bin
    area_dict = {}
    for (start, end), bin_gdf in bin_geodfs.items():
        intersection = gpd.overlay(cart_gdf, bin_gdf, how="intersection")
        area_km2 = intersection.geometry.area.sum() / 1e7  # convert m² to km²
        area_dict[(start, end)] = area_km2

    # Generate DataFrame for area expanded by area in km² (repeat)
    rows = [{"mid": (s + e) / 2, "area_sqkm": a} for (s, e), a in area_dict.items()]
    area_df = pd.DataFrame(rows)

    # Handle empty or zero-area cases gracefully before repeat
    if area_df.empty or area_df["area_sqkm"].sum() == 0:
        expanded_area_density_df = pd.DataFrame(
            columns=[conf["var_target"], conf["fixed_col"], "hue"]
        )
    else:
        expanded_area_density_df = area_df.loc[
            area_df.index.repeat(area_df["area_sqkm"])
        ].copy()
        expanded_area_density_df[conf["var_target"]] = expanded_area_density_df["mid"]
        expanded_area_density_df[conf["fixed_col"]] = " "
        expanded_area_density_df["hue"] = "Land area density"
        expanded_area_density_df.drop(columns=["area_sqkm", "mid"], inplace=True)

    return node_density, expanded_area_density_df


# Function to rename based on the mapping
def rename_value(value, mapping):
    """
    Rename a value based on substring matching using provided mapping.

    Args:
        value (str): original string value.
        mapping (dict): dictionary of {old_substring: new_substring}.

    Returns:
        str: renamed value if matched, else original.
    """
    for old, new in mapping.items():
        if str(old) in value:
            return value.replace(str(old), str(new))
    return value


def apply_rename_and_convert(df, column_name, rename_map):
    """
    Apply renaming to DataFrame column values and convert column to float.

    Args:
        df (pd.DataFrame): input dataframe.
        column_name (str): column to rename and convert.
        rename_map (dict): mapping for renaming values.

    Returns:
        pd.DataFrame: modified dataframe.
    """
    df = df.copy()
    df[column_name] = df[column_name].apply(lambda x: rename_map.get(x, x))
    df[column_name] = df[column_name].astype(float)
    return df


def node_counts_per_violin(
    direction="vertical", market_share_map=None, all_cost_points=False
):
    """
    Number of nodes each violin of the generation/demand figure is built from,
    i.e. the nodes with at least one row.

    Args:
        direction (str): 'vertical' or 'horizontal'.
        market_share_map (dict, optional): annualized cost (as str) -> market
            share
        all_cost_points (bool): include the 950 and 925 Euro cost points

    Returns:
        pd.DataFrame: one row per scenario, columns Electrolysis,
        Photocatalysis, Demand and Total nodes.
    """
    skip_keywords = (
        [k for k in SKIP_KEYWORDS if k not in ("925", "950")]
        if all_cost_points
        else None
    )
    total_summary_df = load_df_without_unit_row(SUMMARY_PATH)
    swarm_df = prepare_swarm_data(
        load_capex_annualized_cost_map(total_summary_df), direction, skip_keywords
    )

    counts = (
        swarm_df.groupby(["PC capex", "series"])
        .size()
        .unstack("series")
        # PC-0 has no photocatalysis node
        .reindex(columns=["Electrolysis", "Photocatalysis", "Demand"], fill_value=0)
        .fillna(0)
        .astype(int)
        .rename_axis(columns=None)
    )
    counts["Total nodes"] = len(load_bus_locations(direction))

    if market_share_map is not None:
        counts.insert(0, "Annualized cost", counts.index)
        counts.index = [
            float(market_share_map.get(str(cost), cost)) for cost in counts.index
        ]
        counts.index.name = "Market share"
    return counts.sort_index()


def prepare_data(direction="vertical"):
    """
    Build the violin input for one direction only.

    `direction="vertical"` bins the nodes by latitude, `"horizontal"` by
    longitude.

    Args:
        direction (str): 'vertical' or 'horizontal'.

    Returns:
        tuple: (total_summary_df, generation_violin_split_df, demand_violin_df,
        node_density_and_land_density, capex_annualized_cost_map, base_index,
        colors_dict)
    """
    filtered_files = [f for f in os.listdir(DATA_DIR) if filter_filename(f)]

    demand_violin_df, generation_violin_split_df = process_files(
        filtered_files, scaling=SCALING, direction=direction, case="split"
    )
    node_density_and_land_density = pd.concat(get_density_data(direction), axis=0)

    # Create annualized photocatalysis mapping depending on the pc_capex stored in the index
    total_summary_df = load_df_without_unit_row(SUMMARY_PATH)
    capex_annualized_cost_map = load_capex_annualized_cost_map(total_summary_df)

    generation_violin_split_df = apply_rename_and_convert(
        generation_violin_split_df, "PC capex", capex_annualized_cost_map
    )

    demand_violin_df = apply_rename_and_convert(
        demand_violin_df, "PC capex", capex_annualized_cost_map
    )

    base_index, *_ = get_base_index(total_summary_df)

    with open(COLOR_DEFINITION_PATH) as stream:
        colors_dict = yaml.safe_load(stream)

    return (
        total_summary_df,
        generation_violin_split_df,
        demand_violin_df,
        node_density_and_land_density,
        capex_annualized_cost_map,
        base_index,
        colors_dict,
    )
