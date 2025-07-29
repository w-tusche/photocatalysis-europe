# -*- coding: utf-8 -*-
import os
import re

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from shapely.geometry import box

from scatter_plots import get_base_index

# --- Global paths and constants ---
DATA_DIR = r"./data/generation_data"
BUS_LOCATIONS_PATH = r"./data/bus_locations.csv"
REGIONS_GEOJSON_PATH = r"./data/regions_onshore_elec_s_150.geojson"
SUMMARY_PATH = r"./results/total_summary.csv"
COLOR_DEFINITION_PATH = r"./src/colors.yaml"
SKIP_KEYWORDS = ["6pct", "3pct", "900", "925", "_net_zero"]


# --- Helper functions ---
def filter_filename(filename):
    """
    Returns False if filename contains any keyword in SKIP_KEYWORDS, True
    otherwise.

    Logs skipped filenames.
    """
    if any(keyword in filename for keyword in SKIP_KEYWORDS):
        print(
            f"File '{filename}' skipped (contains keywords: {', '.join(SKIP_KEYWORDS)})"
        )
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


def prepare_all_data():
    filtered_files = [f for f in os.listdir(DATA_DIR) if filter_filename(f)]

    # Process all files once and get both dataframes
    demand_violin_df_vertical, generation_violin_split_df_vertical = process_files(
        filtered_files, scaling=1e4, direction="vertical", case="split"
    )
    node_density_and_land_density_vertical = pd.concat(
        get_density_data("vertical"), axis=0
    )

    demand_violin_df_horizontal, generation_violin_split_df_horizontal = process_files(
        filtered_files, scaling=1e4, direction="horizontal", case="split"
    )
    node_density_and_land_density_horizontal = pd.concat(
        get_density_data("horizontal"), axis=0
    )

    # Create annualized photocatalysis mapping depending on the pc_capex stored in the index
    total_summary_df = pd.read_csv(
        SUMMARY_PATH, index_col=0, header=0, skiprows=range(1, 2)
    )
    capex_annualized_cost_map = {
        re.findall(r"\d+(?=Euro)", idx)[0]: round(float(val), 1)
        for idx, val in total_summary_df.loc[
            [
                i
                for i in list(total_summary_df.index)
                if "pct_efficiency" not in i and "_noUGHS" not in i and "Unit" not in i
            ],
            :,
        ]["pc total annual cost"].items()
    }

    generation_violin_split_df_vertical = apply_rename_and_convert(
        generation_violin_split_df_vertical, "PC capex", capex_annualized_cost_map
    )

    demand_violin_df_vertical = apply_rename_and_convert(
        demand_violin_df_vertical, "PC capex", capex_annualized_cost_map
    )

    generation_violin_split_df_horizontal = apply_rename_and_convert(
        generation_violin_split_df_horizontal, "PC capex", capex_annualized_cost_map
    )

    demand_violin_df_horizontal = apply_rename_and_convert(
        demand_violin_df_horizontal, "PC capex", capex_annualized_cost_map
    )

    base_index, _, _, _, _ = get_base_index(total_summary_df)

    with open(COLOR_DEFINITION_PATH) as stream:
        colors_dict = yaml.safe_load(stream)

    return (
        total_summary_df,
        generation_violin_split_df_vertical,
        demand_violin_df_vertical,
        node_density_and_land_density_vertical,
        generation_violin_split_df_horizontal,
        demand_violin_df_horizontal,
        node_density_and_land_density_horizontal,
        base_index,
        colors_dict,
    )


def plot_hydrogen_generation_and_demand_violins_vertical(
    total_summary_df,
    generation_violin_split_df_vertical,
    demand_violin_df_vertical,
    node_density_and_land_density_vertical,
    base_index,
    colors_dict,
):
    background_map = True
    violin_width = 1

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 6), width_ratios=[0.75, 0.25]
    )

    # Violin for Electrolysis H2 generation (aim to have it on the left side)
    sns.violinplot(
        data=generation_violin_split_df_vertical[
            (generation_violin_split_df_vertical["case"] == "Electrolysis")
            & (generation_violin_split_df_vertical["ughs_state"] == "UGHS")
        ],
        x="PC capex",
        y="latitude",
        # Create only left side
        hue=True,
        hue_order=[False, True],
        split=True,
        palette=[colors_dict["tech_colors"]["H2 Electrolysis"]] * 2,
        dodge=True,
        orient="v",
        inner="quart",  # quartiles of the data
        bw_method="silverman",
        density_norm="count",
        common_norm=True,
        fill=True,
        ax=ax1,
        label=r"Hydrogen generation by electrolysis",
        legend=False,
        native_scale=True,
        width=violin_width,
        zorder=-3,
    )

    # Violin for Photocatalysis H2 generation (aim to have it on the left side)
    sns.violinplot(
        data=generation_violin_split_df_vertical[
            (generation_violin_split_df_vertical["case"] == "Photocatalysis")
            & (generation_violin_split_df_vertical["ughs_state"] == "UGHS")
        ],
        x="PC capex",
        y="latitude",
        # Create only left side
        hue=True,
        hue_order=[False, True],
        split=True,
        palette=[colors_dict["tech_colors"]["photocatalysis"]] * 2,
        dodge=True,
        orient="v",
        inner="quart",  # quartiles of the data
        density_norm="count",
        bw_method="silverman",
        common_norm=True,
        fill=True,
        ax=ax1,
        label=r"Hydrogen generation by photocatalysis",
        legend=False,
        native_scale=True,
        width=violin_width,
        zorder=-2,
    )

    ax1.set_xlim([115, 163])
    ax1_pos = ax1.get_position()
    ax1_2 = fig.add_axes(
        [ax1_pos.x0 + 0.0127, ax1_pos.y0, ax1_pos.width, ax1_pos.height], frame_on=False
    )
    sns.violinplot(
        data=demand_violin_df_vertical[(demand_violin_df_vertical["case"] == "UGHS")],
        x="PC capex",
        y="latitude",
        hue=True,
        hue_order=[True, False],
        palette=[colors_dict["tech_colors"]["hydrogen"]] * 2,
        split=True,  # Does not work as expected!
        orient="v",
        inner="quart",  # quartiles of the data
        bw_method="silverman",
        gap=0.5,
        density_norm="count",
        common_norm=True,
        fill=True,
        legend=False,
        ax=ax1_2,
        label=r"Hydrogen demand",
        native_scale=True,
        width=violin_width,
        zorder=-1,
    )

    ax1_2.axis(False)

    #  Add description for PC-50 case
    ax1.annotate(
        "PC-50",
        xy=(118.4, 69.5),  # Arrow target position (on x-axis)
        xytext=(118.4, 1.05),  # Text position (above x-axis)
        textcoords=("data", "axes fraction"),  # Mix of data and fraction coords
        arrowprops=dict(arrowstyle="-", facecolor="black", linewidth=2),
        ha="center",
    )

    # --- Add market share line
    ax1_y2 = ax1.twinx()

    ax1_y2.plot(
        total_summary_df.loc[base_index, "pc total annual cost"],
        total_summary_df.loc[base_index, "pc market share"],
        ls="--",
        marker="s",
        c=colors_dict["others"]["UGHS_line"],
        alpha=0.7,
        label="Market share",
    )

    ax1_y2.set_ylim([0, 55])
    ax1_y2.set_ylabel("Photocatalysis market share in %", fontsize=15)
    ax1_y2.tick_params(
        labelsize=13,
    )
    # --- Add market share line

    #### Create plot to the right with node density and land area density
    sns.violinplot(
        data=node_density_and_land_density_vertical,
        x="x",
        y="Latitude",
        hue="hue",
        palette=[
            colors_dict["others"]["node density"],
            colors_dict["others"]["land area density"],
        ],
        split=True,
        orient="v",
        inner="quart",  # quartiles of the data
        density_norm="count",
        bw_method="silverman",
        fill=True,
        ax=ax2,
        zorder=-2,
    )

    # Add descriptive annotations with arrows
    df_for_description = node_density_and_land_density_vertical[
        node_density_and_land_density_vertical["hue"] == "Node density"
    ]["Latitude"]
    q1 = df_for_description.quantile(0.25)
    median = df_for_description.median()
    q3 = df_for_description.quantile(0.75)

    ax2.annotate(
        "Quartile 1",
        xy=(-0.1, q1),
        xytext=(-0.45, 37),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    ax2.annotate(
        "Median",
        xy=(-0.2, median),
        xytext=(-0.49, 42),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    ax2.annotate(
        "Quartile 3",
        xy=(-0.08, q3),
        xytext=(-0.49, 57),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    ax2.annotate(
        "Kernel\ndensity",
        xy=(-0.065, 60),
        xytext=(-0.45, 62),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )

    ax1.set_xlabel(
        r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$", fontsize=15
    )
    ax1.set_ylabel("Latitude in °N", fontsize=15)
    ax2.set_ylabel("Latitude in °N", fontsize=15)
    ax2.set_xlabel("PyPSA-Eur 150 nodes", fontsize=15)

    # Gather handles and labels from both axes
    handles, labels = ax1.get_legend_handles_labels()  # Supply EL & PC
    handles2, labels2 = ax1_2.get_legend_handles_labels()  # H2 Demand
    handles3, labels3 = ax1_y2.get_legend_handles_labels()  # Market share

    # Combine and deduplicate
    unique_labels = dict(zip(labels + labels2 + labels3, handles + handles2 + handles3))

    ax1.set_ylim([35, 70])
    ax1_2.set_ylim([35, 70])
    ax2.set_ylim([35, 70])
    # ax1_2.set_xlim(ax1.get_xlim())
    # TODO: When changing aspect ratio or distance between subplots the first value here has to be adjusted to correctly show the demand violins
    ax1_2.set_xlim((112.9, 163.0))

    # Create the legend below the plot
    ax1.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    sns.move_legend(
        ax2,
        "upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1,
        title=None,
        frameon=False,
        fontsize=14,
    )

    ax1.grid(alpha=0.3, zorder=20)
    ax2.grid(alpha=0.3, zorder=20)
    ax1_2.yaxis.grid(True, alpha=0.3, zorder=20)

    # Redraw lines on top to have a grid also on the green violins
    gridlines = ax1.yaxis.get_gridlines()
    gridline_color = gridlines[0].get_color()
    gridline_linestyle = gridlines[0].get_linestyle()
    gridline_linewidth = gridlines[0].get_linewidth()

    for y in ax1.get_yticks():  # Adjust range and step based on your y-axis limits
        ax1_2.axhline(
            y=y,
            color=gridline_color,
            xmin=0,
            xmax=0.95,
            linestyle=gridline_linestyle,
            linewidth=gridline_linewidth,
            zorder=10,
            alpha=0.3,
        )

    ax1.invert_xaxis()
    ax1_2.invert_xaxis()

    ax1.tick_params(
        axis="both",
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelsize=13,
    )
    ax2.tick_params(
        axis="both",
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelsize=13,
    )

    # TODO: When changing this the first value in ax1_2.set_xlim() has to be changed!
    plt.subplots_adjust(wspace=0.3)

    if background_map:
        proj_kwargs = {"name": "EqualEarth"}

        proj_func = getattr(ccrs, proj_kwargs.pop("name"))
        proj = proj_func(**proj_kwargs)

        regions = gpd.read_file(REGIONS_GEOJSON_PATH).set_index("name")
        regions = regions.to_crs(proj.proj4_init)

        ax1_map = fig.add_axes(
            ax1_pos, projection=proj, zorder=-10
        )  # Create overlay axes

        # Plot the map onto the overlay axes
        regions.plot(
            ax=ax1_map, linewidth=1, color="lightgrey", edgecolor="white", alpha=0.4
        )
        ax1_map.set_extent([-9.5, 26.5, 35, 70])  # Set the extent of the map
        ax1_map.axis("off")  # Turn off the axis to show only the map

        # Adjust ax1 to be transparent or semi-transparent to show the map underneath
        ax1.set_facecolor("none")  # Make the background of ax1

    plt.show()


def plot_hydrogen_generation_and_demand_violins_horizontal(
    total_summary_df,
    generation_violin_split_df_horizontal,
    demand_violin_df_horizontal,
    node_density_and_land_density_horizontal,
    base_index,
    colors_dict,
):
    background_map = True
    violin_width = 1

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 10), height_ratios=[0.75, 0.25]
    )

    # Violin for Electrolysis H2 generation (aim to have it on the left side)
    sns.violinplot(
        data=generation_violin_split_df_horizontal[
            (generation_violin_split_df_horizontal["case"] == "Electrolysis")
            & (generation_violin_split_df_horizontal["ughs_state"] == "UGHS")
        ],
        x="longitude",
        y="PC capex",
        # Create only left side
        hue=True,
        hue_order=[False, True],
        split=True,
        palette=[colors_dict["tech_colors"]["H2 Electrolysis"]] * 2,
        dodge=True,
        orient="h",
        inner="quart",  # quartiles of the data
        bw_method="silverman",
        density_norm="count",
        common_norm=True,
        fill=True,
        ax=ax1,
        label=r"Hydrogen generation by electrolysis",
        legend=False,
        native_scale=True,
        width=violin_width,
        zorder=-3,
        alpha=0.85,
    )

    # Violin for Photocatalysis H2 generation (aim to have it on the left side)
    sns.violinplot(
        data=generation_violin_split_df_horizontal[
            (generation_violin_split_df_horizontal["case"] == "Photocatalysis")
            & (generation_violin_split_df_horizontal["ughs_state"] == "UGHS")
        ],
        x="longitude",
        y="PC capex",
        # Create only left side
        hue=True,
        hue_order=[False, True],
        split=True,
        palette=[colors_dict["tech_colors"]["photocatalysis"]] * 2,
        dodge=True,
        orient="h",
        inner="quart",  # quartiles of the data
        density_norm="count",
        bw_method="silverman",
        common_norm=True,
        fill=True,
        ax=ax1,
        label=r"Hydrogen generation by photocatalysis",
        legend=False,
        native_scale=True,
        width=violin_width,
        zorder=-2,
        alpha=0.85,
    )

    ax1.set_ylim([115, 165])
    ax1_pos = ax1.get_position()
    ax1_2 = fig.add_axes(
        [ax1_pos.x0, ax1_pos.y0, ax1_pos.width, ax1_pos.height], frame_on=False
    )

    x0, x1 = ax1.get_xlim()
    ax1_2.set_xlim([x0, x1])
    y0, y1 = ax1.get_ylim()
    dy = 1.4
    ax1_2.set_ylim([y0 + dy, y1 + dy])

    sns.violinplot(
        data=demand_violin_df_horizontal[
            (demand_violin_df_horizontal["case"] == "UGHS")
        ],
        x="longitude",
        y="PC capex",
        hue=True,
        hue_order=[True, False],
        palette=[colors_dict["tech_colors"]["hydrogen"]] * 2,
        split=True,  # Does not work as expected!
        orient="h",
        inner="quart",  # quartiles of the data
        bw_method="silverman",
        gap=0.5,
        density_norm="count",
        common_norm=True,
        fill=True,
        legend=False,
        ax=ax1_2,
        label=r"Hydrogen demand",
        native_scale=True,
        width=violin_width,
        zorder=-1,
        alpha=0.85,
    )

    ax1_2.axis(False)

    #  Add description for PC-50 case
    ax1.annotate(
        "PC-50",
        xy=(30, 118.4),  # Arrow target position (on x-axis)
        xytext=(1.05, 117.9),  # Text position (above x-axis)
        textcoords=("axes fraction", "data"),  # Mix of data and fraction coords
        arrowprops=dict(arrowstyle="-", facecolor="black", linewidth=2),
        ha="center",
    )

    # --- Add market share line
    ax1_y2 = ax1.twiny()

    ax1_y2.plot(
        total_summary_df.loc[base_index, "pc market share"],
        total_summary_df.loc[base_index, "pc total annual cost"],
        ls="--",
        marker="s",
        c=colors_dict["others"]["UGHS_line"],
        alpha=0.7,
        label="Market share",
    )

    ax1_y2.set_xlim([0, 55])
    ax1_y2.set_xlabel("Photocatalysis market share in %", fontsize=15)
    ax1_y2.tick_params(
        labelsize=13,
    )
    # --- Add market share line

    #### Create plot to the right with node density and land area density
    sns.violinplot(
        data=node_density_and_land_density_horizontal,
        x="longitude",
        y="y",
        hue="hue",
        palette=[
            colors_dict["others"]["node density"],
            colors_dict["others"]["land area density"],
        ],
        split=True,
        orient="h",
        inner="quart",  # quartiles of the data
        density_norm="count",
        bw_method="silverman",
        fill=True,
        ax=ax2,
        zorder=-2,
    )

    # Add descriptive annotations with arrows
    df_for_description = node_density_and_land_density_horizontal[
        node_density_and_land_density_horizontal["hue"] == "Node density"
    ]["longitude"]
    q1 = df_for_description.quantile(0.25)
    median = df_for_description.median()
    q3 = df_for_description.quantile(0.75)

    ax1.set_ylabel(
        r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$", fontsize=15
    )
    ax1.set_xlabel("longitude in °N", fontsize=15)
    ax2.set_xlabel("longitude in °N", fontsize=15)
    ax2.set_ylabel("PyPSA-Eur\n150 nodes", fontsize=15)

    # Gather handles and labels from both axes
    handles, labels = ax1.get_legend_handles_labels()  # Supply EL & PC
    handles2, labels2 = ax1_2.get_legend_handles_labels()  # H2 Demand
    handles3, labels3 = ax1_y2.get_legend_handles_labels()  # Market share

    # Combine and deduplicate
    unique_labels = dict(zip(labels + labels2 + labels3, handles + handles2 + handles3))

    # Create the legend below the plot
    ax1.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.65),
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    sns.move_legend(
        ax2,
        "upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
        title=None,
        frameon=False,
        fontsize=14,
    )

    ax1.grid(alpha=0.3, zorder=20)
    ax2.grid(alpha=0.3, zorder=20)
    ax1_2.xaxis.grid(True, alpha=0.3, zorder=20)
    # FIXME: turn off the grid!
    ax1_2.yaxis.grid(False)
    ax1_2.yaxis.grid("off")
    ax2.set_xlim(ax1.get_xlim())

    # Redraw lines on top to have a grid also on the green violins
    gridlines = ax1.yaxis.get_gridlines()
    gridline_color = gridlines[0].get_color()
    gridline_linestyle = gridlines[0].get_linestyle()
    gridline_linewidth = gridlines[0].get_linewidth()

    for y in ax1.get_yticks():  # Adjust range and step based on your y-axis limits
        ax1_2.axhline(
            y=y,
            color=gridline_color,
            xmin=0,
            xmax=0.95,
            linestyle=gridline_linestyle,
            linewidth=gridline_linewidth,
            zorder=10,
            alpha=0.3,
        )

    ax1.tick_params(
        axis="both",
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelsize=13,
    )
    ax2.tick_params(
        axis="both",
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelsize=13,
    )

    # plt.subplots_adjust(wspace=0.3)

    if background_map:
        proj_kwargs = {"name": "EqualEarth"}

        proj_func = getattr(ccrs, proj_kwargs.pop("name"))
        proj = proj_func(**proj_kwargs)

        regions = gpd.read_file(REGIONS_GEOJSON_PATH).set_index("name")
        regions = regions.to_crs(proj.proj4_init)

        ax1_map = fig.add_axes(
            ax1_pos, projection=proj, zorder=-10
        )  # Create overlay axes

        # Plot the map onto the overlay axes
        regions.plot(
            ax=ax1_map, linewidth=1, color="lightgrey", edgecolor="white", alpha=0.4
        )
        ax1_map.set_extent([-9.5, 26.5, 35, 70])  # Set the extent of the map
        ax1_map.axis("off")  # Turn off the axis to show only the map

        # Adjust ax1 to be transparent or semi-transparent to show the map underneath
        ax1.set_facecolor("none")  # Make the background of ax1

    plt.show()


def main():
    (
        total_summary_df,
        generation_violin_split_df_vertical,
        demand_violin_df_vertical,
        node_density_and_land_density_vertical,
        generation_violin_split_df_horizontal,
        demand_violin_df_horizontal,
        node_density_and_land_density_horizontal,
        base_index,
        colors_dict,
    ) = prepare_all_data()

    plot_hydrogen_generation_and_demand_violins_vertical(
        total_summary_df,
        generation_violin_split_df_vertical,
        demand_violin_df_vertical,
        node_density_and_land_density_vertical,
        base_index,
        colors_dict,
    )

    plot_hydrogen_generation_and_demand_violins_horizontal(
        total_summary_df,
        generation_violin_split_df_horizontal,
        demand_violin_df_horizontal,
        node_density_and_land_density_horizontal,
        base_index,
        colors_dict,
    )


if __name__ == "__main__":
    main()
