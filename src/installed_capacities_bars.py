# -*- coding: utf-8 -*-
import copy
import os
import re

import cartopy.crs as ccrs
import cmcrameri
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml


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


def get_first_appearance_of_pc(tot_sum_filepath, case):
    if case not in ["aghs", "ughs"]:
        raise ValueError("Select valid case (ughs, aghs)")

    """Return a set of all regions where photocatalysis is installed
    Return dict where key gives capex and values represent
    locations (region codes) where for this capex pc was
    installed for the first time

    Args:
        tot_sum_filepath (str): Filepath to the total summary
        case (str): select case here, either ughs (with underground hydrogen storage) or aghs (only above ground hydrogen storage)

    Returns:
        tuple: set of regions with photocatalysis, dict when photocatalysis is installed for the first time
    """
    tot_sum = pd.read_csv(tot_sum_filepath, index_col=0, header=0, skiprows=range(1, 2))

    # For case with underground storage (additionally filter out 3_pct_efficiency and 6_pct_efficiency vals. & net_zero)
    if case == "ughs":
        filtered_tot_sum = tot_sum[
            (tot_sum["ughs allowed"] == 1)
            & (~tot_sum.index.str.contains("_efficiency"))
            & (~tot_sum.index.str.contains("_net_zero"))
        ].copy()
    # For case without underground storage (additionally filter out 3_pct_efficiency and 6_pct_efficiency vals. & net_zero)
    elif case == "aghs":
        filtered_tot_sum = tot_sum[
            (tot_sum["ughs allowed"] == 0)
            & (~tot_sum.index.str.contains("_efficiency"))
            & (~tot_sum.index.str.contains("_net_zero"))
        ].copy()
    else:
        raise ValueError("Select valid case (ughs, aghs)")
    # Sort by descending capex to see where PC is installed first at what capex
    filtered_tot_sum.sort_values("pc capex", inplace=True, ascending=False)
    filtered_tot_sum.fillna({"pc larger 1MW": ""}, inplace=True)
    first_appearance_of_region = {}
    claimed_regions = set()
    for pc_regions, pc_capex in zip(
        filtered_tot_sum["pc larger 1MW"], filtered_tot_sum["pc capex"]
    ):
        if pc_regions == "":
            continue
        new_regions = set(pc_regions.split(";")) - claimed_regions
        claimed_regions = claimed_regions.union(new_regions)
        first_appearance_of_region[int(round(float(pc_capex), 0))] = new_regions
    return claimed_regions, first_appearance_of_region


def extract_buses_for_technology(technology, case):
    if technology not in [
        "photocatalysis",
        "H2 Electrolysis",
        "H2 Fuel Cell",
        "H2 turbine",
    ]:
        raise ValueError(
            "Select valid case ['photocatalysis', 'H2 Electrolysis', 'H2 Fuel Cell	', 'H2 turbine']"
        )

    bus_files = os.listdir("./data/bus_sizes")
    assert all(
        True if "Euro" in file else False for file in bus_files
    ), "Check naming convention before continuing, there should be an Euro in the file_name. (Go to the storage on T and correct this.)"

    if case == "ughs":
        filtered_files = [
            file_name
            for file_name in bus_files
            if "noUGHS" not in file_name
            and "efficiency" not in file_name
            and not "_net_zero" in file_name
        ]
    elif case == "aghs":
        filtered_files = [
            file_name
            for file_name in bus_files
            if "noUGHS" in file_name
            and "efficiency" not in file_name
            and not "_net_zero" in file_name
        ]

    # Store (pc_capex, buses pd.Series)
    bus_sizes_list = []
    for file in filtered_files:
        # FIXME: once moved to non ipynb remove one of the leading dots
        bus_sizes = pd.read_csv(
            os.path.join(r".\data\bus_sizes", file), index_col=[0, 1]
        )
        bus_sizes = pd.Series(bus_sizes["p_nom_opt"], index=bus_sizes.index)
        # Extract the bus data
        buses = bus_sizes[(bus_sizes.index.get_level_values(1) == technology)]
        bus_sizes_list.append((int(re.findall(r"\d+(?=Euro)", file)[0]), buses))
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
    for i, capex in enumerate(
        sorted(["650", "700", "750", "800", "850", "875", "900", "925"])
    ):
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
    for idx, node in enumerate(df.index):
        if node in [
            "ES1 8",
            "ES4 0",
            "PT1 1",
            "GR1 1",
            "ES1 6",
            "GR1 0",
            "IT1 11",
            "ES1 10",
            "ES1 4",
            "IT1 7",
            "ES1 2",
            "ES1 1",
            "ES1 5",
            "PT1 0",
            "FR1 1",
            "ES1 0",
            "IT1 0",
        ]:
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
    for i, capex in enumerate(
        sorted(["650", "700", "750", "800", "850", "875", "900", "925"], reverse=True)
    ):
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
):
    # Initial dimensions and increments
    init_heigth = 0.4
    init_length = 1
    heigth_increment = 0.13
    length_increment = 0.3

    # Number of rectangles
    num_rectangles = 8

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
            rect = patches.Rectangle(
                (0.1 * dir, 0),
                current_length,
                current_heigth,
                edgecolor="black",
                facecolor=colors_to_use[idx],
            )

            # Add the rectangle to the axes
            ax.add_patch(rect)

        ax.text(
            (init_length + (num_rectangles - 1) * length_increment) * dir,
            1.4 + y_label_adjustment,
            text_label,
            # fontsize=10,
            horizontalalignment=ha,
            verticalalignment="center",
            color="black",
            weight=font_weight,
        )

    ax.set_title(
        "Bar stacking visualization",
        fontdict=dict(
            # size=11,
            weight=font_weight
        ),
        pad=-5 + title_adjustment,
    )

    # # Set limits for x and y axis to accommodate all rectangles
    ax.set_xlim(-3.4, 3.4)
    # ax_overlay_stacked.set_xlim(-3.4,3.4)
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
    # The pipeline infrastructure of this network is shown in the plot!
    network_for_the_basis_filename = r".\results\raw\150_lv1.25_I_H_2045_3H_PC_925Euro\elec_s_150_lv1.25__I-H_2045.nc"
    total_summary_filename = r".\results\total_summary.csv"

    n = pypsa.Network(network_for_the_basis_filename)

    # FIXME: once moved to non ipynb remove one of the leading dots
    regions = gpd.read_file(r".\data\regions_onshore_elec_s_150.geojson").set_index(
        "name"
    )

    claimed_regions, first_appearances = get_first_appearance_of_pc(
        total_summary_filename, "ughs"
    )

    bus_sizes_list_pc = extract_buses_for_technology("photocatalysis", "ughs")
    bus_sizes_list_el = extract_buses_for_technology("H2 Electrolysis", "ughs")

    bus_size_factor_to_GW = 1e2

    tot_sum = pd.read_csv(r".\results\total_summary.csv", index_col=0)
    claimed_regions_set = set(
        tot_sum.loc["150_lv1.25_I_H_2045_3H_PC_650Euro", "pc larger 1MW"].split(";")
    ) - set(["IT3 0", "FR1 11"])

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

    regions = gpd.read_file(r".\data\regions_onshore_elec_s_150.geojson").set_index(
        "name"
    )
    regions = regions.to_crs(proj.proj4_init)
    pc_regions = regions.loc[list(claimed_regions)].copy()

    # Find best electrolysis regions
    df_el = create_bus_df_for_selected_buses(bus_sizes_list_el)
    best_el_regions = list(
        df_el.sort_values(900, ascending=False)[: len(claimed_regions_set)].index
    )
    el_region_choice = list(set(best_el_regions).union(claimed_regions_set))
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

    # Create annualized photocatalysis mapping depending on the pc_capex stored in the index
    total_summary_df = pd.read_csv(total_summary_filename, index_col=0)
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


def plot_vertical(
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
):
    xmax_el = 60
    xmax_pc = 95

    # Set up figure with two parts
    fig = plt.figure(figsize=(10, 15))
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
        capex_annualized_cost_map,
    )

    #  Configure legend positions (and reverse the legend order to have bar on top, on top of the legend), labels and limits
    handles, labels = ax1.get_legend_handles_labels()
    l1 = ax1.legend(handles[::-1], labels[::-1], loc="lower left")
    # l1 = ax1.legend(loc="lower left")
    handles, labels = ax2.get_legend_handles_labels()
    l2 = ax2.legend(handles[::-1], labels[::-1], loc="center right")
    l1.set_title("Annualized\nphotocatalysis\ncost in\n€ a$^{-1}$ kW$^{-1}$")
    l2.set_title("Annualized\nphotocatalysis\ncost in\n€ a$^{-1}$ kW$^{-1}$")
    ax1.set_xlabel("Installed electrolysis capacity (GW)")
    ax2.set_xlabel("Installed photocatalysis capacity (GW)")
    ax1.set_ylim(-0.5, len(joined_df_reordered.index) - 0.5)
    ax1.set_xlim(0, xmax_el)
    ax1.invert_xaxis()
    ax2.set_xlim(0, xmax_pc)
    ax1.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax2.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Highlight nodes where salt cavern storage can be build
    salt_cavern_regions = pd.read_csv(
        r"./data/salt_cavern_potentials_s_150.csv", index_col=0
    )
    highlighted_categories = [
        node for node in joined_df_reordered.index if node in salt_cavern_regions.index
    ]
    for label in ax1.get_yticklabels():
        if label.get_text() in highlighted_categories:
            label.set_color(h2_storage_color)

    #### Add Overlay axis to show bar overlay legend

    x0, y0, width, height = (
        0.445,
        0.52,
        0.3,
        0.06,
    )  # Define position and size of the overlay ax
    # Add an overlay axes for the map

    ax = fig.add_axes([x0, y0, width, height], anchor="SE", zorder=40)

    ax.patch.set_alpha(1.0)
    ax.set_facecolor("white")

    add_stacking_legend(
        ax,
        y_label_adjustment=0.2,
        title_adjustment=21,
        colors_pc=colors_pc,
        colors_el=colors_el,
    )

    #### Add legend image / map (if boxplot visuals are changed the image needs to be adjusted/recreated)

    # Positioning of the image in relative coordinates (x0, y0, width, height)
    # x0, y0 lower-left corner of the image box; width, height size of the image box
    x0, y0, width, height = (
        0.445,
        0.62,
        0.45,
        0.25,
    )  # Define position and size of the overlay ax
    # Add an overlay axes for the map

    ax_overlay = fig.add_axes(
        [x0, y0, width, height], anchor="SE", projection=proj, zorder=50
    )
    ax_overlay.patch.set_alpha(1.0)
    ax_overlay.set_facecolor("white")

    map_font_size = 7

    # Base plot
    regions.plot(
        ax=ax_overlay, linewidth=1, color="lightgrey", edgecolor="white", alpha=0.4
    )

    # Photocatalysis regions
    pc_regions.plot(
        ax=ax_overlay,
        linewidths=1,
        legend=True,
        color=tech_colors["photocatalysis"],
        edgecolor="white",
        alpha=0.3,
    )
    # Electrolysis regions
    el_regions.plot(
        ax=ax_overlay,
        linewidths=1,
        legend=True,
        color=tech_colors["H2 Electrolysis"],
        edgecolor="white",
        alpha=0.3,
    )

    # Define annotation parameters for each zone
    node_annotations_pc = {
        "FR1 11": (600000, 5350000),
        "FR1 1": (354949, 5361033),
        "ES1 2": (180937, 5161688),
        "ES1 4": (-325042.19509547454, 4654292.406069289),
        "IT1 11": (1354146, 4643111),
        "IT1 0": (1131597, 4662965),
    }

    # Common annotation parameters
    common_params = {
        "fontsize": map_font_size,
        "color": "black",
        "ha": "center",
        "va": "center",
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
        if idx == "FR1 1":
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

    fig.show()

    # fig.savefig(r".\img\capacity_per_cost_bar.svg", bbox_inches="tight", format="svg")


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
    for i, capex in enumerate(
        sorted(["650", "700", "750", "800", "850", "875", "900", "925"])
    ):
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
    # Add max limits for photocatalysis (max installation at certain destination)
    for idx, node in enumerate(df.index):
        if node in [
            "ES1 8",
            "ES4 0",
            "PT1 1",
            "GR1 1",
            "ES1 6",
            "GR1 0",
            "IT1 11",
            "ES1 10",
            "ES1 4",
            "IT1 7",
            "ES1 2",
            "ES1 1",
            "ES1 5",
            "PT1 0",
            "FR1 1",
            "ES1 0",
            "IT1 0",
        ]:
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
    for i, capex in enumerate(
        sorted(["650", "700", "750", "800", "850", "875", "900", "925"], reverse=True)
    ):
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
):
    # Changes introduced to get vertical bars
    # "barh()" to "bar()" and in there "left" to "bottom"
    # Change the tick label adjustment to have x instead of y in everywhere (in for loop)
    # Change figure layout to wide format e.g.: figsize=(10,5)
    # Change axis layout in gridspec to 2 rows 1 col + width_ratios to "height_ratios", wspace to hspace + gs.subplots(sharex="col")
    # Change legend position, switch ax1 and ax2 (in input and every where else)
    # Change positions. Etc. etc.

    xmax_el = 60
    xmax_pc = 95

    height_ratio = [95, 60]
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
        capex_annualized_cost_map,
    )

    #  Configure legend positions (and reverse the legend order to have bar on top, on top of the legend), labels and limits
    handles, labels = ax1.get_legend_handles_labels()
    l1 = ax1.legend(handles[::-1], labels[::-1], loc="upper right")
    # l1 = ax1.legend(loc="lower left")
    handles, labels = ax2.get_legend_handles_labels()
    l2 = ax2.legend(handles[::-1], labels[::-1], loc="lower left")
    l1.set_title("Annualized\nphotocatalysis\ncost in\n€ a$^{-1}$ kW$^{-1}$")
    l2.set_title("Annualized\nphotocatalysis\ncost in\n€ a$^{-1}$ kW$^{-1}$")
    ax1.set_ylabel("Installed photocatalysis capacity (GW)")
    ax2.set_ylabel("Installed electrolysis capacity (GW)")
    ax2.set_xlim(-0.5, len(joined_df_reordered.index) - 0.5)
    ax1.set_ylim(0, xmax_pc)
    ax2.set_ylim(0, xmax_el)
    ax2.invert_yaxis()
    ax1.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax2.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Highlight nodes where salt cavern storage can be build
    salt_cavern_regions = pd.read_csv(
        r"./data/salt_cavern_potentials_s_150.csv", index_col=0
    )
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

    # Base plot
    regions.plot(
        ax=ax_overlay, linewidth=1, color="lightgrey", edgecolor="white", alpha=0.4
    )

    # Photocatalysis regions
    pc_regions.plot(
        ax=ax_overlay,
        linewidths=1,
        legend=True,
        color=tech_colors["photocatalysis"],
        edgecolor="white",
        alpha=0.3,
    )
    # Electrolysis regions
    el_regions.plot(
        ax=ax_overlay,
        linewidths=1,
        legend=True,
        color=tech_colors["H2 Electrolysis"],
        edgecolor="white",
        alpha=0.3,
    )

    # Define annotation parameters for each zone
    node_annotations_pc = {
        "FR1 11": (600000, 5350000),
        "FR1 1": (354949, 5361033),
        "ES1 2": (180937, 5161688),
        "ES1 4": (-325042.19509547454, 4654292.406069289),
        "IT1 11": (1354146, 4643111),
        "IT1 0": (1131597, 4662965),
    }

    # Common annotation parameters
    common_params = {
        "fontsize": map_font_size,
        "color": "black",
        "ha": "center",
        "va": "center",
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
        if idx == "FR1 1":
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

    fig.show()
    # fig.savefig(r".\img\capacity_per_cost_bar_horizontal.svg", bbox_inches="tight", format="svg")


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

    with open(r"./src/colors.yaml") as stream:
        tech_colors = yaml.safe_load(stream)["tech_colors"]

    with open("./src/colors.yaml") as stream:
        h2_storage_color = yaml.safe_load(stream)["others"]["h2_storage"]

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
