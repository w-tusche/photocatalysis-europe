# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# TODO: clean this up. Contains lots of duplicated code from plot hydrogen network
"""
...
"""

import logging
import os

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pypsa

try:
    import create_basic_summary as cbs
except:
    import src.create_basic_summary as cbs

logger = logging.getLogger(__name__)


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -2:
                continue
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]


def group_pipes(df, drop_direction=False):
    """
    Group pipes which connect same buses and return overall capacity.
    """
    df = df.copy()
    if drop_direction:
        positive_order = df.bus0 < df.bus2
        df_p = df[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        df_n = df[~positive_order].rename(columns=swap_buses)
        df = pd.concat([df_p, df_n])

    # there are pipes for each investment period rename to AC buses name for plotting
    df["index_orig"] = df.index
    df.index = df.apply(
        lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
        axis=1,
    )
    return df.groupby(level=0).agg(
        {"p_nom_opt": "sum", "bus0": "first", "bus1": "first", "index_orig": "first"}
    )


def collect_bus_sizes(network, regions, projection):
    n = network.copy()

    assign_location(n)

    h2_storage = n.stores.query("carrier == 'H2'")
    regions["H2"] = (
        h2_storage.rename(index=h2_storage.bus.map(n.buses.location))
        .e_nom_opt.groupby(level=0)
        .sum()
        .div(1e6)
    )  # TWh
    regions["H2"] = regions["H2"].where(regions["H2"] > 0.1)

    bus_size_factor = 1e5
    linewidth_factor = 7e3
    # MW below which not drawn
    line_lower_threshold = 750

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    # power system connected hydrogen
    carriers = ["H2 Electrolysis", "H2 Fuel Cell", "H2 turbine"]

    elec = n.links[n.links.carrier.isin(carriers)].index

    # photocatalysis generators
    pc = n.generators[n.generators.carrier == "photocatalysis"].index

    # add copy of bus in bus0 for later merging with links
    n.generators["bus0"] = n.generators["bus"]

    # create concatenated DataFrame of photocatalysis (generator) and h2 links
    total_p_nom_opt = pd.concat(
        (
            n.generators.loc[pc, ["p_nom_opt", "bus0", "carrier"]],
            n.links.loc[elec, ["p_nom_opt", "bus0", "carrier"]],
        ),
        axis=0,
    )

    bus_sizes_comb = (
        total_p_nom_opt.loc[:, "p_nom_opt"]
        .groupby([total_p_nom_opt["bus0"], total_p_nom_opt.carrier])
        .sum()
        / bus_size_factor
    )

    bus_sizes_comb.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    bus_sizes = bus_sizes_comb

    # # make a fake MultiIndex so that area is correct for legend
    # bus_sizes.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)

    # drop all links which are not H2 pipelines
    n.links.drop(
        n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True
    )

    h2_new = n.links[n.links.carrier == "H2 pipeline"]
    h2_retro = n.links[n.links.carrier == "H2 pipeline retrofitted"]

    if not h2_retro.empty:
        # if snakemake.params.foresight != "myopic":
        positive_order = h2_retro.bus0 < h2_retro.bus1
        h2_retro_p = h2_retro[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        h2_retro_n = h2_retro[~positive_order].rename(columns=swap_buses)
        h2_retro = pd.concat([h2_retro_p, h2_retro_n])

        h2_retro["index_orig"] = h2_retro.index
        h2_retro.index = h2_retro.apply(
            lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
            axis=1,
        )
        # end of if statement

        retro_w_new_i = h2_retro.index.intersection(h2_new.index)
        h2_retro_w_new = h2_retro.loc[retro_w_new_i]

        retro_wo_new_i = h2_retro.index.difference(h2_new.index)
        h2_retro_wo_new = h2_retro.loc[retro_wo_new_i]
        h2_retro_wo_new.index = h2_retro_wo_new.index_orig

        to_concat = [h2_new, h2_retro_w_new, h2_retro_wo_new]
        h2_total = pd.concat(to_concat).p_nom_opt.groupby(level=0).sum()

    else:
        h2_total = h2_new.p_nom_opt

    link_widths_total = h2_total / linewidth_factor

    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)
    n.links = n.links.groupby(level=0).first()
    link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.0)
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    retro = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline retrofitted", other=0.0
    )
    link_widths_retro = retro / linewidth_factor
    link_widths_retro[n.links.p_nom_opt < line_lower_threshold] = 0.0

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    regions = regions.to_crs(projection.proj4_init)

    return (
        regions,
        bus_sizes,
        link_widths_total,
        link_widths_retro,
        bus_size_factor,
        linewidth_factor,
        n,
    )


def extract_bus_info(
    results_folder: str, output_folder: str, re_create_files: bool
) -> pd.DataFrame:
    """
    Function for assessing PyPSA-Eur results (i.e., .nc-files) and writing the
    results to a .csv-file.

    Parameters
    ----------
    results_folder : str
        Relative path to the sub-directories containing the .nc-files (in /raw)

    Returns
    -------
    pd.DataFrame
        DataFrame of the assessed summary values

    Raises
    ------
    Exception
        If a sub-directory contains more than one .nc-file, an exception is raised.
    """

    # relative file import!
    regions = gpd.read_file(r".\data\regions_onshore_elec_s_150.geojson").set_index(
        "name"
    )

    proj = ccrs.EqualEarth()

    raw_files_folder = os.path.join(results_folder, "raw/")
    subfolder_list = next(os.walk(raw_files_folder))[1]

    try:
        existing_files = os.listdir(output_folder)
        print("Reading existing files.")
    except:
        existing_files = []
        print("No files exist, creating new files.")

    for i, subfolder in enumerate(subfolder_list):
        output_filename = f"bus_sizes_{subfolder}.csv"

        if output_filename in existing_files and re_create_files == False:
            print(
                f"Skipped subfolder {subfolder}. Bus file already exists. If you want to recreate it set re_create_files to True."
            )
            continue

        # find all .nc-files within the subfolder (should be only one)
        pypsa_files = [
            file
            for file in os.listdir(os.path.join(raw_files_folder, subfolder))
            if file.endswith(".nc")
        ]

        # more than one .nc-file in the same subfolder raises an exception
        if len(pypsa_files) > 1:
            raise Exception(
                "More than one .nc-file in the subdirectory. "
                + f"Please check data in {os.path.join(raw_files_folder, subfolder)}"
            )

        # import the PyPSA-Eur results
        try:
            n = pypsa.Network(os.path.join(raw_files_folder, subfolder, pypsa_files[0]))
        except IndexError as e:
            print(
                "Warning! There is no .nc-file in the subdirectory. Directory is skipped."
                + f"Please check data in {os.path.join(raw_files_folder, subfolder)}"
                + f"Error: {e}"
            )
            continue

        (
            regions,
            bus_sizes,
            link_widths_total,
            link_widths_retro,
            bus_size_factor,
            linewidth_factor,
            new_network,
        ) = collect_bus_sizes(n, regions, proj)

        # For now save to csv only the bus_size information.
        bus_sizes.to_csv(
            os.path.join(
                output_folder,
                output_filename,
            )
        )
        print(f"Created new file:{os.path.join(output_folder,output_filename)}")


def main():
    # This should be the folder where you place all the raw results
    results_folder = "./results/"
    output_folder = r".\data\bus_sizes"
    re_create_files = True
    extract_bus_info(results_folder, output_folder, re_create_files)


if __name__ == "__main__":
    main()
