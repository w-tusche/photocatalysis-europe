# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
# SPDX-FileCopyrightText: 2025 Wolfram Tuschewitzki, Jelto Lange
#
# SPDX-License-Identifier: MIT
"""
Collects the bus sizes / pipeline widths of the optimised hydrogen network so
that `plots.py` and the supplementary notebooks can draw the maps from them.
"""

import logging
import os

import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import pypsa

from create_basic_summary import REPO_ROOT, scenario_folders

logger = logging.getLogger(__name__)

# Defaults are anchored to the repository, not to the current working directory,
# so the script and its functions work no matter where they are called from.
DEFAULT_BUS_SIZES_FOLDER = os.path.join(REPO_ROOT, "data", "bus_sizes")
DEFAULT_REGIONS_PATH = os.path.join(
    REPO_ROOT, "data", "regions_onshore_elec_s_150.geojson"
)


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            # (str.find returns -1 when it does not find a space)
            if i == -1:
                continue
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]


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


def bus_size_writer(output_folder: str = None, regions_path: str = None):
    """
    Function that writes the bus sizes of one loaded network.

    The returned function is called as `write(n, scenario)` and writes
    `bus_sizes_<scenario>.csv` into `output_folder`. A caller that has the
    network open already (e.g. `create_summary_csv`) can write the file
    without reading the network a second time.

    Parameters
    ----------
    output_folder : str, optional
        Defaults to the repository's `data/bus_sizes/`, created if missing.
    regions_path : str, optional
        Path to the onshore regions GeoJSON. Defaults to the repository's
        `data/regions_onshore_elec_s_150.geojson`.
    """
    if output_folder is None:
        output_folder = DEFAULT_BUS_SIZES_FOLDER
    if regions_path is None:
        regions_path = DEFAULT_REGIONS_PATH

    regions = gpd.read_file(regions_path).set_index("name")
    proj = ccrs.EqualEarth()
    os.makedirs(output_folder, exist_ok=True)

    def write(n, scenario):
        # the remaining return values (regions, link widths, scaling factors,
        # network) are only needed for plotting, see plots.py
        _, bus_sizes, *_ = collect_bus_sizes(n, regions, proj)

        # For now save to csv only the bus_size information.
        path = os.path.join(output_folder, f"bus_sizes_{scenario}.csv")
        bus_sizes.to_csv(path)
        print(f"Created new file:{path}")

    return write


def extract_bus_info(
    output_folder: str = None,
    re_create_files: bool = False,
    raw_folder: str = None,
    only_subfolders: list = None,
    regions_path: str = None,
) -> pd.DataFrame:
    """
    Function for assessing PyPSA-Eur results (i.e., .nc-files) and writing the
    bus sizes to .csv-files. The networks can be read from another folder.

    Parameters
    ----------
    output_folder : str, optional
        Folder the `bus_sizes_<scenario>.csv` files are written to. Defaults to
        the repository's `data/bus_sizes/` and is created if missing.
    re_create_files : bool, optional
        If True, rewrite bus-size files that already exist in `output_folder`.
    raw_folder : str, optional
        Folder containing the per-scenario subfolders with the .nc-files.
        Defaults to data/raw/; scenarios named in `only_subfolders` are then
        also looked up in `cases.EXTRA_RAW_FOLDERS`.
    only_subfolders : list, optional
        Process only these scenario subfolders.
    regions_path : str, optional
        Path to the onshore regions GeoJSON. Defaults to the repository's
        `data/regions_onshore_elec_s_150.geojson`.

    Returns
    -------
    pd.DataFrame
        DataFrame of the assessed summary values

    Raises
    ------
    Exception
        If a sub-directory contains more than one .nc-file, an exception is raised.
    """
    if output_folder is None:
        output_folder = DEFAULT_BUS_SIZES_FOLDER

    write = bus_size_writer(output_folder, regions_path)

    folders = scenario_folders(raw_folder, only_subfolders)

    existing_files = os.listdir(output_folder)
    print(f"Writing to {output_folder}.")

    for subfolder, folder in folders.items():
        output_filename = f"bus_sizes_{subfolder}.csv"

        if output_filename in existing_files and re_create_files == False:
            print(
                f"Skipped subfolder {subfolder}. Bus file already exists. If you want to recreate it set re_create_files to True."
            )
            continue

        # find all .nc-files within the subfolder (should be only one)
        pypsa_files = [file for file in os.listdir(folder) if file.endswith(".nc")]

        # more than one .nc-file in the same subfolder raises an exception
        if len(pypsa_files) > 1:
            raise Exception(
                "More than one .nc-file in the subdirectory. "
                + f"Please check data in {folder}"
            )

        # import the PyPSA-Eur results
        try:
            print(
                f"Importing the following file: {os.path.join(folder, pypsa_files[0])}"
            )
            n = pypsa.Network(os.path.join(folder, pypsa_files[0]))
        except IndexError as e:
            print(
                "Warning! There is no .nc-file in the subdirectory. Directory is skipped."
                + f"Please check data in {folder}"
                + f"Error: {e}"
            )
            continue

        write(n, subfolder)


if __name__ == "__main__":
    # Options, see extract_bus_info()
    raw_folder = None  # default data/raw/, see cases.py
    output_folder = DEFAULT_BUS_SIZES_FOLDER
    regions_path = DEFAULT_REGIONS_PATH
    only_subfolders = None  # e.g. ["150_lv1.25_I_H_2045_PC_660Euro"]
    re_create_files = False

    extract_bus_info(
        output_folder=output_folder,
        re_create_files=re_create_files,
        raw_folder=raw_folder,
        only_subfolders=only_subfolders,
        regions_path=regions_path,
    )
