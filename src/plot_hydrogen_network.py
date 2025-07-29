# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Creates map of optimised hydrogen network, storage and selected other
infrastructure.
"""

import logging
import os

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

logger = logging.getLogger(__name__)

# TODO: Remove hard coded and supply path differently!
# TODO: Get rid of redundancy with extract_bus_information. Best load extract bus information collect_bus_sizes and run it here before plotting.
# SET FOLDER WHERE TO SEARCH FOR THE DATA.


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]


def group_pipes(df, drop_direction=False):
    """
    Group pipes which connect same buses and return overall capacity.
    """
    df = df.copy()
    if drop_direction:
        positive_order = df.bus0 < df.bus1
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


def plot_h2_map(network, regions, projection):
    # TODO: is not part not the same as in extract_bus_information? collect_bus_sizes(network, regions, projection)
    n = network.copy()

    # if "H2 pipeline" not in n.links.carrier.unique():
    #     return

    map_opts = {
        "boundaries": [-11, 30, 34, 71],
        "color_geomap": {"ocean": "white", "land": "whitesmoke"},
    }

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

    # bus_sizes = (
    #     n.links.loc[elec, "p_nom_opt"].groupby([n.links["bus0"], n.links.carrier]).sum()
    #     / bus_size_factor
    # )

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

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": projection})

    # fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    color_h2_pipe = "#b3f3f4"
    color_retrofit = "#499a9c"

    bus_colors = {
        "H2 Electrolysis": "#ff29d9",
        "H2 Fuel Cell": "#805394",
        "H2 turbine": "#380282",
        "photocatalysis": "#dbd40c",
    }

    # Buses (Generators) and H2 pipelines
    n.plot(
        geomap=True,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        link_colors=color_h2_pipe,
        link_widths=link_widths_total,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    # Retrofitted pipelines (H2)
    n.plot(
        geomap=True,
        bus_sizes=0,
        link_colors=color_retrofit,
        link_widths=link_widths_retro,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    # Hydrogen Storage regions
    regions.plot(
        ax=ax,
        column="H2",
        cmap="Blues",
        linewidths=0,
        legend=True,
        vmax=3,
        vmin=0,
        legend_kwds={
            "label": "Hydrogen Storage [TWh]",
            "shrink": 0.7,
            "extend": "max",
        },
    )

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.23, 1),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )

    carriers = list(bus_colors.keys())
    colors = [bus_colors[c] for c in carriers] + [color_h2_pipe, color_retrofit]

    carriers[-1] = "H2 Photocatalysis"
    carriers[-2] = "H2 Turbine"

    labels = carriers + ["H2 pipeline (total)", "H2 pipeline (repurposed)"]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1.16),  # 1.13
        ncol=2,
        frameon=False,
    )

    add_legend_patches(ax, colors, labels, legend_kw=legend_kw)

    ax.set_facecolor("white")

    return fig


def main(nc_filepath, name):
    # import the PyPSA-Eur results
    n = pypsa.Network(nc_filepath)

    regions = gpd.read_file("./data/regions_onshore_elec_s_150.geojson").set_index(
        "name"
    )

    proj = ccrs.EqualEarth()

    save_figure_to = f"img/elec_s_{name}.pdf"
    # check johnson versions of packages ...
    fig = plot_h2_map(n, regions, proj)

    fig.savefig(save_figure_to, bbox_inches="tight", format="pdf")


if __name__ == "__main__":
    nc_filepath = (
        "results/raw/150_lv1.25_I_H_2045_3H_PC_650Euro/elec_s_150_lv1.25__I-H_2045.nc"
    )
    name = "elec_s_150_lv1.25_I_H_2045_3H_PC_650Euro"
    main(nc_filepath, name)
