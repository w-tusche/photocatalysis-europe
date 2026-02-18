# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create energy balance maps for the defined carriers.
# THIS HAS TO BE PLOTTED WITH A NEWER PYPSA VERSION. Use the envrionement_balances.yaml environment
"""

import os
import re

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import yaml
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

# from scripts._helpers import (
#     PYPSA_V1,
#     configure_logging,
#     set_scenario_config,
#     update_config_from_wildcards,
# )
# from scripts.add_electricity import sanitize_carriers
# from scripts.plot_power_network import load_projection

PYPSA_V1 = bool(re.match(r"^1\.\d", pypsa.__version__))
SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1


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


if __name__ == "__main__":
    # if "snakemake" not in globals():
    #     from scripts._helpers import mock_snakemake

    # snakemake = mock_snakemake(
    #     "plot_balance_map",
    #     clusters="10",
    #     opts="",
    #     sector_opts="",
    #     planning_horizons="2050",
    #     carrier="H2",
    # )

    # configure_logging(snakemake)
    # set_scenario_config(snakemake)
    # update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # n = pypsa.Network(snakemake.input.network)

    plot_mapping = {
        "PC0": {
            "file": "150_lv1.25_I_H_2045_3H_PC_925Euro",
            "carriers": ["AC", "H2", "urban central heat", "urban decentral heat"],
        },
        "PC50": {
            "file": "150_lv1.25_I_H_2045_3H_PC_650Euro",
            "carriers": ["AC", "H2", "urban central heat", "urban decentral heat"],
        },
    }

    for case, case_dict in plot_mapping.items():
        n = pypsa.Network(
            os.path.join(
                "results/raw", case_dict["file"], "elec_s_150_lv1.25__I-H_2045.nc"
            )
        )

        with open(r"src/plotting.default.yaml") as stream:
            config = yaml.safe_load(stream)
        sanitize_carriers(n, config)
        pypsa.set_option("params.statistics.round", 3)
        pypsa.set_option("params.statistics.drop_zero", True)
        pypsa.set_option("params.statistics.nice_names", False)

        # regions = gpd.read_file(snakemake.input.regions).set_index("name")
        regions = gpd.read_file("data/regions_onshore_elec_s_150.geojson").set_index(
            "name"
        )
        for carrier in case_dict["carriers"]:
            # config = snakemake.params.plotting
            with open(r"src/plotting.default.yaml") as stream:
                config = yaml.safe_load(stream)["plotting"]
            # carrier = snakemake.wildcards.carrier
            # Currently check carriers in yaml file, old version check in the plotting part of the config!
            # carrier = "AC"  # [AC, H2, gas, oil, methanol, co2 stored, urban central heat]

            # fill empty colors or "" with light grey
            mask = n.carriers.color.isna() | n.carriers.color.eq("")
            n.carriers["color"] = n.carriers.color.mask(mask, "lightgrey")

            # set EU location with location from config
            eu_location = config["eu_node_location"]
            n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]

            # get balance map plotting parameters
            boundaries = config["map"]["boundaries"]
            config = config["balance_map"][carrier]
            conversion = config["unit_conversion"]

            if carrier not in n.buses.carrier.unique():
                raise ValueError(
                    f"Carrier {carrier} is not in the network. Remove from configuration `plotting: balance_map: bus_carriers`."
                )

            # for plotting change bus to location
            n.buses["location"] = n.buses["location"].replace("", "EU").fillna("EU")

            # set location of buses to EU if location is empty and set x and y coordinates to bus location
            n.buses["x"] = n.buses.location.map(n.buses.x)
            n.buses["y"] = n.buses.location.map(n.buses.y)

            # bus_sizes according to energy balance of bus carrier
            # TODO: FIXME: stuff!!! This is the critical one that was changed. Whats the output of this? what do I have to change?
            eb = n.statistics.energy_balance(
                bus_carrier=carrier, groupby=["bus", "carrier"]
            )

            # remove energy balance of transmission carriers which relate to losses
            transmission_carriers = get_transmission_carriers(
                n, bus_carrier=carrier
            ).rename({"name": "carrier"})
            components = transmission_carriers.unique("component")
            carriers = transmission_carriers.unique("carrier")

            # only carriers that are also in the energy balance
            carriers_in_eb = carriers[
                carriers.isin(eb.index.get_level_values("carrier"))
            ]

            eb.loc[components] = eb.loc[components].drop(
                index=carriers_in_eb, level="carrier"
            )
            eb = eb.dropna()
            bus_sizes = eb.groupby(level=["bus", "carrier"]).sum().div(conversion)
            bus_sizes = bus_sizes.sort_values(ascending=False)

            # Get colors for carriers
            with open(r"src/plotting.default.yaml") as stream:
                config = yaml.safe_load(stream)["plotting"]
            n.carriers.update({"color": config["tech_colors"]})  # XXX MANUALLY CHANGED
            carrier_colors = n.carriers.color.copy().replace("", "grey")

            colors = (
                bus_sizes.index.get_level_values("carrier")
                .unique()
                .to_series()
                .map(carrier_colors)
            )

            # line and links widths according to optimal capacity
            flow = n.statistics.transmission(groupby=False, bus_carrier=carrier).div(
                conversion
            )

            if not flow.empty:
                flow_reversed_mask = flow.index.get_level_values(1).str.contains(
                    "reversed"
                )
                flow_reversed = flow[flow_reversed_mask].rename(
                    lambda x: x.replace("-reversed", "")
                )
                flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)

            # if there are not lines or links for the bus carrier, use fallback for plotting
            fallback = pd.Series()
            line_widths = flow.get("Line", fallback).abs()
            link_widths = flow.get("Link", fallback).abs()

            # define maximal size of buses and branch width
            config = config["balance_map"][carrier]
            bus_size_factor = config["bus_factor"]
            branch_width_factor = config["branch_factor"]
            flow_size_factor = config["flow_factor"]

            # get prices per region as colormap
            buses = n.buses.query("carrier in @carrier").index
            weights = n.snapshot_weightings.generators
            prices = weights @ n.buses_t.marginal_price[buses] / weights.sum()
            level = "name" if PYPSA_V1 else "Bus"
            price = prices.rename(n.buses.location).groupby(level=level).mean()

            if carrier == "co2 stored" and "CO2Limit" in n.global_constraints.index:
                co2_price = n.global_constraints.loc["CO2Limit", "mu"]
                price = price - co2_price

            # if only one price is available, use this price for all regions
            if price.size == 1:
                regions["price"] = price.values[0]
                shift = round(abs(price.values[0]) / 20, 0)
            else:
                regions["price"] = price.reindex(regions.index).fillna(0)
                shift = 0

            vmin, vmax = regions.price.min() - shift, regions.price.max() + shift
            if config["vmin"] is not None:
                vmin = config["vmin"]
            if config["vmax"] is not None:
                vmax = config["vmax"]

            # crs = load_projection(snakemake.params.plotting)
            crs = ccrs.EqualEarth()

            fig, ax = plt.subplots(
                figsize=(5, 6.5),
                subplot_kw={"projection": crs},
                layout="constrained",
            )

            line_flow = flow.get("Line")
            link_flow = flow.get("Link")
            transformer_flow = flow.get("Transformer")

            n.plot(
                bus_sizes=bus_sizes * bus_size_factor,
                bus_colors=colors,
                bus_split_circles=True,
                line_widths=line_widths * branch_width_factor,
                link_widths=link_widths * branch_width_factor,
                line_flow=(
                    line_flow * flow_size_factor if line_flow is not None else None
                ),
                link_flow=(
                    link_flow * flow_size_factor if link_flow is not None else None
                ),
                transformer_flow=(
                    transformer_flow * flow_size_factor
                    if transformer_flow is not None
                    else None
                ),
                ax=ax,
                margin=0.2,
                geomap_colors={"border": "darkgrey", "coastline": "darkgrey"},
                geomap=True,
                boundaries=boundaries,
            )

            regions.to_crs(crs.proj4_init).plot(
                ax=ax,
                column="price",
                cmap=config["cmap"],
                vmin=vmin,
                vmax=vmax,
                edgecolor="None",
                linewidth=0,
            )

            ax.set_title(carrier)

            # Add colorbar
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=config["cmap"], norm=norm)
            price_unit = config["region_unit"]
            cbr = fig.colorbar(
                sm,
                ax=ax,
                label=f"Average Marginal Price [{price_unit}]",
                shrink=0.95,
                pad=0.03,
                aspect=50,
                orientation="horizontal",
            )
            cbr.outline.set_edgecolor("None")

            # add legend
            legend_kwargs = {
                "loc": "upper left",
                "frameon": False,
                "alignment": "left",
                "title_fontproperties": {"weight": "bold"},
            }

            pad = 0.18
            n.carriers.loc["", "color"] = "None"

            # Get lists for supply and consumption carriers
            pos_carriers = bus_sizes[bus_sizes > 0].index.unique("carrier")
            neg_carriers = bus_sizes[bus_sizes < 0].index.unique("carrier")

            # Determine larger total absolute value for supply and consumption for a carrier if carrier exists as both supply and consumption
            common_carriers = pos_carriers.intersection(neg_carriers)

            def get_total_abs(carrier, sign):
                values = bus_sizes.loc[:, carrier]
                return values[values * sign > 0].abs().sum()

            supp_carriers = sorted(
                set(pos_carriers) - set(common_carriers)
                | {
                    c
                    for c in common_carriers
                    if get_total_abs(c, 1) >= get_total_abs(c, -1)
                }
            )
            cons_carriers = sorted(
                set(neg_carriers) - set(common_carriers)
                | {
                    c
                    for c in common_carriers
                    if get_total_abs(c, 1) < get_total_abs(c, -1)
                }
            )

            # Add supply carriers
            add_legend_patches(
                ax,
                n.carriers.color[supp_carriers],
                supp_carriers,
                legend_kw={
                    "bbox_to_anchor": (0, -pad),
                    "ncol": 1,
                    "title": "Supply",
                    **legend_kwargs,
                },
            )

            # Add consumption carriers
            add_legend_patches(
                ax,
                n.carriers.color[cons_carriers],
                cons_carriers,
                legend_kw={
                    "bbox_to_anchor": (0.5, -pad),
                    "ncol": 1,
                    "title": "Consumption",
                    **legend_kwargs,
                },
            )

            # Add bus legend
            legend_bus_sizes = config["bus_sizes"]
            carrier_unit = config["unit"]
            if legend_bus_sizes is not None:
                add_legend_semicircles(
                    ax,
                    [
                        s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR
                        for s in legend_bus_sizes
                    ],
                    [f"{s} {carrier_unit}" for s in legend_bus_sizes],
                    patch_kw={"color": "#666"},
                    legend_kw={
                        "bbox_to_anchor": (0, 1),
                        **legend_kwargs,
                    },
                )

            # Add branch legend
            legend_branch_sizes = config["branch_sizes"]
            if legend_branch_sizes is not None:
                add_legend_lines(
                    ax,
                    [s * branch_width_factor for s in legend_branch_sizes],
                    [f"{s} {carrier_unit}" for s in legend_branch_sizes],
                    patch_kw={"color": "#666"},
                    legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
                )

            # fig.savefig(
            #     f"./img/energy_balance_maps/{carrier}_map_{case}.pdf",
            #     dpi=400,
            #     bbox_inches="tight",
            # )
