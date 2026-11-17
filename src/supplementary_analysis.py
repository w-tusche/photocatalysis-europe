# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Wolfram Tuschewitzki
#
# SPDX-License-Identifier: MIT
"""
Tables and the annex-only figures of the Supplementary Material.

One section per Supplementary Note, in the order of the annex. Every
table function returns a DataFrame with the rows and columns of the
printed table. The figures here are the ones drawn for the annex alone
(transport comparison, exogenous demand bars, carbon Sankey); figures
shared with the main text stay in plots.py. Called from
supplementary_material.py.
"""

import gc
import io
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
import yaml

from cases import REPO_ROOT, RESULT_ROOTS, figure_formats, find_network
from create_basic_summary import (  # noqa: F401  (parse_nodal re-exported)
    LHV,
    parse_nodal,
)
from plots import (
    prepare_LCOH_violin_data,
    remove_legend_patch_edges,
    save_or_plot_img,
    set_size,
)
from weighted_stats import weighted_mean, weighted_median, weighted_quantiles

# Keep the two most recently used networks open, a network is freed only by gc
MAX_OPEN_NETWORKS = 2
_networks = {}


def get_network(scenario):
    """
    Network of a scenario name, or of a path to an .nc file.
    """
    if scenario in _networks:
        _networks[scenario] = _networks.pop(scenario)
    else:
        while len(_networks) >= MAX_OPEN_NETWORKS:
            del _networks[next(iter(_networks))]
            gc.collect()
        path = scenario if scenario.endswith(".nc") else find_network(scenario)
        _networks[scenario] = pypsa.Network(path)
    return _networks[scenario]


def energy_balance(n, bus_carrier):
    """
    Annual balance of one bus carrier per carrier in TWh, supply positive.
    """
    balance = n.statistics.energy_balance(
        bus_carrier=bus_carrier, groupby=["carrier"], nice_names=False
    )
    return balance.groupby(level="carrier").sum() / 1e6


def group_balance(balance, groups, noise=0.05):
    """
    Sum a per-carrier balance into the rows of a table.

    Raises if a carrier above `noise` belongs to no group, so nothing drops out
    of the table unnoticed.
    """
    grouped = pd.Series(
        {
            label: balance.reindex(carriers).dropna().sum()
            for label, carriers in groups.items()
        }
    )
    named = {c for carriers in groups.values() for c in carriers}
    unassigned = [
        c for c in balance.index if c not in named and abs(balance[c]) > noise
    ]
    if unassigned:
        raise RuntimeError(f"carriers in no group: {unassigned}")
    return grouped


def compare(pc0, pc50):
    """
    Two series side by side with their difference.
    """
    table = pd.DataFrame({"PC-0": pc0, "PC-50": pc50})
    table["Difference"] = table["PC-50"] - table["PC-0"]
    return table


def pair_differences(table, cases):
    """
    Variant minus base for two case pairs, absolute and in %.

    `table`: one row per case, labels as in `cases`, ordered base, variant,
    base, variant (e.g. PC-0, PC-0 + transport, PC-50, PC-50 + transport).
    The % is empty where the base value is below 0.01.
    """
    labels = list(cases)
    blocks = {}
    for base, variant in [(labels[0], labels[1]), (labels[2], labels[3])]:
        reference = table.loc[base]
        blocks[base] = pd.DataFrame(
            {
                "absolute": table.loc[variant] - reference,
                "in %": 100
                * (table.loc[variant] / reference.where(reference.abs() >= 0.01) - 1),
            }
        )
    return pd.concat(blocks, axis=1)


def market_share_columns(df, scenarios):
    """
    Column labels of a sweep table: the photocatalysis market share in % with one
    decimal.
    """
    return [round(float(df.loc[s, "pc market share"]), 1) for s in scenarios]


def at_site_limit(gens):
    """
    Mask of the generators with `p_nom_opt` at `p_nom_max`. Generators at their
    nodal expansion limit.

    rtol 1e-3, since the barrier solver leaves binding capacities
    slightly below their bound.
    """
    return np.isclose(gens["p_nom_opt"], gens["p_nom_max"], rtol=1e-3)


# --- Supplementary Note 3: LCOH of electrolysis and photocatalysis ---


def nodal_lcoh_statistics(df, scenario, min_capacity=1):
    """
    Unweighted median, production-weighted median and production-weighted mean
    of the nodal LCOH (EUR/kg) of one scenario, per technology ("EL", "PC"),
    with the node count and summed production (MWh/a) behind them.

    Only nodes with at least `min_capacity` MW of a technology count for it,
    and each node is weighted by the hydrogen it produces with that
    technology.
    """
    capacity = parse_nodal(df.loc[scenario, "nodal capacity node:el:pc"], ["EL", "PC"])
    lcoh = parse_nodal(df.loc[scenario, "nodal LCOH node:el:pc"], ["EL", "PC"])
    generation = pd.read_csv(
        os.path.join(
            REPO_ROOT, "data", "generation_data", f"generation_{scenario}.csv"
        ),
        index_col=0,
    )
    production = {
        "EL": generation["h2_generation_Electrolysis_MWh"],
        "PC": generation["h2_generation_Photocatalysis_MWh"],
    }

    stats = {}
    for tech in ["EL", "PC"]:
        nodes = capacity.index[capacity[tech] >= min_capacity]
        values = lcoh.loc[nodes, tech].dropna()
        weights = production[tech].reindex(values.index).to_numpy()
        values = values.to_numpy()
        found = len(values) > 0
        stats[tech] = {
            "n": len(values),
            "production": float(weights.sum()),
            "median": np.median(values) if found else np.nan,
            "weighted median": (weighted_median(values, weights) if found else np.nan),
            "weighted mean": (weighted_mean(values, weights) if found else np.nan),
        }
    return stats


def lcoh_medians(df, scenarios, min_capacity=1):
    """
    Nodal LCOH medians of the base sweep, unweighted and weighted by
    production.

    Only nodes with at least `min_capacity` MW of a technology count for it.
    """
    columns = {}
    for scenario in scenarios:
        stats = nodal_lcoh_statistics(df, scenario, min_capacity)

        def relative_difference(statistic):
            el, pc = stats["EL"][statistic], stats["PC"][statistic]
            return (el - pc) / el * 100

        market_share = round(float(df.loc[scenario, "pc market share"]), 1)
        columns[market_share] = {
            ("Annualized PC cost", "EUR/a/kW"): float(
                df.loc[scenario, "pc total annual cost"]
            ),
            ("EL median LCOH, unweighted", "EUR/kg"): stats["EL"]["median"],
            ("n EL", "nodes"): stats["EL"]["n"],
            ("PC median LCOH, unweighted", "EUR/kg"): stats["PC"]["median"],
            ("n PC", "nodes"): stats["PC"]["n"],
            ("Relative difference, unweighted median", "%"): relative_difference(
                "median"
            ),
            ("Relative difference, weighted median", "%"): relative_difference(
                "weighted median"
            ),
            ("Relative difference, weighted mean", "%"): relative_difference(
                "weighted mean"
            ),
        }
    table = pd.DataFrame(columns).sort_index(axis=1)
    table.columns.name = "PC market share in %"
    return table


def eur_per_kg(eur, mwh):
    """
    EUR per MWh of hydrogen as EUR/kg, at the LHV.
    """
    return eur / mwh / 1e3 * LHV


def hydrogen_output(n):
    """
    Hydrogen output of electrolysis and photocatalysis, in MWh per time step
    and plant (snapshot weightings applied), with the H2 bus of each plant.
    """
    weights = n.snapshot_weightings.generators
    el = n.links.index[n.links.carrier == "H2 Electrolysis"]
    pc = n.generators.index[n.generators.carrier == "photocatalysis"]
    return {
        "Electrolysis": (
            -n.links_t.p1[el].mul(weights, axis=0),
            n.links.loc[el, "bus1"],
        ),
        "Photocatalysis": (
            n.generators_t.p[pc].mul(weights, axis=0),
            n.generators.loc[pc, "bus"],
        ),
    }


def market_value(h2, buses, prices):
    """
    Market value in EUR/kg, the production-weighted price of the hydrogen
    (Hirth 2013, Eq. 2; Brown and Reichenberg 2021, Eq. 15).

    `h2` and `buses` as returned by `hydrogen_output()`; `prices` holds the
    bus prices per time step. Only the time steps of `h2` count, so a slice
    of `h2` gives the price fetched in that period.
    """
    fetched = prices.loc[h2.index, buses].values
    return eur_per_kg((h2 * fetched).sum().sum(), h2.sum().sum())


def lcoh_decomposition(n):
    """
    Production-weighted LCOH, market value, heat by-product and scarcity
    revenue of both technologies, in EUR/kg.

    LCOH = market value + by-product - scarcity revenue, so the scarcity
    revenue is the remainder.
    """
    weights = n.snapshot_weightings.generators
    prices = n.buses_t.marginal_price
    output = hydrogen_output(n)
    h2_el, h2_pc = output["Electrolysis"][0], output["Photocatalysis"][0]

    el = n.links.loc[h2_el.columns]
    power_cost = (
        (n.links_t.p0[el.index].mul(weights, axis=0) * prices[el.bus0].values)
        .sum()
        .sum()
    )
    with_heat = el[el.bus2.isin(n.buses.index)]
    heat = -n.links_t.p2[with_heat.index].mul(weights, axis=0)
    heat_revenue = (heat * prices[with_heat.bus2].values).sum().sum()

    pc = n.generators.loc[h2_pc.columns]

    lcoh_el = eur_per_kg(
        power_cost + (el.capital_cost * el.p_nom_opt).sum(), h2_el.sum().sum()
    )
    lcoh_pc = eur_per_kg((pc.capital_cost * pc.p_nom_opt).sum(), h2_pc.sum().sum())
    value_el = market_value(*output["Electrolysis"], prices)
    value_pc = market_value(*output["Photocatalysis"], prices)
    heat_el = eur_per_kg(heat_revenue, h2_el.sum().sum())

    table = pd.DataFrame(
        {
            "Electrolysis": [lcoh_el, value_el, heat_el, value_el + heat_el - lcoh_el],
            "Photocatalysis": [lcoh_pc, value_pc, 0.0, value_pc - lcoh_pc],
        },
        index=[
            "LCOH",
            "Market value of the hydrogen",
            "Heat by-product",
            "Scarcity revenue",
        ],
    )
    table["Difference"] = table["Electrolysis"] - table["Photocatalysis"]
    table.index.name = "Production-weighted, EUR/kg"
    return table


MV_SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]


def market_value_seasons(n):
    """
    Seasonal and within-month part of the market value of photocatalysis and
    electrolysis.

    Market value as in `market_value()`, the same function as the market
    value row of `lcoh_decomposition()`. Replacing every bus price by
    its monthly mean keeps the seasonal and spatial part and removes
    timing within a month (day/night); the difference of the two gaps is
    that within-month part. Also returns the unweighted mean H2 bus
    price per half-year.
    """
    prices = n.buses_t.marginal_price
    month = prices.index.month
    monthly = prices.groupby(month).transform("mean")
    summer = month.isin(MV_SUMMER_MONTHS)

    rows = {}
    for name, (h2, buses) in hydrogen_output(n).items():
        rows[name] = {
            "Output Apr-Sep, %": 100 * h2[summer].sum().sum() / h2.sum().sum(),
            "Market value, EUR/kg": market_value(h2, buses, prices),
            "Market value with monthly-mean prices, EUR/kg": market_value(
                h2, buses, monthly
            ),
            "Price fetched Apr-Sep, EUR/kg": market_value(h2[summer], buses, prices),
            "Price fetched Oct-Mar, EUR/kg": market_value(h2[~summer], buses, prices),
        }
    table = pd.DataFrame(rows).T
    table.loc["Difference"] = table.loc["Electrolysis"] - table.loc["Photocatalysis"]
    table.loc["Difference", "Output Apr-Sep, %"] = np.nan

    h2_buses = n.buses.index[n.buses.carrier == "H2"]
    mean_price = prices[h2_buses].mean(axis=1) / 1e3 * LHV
    bus_price = pd.Series(
        {
            "Apr-Sep": mean_price[summer].mean(),
            "Oct-Mar": mean_price[~summer].mean(),
        },
        name="H2 bus price, unweighted mean over buses and time steps, EUR/kg",
    )
    return table, bus_price


# Avoid division by small number (production-weighted photocatalysis LCOH)
PC_MIN_PRODUCTION = 1.0  # TWh/a


def photocatalysis_site_limits(n, min_capacity=1.0):
    """
    Photocatalysis nodes above `min_capacity` MW, those at their area limit
    (`capacity_limit_row`), the share of the capacity they hold, and the used
    share of the eligible potential.
    """
    limits = capacity_limit_row(n, details=True)
    gens = n.generators[n.generators.carrier == "photocatalysis"]
    built = gens[gens.p_nom_opt > min_capacity]
    at_limit = built[at_site_limit(built)]
    return {
        ("Photocatalysis nodes", "nodes"): len(built),
        ("of which at their area limit", "nodes"): limits["photocatalysis"],
        ("Capacity at the area limit", "%"): (
            100 * at_limit.p_nom_opt.sum() / built.p_nom_opt.sum()
            if len(built)
            else np.nan
        ),
        ("Eligible potential used", "%"): limits[
            "photocatalysis - potential utilization (%)"
        ],
    }


def lcoh_decomposition_sweep(get_network, scenarios, df, min_capacity=1.0):
    """
    `lcoh_decomposition` per cost point: the production-weighted LCOH
    difference and its three terms, with the photocatalysis nodes at their area
    limit, in EUR/kg.

    Cost points with less than `PC_MIN_PRODUCTION` TWh/a of photocatalysis
    are left out.
    """
    columns = {}
    for scenario, share in zip(scenarios, market_share_columns(df, scenarios)):
        if float(df.loc[scenario, "photocatalysis (Energy)"]) < PC_MIN_PRODUCTION:
            continue
        n = get_network(scenario)
        difference = lcoh_decomposition(n)["Difference"]
        gap = difference["LCOH"]
        heat = difference["Heat by-product"]
        columns[share] = {
            ("LCOH difference", "EUR/kg"): gap,
            ("Market value of the hydrogen", "EUR/kg"): difference[
                "Market value of the hydrogen"
            ],
            ("Heat by-product", "EUR/kg"): heat,
            ("Scarcity revenue of photocatalysis", "EUR/kg"): -difference[
                "Scarcity revenue"
            ],
            ("Heat by-product, share of the difference", "%"): 100 * heat / gap,
        } | photocatalysis_site_limits(n, min_capacity)
        del n  # lets get_network free it before it opens the next one
    table = pd.DataFrame(columns)
    table.columns.name = "PC market share in %"
    return table


def mwh_to_kg(mwh, heating_value=LHV):
    return mwh * 1e3 / heating_value


def h2_infrastructure_against_gap(df, pc0, pc50, min_capacity=1, heating_value=LHV):
    """
    Hydrogen storage and pipeline cost per kg of hydrogen, against the
    difference of the unweighted median LCOH in the PC-50 case. Backs the
    hydrogen infrastructure sentence of the paper.

    Two denominators: all hydrogen, and only the photocatalytic hydrogen
    (attributing the infrastructure to photocatalysis alone). The LCOH uses
    the LHV; `heating_value` changes only the conversion of the
    infrastructure cost.
    """
    stats = nodal_lcoh_statistics(df, pc50, min_capacity)
    median_gap = stats["EL"]["median"] - stats["PC"]["median"]

    columns = ["tot H2 storage cost", "total H2 pipeline capacity cost"]
    cost = df.loc[[pc0, pc50], columns].astype(float)
    storage, pipelines = cost.loc[pc50] - cost.loc[pc0]
    increase = storage + pipelines
    whole = cost.loc[pc50].sum()

    pc_mwh = float(df.loc[pc50, "photocatalysis (Energy)"]) * 1e6
    all_mwh = pc_mwh + float(df.loc[pc50, "H2 Electrolysis (Energy)"]) * 1e6
    pc_kg = mwh_to_kg(pc_mwh, heating_value)
    all_kg = mwh_to_kg(all_mwh, heating_value)

    per_kg_rows = {
        "Increase, per kg of all hydrogen": increase / all_kg,
        "Increase, per kg of photocatalytic hydrogen": increase / pc_kg,
        "PC-50 total, per kg of all hydrogen": whole / all_kg,
        "PC-50 total, per kg of photocatalytic hydrogen": whole / pc_kg,
    }
    table = pd.DataFrame(
        {
            "Value": {
                ("Storage cost increase, PC-0 to PC-50", "Bln EUR/a"): storage / 1e9,
                ("Pipeline cost increase, PC-0 to PC-50", "Bln EUR/a"): pipelines / 1e9,
                ("Storage and pipeline cost, PC-50", "Bln EUR/a"): whole / 1e9,
                ("Median LCOH difference, PC-50", "EUR/kg"): median_gap,
                # the median difference over all photocatalytic hydrogen, as
                # an annual sum for scale
                ("Median difference times photocatalytic hydrogen", "Bln EUR/a"): (
                    median_gap * mwh_to_kg(pc_mwh) / 1e9
                ),
            }
            | {(label, "EUR/kg"): value for label, value in per_kg_rows.items()}
        }
    )
    table["Share of the median difference in %"] = (
        100 * table["Value"] / median_gap
    ).where([label in per_kg_rows for label, _ in table.index])
    return table


# US BLS CPI-U annual averages (CUUR0000SA0) and ECB USD per EUR (2020), fetched 2026-09-30
CPI = {2005: 195.3, 2016: 240.007, 2020: 258.811}
USD_PER_EUR_2020 = 1.1421961089494

# source, cost-basis year, low and high in USD/kg of that year
LITERATURE_COSTS = [
    ("Pinaud et al., type 1 system", 2005, 1.60, 1.60),
    ("Schneidewind, target cost range", 2016, 1.5, 1.6),
    (
        "Schneidewind, within demonstrated or physically possible parameter limits",
        2016,
        1.0,
        2.0,
    ),
]


def literature_cost_conversion():
    """
    Literature photocatalysis costs inflated with the US CPI and converted to
    EUR 2020.
    """
    rows = []
    for source, year, low, high in LITERATURE_COSTS:
        ratio = CPI[2020] / CPI[year]
        rows.append(
            {
                "Source": source,
                "Original value low in USD/kg": low,
                "Original value high in USD/kg": high,
                "Cost-basis year": year,
                "CPI ratio": ratio,
                "USD/EUR (2020)": USD_PER_EUR_2020,
                "EUR 2020 low in EUR/kg": low * ratio / USD_PER_EUR_2020,
                "EUR 2020 high in EUR/kg": high * ratio / USD_PER_EUR_2020,
            }
        )
    return pd.DataFrame(rows).set_index("Source")


# --- Supplementary Note 4: By-products and system services of electrolysis ---

# printed row -> (carriers on urban central heat, carriers on urban decentral heat)
ELHEAT0_ROWS = {
    "Electrolysis waste heat": (["H2 Electrolysis"], []),
    "Air heat pumps": (
        ["urban central air heat pump"],
        ["urban decentral air heat pump"],
    ),
    "Resistive heaters": (
        ["urban central resistive heater"],
        ["urban decentral resistive heater"],
    ),
    "Gas boilers": (["urban central gas boiler"], ["urban decentral gas boiler"]),
    "Gas combined heat and power, with and without capture": (
        ["urban central gas CHP", "urban central gas CHP CC"],
        [],
    ),
    "Fischer-Tropsch and methanol synthesis waste heat": (
        ["Fischer-Tropsch", "methanolisation"],
        [],
    ),
    "Solar thermal": (
        ["urban central solar thermal"],
        ["urban decentral solar thermal"],
    ),
    "Thermal storage (net)": (
        ["urban central water tanks charger", "urban central water tanks discharger"],
        [
            "urban decentral water tanks charger",
            "urban decentral water tanks discharger",
        ],
    ),
    "Direct air capture": (["DAC"], ["DAC"]),
    "Exogenous heat demand": (
        ["urban central heat", "low-temperature heat for industry"],
        ["urban decentral heat", "low-temperature heat for industry"],
    ),
}


def elheat0_heat_balance(get_network, pairs):
    """
    Change of the urban heat balances when the electrolysis heat is removed.

    `pairs` maps a label to (base scenario, scenario without electrolysis heat).
    Every cell is "without heat minus base".
    """
    blocks = {}
    for i, (block, bus) in enumerate(
        [
            ("Urban central", "urban central heat"),
            ("Urban decentral", "urban decentral heat"),
        ]
    ):
        rows = {
            label: carriers[i]
            for label, carriers in ELHEAT0_ROWS.items()
            if carriers[i]
        }
        columns = {}
        for label, (base, variant) in pairs.items():
            columns[label] = group_balance(
                energy_balance(get_network(variant), bus).sub(
                    energy_balance(get_network(base), bus), fill_value=0.0
                ),
                rows,
            )
        table = pd.DataFrame(columns)
        # technologies that do not move
        keep = (table.abs() >= 0.05).any(axis=1) | (
            table.index == "Exogenous heat demand"
        )
        table = table.loc[keep]
        table.loc["Sum"] = table.sum()
        blocks[block] = table
    table = pd.concat(blocks)
    table.index.names = ["Heat bus", "Change of the annual heat balance"]
    return table


def electrolysis_heat_split(costs_path):
    """
    Where the electricity entering an electrolyzer goes, in %.

    `efficiency` and `efficiency-heat` of the cost data are two shares of the
    same input, so the losses are what remains of both.
    """
    costs = pd.read_csv(costs_path, index_col=["technology", "parameter"])["value"]
    hydrogen = costs[("electrolysis", "efficiency")]
    heat = costs[("electrolysis", "efficiency-heat")]
    return 100 * pd.Series(
        {
            "Hydrogen (LHV)": hydrogen,
            "Recoverable low-temperature heat": heat,
            "Conversion losses": 1 - hydrogen - heat,
        },
        name="Share of the electricity input in %",
    )


def heat_vents(n):
    """
    Carriers of any component that could dump heat, by name.
    """
    found = set()
    for component in ("links", "generators", "stores", "storage_units"):
        carriers = getattr(n, component).carrier
        found |= set(carriers[carriers.str.contains("vent", case=False)])
    return sorted(found)


def electrolysis_heat(n):
    """
    Electrolysis heat delivered to the urban central heat buses and its value
    at their marginal price, in TWh/a, EUR/MWh and EUR/kg.

    The value per kg is the heat by-product of `lcoh_decomposition`, per kg of
    all hydrogen from electrolysis, including the nodes without an urban
    central heat bus.
    """
    weights = n.snapshot_weightings.generators
    prices = n.buses_t.marginal_price

    heat = energy_balance(n, "urban central heat")["H2 Electrolysis"]  # TWh/a
    by_product = lcoh_decomposition(n).loc["Heat by-product", "Electrolysis"]
    h2_kg = mwh_to_kg(hydrogen_output(n)["Electrolysis"][0].to_numpy().sum())

    heat_buses = n.buses.index[n.buses.carrier == "urban central heat"]
    average_price = (
        prices[heat_buses].mul(weights, axis=0).sum() / weights.sum()
    ).mean()

    return {
        ("Electrolysis heat delivered", "TWh/a"): heat,
        ("Value at the marginal heat price", "EUR/MWh"): by_product
        * h2_kg
        / (heat * 1e6),
        ("per kg of electrolysis hydrogen", "EUR/kg"): by_product,
        ("Heat price, time and node average", "EUR/MWh"): average_price,
        ("Components that could vent heat", "count"): len(heat_vents(n)),
    }


def electrolysis_heat_value(get_network, scenarios, df):
    """
    `electrolysis_heat` over the cost points of the base sweep.
    """
    columns = {
        share: electrolysis_heat(get_network(scenario))
        for scenario, share in zip(scenarios, market_share_columns(df, scenarios))
    }
    table = pd.DataFrame(columns)
    table.columns.name = "PC market share in %"
    return table


def nodal_heat_credit(n):
    """
    Electrolysis heat valued at the marginal price of the urban central heat
    bus it is delivered to, per kg of the hydrogen electrolysis produces at
    that node, in EUR/kg; zero at nodes without that bus (no urban central heat
    bus available)
    """
    weights = n.snapshot_weightings.generators
    prices = n.buses_t.marginal_price
    el = n.links[n.links.carrier == "H2 Electrolysis"]
    connected = el[el.bus2.isin(n.buses.index)]

    heat = -n.links_t.p2[connected.index].mul(weights, axis=0)
    revenue = (
        (heat * prices[connected.bus2].values).sum().reindex(el.index, fill_value=0.0)
    )
    h2 = -n.links_t.p1[el.index].mul(weights, axis=0).sum()
    credit = eur_per_kg(revenue, h2.where(h2 > 0))
    credit.index = el.index.str.replace(" H2 Electrolysis", "", regex=False)
    return credit


def heat_credited_medians(get_network, scenarios, df, min_capacity=1):
    """
    Nodal LCOH medians with the electrolysis heat credited, unweighted.

    The nodal LCOH and capacities are those of total_summary.csv, so the
    uncredited rows are the medians of `lcoh_medians`; only the heat credit
    comes from the network. The last row sets the production-weighted heat
    by-product against the difference of the unweighted medians, as the
    paper does ("about a third").
    """
    columns = {}
    for scenario, share in zip(scenarios, market_share_columns(df, scenarios)):
        n = get_network(scenario)
        capacity = parse_nodal(
            df.loc[scenario, "nodal capacity node:el:pc"], ["EL", "PC"]
        )
        lcoh = parse_nodal(df.loc[scenario, "nodal LCOH node:el:pc"], ["EL", "PC"])
        el_nodes = capacity.index[capacity.EL >= min_capacity]
        pc_nodes = capacity.index[capacity.PC >= min_capacity]
        credit = nodal_heat_credit(n).reindex(el_nodes)
        el = lcoh.loc[el_nodes, "EL"]
        credited = el - credit
        pc = lcoh.loc[pc_nodes, "PC"].median()
        by_product = lcoh_decomposition(n).loc["Heat by-product", "Electrolysis"]
        del n  # lets get_network free it before it opens the next one
        columns[share] = {
            ("EL median LCOH", "EUR/kg"): el.median(),
            ("EL median LCOH, heat credited", "EUR/kg"): credited.median(),
            ("PC median LCOH", "EUR/kg"): pc,
            ("Median heat credit", "EUR/kg"): credit.median(),
            ("Relative difference", "%"): 100 * (el.median() - pc) / el.median(),
            ("Relative difference, heat credited", "%"): 100
            * (credited.median() - pc)
            / credited.median(),
            ("Heat by-product, production-weighted", "EUR/kg"): by_product,
            ("Heat by-product, share of the median difference", "%"): (
                100 * by_product / (el.median() - pc)
            ),
        }
    table = pd.DataFrame(columns)
    table.columns.name = "PC market share in %"
    return table


def water_tank_losses(n):
    """
    Share of the heat charged into the urban central water tanks that is lost,
    in %.

    Charger and discharger are the only flows between the tanks and the
    heat bus, so the difference is the conversion and standing losses.
    """
    balance = energy_balance(n, "urban central heat")
    charged = -balance["urban central water tanks charger"]
    discharged = balance["urban central water tanks discharger"]
    return 100 * (charged - discharged) / charged


def elheat0_network_rows(df, n, scenario):
    """
    The rows of `elheat0_comparison` that `transport_aggregates` does not have.
    """
    heat_pumps = n.links[n.links.carrier == "urban central air heat pump"]
    return {
        "electrolysis heat delivered (TWh/a)": energy_balance(
            n, "urban central heat"
        ).get("H2 Electrolysis", 0.0),
        "urban central air heat pumps (GW)": heat_pumps.p_nom_opt.sum() / 1e3,
        "hydrogen storage (TWh)": float(df.loc[scenario, "tot H2 storage"]),
        "hydrogen pipelines (TW km)": float(
            df.loc[scenario, "total H2 pipeline capacity"]
        ),
        "fossil oil (TWh/a)": energy_balance(n, "oil").get("oil", 0.0),
        "CO2 captured by direct air capture (Mt/a)": energy_balance(
            n, "co2 stored"
        ).get("DAC", 0.0),
        "water tank losses, share of the heat charged (%)": water_tank_losses(n),
    }


def elheat0_comparison(df, get_network, cases):
    """
    The base cases against their re-optimizations without the electrolysis heat
    by-product.

    `cases` maps a label to a scenario, in the order PC-0, PC-0 without heat,
    PC-50, PC-50 without heat, as for `transport_aggregates`, whose columns
    come first. Returns the table and the differences of each pair.
    """
    table, _ = transport_aggregates(df, cases)
    extra = pd.DataFrame(
        {
            label: elheat0_network_rows(df, get_network(scenario), scenario)
            for label, scenario in cases.items()
        }
    ).T
    table = table.join(extra)
    return table, pair_differences(table, cases)


def elheat0_heat_replacement(df, get_network, pairs):
    """
    System cost if no electrolysis heat is available, compared to the value the
    base case assigns to that heat at the marginal heat price.

    `pairs` maps a label to (base scenario, scenario without electrolysis
    heat). The replacement cost is the system cost increase divided by the
    heat electrolysis delivers in the base case. Per kg, both are divided by
    all hydrogen from electrolysis of the base case, as in
    `electrolysis_heat`.
    """
    columns = {}
    for label, (base, variant) in pairs.items():
        heat = electrolysis_heat(get_network(base))
        heat_twh = heat[("Electrolysis heat delivered", "TWh/a")]
        credit = heat[("Value at the marginal heat price", "EUR/MWh")]
        increase = (
            float(df.loc[variant, "total system cost"])
            - float(df.loc[base, "total system cost"])
        ) / 1e9
        replacement = increase * 1e3 / heat_twh  # EUR/MWh
        # EUR/kg of hydrogen per EUR/MWh of heat
        per_kg_h2 = heat[("per kg of electrolysis hydrogen", "EUR/kg")] / credit
        columns[label] = {
            ("System cost increase", "Bln EUR/a"): increase,
            ("Electrolysis heat of the base case", "TWh/a"): heat_twh,
            ("Its value at the marginal heat price", "Bln EUR/a"): credit
            * heat_twh
            / 1e3,
            ("Replacement cost", "EUR/MWh"): replacement,
            ("Value at the marginal heat price", "EUR/MWh"): credit,
            ("Replacement cost over value", "%"): 100 * replacement / credit,
            ("Replacement cost, per kg of electrolysis hydrogen", "EUR/kg"): (
                replacement * per_kg_h2
            ),
            ("Value, per kg of electrolysis hydrogen", "EUR/kg"): credit * per_kg_h2,
        }
    return pd.DataFrame(columns)


# exogenous heat demand of a heat bus
HEAT_DEMAND_ROWS = ["Heat demand", "Low-temperature heat for industry"]


def elheat0_heat_buses(get_network, pairs):
    """
    Heat demand of each bus type (urban central, urban decentral, rural) and
    the largest change of any row of its grouped heat balance without the
    electrolysis heat by-product (in TWh/a).
    """
    blocks = {}
    for label, (base, variant) in pairs.items():
        with_heat = heat_balances(get_network(base))
        without = heat_balances(get_network(variant))
        rows = {}
        for bus, balance in with_heat.items():
            change = without[bus].sub(balance, fill_value=0.0)
            demand = -balance[HEAT_DEMAND_ROWS].sum()
            rows[bus] = {
                "Heat demand in TWh/a": demand,
                "Heat demand change in TWh/a": -change[HEAT_DEMAND_ROWS].sum(),
                "Largest change of a row in TWh/a": change.abs().max(),
                "Largest change, share of the demand in %": 100
                * change.abs().max()
                / demand,
            }
        blocks[label] = pd.DataFrame(rows).T
    return pd.concat(blocks, axis=1)


def photocatalysis_heat_sinks(n, min_capacity=1.0):
    """
    District heating demand at the nodes with photocatalysis and at all nodes
    (in TWh/a), its April-September share, and the same share of the
    photocatalysis output.

    No heat output of photocatalysis is modeled; the table shows how much
    district heating demand a heat by-product would meet at its nodes and
    when. Demand: all loads at the urban central heat buses (heat demand and
    low-temperature heat for industry, as `HEAT_DEMAND_ROWS`).
    """
    weights = n.snapshot_weightings.generators
    summer = n.snapshots.month.isin(MV_SUMMER_MONTHS)

    pc = n.generators[n.generators.carrier == "photocatalysis"]
    pc = pc[pc.p_nom_opt >= min_capacity]
    pc_nodes = n.buses.loc[pc.bus, "location"].unique()
    h2 = n.generators_t.p[pc.index].mul(weights, axis=0).sum(axis=1)

    heat_buses = n.buses.index[n.buses.carrier == "urban central heat"]
    loads = n.loads[n.loads.bus.isin(heat_buses)]
    demand = n.get_switchable_as_dense("Load", "p_set")[loads.index]
    demand = demand.mul(weights, axis=0)
    demand = demand.T.groupby(n.buses.loc[loads.bus, "location"].values).sum().T

    el_all = n.links[n.links.carrier == "H2 Electrolysis"]
    el_all = el_all[el_all.p_nom_opt >= min_capacity]
    el_nodes = n.buses.loc[el_all.bus1, "location"].unique()
    el = el_all[el_all.bus2.isin(heat_buses)]
    el_heat_t = -n.links_t.p2[el.index].mul(weights, axis=0)
    el_heat = el_heat_t.sum().groupby(n.buses.loc[el.bus2, "location"].values).sum()

    rows = {}
    for label, nodes in {
        "Nodes with photocatalysis": pc_nodes,
        "Nodes with electrolysis": el_nodes,
        "All nodes": n.buses.index[n.buses.carrier == "AC"],
    }.items():
        at_nodes = demand[demand.columns.intersection(nodes)].sum(axis=1)
        rows[label] = {
            "Nodes": len(nodes),
            "Nodes with district heating": len(demand.columns.intersection(nodes)),
            "District heating demand in TWh/a": at_nodes.sum() / 1e6,
            "District heating demand Apr-Sep in %": 100
            * at_nodes[summer].sum()
            / at_nodes.sum(),
            "Electrolysis heat in TWh/a": el_heat.reindex(nodes).sum() / 1e6,
        }
    table = pd.DataFrame(rows).T
    table["Share of the district heating demand in %"] = (
        100
        * table["District heating demand in TWh/a"]
        / table.loc["All nodes", "District heating demand in TWh/a"]
    )
    table.loc["Nodes with photocatalysis", "Photocatalysis hydrogen in TWh/a"] = (
        h2.sum() / 1e6
    )
    table.loc["Nodes with photocatalysis", "Photocatalysis hydrogen Apr-Sep in %"] = (
        100 * h2[summer].sum() / h2.sum()
    )
    table.loc["Nodes with electrolysis", "Electrolysis heat Apr-Sep in %"] = (
        100 * el_heat_t[summer].sum().sum() / el_heat_t.sum().sum()
    )
    return table


# Price electrolysis pays and the price distribution at its nodes: the fleet
# and the distribution over the hours.

EL_SUMMER_MONTHS = [5, 6, 7, 8, 9, 10]
EL_PERIODS = ["May-Oct", "Nov-Apr"]
EL_PRICE_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# row of `electrolysis_fleet` -> printed row
EL_FLEET_ROWS = {
    "links": "Electrolysis links above 1 MW",
    "capacity": "Capacity in GW",
    "energy": "Electricity drawn in TWh/a",
    "full_load_hours": "Full load hours in h",
    "price_paid": "Price paid in EUR/MWh",
    "time_averaged_price": "Time-averaged price at its nodes in EUR/MWh",
    "discount": "Price paid below that average in EUR/MWh",
    "discount_percent": "Price paid below that average in %",
}


def electrolysis_hours(n, min_capacity=1.0):
    """
    One row per snapshot and electrolysis link above `min_capacity` MW:
    capacity, snapshot weighting, electricity drawn, the price at its
    electricity bus and the half-year.
    """
    el = n.links[
        (n.links.carrier == "H2 Electrolysis") & (n.links.p_nom_opt > min_capacity)
    ]
    links = el.index.rename("link")
    hours = (
        pd.concat(
            [
                n.links_t.p0[el.index].set_axis(links, axis=1),
                n.buses_t.marginal_price[el.bus0].set_axis(links, axis=1),
            ],
            axis=1,
            keys=["draw", "price"],
        )
        .rename_axis(index="snapshot", columns=["quantity", "link"])
        .stack("link", future_stack=True)
        .rename_axis(columns=None)
        .reset_index()
    )
    hours["capacity"] = hours.link.map(el.p_nom_opt)
    hours["weight"] = hours.snapshot.map(n.snapshot_weightings.generators)
    hours["energy"] = hours.draw * hours.weight
    hours["period"] = np.where(
        hours.snapshot.dt.month.isin(EL_SUMMER_MONTHS), EL_PERIODS[0], EL_PERIODS[1]
    )
    return hours


def electrolysis_links(hours):
    """
    One row per electrolysis link: capacity in MW, electricity drawn in MWh/a,
    full load hours, the price paid (weighted by the electricity drawn) and the
    time-averaged price at its bus (every snapshot by its weighting).
    """
    by_link = hours.groupby("link")
    table = pd.DataFrame(
        {"capacity": by_link.capacity.first(), "energy": by_link.energy.sum()}
    )
    table["full_load_hours"] = table.energy / table.capacity
    table["price_paid"] = weighted_mean(hours.price, hours.energy, by=hours.link)
    table["time_averaged_price"] = weighted_mean(
        hours.price, hours.weight, by=hours.link
    )
    return table


def electrolysis_fleet_summary(hours):
    """
    The electrolysis fleet as one column: totals of `electrolysis_links`, and
    its prices averaged over the links, each weighted by the electricity it
    draws.
    """
    links = electrolysis_links(hours)
    fleet = weighted_mean(links[["price_paid", "time_averaged_price"]], links.energy)
    fleet["links"] = len(links)
    fleet["capacity"] = links.capacity.sum() / 1e3
    fleet["energy"] = links.energy.sum() / 1e6
    fleet["full_load_hours"] = links.energy.sum() / links.capacity.sum()
    fleet["discount"] = fleet.time_averaged_price - fleet.price_paid
    fleet["discount_percent"] = fleet.discount / fleet.time_averaged_price * 100
    return fleet


def electrolysis_fleet(hours_pc0, hours_pc50):
    """
    Price electrolysis pays against the time-averaged price at its nodes, with
    capacity, electricity drawn and full load hours, both cases.
    """
    table = compare(
        electrolysis_fleet_summary(hours_pc0), electrolysis_fleet_summary(hours_pc50)
    )
    return table.loc[list(EL_FLEET_ROWS)].rename(index=EL_FLEET_ROWS)


def electrolysis_price_percentiles(hours_pc0, hours_pc50):
    """
    Electricity price percentiles at the links with electrolysis in both cases,
    over the year and per half-year.

    Each node-hour is weighted by its snapshot weighting and the node's
    PC-0 electrolysis capacity.
    """
    common = set(hours_pc0.link) & set(hours_pc50.link)
    pc0 = hours_pc0[hours_pc0.link.isin(common)]
    pc50 = hours_pc50[hours_pc50.link.isin(common)]
    pair = (
        pc0[["snapshot", "link", "period"]]
        .assign(node_weight=pc0.weight * pc0.capacity, **{"PC-0": pc0.price})
        .merge(
            pc50[["snapshot", "link", "price"]].rename(columns={"price": "PC-50"}),
            on=["snapshot", "link"],
        )
    )
    periods = pd.concat([pair.assign(period="Year"), pair])
    labels = [f"P{round(q * 100)}" for q in EL_PRICE_QUANTILES]

    def percentiles(group):
        return pd.DataFrame(
            [
                weighted_quantiles(group[case], group.node_weight, EL_PRICE_QUANTILES)
                for case in ["PC-0", "PC-50"]
            ],
            index=["PC-0", "PC-50"],
            columns=labels,
        )

    table = periods.groupby("period")[["node_weight", "PC-0", "PC-50"]].apply(
        percentiles
    )
    table = table.reindex(
        pd.MultiIndex.from_product([["Year", *EL_PERIODS], ["PC-0", "PC-50"]])
    )
    table.index.names = ["Period", "Case"]
    return table


# Dark lulls: positive residual load (PRL) events found with the Sequent Peak
# Algorithm (SPA) of Kittel & Schill 2024 (Environ. Res.: Energy 1, 035007,
# section 3.2, eq. 7, p. 12), and what covers the drop in wind and solar output
# during them. Residual load is exogenous electricity load minus wind and solar
# generation. All power values are in GW per snapshot, energies in GWh.

ELEC_BUS_CARRIERS = ["AC", "low voltage"]

WIND_SOLAR_CARRIERS = [
    "solar",
    "solar-hsat",
    "solar rooftop",
    "onwind",
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
]

# Every other technology in these networks that can supply the electricity
# buses. Run-of-river follows an inflow profile but is not part of the residual
# load. `electricity_by_carrier` raises if anything else supplies electricity.
SUPPLY_CARRIERS = [
    "OCGT",
    "urban central gas CHP",
    "urban central gas CHP CC",
    "H2 turbine",
    "H2 Fuel Cell",
    "nuclear",
    "hydro",
    "ror",
    "PHS",
    "battery discharger",
    "home battery discharger",
]

# rows of the coverage table; all other carriers form the remainder
COVERAGE_GROUPS = {
    "Electrolysis": ["H2 Electrolysis"],
    "Heat pumps and resistive heaters": [
        "rural air heat pump",
        "rural ground heat pump",
        "urban central air heat pump",
        "urban decentral air heat pump",
        "rural resistive heater",
        "urban central resistive heater",
        "urban decentral resistive heater",
    ],
    "Batteries and pumped hydro": [
        "battery charger",
        "battery discharger",
        "home battery charger",
        "home battery discharger",
        "PHS",
    ],
    "Hydro and nuclear": ["hydro", "nuclear"],
    "Gas combined heat and power": [
        "urban central gas CHP",
        "urban central gas CHP CC",
    ],
    "Open-cycle gas turbines": ["OCGT"],
}

# heating season, for the winter reference event
WINTER_MONTHS = {11, 12, 1, 2, 3}


def elec_buses(n):
    return n.buses.index[n.buses.carrier.isin(ELEC_BUS_CARRIERS)]


def electricity_by_carrier(n, noise=0.05):
    """
    Power delivered to (positive) or taken from (negative) the electricity
    buses per snapshot and carrier, in GW. Loads are not included.

    AC and low voltage buses are one set, so the distribution grid and the DC
    links, which connect two buses of that set, net to their losses. Raises if
    a carrier outside `WIND_SOLAR_CARRIERS` and `SUPPLY_CARRIERS` supplies more
    than `noise` TWh/a.
    """
    buses = elec_buses(n)
    parts = []
    for port in range(5):
        p = n.links_t.get(f"p{port}")
        if f"bus{port}" not in n.links.columns or p is None:
            continue
        links = n.links.index[n.links[f"bus{port}"].isin(buses)]
        links = links.intersection(p.columns)
        parts.append(-p[links].T.groupby(n.links.carrier[links]).sum().T)
    for component, p in [
        (n.generators, n.generators_t.p),
        (n.storage_units, n.storage_units_t.p),
        (n.stores, n.stores_t.p),
    ]:
        index = component.index[component.bus.isin(buses)]
        parts.append(p[index].T.groupby(component.carrier[index]).sum().T)
    power = pd.concat(parts, axis=1).T.groupby(level=0).sum().T / 1e3

    supply = power.clip(lower=0).mul(n.snapshot_weightings.generators, axis=0).sum()
    known = set(WIND_SOLAR_CARRIERS) | set(SUPPLY_CARRIERS)
    unknown = [c for c in supply.index if c not in known and supply[c] / 1e3 > noise]
    if unknown:
        raise RuntimeError(f"electricity supply outside the carrier lists: {unknown}")
    return power


def residual_load(n, power):
    """
    Exogenous electricity load, wind and solar generation and residual load per
    snapshot in GW.

    `power` is `electricity_by_carrier(n)`.
    """
    loads = n.loads.index[n.loads.bus.isin(elec_buses(n))]
    load = n.loads_t.p[loads].sum(axis=1) / 1e3
    wind_solar = power[power.columns.intersection(WIND_SOLAR_CARRIERS)].sum(axis=1)
    return pd.DataFrame(
        {"load": load, "wind and solar": wind_solar, "residual": load - wind_solar}
    )


def spa_deficit(residual, weights):
    """
    Cumulative energy deficit of the SPA in GWh (Kittel & Schill 2024, eq.

    7),
    ED_t = max(0, ED_{t-1} + rl_t), with rl_t the residual energy of snapshot t:
    `residual` in GW times the snapshot length `weights` in hours.
    """
    deficit = np.empty(len(residual))
    previous = 0.0
    for snapshot_i, energy in enumerate(residual.to_numpy() * weights.to_numpy()):
        previous = max(0.0, previous + energy)
        deficit[snapshot_i] = previous
    return pd.Series(deficit, index=residual.index)


def spa_events(residual, weights):
    """
    All PRL events of a residual load series, in time order.

    After Kittel & Schill 2024 (section 3.2, p. 12), an event lasts from
    the first snapshot with a positive cumulative deficit to the
    snapshot of its peak ("duration"), and the recovery period from the
    peak until the deficit is back at zero ("recovery"). "deficit" is
    the peak cumulative deficit in GWh. An event still in deficit at the
    end of the year gets no recovery.
    """
    deficit = spa_deficit(residual, weights)
    positive = (deficit > 1e-6).to_numpy().astype(int)
    # +1 where the deficit turns positive, -1 where it is back at zero
    edges = np.diff(np.concatenate([[0], positive, [0]]))
    rows = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        # stop is the first snapshot with zero deficit again
        peak = start + int(np.argmax(deficit.iloc[start:stop].to_numpy()))
        recovered = stop < len(deficit)
        rows.append(
            {
                "start": deficit.index[start],
                "peak": deficit.index[peak],
                "recovered": deficit.index[stop] if recovered else pd.NaT,
                "deficit [GWh]": deficit.iloc[peak],
                "duration [h]": weights.iloc[start : peak + 1].sum(),
                "recovery [h]": (
                    weights.iloc[peak + 1 : stop + 1].sum() if recovered else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def reference_events(n):
    """
    The two reference events, found on the residual load of `n`: the largest
    event of the year, and the largest whose start falls in `WINTER_MONTHS`.

    Exogenous load and per-unit wind and solar availability are the same
    in every base case, but realized wind and solar generation also
    depends on the optimized capacities. The events are therefore found
    on one case and applied to all, so the cases differ in their
    response, not in the event. "rank" is the event's rank by peak
    deficit among the "events in the year".
    """
    events = spa_events(
        residual_load(n, electricity_by_carrier(n))["residual"],
        n.snapshot_weightings.generators,
    )
    events["rank"] = events["deficit [GWh]"].rank(ascending=False).astype(int)
    events["events in the year"] = len(events)
    winter = events[events["start"].dt.month.isin(WINTER_MONTHS)]
    table = events.loc[
        [events["deficit [GWh]"].idxmax(), winter["deficit [GWh]"].idxmax()]
    ]
    table.index = ["Largest of the year", "Largest of the heating season"]
    return table


def electricity_supply_capacity(networks):
    """
    Capacity, annual generation and full load hours of every carrier in
    `SUPPLY_CARRIERS`, per case. `networks` maps a case label to its network.

    Only components on the electricity buses count, so a carrier's
    capacity is its electric output capacity.
    """
    columns = {}
    for label, n in networks.items():
        buses = elec_buses(n)
        capacity = pd.concat(
            [
                n.links[n.links.bus1.isin(buses)].groupby("carrier").p_nom_opt.sum(),
                n.generators[n.generators.bus.isin(buses)]
                .groupby("carrier")
                .p_nom_opt.sum(),
                n.storage_units[n.storage_units.bus.isin(buses)]
                .groupby("carrier")
                .p_nom_opt.sum(),
            ]
        )
        capacity = capacity.groupby(level=0).sum().reindex(SUPPLY_CARRIERS) / 1e3
        generation = (
            electricity_by_carrier(n)
            .clip(lower=0)
            .mul(n.snapshot_weightings.generators, axis=0)
            .sum()
            .reindex(SUPPLY_CARRIERS)
            / 1e3
        )
        columns[label] = pd.DataFrame(
            {
                "Capacity in GW": capacity,
                "Generation in TWh/a": generation,
                "Full load hours": (1e3 * generation / capacity).where(capacity > 1e-3),
            }
        )
    return pd.concat(columns, axis=1)


def dark_lull_coverage(networks, events):
    """
    What covers the drop in wind and solar output during each reference event.

    Every value is the event mean minus the annual mean, in GW, and its share of
    the drop. By the energy balance of the electricity buses, the drop equals
    the change of all other carriers plus the fall of the exogenous load, so the
    shares add to 100 %. Positive values offset the drop: more supply, or less
    electricity drawn. The event lasts from its start to its peak (see
    `spa_events`).

    Returns three tables, with a column per event and case:
    - the annual and event means of exogenous load and of wind and solar
      generation, in GW;
    - the coverage, in GW and in % of the drop;
    - the carriers of the remainder row larger than 0.5 GW.
    """
    grouped = {c for carriers in COVERAGE_GROUPS.values() for c in carriers}
    means, coverage, remainders = {}, {}, {}
    for label, n in networks.items():
        weights = n.snapshot_weightings.generators
        power = electricity_by_carrier(n)
        balance = residual_load(n, power)[["load", "wind and solar"]]
        annual_power = weighted_mean(power, weights)
        annual_balance = weighted_mean(balance, weights)

        for event, (start, peak) in events[["start", "peak"]].iterrows():
            in_event = weights.loc[start:peak]
            event_balance = weighted_mean(balance.loc[start:peak], in_event)
            means[(event, label)] = pd.Series(
                {
                    "Exogenous load, annual mean": annual_balance["load"],
                    "Exogenous load, event mean": event_balance["load"],
                    "Wind and solar, annual mean": annual_balance["wind and solar"],
                    "Wind and solar, event mean": event_balance["wind and solar"],
                }
            )

            change = weighted_mean(power.loc[start:peak], in_event) - annual_power
            others = change.drop(change.index.intersection(WIND_SOLAR_CARRIERS))
            remainder = others.drop(others.index.intersection(grouped))
            drop = annual_balance["wind and solar"] - event_balance["wind and solar"]
            gw = pd.Series(
                {
                    "Drop in wind and solar output": drop,
                    **{
                        name: others.reindex(carriers).dropna().sum()
                        for name, carriers in COVERAGE_GROUPS.items()
                    },
                    "Exogenous load falls": annual_balance["load"]
                    - event_balance["load"],
                    "Remainder": remainder.sum(),
                }
            )
            coverage[(event, label, "GW")] = gw
            coverage[(event, label, "%")] = 100 * gw / drop
            remainders[(event, label)] = remainder[remainder.abs() > 0.5]

    remainders = pd.DataFrame(remainders)
    remainders = remainders.loc[
        remainders.abs().max(axis=1).sort_values(ascending=False).index
    ]
    return pd.DataFrame(means), pd.DataFrame(coverage), remainders


# --- Supplementary Note 5: Exogenous demand ---
#
# Every exogenous demand of a PyPSA-Eur sector network is a Load. All load
# carriers are read and grouped.

PLOTTING_CONFIG = os.path.join(REPO_ROOT, "src", "plotting.default.yaml")

# process emissions are a load as well, but a CO2 source in t/h
NON_ENERGY = ["process emissions"]

DEMAND_GROUPS = {
    "Heat": [
        "urban central heat",
        "urban decentral heat",
        "rural heat",
        "low-temperature heat for industry",
    ],
    "Electricity": ["electricity", "industry electricity"],
    "Liquid fuels": [
        "kerosene for aviation",
        "naphtha for industry",
        "shipping oil",
        "shipping methanol",
        "industry methanol",
    ],
    "Solid fuels": ["solid biomass for industry", "coal for industry"],
    "Gas": ["gas for industry"],
    "Hydrogen": ["H2 for industry"],
}

DEMAND_BARS = {
    # carriers bottom to top
    "Heat": [
        "urban central heat",
        "urban decentral heat",
        "rural heat",
        "low-temperature heat for industry",
    ],
    "Electricity": ["electricity", "industry electricity"],
    "Other industry\ndemands": [
        "kerosene for aviation",
        "shipping oil",
        "shipping methanol",
        "industry methanol",
        "naphtha for industry",
        "H2 for industry",
        "gas for industry",
        "coal for industry",
        "solid biomass for industry",
    ],
    "Transport": ["land transport EV", "land transport oil"],
}

# hydrogen consumers whose product is a liquid fuel
H2_TO_LIQUID_FUELS = ["Fischer-Tropsch", "methanolisation"]


def load_plotting_config():
    with open(PLOTTING_CONFIG, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_demand(n):
    """
    Annual energy of every load carrier in TWh, without process emissions.
    """
    balance = n.statistics.energy_balance(
        comps=["Load"], groupby=["carrier"], nice_names=False
    )
    return -balance.groupby(level="carrier").sum().drop(NON_ENERGY) / 1e6


def demand_table(networks):
    """
    Exogenous demand per carrier in TWh and as a share of the total.

    The demands are the same in every network; raises if they are not,
    or if a load carrier belongs to no group.
    """
    demands = pd.DataFrame({label: load_demand(n) for label, n in networks.items()})
    if (demands.max(axis=1) - demands.min(axis=1)).max() > 1e-6:
        raise ValueError("the exogenous demands differ between the networks")
    demand = demands.iloc[:, 0]

    carriers = [c for group in DEMAND_GROUPS.values() for c in group]
    unknown = set(demand.index) - set(carriers)
    if unknown:
        raise ValueError(f"load carriers in no group: {sorted(unknown)}")

    nice_names = load_plotting_config()["plotting"]["nice_names"]
    index = pd.MultiIndex.from_tuples(
        [(g, nice_names[c]) for g, cs in DEMAND_GROUPS.items() for c in cs],
        names=["group", "demand"],
    )
    table = pd.DataFrame({"TWh/a": demand.reindex(carriers).to_numpy()}, index=index)
    table["share %"] = 100 * table["TWh/a"] / table["TWh/a"].sum()
    return table


def group_shares(table):
    """
    The demand table summed per group, with a total row.
    """
    groups = table.groupby(level="group", sort=False).sum()
    groups.loc["Total"] = groups.sum()
    return groups


def hydrogen_to_liquid_fuels(networks):
    """
    Hydrogen production and the part of it converted into liquid fuels, per
    case.

    The full hydrogen balance is the one of the Energy balances note.
    """
    rows = {}
    for label, n in networks.items():
        balance = energy_balance(n, "H2")
        production = balance.clip(lower=0).sum()
        to_liquid_fuels = -balance.reindex(H2_TO_LIQUID_FUELS).sum()
        rows[label] = {
            "production TWh/a": production,
            "to liquid fuels TWh/a": to_liquid_fuels,
            "to liquid fuels %": 100 * to_liquid_fuels / production,
        }
    return pd.DataFrame(rows).T


def exogenous_demands(n_base, n_transport):
    """
    Annual demand per bar and carrier, base case plus the transport sector.
    """
    base = load_demand(n_base)
    with_transport = load_demand(n_transport)

    common = base.index.intersection(with_transport.index)
    if (base[common] - with_transport[common]).abs().max() > 1e-6:
        raise ValueError("the transport run changes a load of the base run")

    group_of = {c: group for group, carriers in DEMAND_BARS.items() for c in carriers}
    unknown = set(with_transport.index) - set(group_of)
    if unknown:
        raise ValueError(f"load carriers in no bar: {sorted(unknown)}")

    carriers = [c for group in DEMAND_BARS.values() for c in group]
    table = pd.DataFrame(0.0, index=list(DEMAND_BARS), columns=carriers)
    for carrier, value in with_transport.items():
        table.loc[group_of[carrier], carrier] = value
    return table, base.sum(), with_transport.sum()


def exogenous_demands_bar(
    table, base_total, with_transport_total, output_folder, save, export_format
):
    plotting = load_plotting_config()["plotting"]
    colors = plotting["tech_colors"]
    nice_names = plotting["nice_names"]

    fig, ax = plt.subplots(figsize=set_size(fraction=0.8))
    table.plot(
        kind="bar",
        stacked=True,
        color=[colors[c] for c in table.columns],
        ax=ax,
        width=0.65,
        legend=False,
    )
    # energy basis per group (thermal, electrical, LHV) is stated in the caption
    ax.set_ylabel("TWh")
    ax.set_xlabel("")
    ax.set_ylim(top=table.sum(axis=1).max() * 1.22)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax.tick_params(axis="x", rotation=0)
    # transport only exists in the transport run
    ax.axvline(len(table.index) - 1.5, color="black", linewidth=0.7, linestyle="--")

    # Legend top to bottom, bar by bar
    handles, labels = [], []
    for carriers in DEMAND_BARS.values():
        for carrier in reversed(carriers):
            if table[carrier].any():
                handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[carrier]))
                labels.append(nice_names[carrier])
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=7,
    )

    # Values inside the segments that are large enough
    for p in ax.patches:
        if p.get_height() > 40:
            x, y = p.get_xy()
            ax.annotate(
                f"{p.get_height():.0f}",
                (x + p.get_width() / 2, y + p.get_height() / 2),
                ha="center",
                va="center",
                fontsize=7,
                bbox=dict(
                    boxstyle="round,pad=0",
                    edgecolor="none",
                    facecolor="white",
                    alpha=0.5,
                ),
            )

    # Shares of the base total, in brackets of the total with transport
    totals = table.sum(axis=1)
    for i, (group, top) in enumerate(totals.items()):
        share = rf"$({100 * top / with_transport_total:.0f}~\mathrm{{\%}})$"
        if group != "Transport":
            share = rf"${100 * top / base_total:.0f}~\mathrm{{\%}}$" + "\n" + share
        ax.annotate(
            share,
            (i, top + 0.02 * totals.max()),
            ha="center",
            va="bottom",
            fontsize=8,
            linespacing=1.4,
        )

    fig.tight_layout()
    save_or_plot_img(
        fig, output_folder, "exogenous_demands", save=save, export_format=export_format
    )


# --- Supplementary Note 6: Carbon sequestration ---


def ccs_smr_crossover(get_network, scenarios):
    """
    Hydrogen production by route per CO2 sequestration limit.

    `scenarios` maps the printed limit to a scenario. Shadow price and
    sequestered amount are not printed in the table but quoted in the text.
    """
    rows = {}
    for label, scenario in scenarios.items():
        n = get_network(scenario)
        h2 = energy_balance(n, "H2")
        # rounded first, so the share of two 0.0 values is 0.0 as well
        smr_cc = round(h2.get("SMR CC", 0.0), 1)
        el = round(h2.get("H2 Electrolysis", 0.0), 1)
        pc = round(h2.get("photocatalysis", 0.0), 1)
        sequestered = n.stores.loc[n.stores.carrier == "co2 sequestered", "e_nom_opt"]
        rows[label] = {
            "SMR CC in TWh/a": smr_cc,
            "Electrolysis in TWh/a": el,
            "Photocatalysis in TWh/a": pc,
            "Total H2 supply in TWh/a": round(h2[h2 > 0].sum(), 1),
            "PC market share in %": (
                round(100 * pc / (pc + el), 1) if pc + el > 0 else 0.0
            ),
            "H2 to methanation in TWh/a": round(abs(h2.get("Sabatier", 0.0)), 1),
            "sequestration shadow price in EUR/t": abs(
                n.global_constraints.loc["co2_sequestration_limit", "mu"]
            ),
            "sequestered in Mt/a": sequestered.sum() / 1e6,
        }
    table = pd.DataFrame(rows).T
    table.index.name = "CO2 sequestration limit in Mt/a"
    return table


# --- Supplementary Note 7: Transport sector ---


def transport_aggregates(df, cases):
    """
    System cost, hydrogen production and capacities with and without transport.
    """
    columns = [
        "total system cost",
        "H2 Electrolysis (Energy)",
        "photocatalysis (Energy)",
        "pc market share",
        "H2 Electrolysis",
        "photocatalysis",
        "flh electrolysis",
        "flh photocatalysis",
    ]
    table = df.loc[list(cases.values()), columns].astype(float)
    table.index = list(cases.keys())
    table["total system cost"] /= 1e9
    table["total H2 (Energy)"] = (
        table["H2 Electrolysis (Energy)"] + table["photocatalysis (Energy)"]
    )
    return table, pair_differences(table, cases)


# The transport comparison figures take `cases` as {label: scenario folder
# name}, in the order PC-0, PC-0 (+transport), PC-50, PC-50 (+transport).


def h2_gen_split_and_lcoh_violins(
    df,
    cases,
    output_folder_path,
    tech_colors,
    UGHS_line_color,
    save=False,
    export_format="svg",
    name="h2_gen_split_lcoh_violins_transport_vs_base",
):
    """
    Hydrogen generation by technology (left) and nodal LCOH violins (right) for
    a set of cases.

    Returns the generation table and the nodal LCOH rows.
    """
    cols = {
        "H2 Electrolysis (Energy)": "Electrolysis (TWh/a)",
        "photocatalysis (Energy)": "Photocatalysis (TWh/a)",
        "pc market share": "PC market share (%)",
    }
    table = df.loc[list(cases.values()), list(cols.keys())].astype(float)
    table.rename(index={v: k for k, v in cases.items()}, columns=cols, inplace=True)
    table["Total H2 (TWh/a)"] = (
        table["Electrolysis (TWh/a)"] + table["Photocatalysis (TWh/a)"]
    )

    labels = list(cases.keys())
    xtick_labels = [
        label.replace(" (+transport)", "\n(+transport)") for label in labels
    ]
    x = np.arange(len(labels))
    width = 0.7

    # per case, a case and its +transport variant share the same cost
    case_dfs = []
    for label, idx in cases.items():
        case_df = prepare_LCOH_violin_data(df=df, index=pd.Index([idx]))
        case_df["Case"] = label.replace(" (+transport)", "\n(+transport)")
        case_dfs.append(case_df)
    tot_caps = pd.concat(case_dfs, ignore_index=True)

    # same >=1 MW installed-capacity cut as the main LCOH figure (clip_capacity)
    tot_caps = tot_caps.drop(tot_caps[tot_caps.Capacity < 1].index)

    # narrower canvas, the annex prints it at text width
    fig, (ax1, ax3) = plt.subplots(
        ncols=2,
        nrows=1,
        figsize=set_size(use=True, fraction=1.25, subplots=(1, 2), scale=1.45),
        gridspec_kw={"wspace": 0.42},
    )

    ax1.bar(
        x,
        table["Electrolysis (TWh/a)"],
        width,
        label="Electrolysis",
        color=tech_colors["H2 Electrolysis"],
    )
    ax1.bar(
        x,
        table["Photocatalysis (TWh/a)"],
        width,
        bottom=table["Electrolysis (TWh/a)"],
        label="Photocatalysis",
        color=tech_colors["photocatalysis"],
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_ylabel("Hydrogen generation in TWh a$^{-1}$")
    ax1.grid(alpha=0.3, axis="y")
    ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        table["PC market share (%)"],
        ls="none",
        marker="x",
        c=UGHS_line_color,
        label="Photocatalysis market share",
    )
    ax2.set_ylabel(r"Photocatalysis market share in \%")
    ax2.set_ylim(0, max(60, table["PC market share (%)"].max() * 1.2))
    ax2.tick_params(axis="y", direction="in", right=True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    sns.violinplot(
        data=tot_caps,
        x="Case",
        y=r"$LCOH$ in € kg$^{-1}$",
        hue="Technology",
        order=xtick_labels,
        split=True,
        inner="quart",
        density_norm="count",
        bw_method="silverman",
        width=0.9,
        palette=[tech_colors["H2 Electrolysis"], tech_colors["photocatalysis"]],
        ax=ax3,
    )
    for violin_body in ax3.collections:
        violin_body.set_edgecolor("none")
    ax3.set_xlabel("")
    ax3.grid(alpha=0.3, axis="y")
    ax3.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax3.get_legend().remove()

    # One legend for both panels: the two technologies mean the same thing on
    # either side, so a shared row below the figure replaces two half-empty
    # per-axes legends and lets the panels themselves grow.
    figure_legend = fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
    )
    remove_legend_patch_edges(figure_legend)

    save_or_plot_img(
        fig, output_folder_path, name, save=save, export_format=export_format
    )

    return table, tot_caps


def transport_regional_diffs(df, cases):
    """
    Per-node capacity difference "variant minus base" in GW, for the two case
    pairs.
    """
    nodal_caps = {
        label: parse_nodal(df.loc[case, "nodal capacity node:el:pc"], ["EL", "PC"])
        / 1e3
        for label, case in cases.items()
    }
    labels = list(cases.keys())
    return {
        "PC-0: +transport minus base": nodal_caps[labels[1]].fillna(0)
        - nodal_caps[labels[0]].fillna(0),
        "PC-50: +transport minus base": nodal_caps[labels[3]].fillna(0)
        - nodal_caps[labels[2]].fillna(0),
    }


def regional_capacity_difference_maps(
    diffs,
    regions,
    output_folder_path,
    save=False,
    export_format="svg",
    name="regional_capacity_diff_transport_vs_base",
):
    """
    Four maps of the per-node electrolysis and photocatalysis capacity
    difference between a variation and its base case.
    """
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=set_size(use=True, subplots=(2, 2), scale=1.8)
    )

    panels = [
        (
            "PC-0: +transport minus base",
            "EL",
            axes[0, 0],
            "Electrolysis capacity diff. (GW)\nPC-0: +transport - base",
        ),
        (
            "PC-0: +transport minus base",
            "PC",
            axes[0, 1],
            "Photocatalysis capacity diff. (GW)\nPC-0: +transport - base",
        ),
        (
            "PC-50: +transport minus base",
            "EL",
            axes[1, 0],
            "Electrolysis capacity diff. (GW)\nPC-50: +transport - base",
        ),
        (
            "PC-50: +transport minus base",
            "PC",
            axes[1, 1],
            "Photocatalysis capacity diff. (GW)\nPC-50: +transport - base",
        ),
    ]

    # Shared symmetric color scale per column (EL and PC differ in magnitude)
    el_vmax = max(diffs[k]["EL"].abs().max() for k in diffs)
    pc_vmax = max(diffs[k]["PC"].abs().max() for k in diffs)
    vmax_map = {"EL": el_vmax, "PC": pc_vmax}

    for diff_key, col, ax, title in panels:
        plot_regions = regions.copy()
        plot_regions["diff"] = diffs[diff_key][col]
        vmax = vmax_map[col]
        plot_regions.plot(
            ax=ax,
            column="diff",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            linewidth=0.1,
            edgecolor="grey",
            missing_kwds={"color": "lightgrey"},
            legend=True,
            legend_kwds={"shrink": 0.7, "label": "GW"},
        )
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    save_or_plot_img(
        fig, output_folder_path, name, save=save, export_format=export_format
    )


# --- Supplementary Note 8: Hydrogen transport capacity ---

H2_PIPELINE_CARRIERS = ["H2 pipeline", "H2 pipeline retrofitted"]
LEAKAGE_PER_1000KM = 0.012  # Hofmann et al. 2025
# A corridor whose repurposed capacity reaches this share of its potential is
# at its limit; below the second one it is unused.
AT_LIMIT, UNUSED = 0.99, 0.01


def h2_pipelines(df, pc0, pc50):
    """
    Hydrogen pipeline capacity and cost from total_summary.csv.
    """
    rows = {
        ("Total", "TW km"): "total H2 pipeline capacity",
        ("Newly built", "TW km"): "total new H2 pipeline capacity",
        ("Repurposed", "TW km"): "total repurposed H2 pipeline capacity",
        ("Annualized cost", "Bln EUR/a"): "total H2 pipeline capacity cost",
    }
    values = df.loc[[pc0, pc50], list(rows.values())].astype(float).T
    values.index = pd.MultiIndex.from_tuples(rows, names=["Quantity", "Unit"])
    values.loc[("Annualized cost", "Bln EUR/a")] /= 1e9
    return compare(values[pc0], values[pc50])


def retrofit_potential(n_pc0, n_pc50):
    """
    Repurposed pipeline capacity against its potential, the exogenous upper
    bound per corridor (p_nom_max, a share of the existing methane grid).
    """
    columns = {}
    for label, n in [("PC-0", n_pc0), ("PC-50", n_pc50)]:
        ret = n.links[n.links.carrier == "H2 pipeline retrofitted"]
        ret = ret[ret.p_nom_max > 0]
        used = ret.p_nom_opt / ret.p_nom_max
        built = (ret.p_nom_opt * ret.length).sum() / 1e6  # TW km
        potential = (ret.p_nom_max * ret.length).sum() / 1e6
        columns[label] = {
            ("Built", "GW"): ret.p_nom_opt.sum() / 1e3,
            ("Potential", "GW"): ret.p_nom_max.sum() / 1e3,
            ("Built", "TW km"): built,
            ("Potential", "TW km"): potential,
            ("Unused potential", "TW km"): potential - built,
            ("Potential used", "%"): 100 * built / potential,
            ("Corridors with potential", "-"): len(ret),
            ("Corridors at their limit", "-"): (used >= AT_LIMIT).sum(),
            ("Corridors unused", "-"): (used < UNUSED).sum(),
        }
    table = pd.DataFrame(columns)
    table.index.names = ["Quantity", "Unit"]
    return compare(table["PC-0"], table["PC-50"])


def pipeline_lengths(n):
    """
    Length per H2 pipeline; -reversed twins are stored with length 0 and take
    the length of their parent link.
    """
    pipes = n.links[n.links.carrier.isin(H2_PIPELINE_CARRIERS)]
    length = pipes.length.copy()
    reversed_links = length.index.str.endswith("-reversed")
    if (length[reversed_links] != 0).any():
        raise RuntimeError("a -reversed pipeline has a length of its own")
    parents = length.index[reversed_links].str.replace("-reversed", "", regex=False)
    length.loc[reversed_links] = length.reindex(parents).to_numpy()
    if length.isna().any():
        raise RuntimeError("reversed pipeline without a parent link")
    return length


def h2_transport_losses(n_pc0, n_pc50):
    """
    Transport work and the pipeline leakage that is not modeled.

    Raises if a pipeline or a hydrogen store has losses, since the table
    is only meaningful for a loss-free model.
    """
    columns = {}
    for label, n in [("PC-0", n_pc0), ("PC-50", n_pc50)]:
        weights = n.snapshot_weightings.generators
        length = pipeline_lengths(n)
        stores = n.stores[n.stores.carrier.str.contains("H2", case=False)]
        if not (
            np.allclose(n.links.loc[length.index, "efficiency"], 1.0)
            and (stores.standing_loss == 0).all()
        ):
            raise RuntimeError(f"{label}: hydrogen pipelines or stores have losses")
        throughput = n.links_t.p0[length.index].abs().mul(weights, axis=0).sum()
        work = (throughput * length).sum() / 1e6  # TWh km

        generation = sum(h2.sum().sum() for h2, _ in hydrogen_output(n).values()) / 1e6
        leakage = LEAKAGE_PER_1000KM * work / 1000

        columns[label] = {
            ("Hydrogen generation", "TWh/a"): generation,
            ("Transported hydrogen", "TWh/a"): throughput.sum() / 1e6,
            ("Transport work", "TWh km/a"): work,
            ("Leakage at 1.2 % per 1000 km (not modeled)", "TWh/a"): leakage,
            ("Additional generation required to compensate", "%"): 100
            * leakage
            / generation,
        }
    table = pd.DataFrame(columns)
    table.index.names = ["Quantity", "Unit"]
    return table


# Corridors of `h2_corridors()`, flows from side A to side B. A side is a
# country code, a node, or "*" for any node.
H2_CORRIDORS = {
    "GR-IT": ("GR", "IT"),
    "ES-PT": ("ES", "PT"),
    "ES1 0-ES1 4": ("ES1 0", "ES1 4"),
    "ES1 4, all pipelines": ("*", "ES1 4"),
}


def h2_pipes(n):
    """
    H2 pipelines with their end nodes, and whether a link is a `-reversed`
    twin.
    """
    pipes = n.links[n.links.carrier.isin(H2_PIPELINE_CARRIERS)]
    return pd.DataFrame(
        {
            "bus0": pipes.bus0.str.replace(" H2", ""),
            "bus1": pipes.bus1.str.replace(" H2", ""),
            "twin": pipes.index.str.endswith("-reversed"),
            "p_nom_opt": pipes.p_nom_opt,
        }
    )


def cross_border_h2_capacity(n_pc0, n_pc50):
    """
    Cross-border H2 pipeline capacity per country pair in GW, new and
    repurposed pipelines, without the `-reversed` twins.
    """

    def per_pair(n):
        pipes = h2_pipes(n)
        pipes = pipes[~pipes.twin & (pipes.bus0.str[:2] != pipes.bus1.str[:2])]
        pair = [
            "-".join(sorted((a[:2], b[:2]))) for a, b in zip(pipes.bus0, pipes.bus1)
        ]
        return pipes.p_nom_opt.groupby(pair).sum() / 1e3

    table = compare(per_pair(n_pc0), per_pair(n_pc50)).fillna(0)
    table["Difference"] = table["PC-50"] - table["PC-0"]
    return table.sort_values("Difference", ascending=False)


def _on(node, side):
    return side == "*" or node == side or node[:2] == side


def _direction(bus0, bus1, a, b):
    """
    +1 for a link from side A to side B, -1 for the opposite, 0 otherwise.
    """
    if _on(bus0, a) and _on(bus1, b):
        return 1.0
    if _on(bus0, b) and _on(bus1, a):
        return -1.0
    return 0.0


def h2_corridors(networks):
    """
    Capacity and annual flow of the corridors in `H2_CORRIDORS`, one row per
    corridor and case.

    The flow sums all links of a corridor per timestep, the `-reversed`
    twins included, so opposite flows cancel; links and capacity leave
    the twins out.
    """
    rows = {}
    for case, n in networks.items():
        pipes = h2_pipes(n)
        weights = n.snapshot_weightings.generators
        for corridor, (a, b) in H2_CORRIDORS.items():
            sign = pd.Series(
                [_direction(x, y, a, b) for x, y in zip(pipes.bus0, pipes.bus1)],
                index=pipes.index,
            )
            links = pipes[sign != 0]
            flow = n.links_t.p0[links.index].mul(sign[links.index], axis=1).sum(axis=1)
            built = links[~links.twin & (links.p_nom_opt > 10)]
            rows[(corridor, case)] = {
                ("Links", "-"): len(built),
                ("Capacity", "GW"): built.p_nom_opt.sum() / 1e3,
                ("Flow A to B", "TWh/a"): flow.clip(lower=0).mul(weights).sum() / 1e6,
                ("Flow B to A", "TWh/a"): -flow.clip(upper=0).mul(weights).sum() / 1e6,
            }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table[("Net flow A to B", "TWh/a")] = (
        table[("Flow A to B", "TWh/a")] - table[("Flow B to A", "TWh/a")]
    )
    table.columns.names = ["Quantity", "Unit"]
    table.index.names = ["Corridor", "Case"]
    return table.sort_index(level="Corridor", sort_remaining=False)


# --- Supplementary Note 9: Underground hydrogen storage ---

UGHS_COUNTRIES = {
    "ES": "Spain",
    "GR": "Greece",
    "PT": "Portugal",
    "IT": "Italy",
    "FR": "France",
    "Europe": "Europe",
}


def photocatalysis_per_node(n, column):
    pc = n.generators[n.generators.carrier == "photocatalysis"]
    return pc[column].rename(lambda name: name.replace(" photocatalysis", ""))


def ughs_siting(n_ughs, n_no_ughs):
    """
    Photocatalysis capacity per country with and without UGHS, in GW and %.
    """
    nodal = pd.DataFrame(
        {
            "limit": photocatalysis_per_node(n_ughs, "p_nom_max"),
            "ughs": photocatalysis_per_node(n_ughs, "p_nom_opt"),
            "no_ughs": photocatalysis_per_node(n_no_ughs, "p_nom_opt"),
        }
    )
    country = nodal.groupby(nodal.index.str[:2]).sum() / 1e3
    country.loc["Europe"] = country.sum()

    table = pd.DataFrame(
        {
            ("Expansion limit", "GW"): country["limit"],
            ("PC-50", "GW"): country["ughs"],
            ("'PC-50' no UGHS", "GW"): country["no_ughs"],
            ("Change", "GW"): country["no_ughs"] - country["ughs"],
            ("Share PC-50", "%"): 100 * country["ughs"] / country.loc["Europe", "ughs"],
            ("Share no UGHS", "%"): 100
            * country["no_ughs"]
            / country.loc["Europe", "no_ughs"],
        }
    ).T
    table = table[list(UGHS_COUNTRIES)].rename(columns=UGHS_COUNTRIES)
    table.columns.name = None
    table.index.names = ["Quantity", "Unit"]
    return table


def ughs_cases(df, scenario, no_ughs_scenario):
    """
    Cost point, market share and hydrogen storage with and without UGHS.
    """
    columns = {
        ("CAPEX", "EUR/kW"): "pc capex",
        ("Annualized cost", "EUR/a/kW"): "pc total annual cost",
        ("Market share", "%"): "pc market share",
        ("UGHS", "TWh"): "ughs",
        ("AGHS", "TWh"): "aghs",
    }
    table = df.loc[[scenario, no_ughs_scenario], list(columns.values())].astype(float)
    table.index = ["PC-50", "PC-50 no UGHS"]
    table.columns = pd.MultiIndex.from_tuples(columns, names=["Quantity", "Unit"])
    return table.T


def h2_stores_per_node(n):
    """
    UGHS potential and built UGHS and AGHS per node in TWh.

    UGHS and AGHS are told apart by capital cost, UGHS being the cheaper
    one, as in create_basic_summary.py; with a single cost, all stores
    are AGHS.
    """
    stores = n.stores[n.stores.carrier == "H2"]
    costs = stores.capital_cost.unique()
    ughs = (stores.capital_cost == costs.min()) & (len(costs) > 1)
    table = pd.DataFrame(
        {
            "UGHS potential": stores.e_nom_max.where(ughs),
            "UGHS": stores.e_nom_opt.where(ughs, 0),
            "AGHS": stores.e_nom_opt.where(~ughs, 0),
        }
    ).rename(lambda name: name.replace(" H2 Store", ""))
    return table / 1e6


def electrolysis_per_node(n):
    el = n.links[n.links.carrier == "H2 Electrolysis"]
    return el.p_nom_opt.rename(lambda name: name.replace(" H2 Electrolysis", ""))


def add_utilization(table):
    """
    Installed photocatalysis as a share of the expansion limit, both cases.
    """
    limit = table[("Expansion limit", "GW")]
    table[("Utilization PC-50", "%")] = 100 * table[("PC PC-50", "GW")] / limit
    table[("Utilization no UGHS", "%")] = 100 * table[("PC no UGHS", "GW")] / limit
    return table


def ughs_nodes(n_ughs, n_no_ughs, total_eval):
    """
    Photocatalysis, electrolysis and hydrogen storage per node with and without
    UGHS, next to the node properties that drive the siting.

    The network distance to the nearest UGHS node is taken from
    total_eval.csv.
    """
    pc = n_ughs.generators[n_ughs.generators.carrier == "photocatalysis"]
    irradiance = (
        n_ughs.generators_t.p_max_pu[pc.index]
        .mul(n_ughs.snapshot_weightings.generators, axis=0)
        .sum()
    )
    stores = h2_stores_per_node(n_ughs)
    stores_no_ughs = h2_stores_per_node(n_no_ughs)
    properties = {
        ("Latitude", "°"): n_ughs.buses.y,
        ("Longitude", "°"): n_ughs.buses.x,
        ("Solar irradiance", "kWh/m2/a"): irradiance.rename(
            lambda name: name.replace(" photocatalysis", "")
        ),
        ("Distance to UGHS", "km"): total_eval.loc["shortest_path_length_to_UGHS"],
        ("UGHS potential", "TWh"): stores["UGHS potential"],
        ("Expansion limit", "GW"): photocatalysis_per_node(n_ughs, "p_nom_max") / 1e3,
    }
    built = {
        ("PC PC-50", "GW"): photocatalysis_per_node(n_ughs, "p_nom_opt") / 1e3,
        ("PC no UGHS", "GW"): photocatalysis_per_node(n_no_ughs, "p_nom_opt") / 1e3,
        ("EL PC-50", "GW"): electrolysis_per_node(n_ughs) / 1e3,
        ("EL no UGHS", "GW"): electrolysis_per_node(n_no_ughs) / 1e3,
        ("UGHS PC-50", "TWh"): stores["UGHS"],
        ("AGHS PC-50", "TWh"): stores["AGHS"],
        ("AGHS no UGHS", "TWh"): stores_no_ughs["AGHS"],
    }
    nodes = total_eval.columns
    table = pd.concat(
        [
            pd.DataFrame({key: col.reindex(nodes) for key, col in properties.items()}),
            pd.DataFrame(
                {key: col.reindex(nodes) for key, col in built.items()}
            ).fillna(0),
        ],
        axis=1,
    )
    table.columns.names = ["Quantity", "Unit"]
    table.index.name = "Node"
    return add_utilization(table).sort_values(("PC PC-50", "GW"), ascending=False)


def ughs_countries(nodes):
    """
    The node table summed per country, plus a Europe row: node counts,
    utilization of the expansion limit, electrolysis and hydrogen storage.

    The photocatalysis capacities and shares are in `ughs_siting()`.
    """
    additive = [
        "Expansion limit", "PC PC-50", "PC no UGHS", "EL PC-50", "EL no UGHS",
        "UGHS PC-50", "AGHS PC-50", "AGHS no UGHS",
    ]  # fmt: skip
    country = nodes.index.str[:2]
    summed = nodes[additive].groupby(country).sum()
    summed.loc["Europe"] = summed.sum()
    table = pd.DataFrame(
        {
            ("Nodes", "-"): nodes.groupby(country).size(),
            ("Nodes with UGHS potential", "-"): (nodes[("UGHS potential", "TWh")] > 0)
            .groupby(country)
            .sum(),
        }
    )
    table.loc["Europe"] = table.sum()
    table = add_utilization(table.astype(int).join(summed))
    table[("EL change", "GW")] = table[("EL no UGHS", "GW")] - table[("EL PC-50", "GW")]
    table.columns.names = ["Quantity", "Unit"]
    table.index.name = "Country"
    return table.drop(columns=["PC PC-50", "PC no UGHS", "Expansion limit"], level=0)


# --- Supplementary Note 10: Wind power land use ---

CAPACITY_LIMIT_CARRIERS = {
    "onwind": "onshore wind",
    "offwind-dc": "offshore wind (DC)",
    "offwind-ac": "offshore wind (AC)",
    "solar": "solar",
    "photocatalysis": "photocatalysis",
}
HOFMANN_BASELINE = "hofmann_2025_baseline.nc"


def hofmann_baseline_path():
    """
    Path of the Hofmann et al.

    2025 baseline network, fails if it is missing.
    """
    for root in RESULT_ROOTS:
        path = os.path.join(root, HOFMANN_BASELINE)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"{HOFMANN_BASELINE} not found in {RESULT_ROOTS}.\n"
        "Download the baseline network of Hofmann et al. (2025), 'H2 and CO2 "
        "Network Strategies for the European Energy System', from figshare "
        "(https://doi.org/10.6084/m9.figshare.24923028), and save it as "
        f"data/raw/{HOFMANN_BASELINE}."
    )


def capacity_limit_row(n, details=False):
    """
    Node count and the number of sites at their expansion limit per carrier.

    `details=True` adds, per carrier, the sites with a finite limit, the
    share of them at the limit and the used share of the summed
    potential, plus whether electrolysis is unlimited everywhere (Note 7
    printout).
    """
    row = {"node count": len(n.buses[n.buses.carrier == "AC"])}
    for carrier, label in CAPACITY_LIMIT_CARRIERS.items():
        gens = n.generators[n.generators.carrier == carrier]
        # photocatalysis does not exist in the Hofmann baseline
        row[label] = int(at_site_limit(gens).sum()) if len(gens) else "not applicable"
        if details:
            limited = gens[np.isfinite(gens["p_nom_max"])]
            row[f"{label} - sites available"] = len(limited)
            row[f"{label} - share of sites at limit (%)"] = (
                100 * at_site_limit(limited).sum() / len(limited)
                if len(limited)
                else np.nan
            )
            row[f"{label} - potential utilization (%)"] = (
                100 * limited["p_nom_opt"].sum() / limited["p_nom_max"].sum()
                if len(limited)
                else np.nan
            )
    if details:
        el = n.links[n.links.carrier == "H2 Electrolysis"]
        row["H2 Electrolysis - p_nom_max is inf (all sites)"] = bool(
            np.all(np.isinf(el.p_nom_max))
        )
    return row


def capacity_limits(get_network, scenarios, hofmann_path):
    rows = {label: capacity_limit_row(get_network(s)) for label, s in scenarios.items()}
    rows["Hofmann 2025 baseline"] = capacity_limit_row(get_network(hofmann_path))
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Case"
    return table


# --- Supplementary Note 13: Nodes of the generation and demand violins ---


def gen_dem_node_counts(market_share_map):
    import hydrogen_gen_and_dem_violins as gdv

    counts = gdv.node_counts_per_violin(
        market_share_map=market_share_map, all_cost_points=True
    )
    return counts.drop(columns="Total nodes", errors="ignore")


# --- Supplementary Note 15: Fossil oil, natural gas and the emission limit ---

# printed row -> carriers of the co2 (atmosphere) balance
ATMOSPHERE_ROWS = {
    "Kerosene for aviation (fixed)": ["kerosene for aviation"],
    "Methanol for shipping (fixed)": ["shipping methanol"],
    "Naphtha for industry (fixed)": ["naphtha for industry"],
    "Fuel oil for shipping (fixed)": ["shipping oil"],
    "Methanol for industry (fixed)": ["industry methanol"],
    "Gas boilers (variable)": [
        "urban central gas boiler",
        "urban decentral gas boiler",
        "rural gas boiler",
    ],
    "Gas combined heat and power and open-cycle turbines (variable)": [
        "urban central gas CHP",
        "urban central gas CHP CC",
        "OCGT",
    ],
    "Industrial gas and process emissions (fixed)": [
        "gas for industry",
        "gas for industry CC",
        "process emissions",
        "process emissions CC",
        "SMR",
        "SMR CC",
    ],
    "Coal for industry (fixed)": ["coal for industry"],
    "Direct air capture (variable)": ["DAC"],
    "Biomass with carbon capture (fixed)": ["solid biomass for industry CC"],
    "Net emissions (limited)": ["co2"],
}
# printed row -> carriers of the co2 stored balance
POOL_ROWS = {
    "Biomass with carbon capture (fixed)": ["solid biomass for industry CC"],
    "Capture at industrial and process sources (variable)": [
        "process emissions CC",
        "gas for industry CC",
        "SMR CC",
        "urban central gas CHP CC",
    ],
    "Direct air capture (variable)": ["DAC"],
    "Fischer-Tropsch synthesis (variable)": ["Fischer-Tropsch"],
    "Permanent sequestration (limited)": ["co2 sequestered"],
    "Methanol synthesis (variable)": ["methanolisation"],
    "Methanation (Sabatier)": ["Sabatier"],
}


def co2_intensities(costs_path):
    """
    CO2 intensity of oil, gas and coal in t/MWh from a costs_2040.csv.
    """
    costs = pd.read_csv(costs_path, index_col=[0, 1])["value"]
    return {
        fuel: float(costs[(fuel, "CO2 intensity")]) for fuel in ["oil", "gas", "coal"]
    }


def carbon_flows(n, intensities):
    """
    The two CO2 balances per carrier and the fossil carbon, in Mt CO2/a.

    Fossil carbon comes from the generator volumes, since the fuel buses
    mix fossil and synthetic carbon.
    """
    weights = n.snapshot_weightings.generators
    fuels = {}
    for fuel, label in [
        ("oil", "fossil oil"),
        ("gas", "natural gas"),
        ("coal", "coal"),
    ]:
        gens = n.generators.index[n.generators.carrier == fuel]
        energy = n.generators_t.p[gens].mul(weights, axis=0).sum().sum() / 1e6
        fuels[label] = energy * intensities[fuel]

    atmosphere = energy_balance(n, "co2")
    # naphtha takes more carbon off the oil bus than it emits
    naphtha = abs(energy_balance(n, "oil").get("naphtha for industry", 0.0))
    fuels["carbon retained in products"] = naphtha * intensities[
        "oil"
    ] - atmosphere.get("naphtha for industry", 0.0)
    return pd.concat(
        {
            "atmosphere": atmosphere,
            "stored": energy_balance(n, "co2 stored"),
            "fuels": pd.Series(fuels),
        }
    )


def carbon_accounts(flows_pc0, flows_pc50):
    """
    CO2 exchanged with the atmosphere, and the captured CO2 and its use.
    """
    tables = []
    for account, rows in [("atmosphere", ATMOSPHERE_ROWS), ("stored", POOL_ROWS)]:
        pc0 = group_balance(flows_pc0[account], rows)
        pc50 = group_balance(flows_pc50[account], rows)
        tables.append(compare(pc0, pc50))
    balance, pool = tables

    # the net emission is the co2 store withdrawing from the atmosphere
    balance.loc["Net emissions (limited)"] *= -1
    pool = pool.loc[(pool[["PC-0", "PC-50"]].abs() > 0.05).any(axis=1)]
    pool.loc["Balance"] = pool.sum()
    return balance, pool


def heat_rows(prefix):
    """
    Printed rows of one heat bus -> carriers.
    """
    return {
        "Electrolysis waste heat": ["H2 Electrolysis"],
        "Heat pumps": [f"{prefix} air heat pump", f"{prefix} ground heat pump"],
        "Resistive heaters": [f"{prefix} resistive heater"],
        "Gas boilers": [f"{prefix} gas boiler"],
        "Gas combined heat and power, with and without capture": [
            f"{prefix} gas CHP",
            f"{prefix} gas CHP CC",
        ],
        "Fischer-Tropsch waste heat": ["Fischer-Tropsch"],
        "Methanol synthesis waste heat": ["methanolisation"],
        "Solar thermal": [f"{prefix} solar thermal"],
        "Thermal storage (net)": [
            f"{prefix} water tanks charger",
            f"{prefix} water tanks discharger",
        ],
        "Direct air capture": ["DAC"],
        "Low-temperature heat for industry": ["low-temperature heat for industry"],
        "Heat demand": [f"{prefix} heat"],
        "Hydrogen fuel cell and methanation": ["H2 Fuel Cell", "Sabatier"],
    }


HEAT_BUSES = {
    "Urban central": "urban central",
    "Urban decentral": "urban decentral",
    "Rural": "rural",
}


def heat_balances(n):
    """
    Grouped balance of the three heat buses, TWh/a.
    """
    return {
        label: group_balance(energy_balance(n, f"{prefix} heat"), heat_rows(prefix))
        for label, prefix in HEAT_BUSES.items()
    }


def district_heat(n_pc0, n_pc50):
    """
    Heat balances of the three heat buses and of all of them together.
    """
    pc0, pc50 = heat_balances(n_pc0), heat_balances(n_pc50)
    pc0["All heat buses"] = sum(pc0.values())
    pc50["All heat buses"] = sum(pc50.values())

    blocks = {}
    for bus in pc0:
        block = compare(pc0[bus], pc50[bus])
        # rows that do not exist on this bus
        block = block.loc[(block[["PC-0", "PC-50"]].abs() >= 0.05).any(axis=1)]
        if bus in ["Urban central", "Urban decentral"]:
            block = block.rename(index={"Heat pumps": "Air heat pumps"})
        block.loc["Sum"] = block.sum()
        blocks[bus] = block
    table = pd.concat(blocks)
    table.index.names = ["Heat bus", "Source or sink"]
    return table


SANKEY_PAD = 26  # px between nodes of one column
SANKEY_PANEL_HEIGHT = 1000
SANKEY_COMPRESSION = 0.62  # plotly keeps bands free for the two loops
SANKEY_CENTRE = 0.46


def sankey_node_scale(columns, pad_frac):
    """
    Panel fraction per Mt/a, set by the column that fills the panel first.
    """
    return min(
        (1.0 - pad_frac * (len(sizes) - 1)) / sum(sizes)
        for sizes in columns
        if sum(sizes) > 0
    )


def sankey_stack_column(sizes, scale, pad_frac, align="center"):
    """
    Centre of each node of one column, stacked with a fixed gap.
    """
    heights = [size * scale for size in sizes]
    total = sum(heights) + pad_frac * (len(sizes) - 1)
    start = {"bottom": 1.0 - total, "top": 0.0}.get(align, (1.0 - total) / 2.0)
    centres, cursor = [], start
    for height in heights:
        centres.append(cursor + height / 2.0)
        cursor += height + pad_frac
    return centres


def carbon_sankey_trace(flows, colors, pad_frac):
    """
    The carbon cycle of one case as a plotly Sankey trace.
    """
    import plotly.graph_objects as go

    def a(account, carrier):
        return abs(float(flows[account].get(carrier, 0.0)))

    kerosene = a("atmosphere", "kerosene for aviation")
    naphtha_burnt = a("atmosphere", "naphtha for industry")
    shipping_oil = a("atmosphere", "shipping oil")
    meoh_shipping = a("atmosphere", "shipping methanol")
    meoh_industry = a("atmosphere", "industry methanol")
    boilers = sum(
        a("atmosphere", c)
        for c in [
            "urban central gas boiler",
            "urban decentral gas boiler",
            "rural gas boiler",
        ]
    )
    gas_ind_air = a("atmosphere", "gas for industry CC")
    gas_ind_pool = a("stored", "gas for industry CC")
    chp_air = sum(
        a("atmosphere", c)
        for c in ["urban central gas CHP", "urban central gas CHP CC", "OCGT"]
    )
    chp_pool = a("stored", "urban central gas CHP CC")
    proc_air = a("atmosphere", "process emissions CC")
    proc_pool = a("stored", "process emissions CC")
    coal_burnt = a("atmosphere", "coal for industry")
    dac = a("atmosphere", "DAC")
    biomass = a("atmosphere", "solid biomass for industry CC")
    net = a("atmosphere", "co2")
    ft = a("stored", "Fischer-Tropsch")
    meoh = a("stored", "methanolisation")
    seq = a("stored", "co2 sequestered")
    fossil_oil = a("fuels", "fossil oil")
    natural_gas = a("fuels", "natural gas")
    coal = a("fuels", "coal")
    retained = a("fuels", "carbon retained in products")

    naphtha = naphtha_burnt + retained
    gas_ind = gas_ind_air + gas_ind_pool
    chp = chp_air + chp_pool
    process = proc_air + proc_pool
    atmosphere_total = (
        kerosene
        + naphtha_burnt
        + shipping_oil
        + meoh_shipping
        + meoh_industry
        + boilers
        + gas_ind_air
        + chp_air
        + coal_burnt
        + proc_air
    )
    pool_total = dac + biomass + gas_ind_pool + chp_pool + proc_pool

    nodes, node_color = [], []
    links = {"source": [], "target": [], "value": [], "color": []}

    def add_node(label, value, color):
        nodes.append(f"{label}<br>{value:.1f}")
        node_color.append(color)
        return len(nodes) - 1

    def add_link(source, target, value, color, alpha=0.45):
        h = color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        links["source"].append(source)
        links["target"].append(target)
        links["value"].append(value)
        links["color"].append(f"rgba({r},{g},{b},{alpha})")

    fossil = colors["fossil source"]
    proc_c = colors["process"]
    fuel_c = colors["combustion"]
    synth = colors["synthesis"]

    oil_src = add_node("Fossil oil", fossil_oil, fossil)
    gas_src = add_node("Natural gas", natural_gas, fossil)
    coal_src = add_node("Coal", coal, fossil)

    oil_b = add_node("Oil bus", fossil_oil + ft, fuel_c)
    meoh_b = add_node("Methanol bus", meoh, fuel_c)
    gas_b = add_node("Gas bus", natural_gas, fossil)
    coal_b = add_node("Coal bus", coal, fossil)
    proc_b = add_node("Process emissions", process, proc_c)

    n_kero = add_node("Kerosene (aviation)", kerosene, fuel_c)
    n_naph = add_node("Naphtha (industry)", naphtha, fuel_c)
    n_ship = add_node("Fuel oil (shipping)", shipping_oil, fuel_c)
    n_mesh = add_node("Methanol (shipping)", meoh_shipping, fuel_c)
    n_mein = add_node("Methanol (industry)", meoh_industry, fuel_c)
    n_boil = add_node("Gas boilers", boilers, fossil)
    n_gind = add_node("Gas for industry", gas_ind, fossil)
    n_chp = add_node("Gas CHP and OCGT", chp, fossil)
    n_coal = add_node("Coal for industry", coal_burnt, fossil)

    atmos = add_node("Atmosphere", atmosphere_total, colors["atmosphere"])
    n_prod = add_node("Carbon retained<br>in products", retained, colors["products"])

    n_dac = add_node("Direct air capture", dac, colors["direct air capture"])
    n_bio = add_node("Biomass with<br>carbon capture", biomass, colors["biomass"])
    n_net = add_node("Net emissions", net, colors["net emissions"])

    n_pool = add_node("Captured CO₂ pool", pool_total, colors["pool"])

    n_ft = add_node("Fischer-Tropsch<br>synthesis", ft, synth)
    n_seq = add_node("Permanent<br>sequestration", seq, colors["sequestration"])
    n_meoh = add_node("Methanol<br>synthesis", meoh, synth)

    add_link(oil_src, oil_b, fossil_oil, fossil)
    add_link(gas_src, gas_b, natural_gas, fossil)
    add_link(coal_src, coal_b, coal, fossil)

    add_link(oil_b, n_kero, kerosene, fuel_c)
    add_link(oil_b, n_naph, naphtha, fuel_c)
    add_link(oil_b, n_ship, shipping_oil, fuel_c)
    add_link(meoh_b, n_mesh, meoh_shipping, fuel_c)
    add_link(meoh_b, n_mein, meoh_industry, fuel_c)
    add_link(gas_b, n_boil, boilers, fossil)
    add_link(gas_b, n_gind, gas_ind, fossil)
    add_link(coal_b, n_coal, coal, fossil)

    for source, value in [
        (n_kero, kerosene),
        (n_naph, naphtha_burnt),
        (n_ship, shipping_oil),
        (n_mesh, meoh_shipping),
        (n_mein, meoh_industry),
    ]:
        add_link(source, atmos, value, fuel_c)
    add_link(n_naph, n_prod, retained, colors["products"])
    add_link(n_boil, atmos, boilers, fossil)
    add_link(n_gind, atmos, gas_ind_air, fossil)
    add_link(n_gind, n_pool, gas_ind_pool, fossil)
    add_link(n_chp, atmos, chp_air, fossil)
    add_link(n_chp, n_pool, chp_pool, fossil)
    add_link(n_coal, atmos, coal_burnt, fossil)
    add_link(proc_b, atmos, proc_air, proc_c)
    add_link(proc_b, n_pool, proc_pool, proc_c)

    add_link(atmos, n_dac, dac, colors["atmosphere"])
    add_link(atmos, n_bio, biomass, colors["atmosphere"])
    add_link(atmos, n_net, net, colors["net emissions"])
    add_link(n_dac, n_pool, dac, colors["direct air capture"])
    add_link(n_bio, n_pool, biomass, colors["biomass"])

    add_link(n_pool, n_ft, ft, colors["pool"])
    add_link(n_pool, n_seq, seq, colors["pool"])
    add_link(n_pool, n_meoh, meoh, colors["pool"])

    # the two links that close the cycle
    add_link(n_ft, oil_b, ft, synth, alpha=0.62)
    add_link(n_meoh, meoh_b, meoh, synth, alpha=0.62)

    # Every node has to balance
    checks = {
        "oil bus": (fossil_oil + ft, kerosene + naphtha + shipping_oil),
        "methanol bus": (meoh, meoh_shipping + meoh_industry),
        "gas bus": (natural_gas, boilers + gas_ind + chp),
        "coal bus": (coal, coal_burnt),
        "atmosphere": (atmosphere_total, dac + biomass + net),
        "captured pool": (pool_total, ft + seq + meoh),
    }
    for what, (left, right) in checks.items():
        if abs(left - right) > 0.05:
            raise RuntimeError(f"{what} does not close: {left:.3f} against {right:.3f}")

    # (nodes, values, x, alignment, vertical offset)
    columns = [
        (
            [oil_src, gas_src, coal_src],
            [fossil_oil, natural_gas, coal],
            0.02,
            "center",
            0.0,
        ),
        (
            [oil_b, meoh_b, gas_b, coal_b, proc_b],
            [fossil_oil + ft, meoh, natural_gas, coal, process],
            0.17,
            "center",
            -0.03,
        ),
        (
            [n_kero, n_naph, n_ship, n_mesh, n_mein, n_boil, n_gind, n_chp, n_coal],
            [
                kerosene,
                naphtha,
                shipping_oil,
                meoh_shipping,
                meoh_industry,
                boilers,
                gas_ind,
                chp,
                coal_burnt,
            ],
            0.35,
            "center",
            -0.04,
        ),
        ([atmos], [atmosphere_total], 0.52, "center", 0.0),
        ([n_prod], [retained], 0.52, "center", -0.24),
        ([n_dac, n_bio, n_net], [dac, biomass, net], 0.64, "top", 0.0),
        ([n_pool], [pool_total], 0.79, "center", 0.0),
        # short of the right edge, the loops run round the outside
        ([n_ft, n_seq, n_meoh], [ft, seq, meoh], 0.96, "center", 0.0),
    ]
    node_x, node_y = [0.0] * len(nodes), [0.5] * len(nodes)
    node_offset = [0.0] * len(nodes)
    scale = sankey_node_scale([sizes for _, sizes, _, _, _ in columns], pad_frac)
    for indices, sizes, x, align, offset in columns:
        for index, y in zip(
            indices, sankey_stack_column(sizes, scale, pad_frac, align)
        ):
            node_x[index] = x
            node_y[index] = y
            node_offset[index] = offset

    # Squeeze into the middle band, nudge two nodes out of the way of ribbons
    nudge = {n_net: -0.06, proc_b: 0.08}
    node_y = [
        SANKEY_CENTRE + (y - 0.5) * SANKEY_COMPRESSION + offset + nudge.get(index, 0.0)
        for index, (y, offset) in enumerate(zip(node_y, node_offset))
    ]

    return go.Sankey(
        arrangement="fixed",
        node=dict(
            label=nodes,
            x=node_x,
            y=node_y,
            color=node_color,
            pad=SANKEY_PAD,
            thickness=16,
            line=dict(color="rgba(0,0,0,0.35)", width=0.6),
        ),
        link=dict(**links),
        valueformat=".1f",
        valuesuffix=" Mt/a",
    )


def normalize_multiline_labels(path):
    """
    Replace the relative dy of plotly's multi-line labels by absolute y.

    Word ignores dy after a reset y and stacks all lines on the first.
    """
    with open(path, encoding="utf-8") as stream:
        svg = stream.read()

    def fix_text(match):
        block = match.group(0)
        size = re.search(r"font-size:\s*([\d.]+)px", block)
        font_size = float(size.group(1)) if size else 12.0

        def fix_tspan(tspan):
            attrs = re.sub(r'\sy="[-\d.]+"', "", tspan.group(1) + tspan.group(3))
            return f'<tspan{attrs} y="{float(tspan.group(2)) * font_size:.4g}">'

        return re.sub(r'<tspan([^>]*?)dy="([-\d.]+)em"([^>]*?)>', fix_tspan, block)

    svg = re.sub(r"<text\b.*?</text>", fix_text, svg, flags=re.S)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(svg)


def svg_to_pdf(svg_path, pdf_path):
    """
    Print an SVG file to a one-page PDF of the same size, in Chrome.

    The way Kaleido exports plotly figures to PDF (an <img> on a page whose
    @page size is the figure, then Page.printToPDF), applied to the finished
    SVG, so the PDF has the cropped canvas of crop_to_content(). Runs in a
    thread of its own, so it also works where an event loop is running
    (notebooks).
    """
    import asyncio
    import base64
    import concurrent.futures
    import pathlib
    import tempfile

    import choreographer

    with open(svg_path, encoding="utf-8") as stream:
        svg = stream.read()
    header = re.match(r"<svg\b[^>]*>", svg).group(0)
    width = float(re.search(r'\swidth="([\d.]+)"', header).group(1))
    height = float(re.search(r'\sheight="([\d.]+)"', header).group(1))
    source = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    html = (
        "<!doctype html><html><head><style>"
        f"@page {{ size: {width}px {height}px; margin: 0 }}"
        "html, body { margin: 0; padding: 0; background: white }"
        f"img {{ display: block; width: {width}px; height: {height}px }}"
        "</style></head><body>"
        f'<img src="data:image/svg+xml;base64,{source}">'
        "</body></html>"
    )

    async def print_page(url):
        async with choreographer.Browser(headless=True) as browser:
            tab = await browser.create_tab()
            loaded = tab.subscribe_once("Page.loadEventFired")
            await tab.send_command("Page.enable")
            await tab.send_command("Page.navigate", params={"url": url})
            await loaded
            response = await tab.send_command(
                "Page.printToPDF",
                params={
                    "printBackground": True,
                    "preferCSSPageSize": True,
                    "marginTop": 0,
                    "marginBottom": 0,
                    "marginLeft": 0,
                    "marginRight": 0,
                    "pageRanges": "1",
                },
            )
            if "error" in response:
                raise RuntimeError(f"Page.printToPDF failed: {response['error']}")
            return base64.b64decode(response["result"]["data"])

    with tempfile.TemporaryDirectory() as folder:
        page = pathlib.Path(folder) / "page.html"
        page.write_text(html, encoding="utf-8")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            pdf = executor.submit(asyncio.run, print_page(page.as_uri())).result()
    with open(pdf_path, "wb") as stream:
        stream.write(pdf)


def crop_to_content(fig, path, padding=8):
    """
    Cut the SVG canvas down to the drawn area, measured on a PNG render.
    """
    from PIL import Image, ImageChops

    image = Image.open(io.BytesIO(fig.to_image(format="png"))).convert("RGB")
    left, top, right, bottom = ImageChops.invert(image).getbbox()
    left, top = max(left - padding, 0), max(top - padding, 0)
    right = min(right + padding, image.width)
    bottom = min(bottom + padding, image.height)
    width, height = right - left, bottom - top

    with open(path, encoding="utf-8") as stream:
        svg = stream.read()
    opening = re.match(r"<svg\b[^>]*>", svg)
    header = opening.group(0)
    for attribute, value in [
        ("width", width),
        ("height", height),
        ("viewBox", f"0 0 {width} {height}"),
    ]:
        header = re.sub(
            rf'\s{attribute}="[^"]*"', f' {attribute}="{value}"', header, count=1
        )
    body = svg[opening.end() : svg.rindex("</svg>")]
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(
            f'{header}<g transform="translate({-left},{-top})">{body}</g></svg>'
        )


def carbon_sankey(flows, output_folder, save, name="carbon_flow_sankey"):
    """
    Carbon flows of one case as a Sankey, written as `name`.svg, or in the
    formats of PC_FIGURE_FORMATS (cases.figure_formats): "svg" and "pdf".

    The SVG is edited for Word and cropped; the PDF is printed from it.
    """
    import plotly.graph_objects as go

    with open(os.path.join(REPO_ROOT, "src", "colors.yaml")) as stream:
        colors = yaml.safe_load(stream)["carbon_sankey"]

    # PAD is in pixels, the layout in panel fractions
    margin = dict(l=30, r=80, t=34, b=12)
    figure_height = 80 + SANKEY_PANEL_HEIGHT
    panel_px = (1.0 - 0.02 - 0.052) * (figure_height - margin["t"] - margin["b"])
    trace = carbon_sankey_trace(flows, colors, SANKEY_PAD / panel_px)
    trace.domain = dict(x=[0.0, 1.0], y=[0.052, 0.98])

    fig = go.Figure(trace)
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color="#1a1a1a"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=margin,
        width=1150,
        height=figure_height,
    )
    if not save:
        fig.show()
        return
    formats = figure_formats(["svg"])
    unsupported = set(formats) - {"svg", "pdf"}
    if unsupported:
        raise ValueError(f"carbon_sankey writes svg and pdf, not {unsupported}")
    path = os.path.join(output_folder, f"{name}.svg")
    fig.write_image(path)
    normalize_multiline_labels(path)
    crop_to_content(fig, path)
    if "pdf" in formats:
        svg_to_pdf(path, os.path.join(output_folder, f"{name}.pdf"))
    if "svg" not in formats:
        os.remove(path)
