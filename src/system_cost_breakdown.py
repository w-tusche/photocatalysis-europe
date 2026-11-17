# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2025 Wolfram Tuschewitzki, Jelto Lange
#
# SPDX-License-Identifier: MIT
"""
Total annualized system cost of every base-sweep case, as stacked bars by
technology group (Supplementary Note 12).

Two stages:

    python src/system_cost_breakdown.py extract
    python src/system_cost_breakdown.py plot

`extract` writes capex and opex per component and carrier of every base case to
data/system_cost_breakdown/cost_by_carrier.csv, `plot` draws the figure from it
into img/ (FIGURE_FOLDER in cases.py).

The bars are capex + opex of all components, including fixed capacities (the
"total system cost" of total_summary.csv). The groups follow the boxes of the
system cost difference figure, plus "Nuclear", "Hydropower" and "Other".
Colors from colors.yaml, PyPSA-Eur defaults otherwise.
"""

import argparse
import os
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plots as plots_module
from cases import FIGURE_FOLDER, PC0, PC50, REPO_ROOT, figure_formats, find_network
from create_basic_summary import get_base_index, load_df_without_unit_row

# PyPSA-Eur's default colors, the fallback wherever colors.yaml has no entry
PYPSA_COLORS_YAML = os.path.join(REPO_ROOT, "src", "plotting.default.yaml")
CACHE_FILE = os.path.join(
    REPO_ROOT, "data", "system_cost_breakdown", "cost_by_carrier.csv"
)
SUMMARY_CSV = os.path.join(REPO_ROOT, "results", "total_summary.csv")
OUTPUT_FOLDER = FIGURE_FOLDER
FIG_NAME = "system_cost_breakdown"

# group -> (carriers, color source, color key); listed bottom to top of the stack
GROUPS = {
    "Nuclear": (["nuclear"], "pypsa", "nuclear"),
    "Hydropower": (["hydro", "ror", "PHS"], "pypsa", "hydro"),
    "Other": (
        [
            "coal",
            "solid biomass for industry CC",
            "process emissions CC",
            "gas for industry CC",
            "co2 sequestered",
            "OCGT",
            "urban central solar thermal",
            "urban decentral solar thermal",
            "rural solar thermal",
        ],
        "pypsa",
        "other",
    ),
    "Electricity grid": (
        ["electricity distribution grid", "AC", "DC"],
        "pypsa",
        "electricity distribution grid",
    ),
    "Electricity storage": (["battery", "home battery"], "pypsa", "battery"),
    "Battery chargers": (
        [
            "battery charger",
            "battery discharger",
            "home battery charger",
            "home battery discharger",
        ],
        "pypsa",
        "battery charger",
    ),
    "Heat pumps": (
        [
            "urban central air heat pump",
            "rural air heat pump",
            "rural ground heat pump",
            "urban decentral air heat pump",
        ],
        "pypsa",
        "heat pump",
    ),
    "Resistive heaters": (
        [
            "urban central resistive heater",
            "rural resistive heater",
            "urban decentral resistive heater",
        ],
        "pypsa",
        "resistive heater",
    ),
    "Gas boilers": (
        [
            "urban central gas boiler",
            "rural gas boiler",
            "urban decentral gas boiler",
        ],
        "pypsa",
        "gas boiler",
    ),
    "Gas CHP": (
        ["urban central gas CHP", "urban central gas CHP CC"],
        "pypsa",
        "CHP",
    ),
    "Thermal storage": (
        [
            "urban central water tanks",
            "rural water tanks",
            "urban decentral water tanks",
            "urban central water tanks charger",
            "urban central water tanks discharger",
            "rural water tanks charger",
            "rural water tanks discharger",
            "urban decentral water tanks charger",
            "urban decentral water tanks discharger",
        ],
        "pypsa",
        "water tanks",
    ),
    "Fischer-Tropsch": (["Fischer-Tropsch"], "pypsa", "Fischer-Tropsch"),
    "Methanol synthesis": (["methanolisation"], "pypsa", "methanolisation"),
    "Direct air capture": (["DAC"], "pypsa", "DAC"),
    "Natural gas (fuel)": (["gas"], "pypsa", "gas"),
    "Fossil oil (fuel)": (["oil"], "pypsa", "oil"),
    "Hydrogen storage and pipelines": (
        ["H2", "H2 pipeline", "H2 pipeline retrofitted"],
        "user",
        "hydrogen",
    ),
    "Wind onshore": (["onwind"], "user", "onwind"),
    "Wind offshore": (["offwind-ac", "offwind-dc", "offwind-float"], "user", "offwind"),
    "Solar": (["solar", "solar-hsat", "solar rooftop"], "user", "solar"),
    "Electrolysis": (["H2 Electrolysis"], "user", "H2 Electrolysis"),
    "Photocatalysis": (["photocatalysis"], "user", "photocatalysis"),
}

# max cost of carriers outside GROUPS that go into "Other" (Bln EUR/a)
UNMAPPED_TOLERANCE = 0.5

BAR_WIDTH = 0.7
# segments at least this large (Bln EUR/a) get a value label
SEGMENT_LABEL_MIN = 35
SEGMENT_LABEL_FONTSIZE = 6


def base_sweep(df):
    """
    The base-sweep scenarios, highest annualized photocatalysis cost first
    (PC-0 on the left, as in the other sweep figures of the note).
    """
    index = get_base_index(df)[0]
    return list(
        df.loc[index, "pc total annual cost"]
        .astype(float)
        .sort_values(ascending=False)
        .index
    )


def fixed_capacity_cost(n):
    """
    Annuities that n.objective does not contain: capacities that are not
    extendable at all, in full.
    """
    total = 0.0
    for component, attr in (
        ("generators", "p_nom"),
        ("links", "p_nom"),
        ("storage_units", "p_nom"),
        ("stores", "e_nom"),
        ("lines", "s_nom"),
    ):
        table = getattr(n, component)
        fixed = ~table[f"{attr}_extendable"]
        total += (table.loc[fixed, "capital_cost"] * table.loc[fixed, attr]).sum()
    return total


def extract():
    import pypsa

    df = load_df_without_unit_row(SUMMARY_CSV)
    rows = []
    for scenario in base_sweep(df):
        n = pypsa.Network(find_network(scenario))
        capex = n.statistics.capex(groupby=["carrier"], nice_names=False)
        opex = n.statistics.opex(groupby=["carrier"], nice_names=False)
        costs = pd.concat({"capex": capex, "opex": opex}, axis=1).fillna(0.0)
        costs = costs.reset_index()
        costs.columns = ["component", "carrier", "capex", "opex"]
        costs["scenario"] = scenario
        costs["objective"] = n.objective
        costs["objective_constant"] = n.objective_constant
        costs["fixed_capacity_cost"] = fixed_capacity_cost(n)
        rows.append(costs)

        total = costs[["capex", "opex"]].to_numpy().sum()
        # Ensure that the sum leaves only a minimal residual
        residual = (
            total - n.objective - n.objective_constant - fixed_capacity_cost(n)
        ) / 1e9
        print(
            f"{scenario}: capex + opex {total / 1e9:.2f}, objective "
            f"{n.objective / 1e9:.2f}, constant {n.objective_constant / 1e9:.2f}, "
            f"fixed capacities {fixed_capacity_cost(n) / 1e9:.2f} Bln EUR/a, "
            f"residual {residual:.4f}"
        )
        if abs(residual) > 0.01:
            raise RuntimeError(f"{scenario}: cost does not reconcile with objective")

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    pd.concat(rows).to_csv(CACHE_FILE, index=False)
    print(f"written {CACHE_FILE}")


def grouped_costs(cache):
    """
    Bln EUR/a per group (rows, in stack order) and scenario (columns).
    """
    cache = cache.assign(cost=(cache["capex"] + cache["opex"]) / 1e9)
    per_carrier = cache.pivot_table(
        index="carrier", columns="scenario", values="cost", aggfunc="sum"
    ).fillna(0.0)

    carrier_to_group = {
        carrier: group
        for group, (carriers, _, _) in GROUPS.items()
        for carrier in carriers
    }
    unmapped = per_carrier.loc[~per_carrier.index.isin(carrier_to_group)]
    unmapped = unmapped.loc[unmapped.abs().max(axis=1) > 1e-3]
    if not unmapped.empty:
        print("Carriers outside GROUPS, counted as 'Other' (Bln EUR/a):")
        print(unmapped.round(3).to_string())
        if unmapped.abs().to_numpy().max() > UNMAPPED_TOLERANCE:
            raise RuntimeError("an unmapped carrier is too large to hide in 'Other'")

    groups = per_carrier.index.map(
        lambda carrier: carrier_to_group.get(carrier, "Other")
    )
    return per_carrier.groupby(groups).sum().reindex(list(GROUPS)).fillna(0.0)


def group_colors():
    with open(plots_module.COLORS_YAML) as stream:
        user_colors = yaml.safe_load(stream)["tech_colors"]
    with open(PYPSA_COLORS_YAML) as stream:
        pypsa_colors = yaml.safe_load(stream)["plotting"]["tech_colors"]
    sources = {"user": user_colors, "pypsa": pypsa_colors}
    return {group: sources[source][key] for group, (_, source, key) in GROUPS.items()}


def plot(output_folder=OUTPUT_FOLDER):
    df = load_df_without_unit_row(SUMMARY_CSV)
    scenarios = base_sweep(df)
    cache = pd.read_csv(CACHE_FILE)
    missing = set(scenarios) - set(cache["scenario"])
    if missing:
        raise RuntimeError(f"not in {CACHE_FILE}, run extract first: {missing}")
    costs = grouped_costs(cache)[scenarios]
    colors = group_colors()

    annual_cost = df.loc[scenarios, "pc total annual cost"].astype(float)
    share = df.loc[scenarios, "pc market share"].astype(float)
    totals = costs.sum()
    print("Totals (Bln EUR/a):", totals.round(1).to_dict())

    mpl.rcParams.update(plots_module.get_pgf_with_latex(plots_module.FIGURE_FONT_SCALE))
    fig, ax = plt.subplots(figsize=plots_module.set_size(fraction=1, scale=1.15))

    x = np.arange(len(scenarios))
    bottom = np.zeros(len(scenarios))
    for group in GROUPS:
        values = costs.loc[group].to_numpy()
        ax.bar(
            x,
            values,
            BAR_WIDTH,
            bottom=bottom,
            color=colors[group],
            label=group,
            edgecolor="white",
            linewidth=0.2,
        )
        for xi, value, low in zip(x, values, bottom):
            if value >= SEGMENT_LABEL_MIN:
                ax.text(
                    xi,
                    low + value / 2,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=SEGMENT_LABEL_FONTSIZE,
                    color="black",
                )
        bottom += values

    for xi, total in zip(x, totals):
        ax.text(xi, total + 8, f"{total:.0f}", ha="center", va="bottom")
    for scenario, case in ((PC0, "PC-0"), (PC50, "PC-50")):
        xi = scenarios.index(scenario)
        ax.text(xi, totals[scenario] + 45, case, ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{cost:.1f}\n{s:.1f}" for cost, s in zip(annual_cost, share)])
    ax.set_xlabel(
        r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$ (top)"
        "\n"
        r"and photocatalysis market share in \% (bottom)"
    )
    ax.set_ylabel(r"Annualized system cost in Bln. € a$^{-1}$")
    ax.set_xlim(-0.6, len(scenarios) - 0.4)
    ax.set_ylim(0, np.ceil((totals.max() + 110) / 100) * 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=1.2,
    )

    os.makedirs(output_folder, exist_ok=True)
    for export_format in figure_formats(["svg", "png"]):
        path = os.path.join(output_folder, f"{FIG_NAME}.{export_format}")
        fig.savefig(path, bbox_inches="tight", dpi=400)
        print(f"written {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["extract", "plot"])
    args = parser.parse_args()
    {"extract": extract, "plot": plot}[args.stage]()


if __name__ == "__main__":
    main()
