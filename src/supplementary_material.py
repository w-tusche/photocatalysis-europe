# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Wolfram Tuschewitzki
#
# SPDX-License-Identifier: MIT
# %% [markdown]
# # Supplementary Material: all figures and tables
#
# One section per Supplementary Note, in the order of the annex. Figures go to
# img/ (FIGURE_FOLDER in cases.py), tables are printed. Runs as a script or
# cell by cell (`# %%`).
#
#     python src/supplementary_material.py
#
# Run from the repository root. Needs the solved networks in data/raw/ (more
# folders via EXTRA_RAW_FOLDERS in cases.py). run_pipeline.py runs this script
# as its last stage.

# %%
import os
import subprocess
import sys

import geopandas as gpd
import matplotlib as mpl
import pandas as pd
import yaml

sys.path.insert(0, "src")

import supplementary_analysis as sa
import system_cost_breakdown
from cases import (
    FIGURE_FOLDER,
    PC0,
    PC0_LABEL,
    PC50,
    PC50_LABEL,
    SWEEP,
    balances_python,
    find_costs,
    variation_of,
)
from create_basic_summary import (
    VARIATION_PATTERNS,
    get_base_index,
    load_df_without_unit_row,
)
from plots import (
    COLORS_YAML,
    FIGURE_FONT_SCALE,
    ccs_market_share_scatter,
    get_pgf_with_latex,
    hydrogen_generation_and_demand_violins,
    market_share,
    photocatalysis_market_share_mapping_STH_10_UGHS,
    plot_electrolysis_only_vs_balanced_mix_map,
    plot_hydrogen_salt_cavern_storage_potential,
    plot_LCOH_violins_of_electrolysis_and_photocatalysis,
    plot_system_cost_differences_bars,
    system_development,
)
from supplementary_analysis import get_network

# %% [markdown]
# ## Setup
#
# `SAVE = False` shows the figures instead of writing them.

# %%
SAVE = True
EXPORT_FORMAT = "svg"
OUTPUT_FOLDER = FIGURE_FOLDER
# the energy balance figures need a second environment (PyPSA 1.0, see
# balances_python() in cases.py); set PC_SKIP_BALANCE_PAIRS to leave them out
RUN_BALANCE_PAIRS = "PC_SKIP_BALANCE_PAIRS" not in os.environ

if SAVE:
    mpl.use("Agg")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

total_summary_df = load_df_without_unit_row("results/total_summary.csv")
total_eval_df = pd.read_csv("results/total_eval.csv", index_col=0)

with open(COLORS_YAML) as stream:
    _all_colors = yaml.safe_load(stream)
tech_colors = _all_colors["tech_colors"]
colors_dict = _all_colors["others"]

regions = gpd.read_file("data/regions_onshore_elec_s_150.geojson").set_index("name")

# keyed by the variation names of get_base_index(), in its order
INDICES = dict(
    zip(["base", *VARIATION_PATTERNS], get_base_index(total_summary_df), strict=True)
)
BASE_INDEX = INDICES["base"]

MARKET_SHARE_MAP = photocatalysis_market_share_mapping_STH_10_UGHS

# %% [markdown]
# ## Supplementary Note 1: Market shares and parameter variation

# %%
mpl.rcParams.update(get_pgf_with_latex(scale_factor=FIGURE_FONT_SCALE))
market_share(
    total_summary_df,
    OUTPUT_FOLDER,
    colors_dict["UGHS_line"],
    colors_dict["no_UGHS_line"],
    save=SAVE,
    all=True,
    indices=tuple(INDICES[key] for key in ("base", "noUGHS", "pct6", "pct3")),
    export_format=EXPORT_FORMAT,
    name="market_share_over_annualized_PC_cost",
)

# %% [markdown]
# ## Supplementary Note 2: PyPSA and atlite model extensions
#
# No figure or table.

# %% [markdown]
# ## Supplementary Note 3: LCOH of electrolysis and photocatalysis

# %%
lcoh_medians = sa.lcoh_medians(total_summary_df, SWEEP)
print(lcoh_medians.round(2).to_string())

lcoh_decomposition = sa.lcoh_decomposition(get_network(PC50))
print(lcoh_decomposition.round(2).to_string())

# seasonal part of the market value gap
market_value_seasons, h2_bus_price_halves = sa.market_value_seasons(get_network(PC50))
print(market_value_seasons.round(3).to_string())
print(h2_bus_price_halves.round(3).to_string())

# the decomposition per cost point, with the photocatalysis nodes at their
# area limit
lcoh_decomposition_sweep = sa.lcoh_decomposition_sweep(
    get_network, SWEEP, total_summary_df
)
print(lcoh_decomposition_sweep.round(3).to_string())

# hydrogen storage and pipeline cost against the median LCOH difference
h2_infrastructure = sa.h2_infrastructure_against_gap(total_summary_df, PC0, PC50)
print(h2_infrastructure.round(3).to_string())

# %%
plot_LCOH_violins_of_electrolysis_and_photocatalysis(
    total_summary_df,
    OUTPUT_FOLDER,
    tech_colors,
    colors_dict["UGHS_line"],
    represent_regarding_market_share=True,
    line_cost_or_market_share=False,
    beeswarm=True,
    save=SAVE,
    export_format=EXPORT_FORMAT,
    name="LCOH_beeswarm",
)

# %%
literature_cost_conversion = sa.literature_cost_conversion()
print(literature_cost_conversion.round(4).to_string())

# %% [markdown]
# ## Supplementary Note 4: By-products and system services of electrolysis

# %%
# Electricity usage inside the electrolyzer: what the heat by-product is
# worth over the sweep, and the nodal LCOH medians with it credited
print(sa.electrolysis_heat_split(find_costs(PC50)).round(1).to_string())

electrolysis_heat_value = sa.electrolysis_heat_value(
    get_network, SWEEP, total_summary_df
)
print(electrolysis_heat_value.round(3).to_string())

heat_credited_medians = sa.heat_credited_medians(get_network, SWEEP, total_summary_df)
print(heat_credited_medians.round(3).to_string())

# %%
# The two re-optimizations without the electrolysis heat by-product
ELHEAT0_PAIRS = {
    PC0_LABEL: (PC0, variation_of(PC0, "elheat0")),
    PC50_LABEL: (PC50, variation_of(PC50, "elheat0")),
}
ELHEAT0_CASES = {
    case: scenario
    for label, pair in ELHEAT0_PAIRS.items()
    for case, scenario in zip([label, f"{label} (no el. heat)"], pair)
}

elheat0_heat_balance = sa.elheat0_heat_balance(get_network, ELHEAT0_PAIRS)
print(elheat0_heat_balance.round(1).to_string())

# demands and the largest change per heat bus
elheat0_heat_buses = sa.elheat0_heat_buses(get_network, ELHEAT0_PAIRS)
print(elheat0_heat_buses.round(2).to_string())

elheat0_aggregates, elheat0_differences = sa.elheat0_comparison(
    total_summary_df, get_network, ELHEAT0_CASES
)
print(elheat0_aggregates.T.round(2).to_string())
print(elheat0_differences.round(2).to_string())

elheat0_heat_replacement = sa.elheat0_heat_replacement(
    total_summary_df, get_network, ELHEAT0_PAIRS
)
print(elheat0_heat_replacement.round(3).to_string())

# %%
# District heating demand at the photocatalysis nodes (no heat output of
# photocatalysis is modeled)
photocatalysis_heat_sinks = sa.photocatalysis_heat_sinks(get_network(PC50))
print(photocatalysis_heat_sinks.round(1).T.to_string())

# %%
# Price electrolysis pays and the price percentiles at its
# nodes
el_hours_pc0 = sa.electrolysis_hours(get_network(PC0))
el_hours_pc50 = sa.electrolysis_hours(get_network(PC50))

el_fleet = sa.electrolysis_fleet(el_hours_pc0, el_hours_pc50)
print(el_fleet.round(2).to_string())

el_price_percentiles = sa.electrolysis_price_percentiles(el_hours_pc0, el_hours_pc50)
print(el_price_percentiles.round(1).to_string())

# %%
# Dark lulls: two reference events found on PC-0 and shared by both cases
dark_lull_networks = {PC0_LABEL: get_network(PC0), PC50_LABEL: get_network(PC50)}
dark_lull_events = sa.reference_events(dark_lull_networks[PC0_LABEL])
print(dark_lull_events.round(0).to_string())

supply_capacity = sa.electricity_supply_capacity(dark_lull_networks)
print(supply_capacity.round(1).to_string())

dark_lull_means, dark_lull_coverage, dark_lull_remainder = sa.dark_lull_coverage(
    dark_lull_networks, dark_lull_events
)
print(dark_lull_means.round(0).to_string())
print(dark_lull_coverage.round(1).to_string())
print(dark_lull_remainder.round(1).to_string())

# %% [markdown]
# ## Supplementary Note 5: Exogenous demand composition
#
# The demand split by energy carrier and the hydrogen share converted into
# liquid fuels

# %%
base_networks = {PC0_LABEL: get_network(PC0), PC50_LABEL: get_network(PC50)}
demands = sa.demand_table(base_networks)
print(demands.round(1).to_string())
print(sa.group_shares(demands).round(1).to_string())
print(sa.hydrogen_to_liquid_fuels(base_networks).round(1).to_string())
del base_networks

# %%
mpl.rcParams.update(get_pgf_with_latex(scale_factor=1.0))
exogenous_demands, base_total, with_transport_total = sa.exogenous_demands(
    get_network(PC0), get_network(variation_of(PC0, "transport"))
)
print(exogenous_demands.sum(axis=1).round(1).to_string())
print(f"base total {base_total:.1f}, with transport {with_transport_total:.1f} TWh/a")
sa.exogenous_demands_bar(
    exogenous_demands,
    base_total,
    with_transport_total,
    OUTPUT_FOLDER,
    save=SAVE,
    export_format=EXPORT_FORMAT,
)

# %% [markdown]
# ## Supplementary Note 6: Carbon sequestration

# %%
mpl.rcParams.update(get_pgf_with_latex(scale_factor=FIGURE_FONT_SCALE))

ccs_market_share_scatter(
    total_summary_df,
    OUTPUT_FOLDER,
    base_index=pd.Index([PC50, PC0]),
    ccs_indices={
        level: INDICES[f"CCS_{level}"]
        for level in ["50", "100", "300", "400", "500", "600"]
    },
    case_scenarios=(PC0, PC50),
    save=SAVE,
    export_format=EXPORT_FORMAT,
    name="CCS_scatter",
)

# %%
ccs_levels = {"200 (base)": PC50}
ccs_levels.update(
    {
        level: variation_of(PC50, f"CCS_{level}")
        for level in ["50", "100", "300", "400", "500", "600", "900", "1200"]
    }
)
ccs_smr_crossover = sa.ccs_smr_crossover(get_network, ccs_levels)
ccs_smr_crossover = ccs_smr_crossover.loc[
    ["50", "100", "200 (base)", "300", "400", "500", "600", "900", "1200"]
]
print(ccs_smr_crossover.round(1).to_string())

# %%
ccs_cost_differences = {
    "system_cost_difference_PC0_200Mt_vs_50Mt": (
        PC0,
        variation_of(PC0, "CCS_50"),
        ["'PC-0', CCS 50", "PC-0 (CCS 200)"],
    ),
    "system_cost_difference_PC50_200Mt_vs_50Mt": (
        PC50,
        variation_of(PC50, "CCS_50"),
        ["'PC-50', CCS 50", "PC-50 (CCS 200)"],
    ),
    "system_cost_difference_PC50_200Mt_vs_600Mt": (
        PC50,
        variation_of(PC50, "CCS_600"),
        ["'PC-50', CCS 600", "PC-50 (CCS 200)"],
    ),
}
for name, (base, variant, legend_labels) in ccs_cost_differences.items():
    plot_system_cost_differences_bars(
        get_network(base),
        get_network(variant),
        OUTPUT_FOLDER,
        scale=5,
        save=SAVE,
        export_format=EXPORT_FORMAT,
        name=name,
        legend_labels=legend_labels,
    )

# %% [markdown]
# ## Supplementary Note 7: Transport sector

# %%
TRANSPORT_CASES = {
    PC0_LABEL: PC0,
    f"{PC0_LABEL} (+transport)": variation_of(PC0, "transport"),
    PC50_LABEL: PC50,
    f"{PC50_LABEL} (+transport)": variation_of(PC50, "transport"),
}

transport_aggregates, transport_growth = sa.transport_aggregates(
    total_summary_df, TRANSPORT_CASES
)
print(transport_aggregates.round(1).to_string())
print("\nadded by the transport sector (+transport minus base):")
print(transport_growth.round(1).to_string())

# %%
mpl.rcParams.update(get_pgf_with_latex(scale_factor=FIGURE_FONT_SCALE))
transport_gen_split, _ = sa.h2_gen_split_and_lcoh_violins(
    total_summary_df,
    TRANSPORT_CASES,
    OUTPUT_FOLDER,
    tech_colors,
    colors_dict["UGHS_line"],
    save=SAVE,
    export_format=EXPORT_FORMAT,
    name="h2_gen_split_lcoh_violins_transport_vs_base",
)
print(transport_gen_split.round(1).to_string())

# %%
for name, case, label in [
    ("system_cost_difference_PC0_vs_PC0_transport", PC0, PC0_LABEL),
    ("system_cost_difference_PC50_vs_PC50_transport", PC50, PC50_LABEL),
]:
    plot_system_cost_differences_bars(
        get_network(case),
        get_network(variation_of(case, "transport")),
        OUTPUT_FOLDER,
        scale=5,
        save=SAVE,
        export_format=EXPORT_FORMAT,
        name=name,
        legend_labels=[f"{label} +transport", f"{label} (base)"],
    )

# %%
transport_regional_diffs = sa.transport_regional_diffs(
    total_summary_df, TRANSPORT_CASES
)
for key, diff in transport_regional_diffs.items():
    print(
        f"{key}: net EL {diff['EL'].sum():+.1f} GW, net PC {diff['PC'].sum():+.1f} GW"
    )

sa.regional_capacity_difference_maps(
    transport_regional_diffs,
    regions,
    OUTPUT_FOLDER,
    save=SAVE,
    export_format=EXPORT_FORMAT,
    name="regional_capacity_diff_transport_vs_base",
)

# %%
# nodes at their photocatalysis limit, one network at a time
transport_capacity_limits = pd.DataFrame.from_dict(
    {
        label: sa.capacity_limit_row(get_network(case), details=True)
        for label, case in TRANSPORT_CASES.items()
    },
    orient="index",
)
print(transport_capacity_limits.T.to_string())

# %% [markdown]
# ## Supplementary Note 8: Hydrogen transport capacity

# %%
h2_pipelines = sa.h2_pipelines(total_summary_df, PC0, PC50)
print(h2_pipelines.round(2).to_string())

retrofit_potential = sa.retrofit_potential(get_network(PC0), get_network(PC50))
print(retrofit_potential.round(1).to_string())

h2_transport_losses = sa.h2_transport_losses(get_network(PC0), get_network(PC50))
print(h2_transport_losses.round(2).to_string())

# %%
cross_border_h2 = sa.cross_border_h2_capacity(get_network(PC0), get_network(PC50))
print(cross_border_h2[cross_border_h2.Difference.abs() > 1].round(2).to_string())

h2_corridors = sa.h2_corridors({"PC-0": get_network(PC0), "PC-50": get_network(PC50)})
print(h2_corridors.round(1).to_string())

# %% [markdown]
# ## Supplementary Note 9: Underground hydrogen storage

# %%
mpl.rcParams.update(get_pgf_with_latex(scale_factor=1.0))
plot_hydrogen_salt_cavern_storage_potential(
    OUTPUT_FOLDER, scale=1, save=SAVE, export_format=EXPORT_FORMAT
)

# %%
ughs_siting = sa.ughs_siting(
    get_network(PC50), get_network(variation_of(PC50, "noUGHS"))
)
print(ughs_siting.round(1).to_string())

# %%
ughs_cases = sa.ughs_cases(total_summary_df, PC50, variation_of(PC50, "noUGHS"))
print(ughs_cases.round(2).to_string())

ughs_nodes = sa.ughs_nodes(
    get_network(PC50), get_network(variation_of(PC50, "noUGHS")), total_eval_df
)
ughs_countries = sa.ughs_countries(ughs_nodes)
active = ughs_countries[["EL PC-50", "EL no UGHS", "UGHS PC-50", "AGHS no UGHS"]]
print(ughs_countries[(active > 0.05).any(axis=1)].round(1).to_string())
built = ughs_nodes[["PC PC-50", "PC no UGHS"]]
print(ughs_nodes[(built > 1e-3).any(axis=1)].round(1).to_string())

# %% [markdown]
# ## Supplementary Note 10: Wind power land use

# %%
for name, case, label in [
    ("pc-0-wind_3_vs_wind_1_2", PC0, "PC-0"),
    ("pc-50-wind_3_vs_wind_1_2", PC50, "PC-50"),
]:
    plot_electrolysis_only_vs_balanced_mix_map(
        get_network(case),
        get_network(variation_of(case, "wind_1_2")),
        regions,
        tech_colors,
        OUTPUT_FOLDER,
        save=SAVE,
        export_format=EXPORT_FORMAT,
        titles=(
            f"{label} case; Wind 3.0 MW/km2 (base)",
            f"{label} case; Wind 1.2 MW/km2",
        ),
        name=name,
    )

# %%
capacity_limits = sa.capacity_limits(
    get_network,
    {
        "PC-0": PC0,
        "PC-50": PC50,
        "'PC-0' wind 1.2": variation_of(PC0, "wind_1_2"),
        "'PC-50' wind 1.2": variation_of(PC50, "wind_1_2"),
    },
    sa.hofmann_baseline_path(),
)
print(capacity_limits.to_string())

# %% [markdown]
# ## Supplementary Note 11: Energy balances
#
# Drawn by plot_balance_pairs.py in the balance environment (PyPSA 1.0).
# `extract` writes data/balance_pairs/ and runs only when that cache is missing.

# %%
if RUN_BALANCE_PAIRS:
    balance_pairs = balances_python()
    balance_pairs += [os.path.join("src", "plot_balance_pairs.py")]
    if not all(
        os.path.isdir(os.path.join("data", "balance_pairs", s)) for s in (PC0, PC50)
    ):
        subprocess.run(balance_pairs + ["extract"], check=True)
    subprocess.run(balance_pairs + ["plot", "--outdir", OUTPUT_FOLDER], check=True)

# %% [markdown]
# ## Supplementary Note 12: Energy system development

# %%
mpl.rcParams.update(get_pgf_with_latex(scale_factor=FIGURE_FONT_SCALE))
system_development(
    total_summary_df,
    OUTPUT_FOLDER,
    tech_colors,
    save=SAVE,
    index=BASE_INDEX,
    export_format=EXPORT_FORMAT,
    name="overall_system_development_over_annualized_PC_cost",
)

# %%
# opens the eight base networks only when the cost cache is missing
if not os.path.exists(system_cost_breakdown.CACHE_FILE):
    system_cost_breakdown.extract()
system_cost_breakdown.plot(OUTPUT_FOLDER)

total_system_cost = total_summary_df.loc[SWEEP, "total system cost"].astype(float) / 1e9
print("Total system cost (Bln EUR/a):")
print(total_system_cost.round(1).to_string())

# %% [markdown]
# ## Supplementary Note 13: Nodes of the generation and demand distributions

# %%
gen_dem_node_counts = sa.gen_dem_node_counts(MARKET_SHARE_MAP)
print(gen_dem_node_counts.to_string())

hydrogen_generation_and_demand_violins(
    OUTPUT_FOLDER,
    represent_regarding_market_share=True,
    line_cost_or_market_share=False,
    orientation="vertical",
    beeswarm=True,
    save=SAVE,
    export_format=EXPORT_FORMAT,
    name="gen_dem_violin_beeswarm",
)

# %% [markdown]
# ## Supplementary Note 14: Non-linearity in the market share curve
#
# No figure or table.

# %% [markdown]
# ## Supplementary Note 15: Fossil oil, natural gas and the emission limit

# %%
carbon_flows = {
    label: sa.carbon_flows(get_network(case), sa.co2_intensities(find_costs(case)))
    for label, case in [(PC0_LABEL, PC0), (PC50_LABEL, PC50)]
}
carbon_balance, carbon_pool = sa.carbon_accounts(
    carbon_flows[PC0_LABEL], carbon_flows[PC50_LABEL]
)
print(carbon_balance.round(1).to_string())
print(carbon_pool.round(1).to_string())

sa.carbon_sankey(carbon_flows[PC50_LABEL], OUTPUT_FOLDER, save=SAVE)

# %%
district_heat = sa.district_heat(get_network(PC0), get_network(PC50))
print(district_heat.round(1).to_string())
