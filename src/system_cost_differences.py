# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2025 Wolfram Tuschewitzki, Jelto Lange
#
# SPDX-License-Identifier: MIT
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from mpl_toolkits.axisartist.axislines import AxesZero

from cases import REPO_ROOT
from create_basic_summary import calculate_pipeline_TWkm, load_df_without_unit_row


def dim_opcost_invest(
    n, comp_type, dim_col, op_col, carrier, dim_scaling, dim_type, timestep
):
    """
    Calculate dimension, operational cost and investment cost of a technology
    (carrier)

    Args:
        n (pypsa.components.Network): network to search to buses and values in
        comp_type (str): type of component ["generators","links","stores"]
        dim_col (str): dim col to look at for each component e.g. ["p1","p","e"...]
        op_col (str): operation cost col to look at for each component e.g. ["p1","p",...]
        carrier (str): name of the carrier to look for e.g. ["H2 Electrolysis", ... find them with e.g. n.generators.carrier.unique()]
        dim_scaling (int): divisor for scaling of the dim
        dim_type (str): type of dimension that should be supplied [time dependent: "time_dependent", non time dependent: "capacity"]
        timestep (int): time step of PyPSA network calculation, could be calculated internally int(8760 / len(n.snapshots)) in hours

    Returns:
        tuple: tuple of floats (dimension, operational cost, investment cost)
    """

    assert comp_type in [
        "generators",
        "links",
        "stores",
    ], f'Check for typos, not known comp_type: {comp_type} should be in ["generators","links", "stores"] or extend logic/documentation'
    assert isinstance(timestep, int)

    if comp_type == "stores":
        # For stores there is no p_nom_opt but e_nom_opt
        p_or_e = "e_nom_opt"
    else:
        # for generators and links there is p_nom_opt
        # p_nom_opt: Optimized nominal power. [MW]
        p_or_e = "p_nom_opt"

    carrier_filter = getattr(n, comp_type)["carrier"] == carrier
    component_table = getattr(n, comp_type)
    component_table_t = getattr(n, comp_type + "_t")
    component_table_filtered = component_table[carrier_filter]
    # component_table_t_filtered = component_table_t[dim_col].loc[:, carrier_filter]

    if dim_type == "time_dependent":
        # Time dependent dimension like energy production
        dim = (
            component_table_t[dim_col].loc[:, carrier_filter].sum().sum()
            / dim_scaling
            * timestep
        )
    elif dim_type == "capacity":
        capacity_table = component_table_filtered
        if comp_type == "links":
            # Bidirectional links (DC, electricity distribution grid) have a
            # "-reversed" twin with the same p_nom and capital_cost 0
            capacity_table = capacity_table[
                ~capacity_table.index.str.endswith("-reversed")
            ]
        dim = capacity_table[p_or_e].sum() / dim_scaling
    else:
        raise ValueError(
            "dim_type should be energy or capacity depending on the fact if a time dependent or time independent value should be returned"
        )

    # Calculation sum of all operational costs
    op_cost = (
        (
            component_table_t[op_col].loc[:, carrier_filter].sum()
            * component_table_filtered.marginal_cost  # marginal_cost: Marginal cost of production of 1 MWh. (currency/MWh)
        ).sum()
        / 1e9
        * timestep
    )

    # Calculation of sum of all investment costs
    inv_cost = (
        component_table_filtered.capital_cost
        * component_table_filtered[
            p_or_e
        ]  # capital_cost Fixed period costs of extending p_nom by 1 MW, including periodized investment costs and periodic fixed O&M costs (e.g. annuitized investment costs). [currency/MW]
    ).sum() / 1e9

    return dim, op_cost, inv_cost


def calc_diffs(
    n1, n2, comp_type, dim_col, op_col, carrier, dim_scaling, dim_type, timestep
):
    """
    Calculate the difference between the two systems n1, n2!

    Args:
        n1 (pypsa.components.Network): base network
        n2 (pypsa.components.Network): network to form reference with
        comp_type (str): type of component ["generators","links","stores"]
        dim_col (str): dim col to look at for each component e.g. ["p1","p","e"...]
        op_col (str): operation cost col to look at for each component e.g. ["p1","p",...]
        carrier (str): name of the carrier to look for e.g. ["H2 Electrolysis", ... find them with e.g. n.generators.carrier.unique()]
        dim_scaling (int): divisor for scaling of the dim
        dim_type (str): type of dimension that should be supplied [time dependent: "time_dependent", non time dependent: "capacity"]
        timestep (int): time step of PyPSA network calculation, could be calculated internally int(8760 / len(n.snapshots)) in hours

    Returns:
        float: Difference in dimension, operational cost, investment cost
    """
    dim1, op1, inv1 = dim_opcost_invest(
        n1, comp_type, dim_col, op_col, carrier, dim_scaling, dim_type, timestep
    )
    dim2, op2, inv2 = dim_opcost_invest(
        n2, comp_type, dim_col, op_col, carrier, dim_scaling, dim_type, timestep
    )
    # abs required for carriers that are taken from an exit bus (e.g. electrolysis)
    return (
        np.round(abs(dim2) - abs(dim1), 2),
        np.round(abs(op2) - abs(op1), 2),
        np.round(abs(inv2) - abs(inv1), 2),
    )


def calc_energy_box_values(n_el, n_pc, verbose=False):
    timestep = int(8760 / len(n_el.snapshots))

    ######################################################################################
    # Energy carriers Box

    # Key aspects of PC-system in comparison to EL-only system:
    # Energy carriers:
    # - less "fossil oil" consumption in TWh/a and cost in Bln. €/a
    # - more "fossil gas" consumption in TWh/a and cost in Bln. €/a
    # - less "renewable electricity" in TWh/a and cost in Bln. €/a
    # - more "green hydrogen" in TWh/a and cost in Bln. €/a

    # Calculate differences in dimension, operational cost and investment cost for energy carriers
    df_energy_carriers_all = pd.DataFrame(columns=["Installation", "Opex", "Invest"])
    dim_col = op_col = "p"
    dim_scaling = 1e6
    dim_type = "time_dependent"
    for carrier in n_el.generators.carrier.unique():
        dim_diff, op_diff, inv_diff = calc_diffs(
            n_el,
            n_pc,
            "generators",
            dim_col,
            op_col,
            carrier,
            dim_scaling,
            dim_type,
            timestep,
        )
        df_energy_carriers_all.loc[carrier] = [dim_diff, op_diff, inv_diff]

        if verbose:
            print(
                f"PC-system shows difference in annual generation for {carrier}: {dim_diff} TWh/a"
            )
            print(
                f"Difference of annual operative cost for {carrier}: {op_diff} Bln. €"
            )
            print(
                f"Difference in annuities of investment in {carrier}: {inv_diff} Bln. €"
            )

    # Add electrolysis
    carrier = "H2 Electrolysis"
    dim_diff, op_diff, inv_diff = calc_diffs(
        n_el, n_pc, "links", "p1", "p0", carrier, dim_scaling, dim_type, timestep
    )
    df_energy_carriers_all.loc[carrier] = [dim_diff, op_diff, inv_diff]

    # Group some of the technologies
    carrier_energy_summary_map = {
        "gas": ["gas"],
        "oil": ["oil"],
        "solar": ["solar", "solar-hsat", "solar rooftop"],
        "wind onshore": ["onwind"],
        "wind offshore": ["offwind-ac", "offwind-float", "offwind-dc"],
        # "others": ["ror", "nuclear"],
        "photocatalysis": ["photocatalysis"],
        "electrolysis": ["H2 Electrolysis"],
    }

    df_energy_carriers = pd.DataFrame(columns=["Installation", "Opex", "Invest"])
    for new_name, components in carrier_energy_summary_map.items():
        # Sum the specified rows for each component group
        new_row = df_energy_carriers_all.loc[components, :].sum(axis=0)
        # Assign the new row to the DataFrame with the new name
        df_energy_carriers.loc[new_name] = new_row

    # Ordering the content
    df_energy_carriers = df_energy_carriers.loc[
        [
            "photocatalysis",
            "electrolysis",
            "solar",
            "wind onshore",
            "wind offshore",
            "oil",
            "gas",
        ],
        :,
    ]

    x_energy_carriers = np.arange(len(df_energy_carriers))

    # Energy carriers
    name_map_energy_carriers = {
        "gas": "Natural\ngas",
        "oil": "Oil",
        "solar": "Solar",
        "wind onshore": "Wind\nonshore",
        "wind offshore": "Wind\noffshore",
        "others": "others",
        "photocatalysis": "Photo-\ncatalysis",
        "electrolysis": "Electro-\nlysis",
    }

    return df_energy_carriers, x_energy_carriers, name_map_energy_carriers


def calc_syn_fuel_box_values(n_el, n_pc):
    timestep = int(8760 / len(n_el.snapshots))

    # Syn. Fuels:
    # - more "DAC" in Mt CO2/a and annualized cost in Bln. €/a
    # - more "Fischer-Tropsch" in TWh/a and annualized cost in Bln. €/a
    # - more "Methanolisation" in TWh/a and annualized cost in Bln. €/a

    syn_fuel_mapping = {
        "DAC": {
            "carrier": "DAC",
            "dim_col": "p3",
            "op_col": "p0",
        },  # [Mt-CO2, Bln. €/a] bus 3 is the local co2 bus; negative values equate to co2 going out of the link (into the bus)
        "MeOH": {
            "carrier": "methanolisation",
            "dim_col": "p1",
            "op_col": "p0",
        },  # [TWh/a, Bln. €/a] bus 1 is the local oil or  methanol bus; negative values equate to oil or methanol going out of the link (into the bus)
        "FT": {
            "carrier": "Fischer-Tropsch",
            "dim_col": "p1",
            "op_col": "p0",
        },  # [TWh/a, Bln. €/a] bus 1 is the local oil or  methanol bus; negative values equate to oil or methanol going out of the link (into the bus). marginal_cost is charged on p0 (the hydrogen input)
        "Sabatier": {
            "carrier": "Sabatier",
            "dim_col": "p1",
            "op_col": "p0",
        },  # [TWh/a, Bln. €/a] bus 1 is the EU gas bus; negative values equate to methane going out of the link (into the bus). bus 0 is the hydrogen input the marginal cost applies to
    }

    dim_scaling = 1e6
    dim_type = "time_dependent"

    df_syn_fuels = pd.DataFrame(columns=["Installation", "Opex", "Invest"])
    for technology, vals in syn_fuel_mapping.items():
        dim_diff, op_diff, inv_diff = calc_diffs(
            n_el,
            n_pc,
            "links",
            vals["dim_col"],
            vals["op_col"],
            vals["carrier"],
            dim_scaling,
            dim_type,
            timestep,
        )
        df_syn_fuels.loc[technology] = [dim_diff, op_diff, inv_diff]

    # Methanation is only built below 200 Mt CO2/a sequestration
    sabatier_built = (
        max(
            abs(
                dim_opcost_invest(
                    network,
                    "links",
                    "p1",
                    "p0",
                    "Sabatier",
                    dim_scaling,
                    dim_type,
                    timestep,
                )[0]
            )
            for network in (n_el, n_pc)
        )
        > 1  # TWh/a of methane, against ~0.0003 in a base case
    )

    # Ordering the content
    bar_order = ["FT", "MeOH", "DAC"]
    if sabatier_built:
        bar_order.insert(bar_order.index("DAC"), "Sabatier")
    df_syn_fuels = df_syn_fuels.loc[bar_order, :]

    x_syn_fuels = np.arange(len(df_syn_fuels))

    # Synfuels
    name_map_syn_fuels = {
        "DAC": "Direct air\ncapture",
        "MeOH": "Methanol\nsynthesis",
        "FT": "Fischer-\nTropsch",
        "Sabatier": "Sabatier",
    }

    return df_syn_fuels, x_syn_fuels, name_map_syn_fuels


# CHP capacity shown in the heating and power box
# ["heat capacity", "electricity capacity", "gas capacity"]
CHP_BOX_COLUMN = "heat capacity"


def calc_power_and_h2_system_box_values(n_el, n_pc):
    timestep = int(8760 / len(n_el.snapshots))

    # Heat system
    # - more gas-based technology (i.e., boilers) in GW and cost in Bln. €/a
    # - less electrification  (i.e., less resistive heating, fewer heat pumps) in GW and cost in Bln. €/a
    # - less storage capacities in TWh and cost in Bln. €/a

    # Get information for heating technologies [GW, Bln. €/a]
    carrs_heat = [
        "urban central air heat pump",
        "rural air heat pump",
        "rural ground heat pump",
        "urban decentral air heat pump",
        "urban central resistive heater",
        "rural resistive heater",
        "urban decentral resistive heater",
        "urban central gas boiler",
        "rural gas boiler",
        "urban decentral gas boiler",
        "urban central gas CHP",
        "urban central gas CHP CC",
        "rural water tanks charger",
        "rural water tanks discharger",
        "urban decentral water tanks charger",
        "urban decentral water tanks discharger",
        "urban central water tanks charger",
        "urban central water tanks discharger",
    ]

    df_heating_all = pd.DataFrame(columns=["Installation", "Opex", "Invest"])
    dim_col = ""
    op_col = "p0"
    dim_type = "capacity"
    dim_scaling = 1e3  # GW
    for carrier in carrs_heat:
        dim_diff, op_diff, inv_diff = calc_diffs(
            n_el,
            n_pc,
            "links",
            dim_col,
            op_col,
            carrier,
            dim_scaling,
            dim_type,
            timestep,
        )
        df_heating_all.loc[carrier] = [dim_diff, op_diff, inv_diff]

    # Get information for heat storages [TWh, Bln. €/a]
    carrs_heat_storage = [
        "urban central water tanks",
        "rural water tanks",
        "urban decentral water tanks",
    ]

    dim_col = ""
    op_col = "p"
    dim_type = "capacity"
    dim_scaling = 1e6  # TWh
    for carrier in carrs_heat_storage:
        dim_diff, op_diff, inv_diff = calc_diffs(
            n_el,
            n_pc,
            "stores",
            dim_col,
            op_col,
            carrier,
            dim_scaling,
            dim_type,
            timestep,
        )
        df_heating_all.loc[carrier] = [dim_diff, op_diff, inv_diff]

    # Group some of the technologies
    heat_carrier_summary_map = {
        "Heat pumps": [
            "urban central air heat pump",
            "rural air heat pump",
            "rural ground heat pump",
            "urban decentral air heat pump",
        ],
        "Resistive heaters": [
            "urban central resistive heater",
            "rural resistive heater",
            "urban decentral resistive heater",
        ],
        "Gas boilers": [
            "urban central gas boiler",
            "rural gas boiler",
            "urban decentral gas boiler",
        ],
        # "Storage piping": [
        #     "rural water tanks charger",
        #     "rural water tanks discharger",
        #     "urban decentral water tanks charger",
        #     "urban decentral water tanks discharger",
        #     "urban central water tanks charger",
        #     "urban central water tanks discharger",
        # ],
        "Thermal Storage": [
            "urban central water tanks",
            "rural water tanks",
            "urban decentral water tanks",
        ],
    }

    # Create dataframe with groups
    df_heating = pd.DataFrame(columns=["Installation", "Opex", "Invest"])
    for new_name, components in heat_carrier_summary_map.items():
        # Sum the specified rows for each component group
        new_row = df_heating_all.loc[components, :].sum(axis=0)
        # Assign the new row to the DataFrame with the new name
        df_heating.loc[new_name] = new_row

    # Power
    # - higher degree of decentralization,
    # - less battery storage GWh and cost in  Bln. €/a

    # GW, Bln. €/a
    carrs_power = [
        "electricity distribution grid",
        "home battery charger",
        "home battery discharger",
        "battery charger",
        "battery discharger",
        "DC",
    ]

    df_power = pd.DataFrame(columns=["Installation", "Opex", "Invest"])
    dim_col = ""
    op_col = "p0"
    dim_type = "capacity"
    dim_scaling = 1e3  # GW
    for carrier in carrs_power:
        dim_diff, op_diff, inv_diff = calc_diffs(
            n_el,
            n_pc,
            "links",
            dim_col,
            op_col,
            carrier,
            dim_scaling,
            dim_type,
            timestep,
        )
        df_power.loc[carrier] = [dim_diff, op_diff, inv_diff]

    # AC lines are limited to extend bei 25 %, both systems would like to extend it even further, thus they cap at the same value. With larger expansion enabled both systems would have more AC, possibly the el-only one more.
    # 's_nom_opt' # Optimised capacity for apparent power. (MVA)
    # capital_cost # Fixed period costs of extending s_nom by 1 MVA, including periodized investment costs and periodic fixed O&M costs (e.g. annuitized investment costs). (currency/MVA)
    # print(n.lines[n.lines.carrier == "AC"].s_nom_opt.sum() / 1e6)  # TVA
    # print(n_pc.lines[n_pc.lines.carrier == "AC"].s_nom_opt.sum() / 1e6)  # TVA

    # Electricity storage
    carrs_el_storage = ["home battery", "battery"]

    dim_col = ""
    op_col = "p"
    dim_type = "capacity"
    dim_scaling = 1e6  # TWh
    for carrier in carrs_el_storage:
        dim_diff, op_diff, inv_diff = calc_diffs(
            n_el,
            n_pc,
            "stores",
            dim_col,
            op_col,
            carrier,
            dim_scaling,
            dim_type,
            timestep,
        )
        df_power.loc[carrier] = [dim_diff, op_diff, inv_diff]

    # carr_co2_storage = ["co2", "co2 stored", "co2 sequestered"]
    # CO2 Storage was tested, no change here! Just using a bit of CO2 storage requires investment into CC which is expansive
    # Rest emissions allowed thus is a lot cheaper than some sequestered CO2

    # HOW IS THE CONNECTION HERE between df_power and power_system_map?
    power_system_map = {
        "Battery chargers": [
            "home battery charger",
            "home battery discharger",
            "battery charger",
            "battery discharger",
        ],
        "Electricity grid": [
            "electricity distribution grid",
            "DC",
        ],
        "Electricity storage": ["home battery", "battery"],
    }

    # Create dataframe with groups
    df_power_heating_and_grid = pd.DataFrame(columns=["Installation", "Opex", "Invest"])
    for new_name, components in power_system_map.items():
        # Sum the specified rows for each component group
        new_row = df_power.loc[components, :].sum(axis=0)
        # Assign the new row to the DataFrame with the new name
        df_power_heating_and_grid.loc[new_name] = new_row

    # Battery charger and discharger are one inverter: same p_nom, discharger cost 0
    df_power_heating_and_grid.loc["Battery chargers", "Installation"] = df_power.loc[
        ["home battery charger", "battery charger"], "Installation"
    ].sum()

    name_map_power = {
        "Gas CHP": "Gas\nCHP",
        "Battery chargers": "Battery\nchargers",
        "Electricity grid": "Electricity\ngrid",
        "Electricity storage": "Electricity\nstorage",
    }

    # Heat system
    name_map_heating = {
        "Heat pumps": "Heat\npumps",
        "Resistive heaters": "Resistive\nheaters",
        "Gas boilers": "Gas\nboilers",
        # "Gas CHP": "Gas CHP",
        # "Storage piping": "Storage\npiping",
        "Thermal Storage": "Thermal\nstorage",
    }

    # Gas CHP sits next to the gas boilers: same fuel, same heat output side
    # columns "Installation CC" / "Cost CC", drawn hatched on top of none CC
    df_chp, _ = calc_chp_box_values(n_el, n_pc)
    chp_base = df_chp.loc[CHP_BASE_CARRIER]
    chp_cc = df_chp.loc[CHP_CC_CARRIER]

    # Prepare new box power and heating system; the .loc also sets the bar order
    df_power_and_heating_system = pd.concat(
        [
            df_heating,
            df_power_heating_and_grid.loc[
                ["Electricity storage", "Battery chargers", "Electricity grid"]
            ],
        ],
        axis=0,
    )
    df_power_and_heating_system["Installation CC"] = 0.0
    df_power_and_heating_system["Cost CC"] = 0.0

    df_power_and_heating_system.loc["Gas CHP"] = {
        "Installation": chp_base[CHP_BOX_COLUMN],
        "Opex": chp_base["Opex"],
        "Invest": chp_base["Invest"],
        "Installation CC": chp_cc[CHP_BOX_COLUMN],
        "Cost CC": chp_cc["Cost"],
    }
    bar_order = [
        "Heat pumps",
        "Resistive heaters",
        "Gas boilers",
        "Gas CHP",
        "Thermal Storage",
        "Electricity storage",
        "Battery chargers",
        "Electricity grid",
    ]
    df_power_and_heating_system = df_power_and_heating_system.loc[bar_order]
    x_power_and_heating_system = np.arange(len(df_power_and_heating_system))
    power_and_heating_system_name_map = name_map_power | name_map_heating

    return (
        df_power_and_heating_system,
        x_power_and_heating_system,
        power_and_heating_system_name_map,
    )


def calc_hydrogen_box(n_el, n_pc):
    # Hydrogen system
    # - more H2-Storage in TWh and cost in Bln. €/a
    # - more H2-pipelines in TWkm and cost in Bln. €/a

    def calc_total_hydrogen_pipeline_costs(n):
        # Hydrogen pipelines [TWkm, Bln. €/a]
        # Pipeline operational costs seem to be not important
        # if you want to calculate them: only pipelines with p_max_pu >= 0 energy bus0 * marginal cost

        # Calculate total pipeline capacity in TWkm
        (
            total_new_h2_pipeline_TWkm,
            total_new_h2_pipeline_cost,
        ) = calculate_pipeline_TWkm(
            n,
            pipeline_type="H2 pipeline",
            output_spec_pipeline_cost=True,
        )

        (
            total_repurposed_h2_pipeline_TWkm,
            total_repurposed_h2_pipeline_cost,
        ) = calculate_pipeline_TWkm(
            n,
            pipeline_type="H2 pipeline retrofitted",
            output_spec_pipeline_cost=True,
        )

        total_h2_pipeline_TWkm = (
            total_new_h2_pipeline_TWkm + total_repurposed_h2_pipeline_TWkm
        )
        total_h2_pipeline_cost = (
            total_new_h2_pipeline_cost + total_repurposed_h2_pipeline_cost
        )
        return total_h2_pipeline_TWkm, total_h2_pipeline_cost

    timestep = int(8760 / len(n_el.snapshots))

    df_h2 = pd.DataFrame(columns=["Installation", "Opex", "Invest"])
    # Hydrogen storage [TWh, Bln. €/a]
    carrier = "H2"
    dim_col = ""
    op_col = "p"
    dim_scaling = 1e6  # TWh
    dim_type = "capacity"
    dim_diff, op_diff, inv_diff = calc_diffs(
        n_el,
        n_pc,
        "stores",
        dim_col,
        op_col,
        carrier,
        dim_scaling,
        dim_type,
        timestep,
    )

    df_h2.loc["Hydrogen storage"] = [dim_diff, op_diff, inv_diff]

    (
        base_total_h2_pipeline_TWkm,
        base_total_h2_pipeline_cost,
    ) = calc_total_hydrogen_pipeline_costs(n_el)
    (
        pc_total_h2_pipeline_TWkm,
        pc_total_h2_pipeline_cost,
    ) = calc_total_hydrogen_pipeline_costs(n_pc)

    df_h2.loc["Hydrogen pipeline"] = [
        np.round(pc_total_h2_pipeline_TWkm - base_total_h2_pipeline_TWkm, 2),
        0,
        np.round(
            pc_total_h2_pipeline_cost / 1e9 - base_total_h2_pipeline_cost / 1e9, 2
        ),
    ]

    hydrogen_system_name_map = {
        "Hydrogen storage": "Hydrogen\nstorage",
        "Hydrogen pipeline": "Hydrogen\npipeline",
    }

    # Prepare new box Hydrogen System
    df_hydrogen_system = df_h2.loc[["Hydrogen storage", "Hydrogen pipeline"]]
    x_hydrogen_system = np.arange(len(df_hydrogen_system))

    return df_hydrogen_system, x_hydrogen_system, hydrogen_system_name_map


CHP_CARRIERS = ("urban central gas CHP", "urban central gas CHP CC")

# Plain gas CHP and its carbon-capture variant are stacks
CHP_BASE_CARRIER = "urban central gas CHP"
CHP_CC_CARRIER = "urban central gas CHP CC"


def chp_flows(n, carrier):
    """
    Capacity and annual energy on all three sides of a CHP carrier.

    `p_nom_opt` is the gas input capacity, bus1 is electricity, bus2 heat.

    Args:
        n (pypsa.components.Network): network to read the links from
        carrier (str): CHP carrier, one of CHP_CARRIERS

    Returns:
        dict: capacities in GW (``gas capacity``, ``electricity capacity``,
        ``heat capacity``) and annual energy in TWh/a (``electricity
        generation``, ``heat generation``)
    """
    links = n.links[n.links.carrier == carrier]
    keys = (
        "gas capacity",
        "electricity capacity",
        "heat capacity",
        "electricity generation",
        "heat generation",
    )
    if links.empty:
        return dict.fromkeys(keys, 0.0)

    bus1_carriers = set(n.buses.carrier[links.bus1])
    bus2_carriers = set(n.buses.carrier[links.bus2])
    assert bus1_carriers == {
        "AC"
    }, f"{carrier}: expected bus1 to be electricity, found {bus1_carriers}"
    assert all(
        "heat" in bus_carrier for bus_carrier in bus2_carriers
    ), f"{carrier}: expected bus2 to be heat, found {bus2_carriers}"

    weights = n.snapshot_weightings.generators
    # p1/p2 are negative for an outflow of the link
    return {
        "gas capacity": links.p_nom_opt.sum() / 1e3,
        "electricity capacity": (links.p_nom_opt * links.efficiency).sum() / 1e3,
        "heat capacity": (links.p_nom_opt * links.efficiency2).sum() / 1e3,
        "electricity generation": (
            -n.links_t.p1[links.index].sum(axis=1) * weights
        ).sum()
        / 1e6,
        "heat generation": (-n.links_t.p2[links.index].sum(axis=1) * weights).sum()
        / 1e6,
    }


def calc_chp_box_values(n_el, n_pc):
    """
    Gas CHP panel (PC-system minus EL-only system).

    The cost is returned once per carrier and not split between heat and
    electricity.

    Args:
        n_el (pypsa.components.Network): EL-only (PC-0) network
        n_pc (pypsa.components.Network): PC-system (PC-50) network

    Returns:
        tuple: (dataframe indexed by CHP_CARRIERS, name map)
    """
    timestep = int(8760 / len(n_el.snapshots))

    rows = {}
    for carrier in CHP_CARRIERS:
        # cost differences on the gas input side, consistent with the boilers
        _, op_diff, inv_diff = calc_diffs(
            n_el,
            n_pc,
            "links",
            "",
            "p0",
            carrier,
            1e3,
            "capacity",
            timestep,
        )
        flows_el = chp_flows(n_el, carrier)
        flows_pc = chp_flows(n_pc, carrier)
        row = {key: np.round(flows_pc[key] - flows_el[key], 2) for key in flows_el}
        row["Opex"] = op_diff
        row["Invest"] = inv_diff
        row["Cost"] = np.round(op_diff + inv_diff, 2)
        rows[carrier] = row

    df_chp = pd.DataFrame(rows).T

    chp_name_map = {
        CHP_BASE_CARRIER: "Gas CHP",
        CHP_CC_CARRIER: "Gas CHP\nwith CC",
    }

    return df_chp, chp_name_map


HEAT_BUS_CARRIERS = ("urban central heat", "urban decentral heat", "rural heat")

# By-product heat: heat on a secondary output (bus2 and higher) of a link.
# Link carrier -> source, as named in the figure legend
BYPRODUCT_HEAT_SOURCES = {
    "H2 Electrolysis": "electrolysis",
    "Fischer-Tropsch": "fuel synthesis",
    "methanolisation": "fuel synthesis",
    "Sabatier": "fuel synthesis",
    "H2 Fuel Cell": "fuel cells",
}


def byproduct_heat_capacity(n):
    """
    Heat output capacity of the by-product heat per link carrier, in GW.

    `p_nom_opt` is on bus0, so the heat capacity is `p_nom_opt` times the
    efficiency of the heat output, the measure of the gas CHP bar. Gas CHP is
    left out: heat is one of its two products, and it has a bar of its own.
    Electrolyzers at nodes without an urban central heat bus deliver no heat
    and do not count.

    Raises for a carrier with heat on a secondary output that is neither gas
    CHP nor in BYPRODUCT_HEAT_SOURCES, so no source drops out unnoticed.
    """
    heat_bus_carriers = set(n.buses.carrier[n.buses.carrier.str.contains("heat")])
    unknown_buses = heat_bus_carriers - set(HEAT_BUS_CARRIERS)
    if unknown_buses:
        raise RuntimeError(
            f"heat bus carriers not in HEAT_BUS_CARRIERS: {unknown_buses}"
        )
    heat_buses = n.buses.index[n.buses.carrier.isin(HEAT_BUS_CARRIERS)]

    ports = sorted(
        int(column[3:])
        for column in n.links.columns
        if column.startswith("bus") and column[3:].isdigit() and int(column[3:]) >= 2
    )
    capacity = {}
    for port in ports:
        efficiency = f"efficiency{port}"
        links = n.links[n.links[f"bus{port}"].isin(heat_buses)]
        if links.empty:
            continue
        varying = n.links_t.get(efficiency, pd.DataFrame()).columns.intersection(
            links.index
        )
        if len(varying):
            raise RuntimeError(
                f"time-varying {efficiency} on heat outputs: {list(varying)}"
            )
        per_carrier = (links.p_nom_opt * links[efficiency]).groupby(
            links.carrier
        ).sum() / 1e3
        for carrier, value in per_carrier.items():
            capacity[carrier] = capacity.get(carrier, 0.0) + value

    capacity = pd.Series(capacity, dtype=float).drop(
        list(CHP_CARRIERS), errors="ignore"
    )
    unknown = capacity.index.difference(list(BYPRODUCT_HEAT_SOURCES))
    if len(unknown):
        raise RuntimeError(f"by-product heat carriers in no source: {list(unknown)}")
    return capacity


def calc_byproduct_heat_box_values(n_el, n_pc, noise=0.05):
    """
    By-product heat bar of the heating and power box (PC-system minus EL-only
    system), heat output capacity in GW.

    Several carriers of one source are netted, as in the other grouped bars.

    Returns:
        pandas.Series: difference per source in the order of
        BYPRODUCT_HEAT_SOURCES, sources below `noise` dropped
    """
    capacity = pd.DataFrame(
        {"el": byproduct_heat_capacity(n_el), "pc": byproduct_heat_capacity(n_pc)}
    ).fillna(0.0)
    difference = (capacity["pc"] - capacity["el"]).groupby(BYPRODUCT_HEAT_SOURCES).sum()
    order = list(dict.fromkeys(BYPRODUCT_HEAT_SOURCES.values()))
    difference = difference.reindex(order).dropna()
    return difference[difference.abs() >= noise].round(2)


def calculate_total_system_difference():
    summary_df = load_df_without_unit_row(
        os.path.join(REPO_ROOT, "results", "total_summary.csv")
    )

    balanced_mix_system_cost = summary_df.loc[
        "150_lv1.25_I_H_2045_PC_660Euro", "total system cost"
    ]
    el_100_system_cost = summary_df.loc[
        "150_lv1.25_I_H_2045_PC_950Euro", "total system cost"
    ]

    cost_reduction_in_percent = (
        (el_100_system_cost - balanced_mix_system_cost) / balanced_mix_system_cost * 100
    )

    return cost_reduction_in_percent


def main():
    print(
        f"Total system cost difference PC-0 vs PC-50: "
        f"{calculate_total_system_difference():.2f} %"
    )


if __name__ == "__main__":
    main()
