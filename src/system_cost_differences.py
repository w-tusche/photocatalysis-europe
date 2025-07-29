# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pypsa
import yaml
from mpl_toolkits.axisartist.axislines import AxesZero

from create_basic_summary import calculate_pipeline_TWkm


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
        dim = component_table_filtered[p_or_e].sum() / dim_scaling
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
            "op_col": "p1",
        },  # [TWh/a, Bln. €/a] bus 1 is the local oil or  methanol bus; negative values equate to oil or methanol going out of the link (into the bus)
    }

    dim_scaling = 1e6
    dim_type = "time_dependent"

    df_syn_fuels = pd.DataFrame(columns=["Installation", "Opex", "Invest"])
    for techology, vals in syn_fuel_mapping.items():
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
        df_syn_fuels.loc[techology] = [dim_diff, op_diff, inv_diff]

    # Ordering the content
    df_syn_fuels = df_syn_fuels.loc[
        [
            "FT",
            "MeOH",
            "DAC",
        ],
        :,
    ]

    x_syn_fuels = np.arange(len(df_syn_fuels))

    # Synfuels
    name_map_syn_fuels = {
        "DAC": "Direct air\ncapture",
        "MeOH": "Methanol\nsynthesis",
        "FT": "Fischer-\nTropsch",
    }

    return df_syn_fuels, x_syn_fuels, name_map_syn_fuels


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
        # "Gas CHP": ["urban central gas CHP", "urban central gas CHP CC"],
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

    # FIXME: Join this with the heating!

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
    # Rest emissions allowed thus is a lot cheaper than som sequestered CO2

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

    name_map_power = {
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

    # Ordering the content
    df_power_heating_and_grid = df_power_heating_and_grid.loc[
        [
            "Electricity grid",
            "Electricity storage",
            "Battery chargers",
        ],
        :,
    ]

    # TODO: Fix this here!
    # Prepare new box power and heating system
    df_power_and_heating_system = pd.concat(
        [
            df_heating,
            df_power_heating_and_grid.loc[
                ["Battery chargers", "Electricity grid", "Electricity storage"]
            ],
        ],
        axis=0,
    )
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
        # if you want to calcualte them: only pipelines with p_max_pu >= 0 energy bus0 * marginal cost

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


def plot_system_cost_differences(n_el, n_pc):
    # Calculate all required data
    (
        df_energy_carriers,
        x_energy_carriers,
        name_map_energy_carriers,
    ) = calc_energy_box_values(n_el, n_pc)
    df_syn_fuels, x_syn_fuels, name_map_syn_fuels = calc_syn_fuel_box_values(n_el, n_pc)
    df_hydrogen_system, x_hydrogen_system, hydrogen_system_name_map = calc_hydrogen_box(
        n_el, n_pc
    )
    (
        df_power_and_heating_system,
        x_power_and_heating_system,
        power_and_heating_system_name_map,
    ) = calc_power_and_h2_system_box_values(n_el, n_pc)

    with open(r"./src/colors.yaml") as stream:
        colors_dict = yaml.safe_load(stream)["others"]

    rotation_deg = 0
    width = 0.35  # Width of the bars

    # Define the layout using a list of strings
    layout = [
        [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
        ],
        [
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
        ],
        [
            "B",
            "B",
            ".",
            "D",
            "D",
            "D",
            "E",
        ],
    ]

    def create_bar_plot(ax, x_values, df, name_map, title, y_label_left, ax1_ylim):
        ax.set_title(title)

        # Plot the Installation bars on ax
        ax.bar(
            x_values - width / 2,
            df["Installation"],
            width,
            label="Installation",
            color=colors_dict["generation installation"],
        )
        ax.set_ylabel(y_label_left, color=colors_dict["generation installation"])
        ax.tick_params(axis="y", labelcolor=colors_dict["generation installation"])

        # Create a secondary y-axis for the stacked bars
        ax2 = ax.twinx()
        # Stack Invest and Opex on top of each other
        ax2.bar(
            x_values + width / 2,
            df["Invest"],
            width,
            label="Invest",
            color=colors_dict["annualized costs"],
        )
        ax2.bar(
            x_values + width / 2,
            df["Opex"],
            width,
            label="Opex",
            color=colors_dict["annualized costs"],
            bottom=df["Invest"],
        )

        # Add labels and titles
        ax2.set_ylabel(
            "Difference in annualized\ncosts in Bln. € a$^{-1}$",
            color=colors_dict["annualized costs"],
        )
        ax2.tick_params(axis="y", labelcolor=colors_dict["annualized costs"])
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [name_map[key] for key in df.index],
            rotation=rotation_deg,
        )

        # Fix axis spread
        ax.set_ylim(ax1_ylim)
        ax.set_xlim([-0.5, len(df) - 0.5])
        ax2.set_ylim(ax1_ylim / 15)

        # Grid and tick customization
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True)
        ax2.tick_params(axis="both", direction="in", right=True)

        return ax2

    def setup_axes_zero(fig, position):
        """
        Creates and configures an AxesZero on the given figure at the specified
        position.
        """
        ax = fig.add_axes(position, axes_class=AxesZero)
        for direction in ["xzero", "yzero"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)

        for direction in ["left", "right", "bottom", "top"]:
            ax.axis[direction].set_visible(False)

        ax.set_ylim([-2.2, 5.2])
        ax.set_yticks([])
        ax.set_xticks([])
        return ax

    def plot_bars(ax):
        """
        Plots bar data on the given axis.
        """

        ax.bar(
            x=[0.2, 1.2, 2.2, 3.8, 4.8, 5.8],
            height=[5, 3, 2, 5, 7, 2],
            width=[0.9] * 6,
            bottom=[0, 2, 0, 0, -2, -2],
            align="edge",
            color=[
                colors_dict["pc-50"],
                colors_dict["pc-0"],
                colors_dict["difference"],
            ],
            label=["PC-50", "PC-0", "Difference", "", "", ""],
        )

    def plot_lines_and_arrows(ax):
        """
        Adds lines and arrows annotations to the given axis.
        """
        lw = 1.5
        lines = [
            ([0.2, 2.1], [5, 5]),
            ([1.2, 3.1], [2, 2]),
            ([3.8, 5.7], [5, 5]),
            ([4.8, 6.7], [-2, -2]),
        ]
        for x, y in lines:
            ax.plot(x, y, "k", linewidth=lw)

        arrows = {
            "bm-1": {"xy": (0.65, 0), "xyt": (0.65, 5), "ls": "--", "a": 0.5},
            "el-1": {"xy": (1.65, 5), "xyt": (1.65, 2), "ls": "--", "a": 0.5},
            "d-1": {"xy": (2.65, 0), "xyt": (2.65, 2), "ls": "-", "a": 1},
            "bm-2": {"xy": (4.25, 0), "xyt": (4.25, 5), "ls": "--", "a": 0.5},
            "el-2": {"xy": (5.25, 5), "xyt": (5.25, -2), "ls": "--", "a": 0.5},
            "d-2": {"xy": (6.25, 0), "xyt": (6.25, -2), "ls": "-", "a": 1},
        }
        for v in arrows.values():
            ax.annotate(
                "",
                xy=v["xy"],
                xytext=v["xyt"],
                color="k",
                arrowprops=dict(
                    arrowstyle="<-",
                    linestyle=v["ls"],
                    alpha=v["a"],
                    color="k",
                ),
            )

    # Create the mosaic plot
    fig, ax_dict = plt.subplot_mosaic(layout, figsize=(8, 10))

    create_bar_plot(
        ax=ax_dict["A"],
        x_values=x_energy_carriers,
        df=df_energy_carriers,
        name_map=name_map_energy_carriers,
        title="Electricity and hydrogen production",
        y_label_left="Difference in energy\ngeneration in TWh a$^{-1}$",
        ax1_ylim=np.array([-1000, 1300]),
    )

    create_bar_plot(
        ax=ax_dict["B"],
        x_values=x_hydrogen_system,
        df=df_hydrogen_system,
        name_map=hydrogen_system_name_map,
        title="Hydrogen system",
        y_label_left="Difference in storage capacity in TWh\nand transport capacity in TWkm ",
        ax1_ylim=np.array([0, 50]),
    )

    create_bar_plot(
        ax=ax_dict["C"],
        x_values=x_power_and_heating_system,
        df=df_power_and_heating_system,
        name_map=power_and_heating_system_name_map,
        title="Heating and power system",
        y_label_left="Difference in installed capacity\nin GW and TWh (storage)",
        ax1_ylim=np.array([-150, 150]),
    )

    ax2 = create_bar_plot(
        ax=ax_dict["D"],
        x_values=x_syn_fuels,
        df=df_syn_fuels,
        name_map=name_map_syn_fuels,
        title="Synthetic fuels",
        y_label_left="Difference in generation in TWh a$^{-1}$\nand Mt CO$_2$ a$^{-1}$ (DAC)",
        ax1_ylim=np.array([0, 150]),
    )

    # Show the plot
    handles1, labels1 = ax_dict["A"].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    labels = ["Yearly generation / total installation", "Annualized costs per year"]
    fig.legend(
        handles1 + [handles2[0]],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        frameon=False,
        fancybox=False,
    )

    plt.subplots_adjust(left=0, bottom=0.07, right=1, top=0.9, wspace=0.5, hspace=0.3)

    ### Add explanatory legend plot
    # Define bounding position and remove the placeholder axis
    bounds_orig = ax_dict["E"].get_position(fig).bounds
    bounds = (0.945, 0.085, 0.15, 0.18)
    fig.delaxes(ax_dict["E"])

    # # Create and set up AxesZero
    ax_explain = setup_axes_zero(fig, bounds)
    # # Plot components
    plot_bars(ax_explain)
    plot_lines_and_arrows(ax_explain)
    ax_explain.set_title("Difference\nexplanation")

    # # Add legend
    ax_explain.legend(
        ["PC-50", "PC-0", "Difference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        frameon=False,
        fancybox=False,
    )

    # Draw a rectangle on the entire figure (not inside any specific axis)
    rect = plt.Rectangle(
        (bounds[0] - 0.015, bounds[1] - 0.081),
        bounds[2] + 0.04,
        bounds[-1] + 0.124,
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="black",
        fill=False,
    )
    fig.patches.append(rect)

    plt.show()

    # fig.savefig(
    #     r"../img/system_cost_difference.svg",
    #     format="svg",
    #     bbox_inches="tight",
    # )


def calculate_total_system_difference():
    summary_df = pd.read_csv(
        "../results/total_summary.csv",
        index_col=0,
        header=0,
        skiprows=range(1, 2),
    )

    balanced_mix_system_cost = summary_df.loc[
        "150_lv1.25_I_H_2045_3H_PC_650Euro", "total system cost"
    ]
    el_100_system_cost = summary_df.loc[
        "150_lv1.25_I_H_2045_3H_PC_925Euro", "total system cost"
    ]

    cost_reduction_in_percent = (
        (el_100_system_cost - balanced_mix_system_cost) / balanced_mix_system_cost * 100
    )

    return cost_reduction_in_percent


def main():
    # Network files for base and mixed scenario

    folder_path = r"results/raw"
    file_el_only = r"150_lv1.25_I_H_2045_3H_PC_925Euro/elec_s_150_lv1.25__I-H_2045.nc"
    file_mix = r"150_lv1.25_I_H_2045_3H_PC_650Euro/elec_s_150_lv1.25__I-H_2045.nc"
    n_el = pypsa.Network(os.path.join(folder_path, file_el_only))
    n_pc = pypsa.Network(os.path.join(folder_path, file_mix))

    plot_system_cost_differences(n_el, n_pc)
    print(calculate_total_system_difference())


if __name__ == "__main__":
    main()
