# -*- coding: utf-8 -*-
"""
This scrip creates a basic summary of PyPSA-Eur .nc-files including
photocatalysis systems.

The script runs through the subdirectories assessing .nc-files regarding system
characteristics (e.g., installed power plant capacities, installed capacities of
hydrogen storage systems) and stores the results in `total_summary.csv`.

The following structure of the (sub)directories is required:

.
├── create_basic_summary.py
└── results
    └── raw
        ├── folder_containing_nc_file_1
        │   ...
        │   └── pypsa_result.nc
        ├── folder_containing_nc_file_2
        │   ...
        │   └── pypsa_result.nc
        ...
        ├── folder_containing_nc_file_n
        │   ...
        │   └── pypsa_result.nc
        └── total_summary.csv
"""

import os

import networkx as nx
import numpy as np
import pandas as pd
import pypsa


def node_solar_irradiance_and_distance_to_UGHS(
    eval_df, n, timestep, system_buses, ughs_systems
):
    eval_df = eval_df.copy()
    # TODO: this could be interesting for electrolysis as well and for Hydrogen demand centers

    # solar irradiance in kWh/a/m^2
    # get solar irradiance at generator locations (photocatalysis)
    solar_irradiance = (
        n.generators_t.p_max_pu.loc[:, n.generators.carrier == "photocatalysis"].sum(
            axis=0
        )
        * timestep
    )
    solar_irradiance.rename(
        index=lambda x: x.replace(" photocatalysis", ""), level=0, inplace=True
    )

    eval_df.loc["solar_irradiance"] = solar_irradiance

    # Extract nodes of the network graph
    graph = n.graph(weight="length")
    nodes = graph.nodes

    dummy_index = system_buses.index.copy()
    dummy_index += " H2"

    dummy_index_ughs = ughs_systems.index.copy()
    dummy_index_ughs += " H2"

    # remove nodes that are not required
    irrelevant_nodes = list(set(nodes) - set(dummy_index))
    for irr_no in irrelevant_nodes:
        graph.remove_node(irr_no)

    # remove edges with weight lower than 1
    to_remove = [
        (a, b) for a, b, attrs in graph.edges(data=True) if attrs["weight"] < 1
    ]
    graph.remove_edges_from(to_remove)

    # Get shortest distances from nodes with photocatalysis to UGHS
    for bus in dummy_index:
        distances = []

        for ughs_loc in dummy_index_ughs:
            shortest_path_length = nx.shortest_path_length(
                graph, source=bus, target=ughs_loc, weight="weight"
            )

            distances.append(shortest_path_length)

        bus = bus.replace(" H2", "")
        eval_df.loc["shortest_path_length_to_UGHS", bus] = np.min(distances)

    return eval_df


def calculate_pipeline_TWkm(
    n: pypsa.Network,
    pipeline_type: str = "H2 pipeline",
    output_spec_pipeline_cost: bool = False,
):
    """
    Function to calculate pipeline capacities in TWkm.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA-Eur Network including pipeline infrastructure
    pipeline_type : str, optional
        String indicating the pipeline type (default is "H2 pipeline")
    output_spec_pipeline_cost : bool, optional
        Boolean to decide whether pipeline-cost should be an output
    Returns
    -------
    float
        Value of total pipeline capacities throughout Europe in TWkm
    """

    pipeline_TWkm = (
        n.links.loc[n.links.carrier == pipeline_type].p_nom_opt
        * n.links.loc[n.links.carrier == pipeline_type].length
    ).sum() / 1e6  # TWkm

    if not output_spec_pipeline_cost:
        return pipeline_TWkm
    else:
        total_pipeline_annual_cost = (
            n.links.loc[n.links.carrier == pipeline_type].capital_cost
            * n.links.loc[n.links.carrier == pipeline_type].p_nom_opt
        ).sum()
        return pipeline_TWkm, total_pipeline_annual_cost


def get_total_brutto_power_generation(n):
    """
    Calculate total initial electricity input due to power generation in the
    system. Thus, all electricity from generators + links that use primary
    energy carriers for generation. Might be mixed a bit however the error
    seems to be small.

    Args:
        n (pypsa.components.Network): PyPSA network

    Returns:
        float: total "brutto" power generation in (TWh)
    """

    # Timestep from network file in hours (e.g. 3 if 3H)
    timestep = int(8760 / len(n.snapshots))  # hours

    # Generators directly supplying electricity
    power_generators = [
        "offwind-ac",
        "offwind-float",
        "onwind",
        "ror",
        "solar",
        "solar-hsat",
        "offwind-dc",
        "nuclear",
        "gas",
        "oil",
        "coal",
        "solar rooftop",
    ]
    # Generators using other primary energy to generate electricity (however could also be indirectly)
    # The dont play a large role here
    power_links = [
        "OCGT",
        "urban central gas CHP",
        "urban central gas CHP CC",
    ]

    total_power_generation = 0
    for carrier in power_generators:
        total_power_generation += (
            (n.generators_t.p.loc[:, n.generators.carrier == carrier].sum().sum())
            / 1e6
            * timestep
        )  # TWh

    for carrier in power_links:
        total_power_generation += (
            abs(n.links_t.p1.loc[:, n.links.carrier == carrier].sum().sum())
            / 1e6
            * timestep
        )  # TWh

    return total_power_generation  # TWh


def create_summary_csv(results_folder: str, re_create_tot_sum: bool) -> pd.DataFrame:
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

    try:
        tot_sum = pd.read_csv(
            os.path.join(results_folder, "total_summary.csv"), index_col=0
        )
        # tot_eval = pd.read_csv(os.path.join(results_folder,"total_eval.csv"),index_col=0)
        files_already_checked = list(tot_sum.index)
    except FileNotFoundError:
        tot_sum = None

    # TODO: think about transposing the total_summary data (so dtypes can be set better. Since the row with strings messes with the type of columns)
    raw_files_folder = os.path.join(results_folder, "raw/")
    subfolder_list = next(os.walk(raw_files_folder))[1]

    tot_eval_distance_ran_once = False
    for i, subfolder in enumerate(subfolder_list):
        if tot_sum is not None and (re_create_tot_sum == False):
            # Only process file if not already processed (if not forced)
            if subfolder in files_already_checked:
                print(
                    f"Skipped subfolder {subfolder}. Was already in index of summary_total file."
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

        # Export demand data to the data/demand_data
        store_demand_and_generation_data(n, subfolder)

        # Column, unit matching:
        col_unit_map = {
            "coal": ["GW"],
            "gas": ["GW"],
            "nuclear": ["GW"],
            "offwind-ac": ["GW"],
            "offwind-dc": ["GW"],
            "offwind-float": ["GW"],
            "oil": ["GW"],
            "onwind": ["GW"],
            "photocatalysis": ["GW"],
            "ror": ["GW"],
            "rural solar thermal": ["GW"],
            "solar": ["GW"],
            "solar rooftop": ["GW"],
            "solar-hsat": ["GW"],
            "urban central solar thermal": ["GW"],
            "urban decentral solar thermal": ["GW"],
            "H2 Electrolysis": ["GW"],
            "H2 Fuel Cell": ["GW"],
            "H2 turbine": ["GW"],
            "H2 storage": ["GWh"],
            "home battery": ["GWh"],
            "battery": ["GWh"],
            "H2 Electrolysis (Energy)": ["TWh"],
            "photocatalysis (Energy)": ["TWh"],
            "total power generation (Energy)": ["TWh"],
            "pc market share": ["%"],
            "pc total annual cost": ["Euro/a/kW"],
            "pc/pv cost relation": ["-"],
            "pc/wind cost relation": ["-"],
            "pc/el cost relation": ["-"],
            "pc capex": ["Euro/kW"],
            "node count": ["-"],
            "flh electrolysis": ["h/a"],
            "flh photocatalysis": ["h/a"],
            "aghs": ["TWh"],
            "ughs": ["TWh"],
            "tot H2 storage": ["TWh"],
            "tot H2 storage cost": ["Euro"],
            "total H2 pipeline capacity": ["TWkm"],
            "total H2 pipeline capacity cost": ["Euro"],
            "total new H2 pipeline capacity": ["TWkm"],
            "total new H2 pipeline capacity cost": ["Euro"],
            "total repurposed H2 pipeline capacity": ["TWkm"],
            "total repurposed H2 pipeline capacity cost": ["Euro"],
            # TODO: check if this is Euro or Euro/year (https://pypsa.readthedocs.io/en/latest/user-guide/optimal-power-flow.html)
            "total system cost": ["Euro"],
            "ughs allowed": ["-"],
            "pc larger 1MW": ["-"],
            "nodal capacity node:el:pc": ["MW"],
            "nodal LCOH node:el:pc": ["Euro/kg"],
        }

        # Create the data frame with units:
        summary_df = pd.DataFrame(col_unit_map, index=["Unit"])

        # Timestep from network file in hours (e.g. 3 if 3H)
        timestep = int(8760 / len(n.snapshots))  # hours

        # Get number of nodes
        node_count = n.buses[n.buses.carrier == "AC"].shape[0]

        # identify system-wide capacities of generators in GW
        gens = n.generators.groupby(["carrier"]).p_nom_opt.sum() / 1e3

        # identify system-wide capacities of electrolysis, fuel cell and h2 turbines in GW
        h2_link_names = ["H2 Electrolysis", "H2 Fuel Cell", "H2 turbine"]
        h2_link_caps = (n.links.groupby(["carrier"]).p_nom_opt.sum() / 1e3).loc[
            h2_link_names
        ]

        # get indices of electrolyzer and photocatalysis locations
        el_ind = n.links[n.links.carrier.isin(["H2 Electrolysis"])].index
        pc_ind = n.generators[n.generators.carrier.isin(["photocatalysis"])].index

        # calculate total hydrogen generation in TWh/a; per definition, p1 leaves the
        # electrolysis link as negative hydrogen power
        el_h2_generation = -(n.links_t.p1.loc[:, el_ind].sum().sum() / 1e6) * timestep
        pc_h2_generation = n.generators_t.p.loc[:, pc_ind].sum().sum() / 1e6 * timestep
        total_h2_generation = el_h2_generation + pc_h2_generation

        # get total annual cost value (CAPEX-annuity and FOM) of photocatalysis
        # this value allows the recalculation of the assumed CAPEX/kW
        pc_tac = n.generators.capital_cost.loc[pc_ind].mean() / 1000  # €/a/kW

        pv_ind = n.generators[n.generators.carrier.isin(["solar"])].index
        pv_tac = n.generators.capital_cost.loc[pv_ind].mean() / 1000  # €/a/kW

        wind_ind = n.generators[n.generators.carrier.isin(["onwind"])].index
        wind_tac = n.generators.capital_cost.loc[wind_ind].mean() / 1000  # €/a/kW

        el_tac = n.links.capital_cost.loc[el_ind].mean() / 1000  # €/a/kW

        pc_pv_cost_relation = pc_tac / pv_tac
        pc_wind_cost_relation = pc_tac / wind_tac
        pc_el_cost_relation = pc_tac / el_tac

        # get pc cost assumptions
        pc_capex = recalc_capex(pc_tac)

        # calculate the operational market share of photocatalysis
        # (i.e., regarding annual hydrogen production)
        pc_market_share = (pc_h2_generation / total_h2_generation) * 100  # in %

        # get storage capacities of selected storage systems
        store_names = ["H2", "home battery", "battery"]
        store_cap = n.stores.groupby(["carrier"]).e_nom_opt.sum() / 1e3  # in GWh
        store_cap = store_cap[store_names]
        store_cap.rename(index={"H2": "H2 storage"}, inplace=True)

        flh_el = calc_full_load_hours(
            installed_capacity_GW=h2_link_caps["H2 Electrolysis"],
            generated_energy_TWh=el_h2_generation,
        )
        flh_pc = calc_full_load_hours(
            installed_capacity_GW=gens.photocatalysis,
            generated_energy_TWh=pc_h2_generation,
        )

        # storage identification and distinction between AGHS and UGHS
        h2_store_ind = n.stores.carrier.filter(like="H2").index

        # distinct between AGHS and UGHS based on storage capital cost (low for UGHS)
        # does not make sense when no UGHS is in there
        unique_costs = n.stores.loc[h2_store_ind].capital_cost.unique()
        aghs_cost = unique_costs.max()
        ughs_cost = unique_costs.min()
        ughs_systems = n.stores.loc[h2_store_ind][
            n.stores.loc[h2_store_ind].capital_cost == ughs_cost
        ].copy()
        ughs_systems.rename(
            index=lambda x: x.replace(" H2 Store", ""), level=0, inplace=True
        )
        aghs_systems = n.stores.loc[h2_store_ind][
            n.stores.loc[h2_store_ind].capital_cost == aghs_cost
        ].copy()
        aghs_systems.rename(
            index=lambda x: x.replace(" H2 Store", ""), level=0, inplace=True
        )

        # Calculate Hydrogen storage capacities
        aghs_storage_capacity = aghs_systems.e_nom_opt.sum() / 1e6  # TWh
        if len(unique_costs) == 1:
            ughs_storage_capacity = 0
            tot_h2_storage_capacity = aghs_storage_capacity
            ughs_expandable = 0
        elif len(unique_costs) == 2:
            ughs_storage_capacity = ughs_systems.e_nom_opt.sum() / 1e6  # TWh
            tot_h2_storage_capacity = (
                aghs_storage_capacity + ughs_storage_capacity
            )  # TWh
            ughs_expandable = 1

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

        total_h2_storage_cost = (
            n.stores.loc[n.stores.carrier == "H2"].e_nom_opt
            * n.stores.loc[n.stores.carrier == "H2"].capital_cost
        ).sum()

        # Get locations of photocatalysis
        pc_locations = n.generators.loc[n.generators.carrier == "photocatalysis"].copy()
        pc_locations.rename(
            index=lambda x: x.replace(" photocatalysis", ""), level=0, inplace=True
        )

        # Get locations of electrolysis
        el_locations = n.links.loc[n.links.carrier == "H2 Electrolysis"].copy()
        el_locations.rename(
            index=lambda x: x.replace(" H2 Electrolysis", ""), level=0, inplace=True
        )

        # Add regions where at least 1 MW PC is installed
        pc_generators = pc_locations.loc[pc_locations["p_nom_opt"] > 1].copy()
        pc_generator_regions = list(pc_generators.index)

        # Levelized cost of hydrogen calculation
        clip_capacity = 1  # MW

        el_capacity = n.links.p_nom_opt.loc[el_ind]
        el_capacity.rename(
            index=lambda x: x.replace(" H2 Electrolysis", ""), level=0, inplace=True
        )

        pc_capacity = n.generators.p_nom_opt.loc[pc_ind]
        pc_capacity.rename(
            index=lambda x: x.replace(" photocatalysis", ""), level=0, inplace=True
        )

        nodal_capacity = pd.DataFrame(index=el_capacity.index, columns=["EL", "PC"])
        nodal_capacity.loc[el_capacity.index, "EL"] = el_capacity
        nodal_capacity.loc[pc_capacity.index, "PC"] = pc_capacity

        # TODO: Check implementation (there should be a more elegant way to do this concatenation!)
        round_digits = 5
        nodal_capacity_string = ""
        for row in nodal_capacity.index:
            nodal_capacity_string += (
                row
                + ":"
                + str(np.round(nodal_capacity.loc[row, "EL"], round_digits))
                + ":"
                + str(np.round(nodal_capacity.loc[row, "PC"], round_digits))
                + ";"
            )

        # calculate h2 generation per location
        el_nodal_h2_generation_ts = -(n.links_t.p1.loc[:, el_ind]) * timestep
        el_nodal_h2_generation_ts.rename(
            columns=lambda x: x.replace(" H2 Electrolysis", ""), level=0, inplace=True
        )

        pc_nodal_h2_generation_ts = n.generators_t.p.loc[:, pc_ind] * timestep
        pc_nodal_h2_generation_ts.rename(
            columns=lambda x: x.replace(" photocatalysis", ""), level=0, inplace=True
        )

        el = n.links.loc[el_ind]
        el.rename(
            index=lambda x: x.replace(" H2 Electrolysis", ""), level=0, inplace=True
        )

        pc = n.generators.loc[pc_ind]
        pc.rename(
            index=lambda x: x.replace(" photocatalysis", ""), level=0, inplace=True
        )

        el_buses = el.index
        pc_buses = pc.index

        # nodal marginal prices
        nmp = n.buses_t.marginal_price.loc[:, el_buses]

        el_nodal_power_cons = n.links_t.p0.loc[:, el_ind] * timestep
        el_nodal_power_cons.rename(
            columns=lambda x: x.replace(" H2 Electrolysis", ""), level=0, inplace=True
        )

        el_nodal_power_cons_cost = nmp * el_nodal_power_cons

        LCOH_el = (
            el_nodal_power_cons_cost.sum() + el.capital_cost * el_capacity
        ) / el_nodal_h2_generation_ts.sum()

        LCOH_el[el_capacity < clip_capacity] = np.nan
        LCOH_pc = (pc.capital_cost * pc_capacity) / pc_nodal_h2_generation_ts.sum()
        LCOH_pc[pc_capacity < clip_capacity] = np.nan

        # LCOH in € / kg
        LCOH_el_per_kg = LCOH_el / 1000 * 33.33
        LCOH_pc_per_kg = LCOH_pc / 1000 * 33.33

        nodal_LCOH = pd.DataFrame(index=el_capacity.index, columns=["EL", "PC"])
        nodal_LCOH.loc[el_capacity.index, "EL"] = LCOH_el_per_kg
        nodal_LCOH.loc[pc_capacity.index, "PC"] = LCOH_pc_per_kg

        # TODO: Check implementation (there should be a more elegant way to do this concatenation!)
        round_digits = 5
        nodal_LCOH_string = ""
        for row in nodal_LCOH.index:
            nodal_LCOH_string += (
                row
                + ":"
                + str(np.round(nodal_LCOH.loc[row, "EL"], round_digits))
                + ":"
                + str(np.round(nodal_LCOH.loc[row, "PC"], round_digits))
                + ";"
            )

        total_power_generation = get_total_brutto_power_generation(n)

        # entry-wise addition of further data
        data_to_add = {
            "H2 Electrolysis (Energy)": el_h2_generation,
            "photocatalysis (Energy)": pc_h2_generation,
            "total power generation (Energy)": total_power_generation,
            "pc market share": pc_market_share,
            "pc total annual cost": pc_tac,
            "pc/pv cost relation": pc_pv_cost_relation,
            "pc/wind cost relation": pc_wind_cost_relation,
            "pc/el cost relation": pc_el_cost_relation,
            "pc capex": int(pc_capex),
            "node count": node_count,
            "flh electrolysis": flh_el,
            "flh photocatalysis": flh_pc,
            "aghs": aghs_storage_capacity,
            "ughs": ughs_storage_capacity,
            "tot H2 storage": tot_h2_storage_capacity,
            "tot H2 storage cost": total_h2_storage_cost,
            "total H2 pipeline capacity": total_h2_pipeline_TWkm,
            "total H2 pipeline capacity cost": total_h2_pipeline_cost,
            "total new H2 pipeline capacity": total_new_h2_pipeline_TWkm,
            "total new H2 pipeline capacity cost": total_new_h2_pipeline_cost,
            "total repurposed H2 pipeline capacity": total_repurposed_h2_pipeline_TWkm,
            "total repurposed H2 pipeline capacity cost": total_repurposed_h2_pipeline_cost,
            "total system cost": n.objective,
            "ughs allowed": ughs_expandable,
            "pc larger 1MW": ";".join(pc_generator_regions),
            "nodal capacity node:el:pc": nodal_capacity_string,
            "nodal LCOH node:el:pc": nodal_LCOH_string,
        }

        additional_columns = pd.DataFrame(
            data={key: [value] for key, value in data_to_add.items()}
        )
        summary = pd.concat(
            (
                gens,
                h2_link_caps,
                store_cap.loc[["home battery", "battery"]],
                additional_columns.iloc[0, :],
            ),
            axis=0,
        )

        # Add new line (subfolder name as index) to the summary_df
        summary_df.loc[subfolder] = summary

        # assess single photocatalysis and electrolysis locations
        system_buses = n.buses[n.buses.carrier == "AC"]

        eval_df = pd.DataFrame(columns=system_buses.index)

        # Run this only when it is case where ughs is activated. Else you get wrong results
        if tot_eval_distance_ran_once == False and not subfolder.endswith("noUGHS"):
            # solar irradiance and distance to UGHS does not change, thus, it only needs to be executed once
            # solar irradiance in kWh/a/m^2
            eval_df = node_solar_irradiance_and_distance_to_UGHS(
                eval_df, n, timestep, system_buses, ughs_systems
            )
            tot_eval_distance_ran_once = True

        eval_df.loc[f"el_cap_{subfolder}"] = el_locations.loc[
            el_locations.index, "p_nom_opt"
        ]
        # TODO: this pc_locations is not clipped, pc_generators is clipped at 1MW, maybe change this! (same for electrolysis!)
        eval_df.loc[f"pc_cap_{subfolder}"] = pc_locations.loc[
            pc_locations.index, "p_nom_opt"
        ]

        if i == 0:
            # create a new DataFrame for storing the results of all .nc-files
            total_summary_df = summary_df.copy()
            # create a new DataFrame for storing location dependent results of all .nc-files
            total_eval_df = eval_df.copy()
        else:
            # TODO: Check if this is the right way to add this!
            # merge summary_df into total_summary_df; delete duplicates (i.e., 'Unit')
            total_summary_df = pd.concat(
                [total_summary_df, summary_df]
            ).drop_duplicates(keep="first")

            # merge eval_df into total_eval_df
            total_eval_df = pd.concat([total_eval_df, eval_df]).drop_duplicates(
                keep="first"
            )

            total_summary_df.to_csv(os.path.join(results_folder, "total_summary.csv"))
            total_eval_df.to_csv(os.path.join(results_folder, "total_eval.csv"))

    if tot_sum is not None and re_create_tot_sum == False:
        # Append data
        try:
            total_summary_df.drop("Unit").to_csv(
                os.path.join(results_folder, "total_summary.csv"),
                mode="a",
                index=True,
                header=False,
            )
            total_eval_df.drop(
                ["solar_irradiance", "shortest_path_length_to_UGHS"]
            ).to_csv(
                os.path.join(results_folder, "total_eval.csv"),
                mode="a",
                index=True,
                header=False,
            )
        except UnboundLocalError:
            print("Nothing to add. All files where already added before.")
            total_summary_df = tot_sum
    else:
        # Create tot sum
        total_summary_df.to_csv(os.path.join(results_folder, "total_summary.csv"))
        total_eval_df.to_csv(os.path.join(results_folder, "total_eval.csv"))

    return total_summary_df


def store_demand_and_generation_data(network, name):
    """
    Function to store the cumulated demand and generation data for each
    location.
    """
    netw = network.copy()

    # Store total hydrogen demand
    technos = extract_demand_and_generation_info(netw, name)
    tot_h2_demand_with_locs = technos["Total H2 demand"]["series"]
    tot_h2_demand_with_locs.name = "total_h2_demand_MWh"
    tot_h2_demand_with_locs.rename_axis(index="Bus", inplace=True)

    # Generation dataframe
    technology = ["H2 Electrolysis", "photocatalysis", "SMR", "SMR CC"]
    generation_df = pd.DataFrame()
    for tecno in technology:
        tot_h2_generation_with_locs = technos[tecno]["series"]
        tot_h2_generation_with_locs.name = (
            f"h2_generation_{technos[tecno]['title']}_MWh"
        )
        tot_h2_generation_with_locs.rename_axis(index="Bus", inplace=True)

        generation_df = pd.concat([generation_df, tot_h2_generation_with_locs], axis=1)

    generation_df.fillna(0, inplace=True)
    generation_df.rename_axis(index="Bus", inplace=True)

    generation_df["h2_generation_total_MWh"] = (
        generation_df["h2_generation_Electrolysis_MWh"]
        + generation_df["h2_generation_Photocatalysis_MWh"]
        + generation_df["h2_generation_SMR_MWh"]
        + generation_df["h2_generation_SMR CC_MWh"]
    )
    generation_df.rename(
        {"h2_generation_SMR CC_MWh": "h2_generation_SMR_CC_MWh"}, axis=1, inplace=True
    )

    generation_df["total_h2_demand_MWh"] = tot_h2_demand_with_locs

    # Store hydrogen production and demand jointly
    generation_df.to_csv(rf"./data/generation_data/generation_{name}.csv")


def extract_demand_and_generation_info(network, name):
    """
    Extract hydrogen demand and generation info from a PyPSA-network.

    Args:
        network (pypsa.components.Network): PyPSA network file
        name: subfolder name


    Returns:
        dict: dict containing information on hydrogen demand
    """
    # TODO: Store all this to only one file instead of multiple files (same with bus_sizes?!)

    n = network.copy()
    timestep = int(8760 / len(n.snapshots))

    technologies_dict = {
        # The connections can be found in prepare_sector_network (n.madd(... ))
        # Generation
        "H2 Electrolysis": {"type": "link", "p_type": "p1"},
        "photocatalysis": {"type": "generator"},
        "SMR": {"type": "link", "p_type": "p1"},
        "SMR CC": {"type": "link", "p_type": "p1"},
        # Demand
        "H2 for industry": {"type": "load"},
        "methanolisation": {"type": "link", "p_type": "p0"},
        "Fischer-Tropsch": {"type": "link", "p_type": "p0"},
        "Sabatier": {"type": "link", "p_type": "p0"},
        "H2 turbine": {"type": "link", "p_type": "p0"},
        "H2 Fuel Cell": {"type": "link", "p_type": "p0"},
        # "land transport fuel cell": {"type": "load"},
        # "Haber-Bosch": {"type": "link", "p_type": "p2"},
        # "ammonia cracker": {"type": "link", "p_type": "p2"},
    }

    # We dont use ammonia, else the following would also be required
    # Mwh_H2_per_tNH3 _electrolysis
    # The energy amount of hydrogen needed to produce a ton of ammonia using Haber–Bosch process. From Wang et al (2018), Base value assumed around 0.197 tH2/tHN3 (>3/17 since some H2 lost and used for energy)

    for technology, vals in technologies_dict.items():
        # Determine the component and timeseries based on the type
        if vals["type"] == "link":
            df = n.links[
                n.links.carrier.isin(
                    [
                        technology,
                    ]
                )
            ]
            if vals["p_type"] == "p0":
                timeseries_data = n.links_t.p0
            elif vals["p_type"] == "p1":
                timeseries_data = n.links_t.p1
            elif vals["p_type"] == "p2":
                timeseries_data = n.links_t.p2
        elif vals["type"] == "load":
            df = n.loads[
                n.loads.carrier.isin(
                    [
                        technology,
                    ]
                )
            ]
            timeseries_data = n.loads_t.p
        elif vals["type"] == "generator":
            df = n.generators[n.generators.carrier.isin([technology])]
            timeseries_data = n.generators_t.p
        else:
            raise KeyError(
                'Check for a valid key in your technologies dict! Valid should be e.g. ["link", "load"]'
            )

        idx = df.index
        bus_info = abs((timeseries_data.loc[:, idx].sum()) * timestep)
        tot = bus_info.sum()
        technologies_dict[technology] = {
            "type": vals["type"],
            "df": df,
            "index": idx,
            "bus_info": bus_info,
            "total": tot,
        }

    total_h2_generation = (
        technologies_dict["H2 Electrolysis"]["total"]
        + technologies_dict["photocatalysis"]["total"]
        + technologies_dict["SMR"]["total"]
        + technologies_dict["SMR CC"]["total"]
    )
    h2_delta = total_h2_generation - (
        technologies_dict["methanolisation"]["total"]
        + technologies_dict["Fischer-Tropsch"]["total"]
        + technologies_dict["H2 for industry"]["total"]
        + technologies_dict["H2 Fuel Cell"]["total"]
        + technologies_dict["H2 turbine"]["total"]
        + technologies_dict["Sabatier"]["total"]
    )
    # If this delta get to big check for hydrogen_fuel_cell and check hydrogen_turbine
    assert (
        abs(h2_delta) < 0.1 * 1e6
    ), f"Delta rather large, check for further technologies with hydrogen demand or generation (e.g. with: n.links[n.links.bus0.str.contains('H2')].carrier.unique()). Case = {name}."

    # Technologies to create map for
    technos = {
        # Generation
        "H2 Electrolysis": {"title": "Electrolysis"},
        "photocatalysis": {"title": "Photocatalysis"},
        "SMR": {"title": "SMR"},
        "SMR CC": {"title": "SMR CC"},
        # Demand
        "H2 for industry": {"title": "H$_2$ for industry"},
        "methanolisation": {"title": "Methanolisation"},
        "Fischer-Tropsch": {"title": "Fischer-Tropsch"},
        "H2 Fuel Cell": {"title": "H2 fuel cell"},
        "H2 turbine": {"title": "H2 turbine"},
        "Sabatier": {"title": "Sabatier"},
    }

    for techno in technos.keys():
        series = technologies_dict[techno]["bus_info"].copy()
        series.rename(
            index=lambda x: x.replace(f" {techno}", ""), level=0, inplace=True
        )
        series.clip(0, inplace=True)
        technos[techno]["series"] = series

    # Initialize Series and sum up all series
    total_series = pd.Series(0, index=technos["H2 for industry"]["series"].index)
    for techno in technos.keys():
        if techno not in ["H2 Electrolysis", "photocatalysis"]:
            total_series += technos[techno]["series"]

    total_series.fillna(0, inplace=True)
    # Add the total to the dict
    technos["Total H2 demand"] = {
        "title": "Total H$_2$ demand (TWh)",
        "series": total_series,
    }

    return technos


def load_total_summary(filepath):
    total_summary_df = pd.read_csv(filepath, index_col=0)
    return total_summary_df


def recalc_capex(total_annuity):
    # annuity factor for PyPSA results to recalculate CAPEX in €/kW
    i = 0.07
    n = 10
    FOM = 0.04
    annuity_factor = ((1 + i) ** n * i) / ((1 + i) ** n - 1)

    # recalculat CAPEX in €/kW from annual cost
    CAPEX_recalc = total_annuity / (annuity_factor + FOM)
    return CAPEX_recalc


def load_df_without_unit_row(filepath):
    # create a DataFrame without the "Unit"-column (to keep the code leaner)
    df_without_unit = pd.read_csv(filepath, index_col=0, header=0, skiprows=range(1, 2))
    return df_without_unit


def calc_full_load_hours(installed_capacity_GW, generated_energy_TWh):
    flh = (generated_energy_TWh * 1000) / installed_capacity_GW
    return flh


if __name__ == "__main__":
    # This should be the folder where all the raw result files are placed.
    results_folder = "./results/"

    # toggle creation of csv-summary
    create_new_csv_bool = True
    print(f"Option for creating CSV-summary is set to {create_new_csv_bool}.")
    if create_new_csv_bool:
        print("New CSV-summary is created!")
        total_summary_df = create_summary_csv(
            results_folder, re_create_tot_sum=create_new_csv_bool
        )
    else:
        print(
            "No new CSV-summary is created.\nNew data appended to existing file and "
            + f"results are read from {os.path.join(results_folder,'total_summary.csv')}"
        )
        total_summary_df = create_summary_csv(
            results_folder, re_create_tot_sum=create_new_csv_bool
        )
