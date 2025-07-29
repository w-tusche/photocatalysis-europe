# -*- coding: utf-8 -*-
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes


def hydrogen_generation_and_storage_over_costs(df, output_folder_path, save=False):
    # Hydrogen generation and storage over photocatalysis CAPEX
    fig, ax = plt.subplots()
    ax_ri = ax.twinx()

    (pc,) = ax.plot(
        df["pc capex"],
        df["photocatalysis (Energy)"],
        ls="none",
        marker="s",
        c="#dbd40c",
    )
    (el,) = ax.plot(
        df["pc capex"],
        df["H2 Electrolysis (Energy)"],
        ls="none",
        marker="s",
        c="#ff29d9",
    )
    (h2_sto,) = ax_ri.plot(
        df["pc capex"],
        df["tot H2 storage"] / 1000,
        ls="none",
        marker="o",
        c="red",
    )

    ax.set(
        xlim=[625, 950],
        xlabel="Photocatalysis CAPEX in €/kW",
        ylabel="Hydrogen generation in TWh/a",
    )
    ax.grid()

    ax_ri.set(
        ylabel="Hydrogen storage in TWh (circles)",
    )

    ax.legend(
        handles=[h2_sto, pc, el],
        labels=["H2-Storage", "Photocatalysis", "Electrolysis"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
    )

    if save:
        fig.savefig(
            os.path.join(output_folder_path, "H2_gen_and_store_over_pc_CAPEX.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    else:
        plt.tight_layout()
        plt.show()


def get_base_index(df: pd.DataFrame):
    """
    Filters the index for different scenarios and outputs the index for the
    base scenario and the scenario variations.

    Parameters
    ----------
    df : pd.DataFrame
        summary DataFrame

    Returns
    -------
    pd.Index, pd.Index, pd.Index, pd.Index
        Returns index of scenario variations
    """

    index_list = df.index.to_list()

    noUGHS_index = pd.Index([index for index in index_list if "_noUGHS" in index])
    pct3_index = pd.Index([index for index in index_list if "_3pct" in index])
    pct6_index = pd.Index([index for index in index_list if "_6pct" in index])
    net_zero_index = pd.Index([index for index in index_list if "net_zero" in index])
    base_index = (
        df.index.difference(noUGHS_index)
        .difference(pct3_index)
        .difference(pct6_index)
        .difference(net_zero_index)
    )

    return base_index, noUGHS_index, pct6_index, pct3_index, net_zero_index


def pipeline_TWkm(df, output_folder_path, save=False):
    # Photocatalysis market share over photocatalysis CAPEX
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))

    (tot_h2,) = ax.plot(
        df["pc capex"],
        df["total H2 pipeline capacity"],
        ls="none",
        marker="s",
        c="#dbd40c",
    )
    (tot_new_h2,) = ax.plot(
        df["pc capex"],
        df["total new H2 pipeline capacity"],
        ls="none",
        marker="s",
        c="#ff29d9",
    )
    (tot_repur_h2,) = ax.plot(
        df["pc capex"],
        df["total repurposed H2 pipeline capacity"],
        ls="none",
        marker="o",
        c="red",
    )

    ax.set(
        xlim=[625, 950],
        xlabel="Photocatalysis CAPEX in €/kW",
        ylabel="Hydrogen pipeline capacity in TWkm",
    )
    ax.grid()

    ax.legend(
        handles=[tot_h2, tot_new_h2, tot_repur_h2],
        labels=["Total capacity", "New capacity", "Repurposed capacity"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
    )

    if save:
        fig.savefig(
            os.path.join(output_folder_path, "H2_pipeline_capacity_over_pc_CAPEX.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    else:
        plt.tight_layout()
        plt.show()


def power_gen_and_storage_over_costs(df, output_folder_path, save=False):
    # Power generation and storage over photocatalysis CAPEX
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    ax_ri = ax.twinx()

    wind_indices = ["offwind-ac", "offwind-dc", "offwind-float", "onwind"]
    solar_indices = ["solar", "solar rooftop", "solar-hsat"]
    battery_indices = ["battery", "home battery"]

    (wind,) = ax.plot(
        df["pc capex"],
        df[wind_indices].sum(axis=1),
        ls="none",
        marker="s",
        c="blue",
    )
    (solar,) = ax.plot(
        df["pc capex"],
        df[solar_indices].sum(axis=1),
        ls="none",
        marker="d",
        c="coral",
    )

    (bat_sto,) = ax_ri.plot(
        df["pc capex"],
        df[battery_indices].sum(axis=1),
        ls="none",
        marker="o",
        c="orchid",
    )

    ax.set(
        xlim=[625, 950],
        xlabel="Photocatalysis CAPEX in €/kW",
        ylabel="Power generation capacity in GW",
    )
    ax.grid()

    ax_ri.set(
        ylabel="Power storage in GWh (circles)",
    )

    ax.legend(
        handles=[bat_sto, wind, solar],
        labels=["Battery", "Wind power", "Photovoltaics"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
    )

    if save:
        fig.savefig(
            os.path.join(output_folder_path, "power_gen_and_store_over_pc_CAPEX.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    else:
        plt.tight_layout()
        plt.show()


def fossil_usage_over_costs(df, output_folder_path, save=False):
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))

    fossils = {
        "oil": {"color": "#7c877c", "label": "Oil"},
        "coal": {"color": "#454545", "label": "Coal"},
        "gas": {"color": "#c9dfdf", "label": "Nataural gas"},
        "nuclear": {"color": "#8d5c5c", "label": "Nuclear power"},
    }

    for fossil, vals in fossils.items():
        ax.plot(
            df["pc capex"],
            df[fossil],
            ls="none",
            marker="s",
            c=vals["color"],
            label=vals["label"],
        )

    ax.set(
        xlim=[625, 950],
        ylim=[-50, None],
        xlabel="Photocatalysis CAPEX in €/kW",
        ylabel="Installed fossil capacity in GW",
    )
    ax.grid()

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=True,
        # loc="best",
        # frameon=False,
    )

    if save:
        fig.savefig(
            os.path.join(output_folder_path, "Fossil_usage_over_pc_CAPEX.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    else:
        plt.tight_layout()
        plt.show()


# TODO: calculate slope of both curves. PC should go up more?!
def installed_capacity_EL_and_PC_over_costs(df, output_folder_path, save=False):
    # Installed capacity Electrolysis and PC
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))

    technologies = {
        "H2 Electrolysis": {"color": "#ff29d9", "label": "Electrolysis", "unit": "GW"},
        "photocatalysis": {
            "color": "#dbd40c",
            "label": "Photocatalysis",
            "unit": "km2",
        },
    }

    for technology, vals in technologies.items():
        y_vals = df[technology]

        ax.plot(
            df["pc capex"],
            y_vals,
            ls="none",
            marker="s",
            c=vals["color"],
            label=vals["label"],
        )

    ax.set(
        xlim=[625, 950],
        ylim=[-50, None],
        xlabel="Photocatalysis CAPEX in €/kW",
        ylabel="Hydrogen generation capacity in GW",
    )
    ax.grid()

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
    )

    if save:
        fig.savefig(
            os.path.join(output_folder_path, "H2_gen_capacity_over_pc_CAPEX.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    else:
        plt.tight_layout()
        plt.show()


def full_loud_hours_EL_and_PC_over_costs(df, output_folder_path, save=False):
    # Full load hours Electrolysis and PC
    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))

    technologies = {
        "Electrolysis": {
            "color": "#ff29d9",
            "label": "Electrolysis",
            "data": df["flh electrolysis"],
        },
        "photocatalysis": {
            "color": "#dbd40c",
            "label": "Photocatalysis (GW)",
            "data": df["flh photocatalysis"],
        },
    }

    for technology, vals in technologies.items():
        ax.plot(
            df["pc capex"],
            vals["data"],
            ls="none",
            marker="s",
            c=vals["color"],
            label=vals["label"],
        )

    ax.set(
        xlim=[625, 950],
        xlabel="Photocatalysis CAPEX in €/kW",
        ylabel="Full load hours",
    )
    ax.grid()

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
    )

    if save:
        fig.savefig(
            os.path.join(output_folder_path, "FLH_over_pc_CAPEX.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
    else:
        plt.tight_layout()
        plt.show()


def main():
    filepath = os.path.join(r"./results/total_summary.csv")

    # Load dataframe but skip the unit column
    total_summary_df = pd.read_csv(
        filepath, index_col=0, header=0, skiprows=range(1, 2)
    )

    with open(r"src/colors.yaml") as stream:
        all_colors = yaml.safe_load(stream)
        tech_colors = all_colors["tech_colors"]
        noUGHS_line_color = all_colors["others"]["no_UGHS_line"]
        UGHS_line_color = all_colors["others"]["UGHS_line"]

    # output_folder_path = r".\img"
    output_folder_path = r"./img"

    hydrogen_generation_and_storage_over_costs(
        total_summary_df, output_folder_path, save=False
    )
    pipeline_TWkm(total_summary_df, output_folder_path, save=False)
    power_gen_and_storage_over_costs(total_summary_df, output_folder_path, save=False)
    fossil_usage_over_costs(total_summary_df, output_folder_path, save=False)
    installed_capacity_EL_and_PC_over_costs(
        total_summary_df, output_folder_path, save=False
    )
    full_loud_hours_EL_and_PC_over_costs(
        total_summary_df, output_folder_path, save=False
    )


if __name__ == "__main__":
    main()
