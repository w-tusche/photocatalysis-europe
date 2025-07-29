# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path

import cartopy.crs as ccrs
import cmcrameri  # To register colormaps for matplotlib e.g. cmc.batlow
import geopandas as gpd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
import yaml
from mpl_toolkits.axisartist.axislines import AxesZero
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

import extract_bus_information as extract_bus_information
import plot_hydrogen_network as plot_hydrogen_network
from create_basic_summary import calculate_pipeline_TWkm

pgf_with_latex = {
    # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    # "svg.fonttype": 'times',
    "font.serif": ["times"],  # blank entries should cause plots
    "font.sans-serif": [],  # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 10,
    "axes.titlesize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "lines.linewidth": 2,
    "lines.markersize": 5,
    "legend.fontsize": 8,  # Make the legend/label fonts
    # "legend.borderaxespad": 0.2,
    "xtick.labelsize": 8,  # a little smaller
    "ytick.labelsize": 8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.pad": 5.6,
    "ytick.major.pad": 5.6,
    "pgf.preamble": "\n".join(
        [  # plots will use this preamble
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[detect-all,locale=DE]{siunitx}",
            r"\usepackage{eurosym}",
            r"\usepackage[version=4]{mhchem}",
            r"\sisetup{per-mode = symbol}",
            r"\DeclareSIUnit{\sieuro}{\mbox{\euro}}",
            r"\DeclareSIUnit{\a}{a}",
        ]
    ),
    "text.latex.preamble": "\n".join(
        [  # plots will use this preamble
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[detect-all,locale=DE]{siunitx}",
            r"\usepackage{eurosym}",
            r"\usepackage[version=4]{mhchem}",
            r"\sisetup{per-mode = symbol}",
            r"\DeclareSIUnit{\sieuro}{\mbox{\euro}}",
            r"\DeclareSIUnit{\a}{a}",
        ]
    ),
}

# rounded_df = total_summary_df.loc[base,["pc total annual cost","pc market share"]].map(lambda val: round(float(val), 1)).sort_values("pc market share")
# photocatalysis_market_share_mapping_STH_10_UGHS = dict(zip(rounded_df["pc total annual cost"], rounded_df["pc market share"]))
photocatalysis_market_share_mapping_STH_10_UGHS = {
    "168.7": "0.0",
    "164.1": "4.3",
    "159.6": "5.1",
    "155.0": "9.6",
    "145.9": "20.3",
    "136.8": "31.1",
    "127.7": "42.8",
    "118.5": "51.2",
}


def get_base_index(df: pd.DataFrame):
    """
    Determines and separates the DataFrame index into different scenario
    categories, returning indices for the base scenario and scenario
    variations.

    Parameters
    ----------
    df : pd.DataFrame
        The summary DataFrame containing scenario indices.

    Returns
    -------
    tuple of pd.Index
        A tuple containing the indices for the base scenario, noUGHS variations,
        6pct variations, 3pct variations, and net_zero variations, respectively.
    """

    # Define scenario patterns and corresponding names
    scenario_patterns = {
        "noUGHS": "_noUGHS",
        "pct3": "_3pct",
        "pct6": "_6pct",
        "net_zero": "net_zero",
    }

    # Create a dictionary to hold the indices for each scenario
    scenario_indices = {
        key: df.index[df.index.str.contains(pattern)]
        for key, pattern in scenario_patterns.items()
    }

    # Calculate the base_index by removing all scenario indices
    base_index = df.index
    for indices in scenario_indices.values():
        base_index = base_index.difference(indices)

    # Return the indices in the desired order
    return (
        base_index,
        scenario_indices["noUGHS"],
        scenario_indices["pct6"],
        scenario_indices["pct3"],
        scenario_indices["net_zero"],
    )


def set_size(use=True, fraction=1, subplots=(1, 1), scale=1):
    """
    Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    ## elsevier guide width
    # single column: 90 mm / 255 pt
    # 1.5 column: 140 mm / 397 pt
    # 2 column: 190 mm / 539 pt

    # Column width Nature Energy
    # (pt with values from above)
    # single column: 88 mm / 250 pt
    # 2 column: 180 mm / 511 pt

    # Width of figure (in pts)
    width_pt = 468
    height_pt = 622
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2 * scale

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    if use == True:
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = fig_width_in * golden_ratio * 1.3 * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def save_or_plot_img(fig, output_folder_path, name, save=False):
    if save:
        fig.savefig(
            os.path.join(output_folder_path, f"{name}.svg"),
            format="svg",
            bbox_inches="tight",
        )
    else:
        fig.tight_layout()
        plt.show()


def market_share(
    df,
    output_folder_path,
    UGHS_line_color,
    noUGHS_line_color,
    save=False,
    invert_axis=True,
    para_axis=False,
    all=False,
):
    # Photocatalysis market share over photocatalysis CAPEX
    # fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))
    fig = plt.figure(figsize=(16 / 2.54, 12 / 2.54))

    host = fig.add_axes([0.08, 0.3, 0.84, 0.55], axes_class=HostAxes)

    if para_axis:
        # Create a parasite axes that shares the x-axis of the host
        par1 = ParasiteAxes(host, sharey=host)
        # Create a second parasite axis for additional ticks
        par2 = ParasiteAxes(host, sharey=host)

        # Append the parasite axes to the host's 'parasites' list
        host.parasites.append(par1)
        host.parasites.append(par2)
        # Make the right axis of par1 visible and configure it
        par1.axis["bottom2"] = par1.new_fixed_axis(loc="bottom", offset=(0, -35))
        par1.axis["bottom2"].set_visible(True)
        par1.axis["bottom2"].major_ticklabels.set_visible(True)
        par1.axis["bottom2"].label.set_visible(True)

        # Create a second right axis for par2 with an offset to distinguish it
        par2.axis["bottom3"] = par2.new_fixed_axis(loc="bottom", offset=(0, -70))
        par2.axis["bottom3"].major_ticklabels.set_visible(True)
        par2.axis["bottom3"].set_visible(True)

    base_index, noUGHS_index, pct6_index, pct3_index, _ = get_base_index(df)

    host.set(
        xlim=[115, 172.5],
        ylim=[-1.0, 55],
        xlabel=r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$",
        ylabel=r"Photocatalysis market share in \%",
    )

    # Add market "segments"
    bounds = [0, 5, 50, host.get_ylim()[1]]
    face_colors = ["0.95", "0.875", "0.8"]
    text_x_pos = [116.5, 171, 171]
    if invert_axis:
        text_x_alignment = ["right", "left", "left"]
    else:
        text_x_alignment = ["left", "right", "right"]
    market_segments = ["Market entry", "Growing market presence", "Market dominance"]
    gap = 0.0  # 0.25

    for bound_i in range(len(bounds) - 1):
        rectangle = patches.Rectangle(
            (host.get_xlim()[0], bounds[bound_i] + gap),
            host.get_xlim()[1] - host.get_xlim()[0],
            bounds[bound_i + 1] - bounds[bound_i] - 2 * gap,
            edgecolor="None",
            facecolor=face_colors[bound_i],
        )
        host.add_patch(rectangle)
        host.text(
            x=text_x_pos[bound_i],
            y=(bounds[bound_i] + bounds[bound_i + 1]) / 2,
            s=market_segments[bound_i],
            ha=text_x_alignment[bound_i],
            va="center",
        )

    (base,) = host.plot(
        df.loc[base_index, "pc total annual cost"],
        df.loc[base_index, "pc market share"],
        ls="--",
        marker="s",
        c=UGHS_line_color,
    )

    if all:
        (noUGHS,) = host.plot(
            df.loc[noUGHS_index, "pc total annual cost"],
            df.loc[noUGHS_index, "pc market share"],
            ls="--",
            marker="s",
            c=noUGHS_line_color,
        )

        (pct3,) = host.plot(
            df.loc[pct3_index, "pc total annual cost"],
            df.loc[pct3_index, "pc market share"],
            ls="-.",
            marker="x",
            c=UGHS_line_color,
        )

        (pct6,) = host.plot(
            df.loc[pct6_index, "pc total annual cost"],
            df.loc[pct6_index, "pc market share"],
            ls=":",
            marker="d",
            c=UGHS_line_color,
        )

    host.grid()

    host.legend(
        handles=[base, pct6, pct3, noUGHS] if all else [base],
        labels=[
            r"$STH=10~\mathrm{\%}$",
            r"$STH=6~\mathrm{\%}$",
            r"$STH=3~\mathrm{\%}$",
            "$STH=10~\mathrm{\%}$,\nno UGHS",
        ]
        if all
        else [r"$STH=10~\mathrm{\%}$"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=4,
        frameon=False,
        fancybox=False,
    )

    if para_axis:
        scaling_fac_par1 = (
            df.loc[base_index, "pc/pv cost relation"]
            / df.loc[base_index, "pc total annual cost"]
        ).mean()
        scaling_fac_par2 = (
            df.loc[base_index, "pc/el cost relation"]
            / df.loc[base_index, "pc total annual cost"]
        ).mean()

        par1.set(
            # xticks = np.array(host.get_xticks())*scaling_fac_par1,
            xlim=[
                host.get_xlim()[0] * scaling_fac_par1,
                host.get_xlim()[1] * scaling_fac_par1,
            ],
            xlabel="Annualized cost relation photocatalysis to photovoltaics",
        )

        par2.set(
            # xticks = np.array(host.get_xticks())*scaling_fac_par2,
            xlim=[
                host.get_xlim()[0] * scaling_fac_par2,
                host.get_xlim()[1] * scaling_fac_par2,
            ],
            xlabel="Annualized cost relation photocatalysis to electrolysis",
        )

    # Annotate PC-0 and PC-50 case
    el_100_x = df.loc[f"150_lv1.25_I_H_2045_3H_PC_925Euro", "pc total annual cost"]
    el_100_y = df.loc[f"150_lv1.25_I_H_2045_3H_PC_925Euro", "pc market share"]
    balanced_mix_x = df.loc[
        f"150_lv1.25_I_H_2045_3H_PC_650Euro", "pc total annual cost"
    ]
    balanced_mix_y = df.loc[f"150_lv1.25_I_H_2045_3H_PC_650Euro", "pc market share"]
    host.annotate(
        "PC-0",
        xy=(el_100_x, el_100_y),
        xytext=(el_100_x, el_100_y + 7),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        ha="center",
    )
    host.annotate(
        "PC-50",
        xy=(balanced_mix_x, balanced_mix_y),
        xytext=(129, 51),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        ha="left",
    )

    if invert_axis:
        host.invert_xaxis()
        if para_axis:
            par1.invert_xaxis()
            par2.invert_xaxis()

    save_or_plot_img(
        fig,
        output_folder_path,
        "market_share_over_annualized_PC_cost"
        if all
        else "market_share_over_annualized_PC_cost_only_10",
        save=save,
    )


def system_development(
    df, output_folder_path, tech_colors, save=False, invert_axis=True
):
    # Development of different system characteristics

    fig = plt.figure(figsize=(16 / 2.54, 12 / 2.54))

    host = fig.add_axes([0.11, 0.3, 0.7, 0.5], axes_class=HostAxes)

    # Create a parasite axes that shares the x-axis of the host
    par1 = ParasiteAxes(host, sharey=host)
    # Create a second parasite axis for additional ticks
    par2 = ParasiteAxes(host, sharey=host)

    par3 = ParasiteAxes(host, sharex=host)

    # Append the parasite axes to the host's 'parasites' list
    host.parasites.append(par1)
    host.parasites.append(par2)
    host.parasites.append(par3)

    # Hide the right spine of the host axis (initial 'right' is allocated to par1)
    host.axis["right"].set_visible(False)

    # Make the right axis of par1 visible and configure it
    par1.axis["bottom2"] = par1.new_fixed_axis(loc="bottom", offset=(0, -35))
    par1.axis["bottom2"].set_visible(True)
    par1.axis["bottom2"].major_ticklabels.set_visible(True)
    par1.axis["bottom2"].label.set_visible(True)

    # Create a second right axis for par2 with an offset to distinguish it
    par2.axis["bottom3"] = par2.new_fixed_axis(loc="bottom", offset=(0, -70))
    par2.axis["bottom3"].major_ticklabels.set_visible(True)
    par2.axis["bottom3"].set_visible(True)

    par3.axis["right2"] = par3.new_fixed_axis(loc="right", offset=(0, 0))
    par3.axis["right2"].major_ticklabels.set_visible(True)
    par3.axis["right2"].set_visible(True)

    (base_index, _, _, _, _) = get_base_index(df)

    host.set(
        xlim=[115, 172.5],
        ylim=[-50, 1250],
        xlabel=r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$",
        ylabel="Generation capacity in GW (dashed)",
    )

    wind_on_indices = ["onwind"]
    wind_off_indices = ["offwind-ac", "offwind-dc", "offwind-float"]
    solar_indices = ["solar", "solar rooftop", "solar-hsat"]

    (wind_on,) = host.plot(
        df.loc[base_index, "pc total annual cost"],
        df.loc[base_index, wind_on_indices].sum(axis=1),
        ls="--",
        marker="s",
        c=tech_colors["onwind"],
    )

    (wind_off,) = host.plot(
        df.loc[base_index, "pc total annual cost"],
        df.loc[base_index, wind_off_indices].sum(axis=1),
        ls="--",
        marker="s",
        c=tech_colors["offwind"],
    )

    (solar,) = host.plot(
        df.loc[base_index, "pc total annual cost"],
        df.loc[base_index, solar_indices].sum(axis=1),
        ls="--",
        marker="s",
        c=tech_colors["solar"],
    )

    (photocatalysis,) = host.plot(
        df.loc[base_index, "pc total annual cost"],
        df.loc[base_index, "photocatalysis"],
        ls="--",
        marker="s",
        c=tech_colors["photocatalysis"],
    )

    (electrolysis,) = host.plot(
        df.loc[base_index, "pc total annual cost"],
        df.loc[base_index, "H2 Electrolysis"],
        ls="--",
        marker="s",
        c=tech_colors["H2 Electrolysis"],
    )

    (h2_gen,) = par3.plot(
        df.loc[base_index, "pc total annual cost"],
        df.loc[base_index, ["photocatalysis (Energy)", "H2 Electrolysis (Energy)"]].sum(
            axis=1
        ),
        ls="-",
        marker="o",
        c=tech_colors["hydrogen"],
    )

    (el_gen,) = par3.plot(
        df.loc[base_index, "pc total annual cost"],
        df.loc[base_index, "total power generation (Energy)"],
        ls="-",
        marker="o",
        c=tech_colors["electricity"],
    )

    host.grid()

    host.legend(
        handles=[
            wind_on,
            wind_off,
            solar,
            photocatalysis,
            electrolysis,
            h2_gen,
            el_gen,
        ],
        labels=[
            "Wind (onshore)",
            "Wind (offshore)",
            "Solar",
            "Photocatalysis",
            "Electrolysis",
            "Hydrogen generation",
            "Electricity generation",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.59),
        ncol=3,
        frameon=False,
        fancybox=False,
    )
    host.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    scaling_fac_par1 = (
        df.loc[base_index, "pc/pv cost relation"]
        / df.loc[base_index, "pc total annual cost"]
    ).mean()
    scaling_fac_par2 = (
        df.loc[base_index, "pc/el cost relation"]
        / df.loc[base_index, "pc total annual cost"]
    ).mean()

    par1.set(
        xlim=[
            host.get_xlim()[0] * scaling_fac_par1,
            host.get_xlim()[1] * scaling_fac_par1,
        ],
        xlabel="Annualized cost relation photocatalysis to photovoltaics",
    )

    par2.set(
        xlim=[
            host.get_xlim()[0] * scaling_fac_par2,
            host.get_xlim()[1] * scaling_fac_par2,
        ],
        xlabel="Annualized cost relation photocatalysis to electrolysis",
    )

    par3.set(
        ylabel=r"Energy generation in TWh a$^{-1}$ (solid)",
        ylim=[-275, 6875],
        yticks=np.arange(0, 6800.1, 1100),
    )
    par3.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    # Add description for PC-0 and PC-50 case
    max_x_val = max(df.loc[base_index, "pc total annual cost"])
    min_x_val = min(df.loc[base_index, "pc total annual cost"])
    rectangle_extent = 2
    rectangle_heigth = 1280
    y_text_position = 1.05  # relative to axis
    for tick_value, text_label in zip([max_x_val, min_x_val], ["PC-0", "PC-50"]):
        host.add_patch(
            plt.Rectangle(
                (tick_value - (rectangle_extent / 2), -40),
                rectangle_extent,
                rectangle_heigth,
                ls=":",
                lw=2,
                ec="k",
                fc="none",
            )
        )
        host.annotate(
            text_label,
            xy=(tick_value, 1240),  # Arrow target position (on x-axis)
            xytext=(tick_value, y_text_position),  # Text position (above x-axis)
            textcoords=("data", "axes fraction"),  # Mix of data and fraction coords
            arrowprops=dict(arrowstyle="-", facecolor="black", linewidth=2),
            ha="center",
        )

    if invert_axis:
        host.invert_xaxis()
        par1.invert_xaxis()
        par2.invert_xaxis()

    save_or_plot_img(
        fig,
        output_folder_path,
        "overall_system_development_over_annualized_PC_cost",
        save=save,
    )


#### ---- EL vs Mix


def plot_electrolysis_only_vs_balanced_mix_map(
    n_el, n_mix, regions, tech_colors, output_folder_path, save=False
):
    projection = ccrs.EqualEarth()

    nrows = 1
    ncols = 2
    fig, axes = plt.subplots(
        figsize=set_size(use=True, fraction=2, subplots=(nrows, ncols), scale=1.7),
        ncols=ncols,
        nrows=nrows,
        subplot_kw={"projection": projection},
    )

    network_map = {
        "el": {"n": n_el, "ax": axes[0], "title": "PC-0 case"},
        "mix": {"n": n_mix, "ax": axes[1], "title": "PC-50 case"},
    }

    # cmap = "Blues"
    # cmap = "cmc.devon_r"
    cmap = "cmc.oslo_r"
    vmax = 3
    vmin = 0

    for idx, vals in enumerate(network_map.values()):
        (
            regions,
            bus_sizes,
            link_widths_total,
            link_widths_retro,
            bus_size_factor,
            linewidth_factor,
            new_network,
        ) = extract_bus_information.collect_bus_sizes(vals["n"], regions, projection)

        map_opts = {
            "boundaries": [-11, 30, 34, 71],
            "color_geomap": {"ocean": "white", "land": "whitesmoke"},
        }

        # Buses (Generators) and H2 pipelines
        vals["n"].plot(
            geomap=True,
            bus_sizes=bus_sizes,
            bus_colors=tech_colors,
            link_colors=tech_colors["H2 pipeline (total)"],
            link_widths=link_widths_total,
            branch_components=["Link"],
            ax=vals["ax"],
            **map_opts,
        )

        # Retrofitted pipelines (H2)
        vals["n"].plot(
            geomap=True,
            bus_sizes=0,
            link_colors=tech_colors["H2 pipeline (repurposed)"],
            link_widths=link_widths_retro,
            branch_components=["Link"],
            ax=vals["ax"],
            **map_opts,
        )

        # Hydrogen Storage regions
        regions.plot(
            ax=vals["ax"],
            column="H2",
            cmap=cmap,
            linewidths=0,
            vmax=vmax,
            vmin=vmin,
        )

        vals["ax"].set_facecolor("white")
        vals["ax"].set_title(vals["title"], fontsize=16, fontweight="bold")
        # Add gridlines with conditional label drawing
        gl = vals["ax"].gridlines(draw_labels=True)
        if idx == 0:
            gl.right_labels = True  # Disable right labels for the left plot
        elif idx == 1:
            gl.left_labels = False  # Disable left labels for the right plot

        # Change tick label color to soft gray
        gl.xlabel_style = {"color": "gray"}
        gl.ylabel_style = {"color": "gray"}

    # define a mappable based on which the colorbar will be drawn
    mappable = cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)

    bbox = axes[1].get_position()
    a, b, c, d = bbox.bounds

    # define position and extent of colorbar
    #                     x_pos y_pos dx  dy
    cb_ax = fig.add_axes([0.95, b + b / 10, 0.014, d - b / 5])

    # draw colorbar
    cbar = fig.colorbar(mappable, cax=cb_ax, orientation="vertical")

    # Add a label to the colorbar
    cbar.set_label("Hydrogen storage in TWh", labelpad=10)

    # Bubble legend for generator size
    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, -0.05),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    plot_hydrogen_network.add_legend_circles(
        axes[0],
        sizes,
        labels,
        srid=n_el.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    # Legend for pipeline size
    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.25, -0.05),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    plot_hydrogen_network.add_legend_lines(
        axes[0],
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )

    carriers = [
        carrier
        for carrier in tech_colors.keys()
        if carrier
        not in [
            "H2 Fuel Cell",
            "H2 turbine",
            "onwind",
            "offwind",
            "hydrogen",
            "electricity",
            "solar",
        ]
    ]
    colors = [tech_colors[c] for c in carriers]

    labels = carriers.copy()
    labels[1] = "H2 Photocatalysis"

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.54, -0.05),
        ncol=2,
        frameon=False,
    )

    plot_hydrogen_network.add_legend_patches(
        axes[0], colors, labels, legend_kw=legend_kw
    )

    fig.subplots_adjust(wspace=+0.1)

    save_or_plot_img(fig, output_folder_path, "el_vs_mix", save=save)


### Levelized cost of hydrogen violins
def prepare_LCOH_violin_data(
    df: pd.DataFrame,
    index: pd.Index,
):
    """
    Function for preparing LCOH data to fit pd.DataFrame structure required for
    sns.violinplot.

    Parameters
    ----------
    df : pd.DataFrame
        summary DataFrame
    index : pd.Index
        Index values to filter the summary DataFrame df
    """

    for ind_iterator, ind in enumerate(index):
        x_data_value = df.loc[ind, "pc total annual cost"]
        nodal_capacity_string = df.loc[ind, "nodal capacity node:el:pc"]
        nodal_LCOH_string = df.loc[ind, "nodal LCOH node:el:pc"]

        nodes_cap = []
        el_cap = []
        pc_cap = []

        for nodal_capacity_string_splitted in nodal_capacity_string.split(";")[:-1]:
            nodal_capacity = nodal_capacity_string_splitted.split(":")

            nodes_cap.append(nodal_capacity[0])
            el_cap.append(float(nodal_capacity[1]))
            pc_cap.append(float(nodal_capacity[2]))

        nodes_LCOH = []
        el_LCOH = []
        pc_LCOH = []

        for nodal_LCOH_string_splitted in nodal_LCOH_string.split(";")[:-1]:
            nodal_LCOH = nodal_LCOH_string_splitted.split(":")

            nodes_LCOH.append(nodal_LCOH[0])
            el_LCOH.append(float(nodal_LCOH[1]))
            pc_LCOH.append(float(nodal_LCOH[2]))

        if not nodes_cap == nodes_LCOH:
            raise Exception(
                "Nodes differ between LCOH and capacity data.\n"
                + "Please check indices (e.g. node-count, order)!"
            )

        capacity_df = pd.DataFrame(
            np.ones((len(el_cap) + len(pc_cap), 4)) * np.nan,
            columns=[
                r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$",
                "Capacity",
                r"LCOH in € kg$^{-1}$",
                "Technology",
            ],
        )
        capacity_df[
            r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$"
        ] = capacity_df[
            r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$"
        ].astype(
            np.float64
        )
        capacity_df["Capacity"] = capacity_df["Capacity"].astype(np.float64)
        capacity_df[r"LCOH in € kg$^{-1}$"] = capacity_df[
            r"LCOH in € kg$^{-1}$"
        ].astype(np.float64)
        capacity_df["Technology"] = capacity_df["Technology"].astype(str)

        capacity_df.loc[
            :, r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$"
        ] = np.ones((len(el_cap) + len(pc_cap), 1)) * x_data_value.astype(
            np.float64
        ).round(
            1
        )
        capacity_df.loc[: len(el_cap) - 1, "Capacity"] = el_cap
        capacity_df.loc[: len(el_LCOH) - 1, r"LCOH in € kg$^{-1}$"] = el_LCOH
        capacity_df.loc[: len(el_cap) - 1, "Technology"] = "Electrolysis"

        capacity_df.loc[len(el_cap) :, "Capacity"] = pc_cap
        capacity_df.loc[len(el_cap) :, r"LCOH in € kg$^{-1}$"] = pc_LCOH
        capacity_df.loc[len(el_cap) :, "Technology"] = "Photocatalysis"

        if ind_iterator == 0:
            total_capacity_df = capacity_df
        else:
            total_capacity_df = pd.concat(
                [total_capacity_df, capacity_df], ignore_index=True
            )

    return total_capacity_df


def plot_LCOH_violins_of_electrolysis_and_photocatalysis(
    df,
    output_folder_path,
    tech_colors,
    UGHS_line_color,
    represent_regarding_market_share,
    line_cost_or_market_share,
    save=False,
):
    # get nodal data
    base_index, _, _, _, _ = get_base_index(df)

    total_LCOH_df = prepare_LCOH_violin_data(
        df=df,
        index=base_index,
    )

    if represent_regarding_market_share:
        total_LCOH_df[
            r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$"
        ] = total_LCOH_df[
            r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$"
        ].map(
            lambda val: float(
                photocatalysis_market_share_mapping_STH_10_UGHS.get(str(val), str(val))
            )
        )

    clip_capacity = 10
    total_LCOH_df = total_LCOH_df.drop(
        total_LCOH_df[total_LCOH_df.Capacity < clip_capacity].index
    )

    fig, ax = plt.subplots(figsize=set_size(use=True, fraction=1, scale=1))

    sns.violinplot(
        data=total_LCOH_df,
        x=r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$",
        y=r"LCOH in € kg$^{-1}$",
        hue="Technology",
        palette={
            "Electrolysis": tech_colors["H2 Electrolysis"],
            "Photocatalysis": tech_colors["photocatalysis"],
        },
        split=True,
        gap=0.0,
        density_norm="width",
        bw_method="silverman",
        # width=1.6,
        width=9 if represent_regarding_market_share else 1.6,
        inner="quart",
        ax=ax,
        native_scale=True,
    )
    if represent_regarding_market_share:
        ax.set_xlabel("Photocatalysis market share in \%")

    if line_cost_or_market_share:
        ax2 = ax.twinx()

        if represent_regarding_market_share:
            ax2_x = df.loc[base_index, "pc market share"]
            ax2_y = df.loc[base_index, "pc total annual cost"]
            ax2_label = r"Annualized photocatalysis cost"
        else:
            ax2_x = df.loc[base_index, "pc total annual cost"]
            ax2_y = df.loc[base_index, "pc market share"]
            ax2_label = "Market share"

        ax2.plot(
            ax2_x,
            ax2_y,
            ls="--",
            marker="s",
            c=UGHS_line_color,
            alpha=0.7,
            label=ax2_label,
        )

    ### Add description for PC-0 and PC-50 cases
    if represent_regarding_market_share:
        label_pos = [0, 51.2]
    else:
        label_pos = [168.7, 118.5]
    for x_val, lab in zip(label_pos, ["PC-0", "PC-50"]):
        ax.annotate(
            lab,
            xy=(x_val, 3.7),  # Arrow target position (on x-axis)
            xytext=(x_val, 1.05),  # Text position (above x-axis)
            textcoords=("data", "axes fraction"),  # Mix of data and fraction coords
            arrowprops=dict(arrowstyle="-", facecolor="black", linewidth=2),
            ha="center",
        )

    if represent_regarding_market_share == False:
        ax.invert_xaxis()
    ax.set_ylim([1.95, 3.80])
    ax.grid(alpha=0.4)
    ax.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    if line_cost_or_market_share:
        if represent_regarding_market_share:
            ax2.set_ylim([108, 182])
            ax2.set_ylabel("Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$")
            ax2_legend = ["Annualized photocatalysis cost"]
        else:
            ax2.set_ylim([-2, 72])
            ax2.set_ylabel("Photocatalysis market share in \%")
            ax2_legend = ["Market share"]
        ax2_handles, _ = ax2.get_legend_handles_labels()
    else:
        ax2_handles = []
        ax2_legend = []

    ax_handles, _ = ax.get_legend_handles_labels()

    ax.legend(
        ax_handles + ax2_handles,
        (
            ["Electrolysis", "Photocatalysis"] + ax2_legend
        ),
        loc="center",
        bbox_to_anchor=(0.5, -0.18),
        ncols=3,
        frameon=False,
        fancybox=False,
    )

    save_or_plot_img(fig, output_folder_path, "LCOH", save=save)


def hydrogen_generation_and_demand_violins_vertical(
    output_folder_path, represent_regarding_market_share, line_cost_or_market_share, save=False
):
    import hydrogen_gen_and_dem_violins

    (
        total_summary_df,
        generation_violin_split_df_vertical,
        demand_violin_df_vertical,
        node_density_and_land_density_vertical,
        generation_violin_split_df_horizontal,
        demand_violin_df_horizontal,
        node_density_and_land_density_horizontal,
        base_index,
        colors_dict,
    ) = hydrogen_gen_and_dem_violins.prepare_all_data()

    if represent_regarding_market_share:
        generation_violin_split_df_vertical[
            r"PC capex"
        ] = generation_violin_split_df_vertical[r"PC capex"].map(
            lambda val: float(
                photocatalysis_market_share_mapping_STH_10_UGHS.get(str(val), str(val))
            )
        )
        demand_violin_df_vertical[r"PC capex"] = demand_violin_df_vertical[
            r"PC capex"
        ].map(
            lambda val: float(
                photocatalysis_market_share_mapping_STH_10_UGHS.get(str(val), str(val))
            )
        )

    background_map = True
    violin_width = 1

    nrows = 1
    ncols = 2
    fig, (ax1, ax2) = plt.subplots(
        figsize=set_size(use=True, fraction=2, subplots=(nrows, ncols)),
        ncols=ncols,
        nrows=nrows,
        width_ratios=[0.75, 0.25],
    )

    # Violin for Electrolysis H2 generation (aim to have it on the left side)
    sns.violinplot(
        data=generation_violin_split_df_vertical[
            (generation_violin_split_df_vertical["case"] == "Electrolysis")
            & (generation_violin_split_df_vertical["ughs_state"] == "UGHS")
        ],
        x="PC capex",
        y="latitude",
        # Create only left side
        hue=True,
        hue_order=[True, False] if represent_regarding_market_share else [False, True],
        split=True,
        palette=[colors_dict["tech_colors"]["H2 Electrolysis"]] * 2,
        dodge=True,
        orient="v",
        inner="quart",  # quartiles of the data
        bw_method="silverman",
        density_norm="count",
        common_norm=True,
        fill=True,
        ax=ax1,
        label=r"Hydrogen generation by electrolysis",
        legend=False,
        native_scale=True,
        width=violin_width,
        zorder=-3,
    )

    # Violin for Photocatalysis H2 generation (aim to have it on the left side)
    sns.violinplot(
        data=generation_violin_split_df_vertical[
            (generation_violin_split_df_vertical["case"] == "Photocatalysis")
            & (generation_violin_split_df_vertical["ughs_state"] == "UGHS")
        ],
        x="PC capex",
        y="latitude",
        # Create only left side
        hue=True,
        hue_order=[True, False] if represent_regarding_market_share else [False, True],
        split=True,
        palette=[colors_dict["tech_colors"]["photocatalysis"]] * 2,
        dodge=True,
        orient="v",
        inner="quart",  # quartiles of the data
        density_norm="count",
        bw_method="silverman",
        common_norm=True,
        fill=True,
        ax=ax1,
        label=r"Hydrogen generation by photocatalysis",
        legend=False,
        native_scale=True,
        width=violin_width,
        zorder=-2,
    )

    if represent_regarding_market_share:
        ax1.set_xlim([1.8, 54.9])
        ax1_pos = ax1.get_position()
        ax1_2 = fig.add_axes(
            [ax1_pos.x0 + 0.0102, ax1_pos.y0, ax1_pos.width, ax1_pos.height],
            frame_on=False,
        )
    else:
        ax1.set_xlim([115, 163])
        ax1_pos = ax1.get_position()
        ax1_2 = fig.add_axes(
            [ax1_pos.x0 + 0.0127, ax1_pos.y0, ax1_pos.width, ax1_pos.height],
            frame_on=False,
        )

    sns.violinplot(
        data=demand_violin_df_vertical[(demand_violin_df_vertical["case"] == "UGHS")],
        x="PC capex",
        y="latitude",
        hue=True,
        hue_order=[False, True] if represent_regarding_market_share else [True, False],
        palette=[colors_dict["tech_colors"]["hydrogen"]] * 2,
        split=True,  # Does not work as expected!
        orient="v",
        inner="quart",  # quartiles of the data
        bw_method="silverman",
        gap=0.5,
        density_norm="count",
        common_norm=True,
        fill=True,
        legend=False,
        ax=ax1_2,
        label=r"Hydrogen demand",
        native_scale=True,
        width=violin_width,
        zorder=-1,
    )

    ax1_2.axis(False)

    if represent_regarding_market_share:
        label_x = 51.2
    else:
        label_x = 118.4
    #  Add description for PC-50 case
    ax1.annotate(
        "PC-50",
        xy=(label_x, 69.5),  # Arrow target position (on x-axis)
        xytext=(label_x, 1.05),  # Text position (above x-axis)
        textcoords=("data", "axes fraction"),  # Mix of data and fraction coords
        arrowprops=dict(arrowstyle="-", facecolor="black", linewidth=2),
        ha="center",
    )

    # --- Add market share line
    if line_cost_or_market_share:
        ax1_y2 = ax1.twinx()

        if represent_regarding_market_share:
            ax2_x = total_summary_df.loc[base_index, "pc market share"]
            ax2_y = total_summary_df.loc[base_index, "pc total annual cost"]
            ax2_label = "Annualized photocatalysis cost"
        else:
            ax2_x = total_summary_df.loc[base_index, "pc total annual cost"]
            ax2_y = total_summary_df.loc[base_index, "pc market share"]
            ax2_label = "Market share"

        ax1_y2.plot(
            ax2_x,
            ax2_y,
            ls="--",
            marker="s",
            c=colors_dict["others"]["UGHS_line"],
            alpha=0.7,
            label=ax2_label,
        )

        if represent_regarding_market_share:
            # ax1_y2.set_ylim([])
            ax1_y2.set_ylabel("Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$")
        else:
            ax1_y2.set_ylim([0, 55])
            ax1_y2.set_ylabel(r"Photocatalysis market share in $\mathrm{\%}$")
        # --- Add market share line

    #### Create plot to the right with node density and land area density
    sns.violinplot(
        data=node_density_and_land_density_vertical,
        x="x",
        y="Latitude",
        hue="hue",
        palette=[
            colors_dict["others"]["node density"],
            colors_dict["others"]["land area density"],
        ],
        split=True,
        orient="v",
        inner="quart",  # quartiles of the data
        density_norm="count",
        bw_method="silverman",
        fill=True,
        ax=ax2,
        zorder=-2,
    )

    # Add descriptive annotations with arrows
    df_for_description = node_density_and_land_density_vertical[
        node_density_and_land_density_vertical["hue"] == "Node density"
    ]["Latitude"]
    q1 = df_for_description.quantile(0.25)
    median = df_for_description.median()
    q3 = df_for_description.quantile(0.75)

    ax2.annotate(
        "Quartile 1",
        xy=(-0.1, q1),
        xytext=(-0.45, 37),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    ax2.annotate(
        "Median",
        xy=(-0.2, median),
        xytext=(-0.49, 42),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    ax2.annotate(
        "Quartile 3",
        xy=(-0.08, q3),
        xytext=(-0.49, 57),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    ax2.annotate(
        "Kernel\ndensity",
        xy=(-0.065, 60),
        xytext=(-0.45, 62),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )

    if represent_regarding_market_share:
        ax1.set_xlabel("Photocatalysis market share in \%")
    else:
        ax1.set_xlabel(r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$")
    ax1.set_ylabel("Latitude in °N")
    ax2.set_ylabel("Latitude in °N")
    ax2.set_xlabel("PyPSA-Eur 150 nodes")

    # Gather handles and labels from both axes
    handles, labels = ax1.get_legend_handles_labels()  # Supply EL & PC
    handles2, labels2 = ax1_2.get_legend_handles_labels()  # H2 Demand
    if line_cost_or_market_share:
        handles3, labels3 = ax1_y2.get_legend_handles_labels()  # Market share
    else: 
        handles3 = labels3 = []

    # Combine and deduplicate
    unique_labels = dict(zip(labels + labels2 + labels3, handles + handles2 + handles3))

    ax1.set_ylim([35, 70])
    ax1_2.set_ylim([35, 70])
    ax2.set_ylim([35, 70])
    # ax1_2.set_xlim(ax1.get_xlim())
    # print(ax1_2)
    # # TODO: When changing aspect ratio or distance between subplots the first value here has to be adjusted to correctly show the demand violins
    if represent_regarding_market_share:
        ax1_2.set_xlim((1.69, 55.1))
    else:
        ax1_2.set_xlim((114.7, 163.0))

    # Create the legend below the plot
    ax1.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    sns.move_legend(
        ax2,
        "upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1,
        title=None,
        frameon=False,
    )

    ax1.grid(alpha=0.3, zorder=20)
    ax2.grid(alpha=0.3, zorder=20)
    ax1_2.yaxis.grid(True, alpha=0.3, zorder=20)

    # Redraw lines on top to have a grid also on the green violins
    gridlines = ax1.yaxis.get_gridlines()
    gridline_color = gridlines[0].get_color()
    gridline_linestyle = gridlines[0].get_linestyle()
    gridline_linewidth = gridlines[0].get_linewidth()

    for y in ax1.get_yticks():  # Adjust range and step based on your y-axis limits
        ax1_2.axhline(
            y=y,
            color=gridline_color,
            xmin=0,
            xmax=0.95,
            linestyle=gridline_linestyle,
            linewidth=gridline_linewidth,
            zorder=10,
            alpha=0.3,
        )

    if represent_regarding_market_share == False:
        ax1.invert_xaxis()
        ax1_2.invert_xaxis()

    ax1.tick_params(
        axis="both",
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax2.tick_params(
        axis="both",
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
    )

    # TODO: When changing this the first value in ax1_2.set_xlim() has to be changed!
    plt.subplots_adjust(wspace=0.215)

    if background_map:
        proj_kwargs = {"name": "EqualEarth"}

        proj_func = getattr(ccrs, proj_kwargs.pop("name"))
        proj = proj_func(**proj_kwargs)

        regions = gpd.read_file(
            hydrogen_gen_and_dem_violins.REGIONS_GEOJSON_PATH
        ).set_index("name")
        regions = regions.to_crs(proj.proj4_init)

        ax1_map = fig.add_axes(
            ax1_pos, projection=proj, zorder=-10
        )  # Create overlay axes

        # Plot the map onto the overlay axes
        regions.plot(
            ax=ax1_map, linewidth=1, color="lightgrey", edgecolor="white", alpha=0.4
        )
        ax1_map.set_extent([-9.5, 26.5, 35, 70])  # Set the extent of the map
        ax1_map.axis("off")  # Turn off the axis to show only the map

        # Adjust ax1 to be transparent or semi-transparent to show the map underneath
        ax1.set_facecolor("none")  # Make the background of ax1

    save_or_plot_img(fig, output_folder_path, "gen_dem_violin", save=save)


def plot_installed_capacities_bars(
    output_folder_path, represent_regarding_market_share, save=False
):
    # represent_regarding_market_share == True has market shares as reference values
    # represent_regarding_market_share == False has annualized photocatalysis costs as reference values

    from installed_capacities_bars import (
        add_stacking_legend,
        plot_capacity_data_both,
        prepare_bar_plot_data,
    )

    (
        capex_annualized_cost_map,
        df_p_nom_max,
        joined_df_reordered,
        colors_pc,
        colors_el,
        regions,
        pc_regions,
        el_regions,
        proj,
    ) = prepare_bar_plot_data()

    with open(r"./src/colors.yaml") as stream:
        tech_colors = yaml.safe_load(stream)["tech_colors"]

    with open("./src/colors.yaml") as stream:
        h2_storage_color = yaml.safe_load(stream)["others"]["h2_storage"]

    xmax_el = 60
    xmax_pc = 95

    # Set up figure with two parts
    fig = plt.figure(figsize=set_size(use=True, fraction=2, scale=2.4))
    width_ratio = [xmax_el, xmax_pc]  # Set split manually to correspond ax limits
    gs = fig.add_gridspec(1, 2, width_ratios=width_ratio, wspace=0.03)
    (ax1, ax2) = gs.subplots(sharey="row")

    plot_capacity_data_both(
        ax2,
        ax1,
        joined_df_reordered,
        colors_pc,
        colors_el,
        df_p_nom_max,
        capex_annualized_cost_map,
    )

    #  Configure legend positions (and reverse the legend order to have bar on top, on top of the legend), labels and limits
    handles, labels = ax1.get_legend_handles_labels()
    if represent_regarding_market_share:
        labels = [
            photocatalysis_market_share_mapping_STH_10_UGHS.get(val, val)
            for val in labels
        ]
    l1 = ax1.legend(handles[::-1], labels[::-1], loc="lower left")
    # l1 = ax1.legend(loc="lower left")
    handles, labels = ax2.get_legend_handles_labels()
    if represent_regarding_market_share:
        labels = [
            photocatalysis_market_share_mapping_STH_10_UGHS.get(val, val)
            for val in labels
        ]
    l2 = ax2.legend(handles[::-1], labels[::-1], loc="center right")
    if represent_regarding_market_share:
        l1.set_title("Photocatalysis\nmarket share\nin $\%$")
        l2.set_title("Photocatalysis\nmarket share\nin $\%$")
    else:
        l1.set_title("Annualized\nphotocatalysis\ncost in\n€ a$^{-1}$ kW$^{-1}$")
        l2.set_title("Annualized\nphotocatalysis\ncost in\n€ a$^{-1}$ kW$^{-1}$")
    ax1.set_xlabel("Installed electrolysis capacity (GW)")
    ax2.set_xlabel("Installed photocatalysis capacity (GW)")
    ax1.set_ylim(-0.5, len(joined_df_reordered.index) - 0.5)
    ax1.set_xlim(0, xmax_el)
    ax1.invert_xaxis()
    ax2.set_xlim(0, xmax_pc)
    ax1.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    ax2.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Highlight nodes where salt cavern storage can be build
    salt_cavern_regions = pd.read_csv(
        r"./data/salt_cavern_potentials_s_150.csv", index_col=0
    )
    highlighted_categories = [
        node for node in joined_df_reordered.index if node in salt_cavern_regions.index
    ]
    for label in ax1.get_yticklabels():
        if label.get_text() in highlighted_categories:
            label.set_color(h2_storage_color)

    #### Add Overlay axis to show bar overlay legend

    x0, y0, width, height = (
        0.445,
        0.52,
        0.3,
        0.06,
    )  # Define position and size of the overlay ax
    # Add an overlay axes for the map

    ax = fig.add_axes([x0, y0, width, height], anchor="SE", zorder=40)

    ax.patch.set_alpha(1.0)
    ax.set_facecolor("white")

    add_stacking_legend(
        ax,
        y_label_adjustment=0.2,
        title_adjustment=21,
        colors_pc=colors_pc,
        colors_el=colors_el,
    )

    #### Add legend image / map (if boxplot visuals are changed the image needs to be adjusted/recreated)

    # Positioning of the image in relative coordinates (x0, y0, width, height)
    # x0, y0 lower-left corner of the image box; width, height size of the image box
    x0, y0, width, height = (
        0.445,
        0.62,
        0.45,
        0.25,
    )  # Define position and size of the overlay ax
    # Add an overlay axes for the map

    ax_overlay = fig.add_axes(
        [x0, y0, width, height], anchor="SE", projection=proj, zorder=50
    )
    ax_overlay.patch.set_alpha(1.0)
    ax_overlay.set_facecolor("white")

    # map_font_size = 7

    # Base plot
    regions.plot(
        ax=ax_overlay, linewidth=1, color="lightgrey", edgecolor="white", alpha=0.4
    )

    # Photocatalysis regions
    pc_regions.plot(
        ax=ax_overlay,
        linewidths=1,
        legend=True,
        color=tech_colors["photocatalysis"],
        edgecolor="white",
        alpha=0.3,
    )
    # Electrolysis regions
    el_regions.plot(
        ax=ax_overlay,
        linewidths=1,
        legend=True,
        color=tech_colors["H2 Electrolysis"],
        edgecolor="white",
        alpha=0.3,
    )

    # Define annotation parameters for each zone
    node_annotations_pc = {
        "FR1 11": (600000, 5350000),
        "FR1 1": (354949, 5361033),
        "ES1 2": (180937, 5161688),
        "ES1 4": (-325042.19509547454, 4654292.406069289),
        "IT1 11": (1354146, 4643111),
        "IT1 0": (1131597, 4662965),
    }

    # Common annotation parameters
    common_params = {
        # "fontsize": map_font_size,
        "color": "black",
        "ha": "center",
        "va": "center",
    }

    # Annotate each node with its bus name (for pc_regions)
    for idx, row in pc_regions.iterrows():
        centroid = row["geometry"].centroid
        # print(idx, centroid.x, centroid.y)
        xytext = node_annotations_pc.get(idx, (centroid.x, centroid.y))
        ax_overlay.annotate(
            str(idx), xy=(centroid.x, centroid.y), xytext=xytext, **common_params
        )

    node_annotations_el = {
        "DK2 0": (895544, 6511730),
        "DK1 0": (709430, 6588491),
        "DE1 5": (685047, 6275656),
        "NL1 0": (342397, 6268247),
        "NL1 4": (554864, 6202336),
    }

    # Annotate each node with its bus name (for el_regions)
    for idx, row in el_regions.iterrows():
        if idx == "FR1 1":
            continue
        centroid = row["geometry"].centroid
        # print(idx, centroid.x, centroid.y)
        xytext = node_annotations_el.get(idx, (centroid.x, centroid.y))
        ax_overlay.annotate(
            str(idx), xy=(centroid.x, centroid.y), xytext=xytext, **common_params
        )

    # ax.get_extent()
    ax_overlay.set_extent([-9.5, 26.5, 34.7, 62])

    gl = ax_overlay.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)

    # Control which labels should be drawn
    gl.left_labels = True
    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = True

    # Setting the property for spacing or clipping
    gl.xlabel_style = {
        # "size": map_font_size,
        "color": "k",
        "clip_on": True,
    }
    gl.ylabel_style = {
        # "size": map_font_size,
        "color": "k",
        "clip_on": True,
    }

    save_or_plot_img(fig, output_folder_path, "capacity_per_cost_bar", save=save)


def plot_system_cost_differences_bars(
    n_el, n_pc, output_folder_path, scale, save=False
):
    from system_cost_differences import (
        calc_energy_box_values,
        calc_hydrogen_box,
        calc_power_and_h2_system_box_values,
        calc_syn_fuel_box_values,
    )

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

    # fmt: off
    layout = [
        ["A","A","A","A","A","A","A","A","A","A",],
        ["C","C","C","C","C","C","C","C","C","C",],
        ["B","B","B",".",".","D","D","D",".","E",],
    ]
    # fmt: on

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
        ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
        )

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
    nrows = 3
    ncols = 7
    fig, ax_dict = plt.subplot_mosaic(
        layout,
        figsize=set_size(use=True, subplots=(nrows, ncols), fraction=1, scale=scale),
    )

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
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=False,
        fancybox=False,
    )

    plt.subplots_adjust(left=0, bottom=0.07, right=1, top=0.9, wspace=0.0, hspace=0.28)

    ### Add explanatory legend plot
    # Define bounding position and remove the placeholder axis
    bounds_orig = ax_dict["E"].get_position(fig).bounds
    bounds = (0.925, 0.085, 0.15, 0.18)
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
        # (bounds[0] - 0.015, bounds[1] - 0.081),
        (bounds[0] - 0.015, bounds[1] - 0.071),
        bounds[2] + 0.04,
        bounds[-1] + 0.124,
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="black",
        fill=False,
    )
    fig.patches.append(rect)

    save_or_plot_img(fig, output_folder_path, "system_cost_difference", save=save)


def main():
    mpl.rcParams.update(pgf_with_latex)
    filepath = os.path.join(r"./results/total_summary.csv")

    # Load dataframe but skip the unit column
    total_summary_df = pd.read_csv(
        filepath, index_col=0, header=0, skiprows=range(1, 2)
    )

    folder_path = r"results/raw"
    file_el_only = r"150_lv1.25_I_H_2045_3H_PC_925Euro/elec_s_150_lv1.25__I-H_2045.nc"
    file_mix = r"150_lv1.25_I_H_2045_3H_PC_650Euro/elec_s_150_lv1.25__I-H_2045.nc"
    n_el = pypsa.Network(os.path.join(folder_path, file_el_only))
    n_mix = pypsa.Network(os.path.join(folder_path, file_mix))
    regions = gpd.read_file("data/regions_onshore_elec_s_150.geojson").set_index("name")

    with open(r"src/colors.yaml") as stream:
        all_colors = yaml.safe_load(stream)
        tech_colors = all_colors["tech_colors"]
        noUGHS_line_color = all_colors["others"]["no_UGHS_line"]
        UGHS_line_color = all_colors["others"]["UGHS_line"]
        colors_dict = all_colors["others"]

    output_folder_path = r"./img/"

    ### Small plots (1 column)

    market_share(
        total_summary_df,
        output_folder_path,
        UGHS_line_color,
        noUGHS_line_color,
        save=True,
        all=False,  # "false" to plot only 10% case
    )

    system_development(total_summary_df, output_folder_path, tech_colors, save=True)

    plot_LCOH_violins_of_electrolysis_and_photocatalysis(
        total_summary_df,
        output_folder_path,
        tech_colors,
        UGHS_line_color,
        represent_regarding_market_share=True,
        line_cost_or_market_share=False,
        save=True,
    )

    # Manually slightly changed in inkscape
    plot_system_cost_differences_bars(
        n_el, n_mix, output_folder_path, save=True, scale=5
    )

    ### Large plots (2 columns)

    # TODO: fix that saved figure and shown figure look the same.
    plot_electrolysis_only_vs_balanced_mix_map(
        n_el, n_mix, regions, tech_colors, output_folder_path, save=True
    )

    hydrogen_generation_and_demand_violins_vertical(
        output_folder_path, represent_regarding_market_share=True,line_cost_or_market_share=False, save=True
    )

    # Manually slightly changed in inkscape
    plot_installed_capacities_bars(output_folder_path, represent_regarding_market_share=True,save=True)


if __name__ == "__main__":
    main()
