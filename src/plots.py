# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2025 Wolfram Tuschewitzki, Jelto Lange
#
# SPDX-License-Identifier: MIT
import os

import cartopy.crs as ccrs
import cmcrameri  # To register colormaps for matplotlib e.g. cmc.batlow
import geopandas as gpd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
import yaml
from matplotlib import legend_handler, transforms
from matplotlib.path import Path
from mpl_toolkits.axisartist.axislines import AxesZero
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

import extract_bus_information as extract_bus_information
from beeswarm import (
    CONSTANT_SIZE,
    VIOLIN_ALPHA,
    WIDTH_FRAC,
    add_beeswarm,
    darken,
    sizes_from_values,
    violin_half_band,
)
from cases import FIGURE_FOLDER, REPO_ROOT, figure_formats
from create_basic_summary import get_base_index, load_df_without_unit_row, parse_nodal
from weighted_stats import interpolated_weighted_quantiles

# Anchored to this module rather than to the working directory, so the plotting
# functions can be called from any working directory.
COLORS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colors.yaml")

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


def get_pgf_with_latex(
    scale_factor=1.0,
    title_scale=1.0,
    annotation_scale=1.0,
    axis_scale=1.0,
    legend_scale=1.0,
):
    """
    `pgf_with_latex` with 8.8 pt text and 7.5 pt legend and ticks, times
    `scale_factor` (the figures use FIGURE_FONT_SCALE).

    The other factors scale single groups further (the figures printed small
    use TEXT_SCALE):
    title_scale: axes titles
    annotation_scale: font.size, i.e. free text such as annotations; cartopy's
        gridline labels take it, too
    axis_scale: axis labels and tick labels, colorbars included
    legend_scale: legend entries and legend titles
    """
    base_axes_labelsize = 8.8
    base_font_size = 8.8
    base_axes_titlesize = 8.8
    base_legend_fontsize = 7.5
    base_xtick_labelsize = 7.5
    base_ytick_labelsize = 7.5

    return {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["times"],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": base_axes_labelsize * scale_factor * axis_scale,
        "font.size": base_font_size * scale_factor * annotation_scale,
        "axes.titlesize": base_axes_titlesize * scale_factor * title_scale,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.fontsize": base_legend_fontsize * scale_factor * legend_scale,
        # matplotlib's default for an unset title size is font.size
        "legend.title_fontsize": base_font_size * scale_factor * legend_scale,
        "xtick.labelsize": base_xtick_labelsize * scale_factor * axis_scale,
        "ytick.labelsize": base_ytick_labelsize * scale_factor * axis_scale,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "pgf.preamble": pgf_with_latex["pgf.preamble"],
        "text.latex.preamble": pgf_with_latex["text.latex.preamble"],
    }


# Font scale of all figures, the figure size stays the same
FIGURE_FONT_SCALE = 1.15

# Further scale of the text groups of get_pgf_with_latex() in the figures that
# are printed small. Those figures set their sizes themselves (rc_context as
# decorator, the save included), so the annex versions get the sizes of the
# main figure whatever the caller set
TEXT_SCALE = 1.5


# Annualized photocatalysis cost -> market share of the base sweep (10 % STH,
# UGHS), both as strings of the 1-decimal rounded values. Regenerate with
#   df = load_df_without_unit_row("results/total_summary.csv")
#   cols = ["pc total annual cost", "pc market share"]
#   rounded = df.loc[get_base_index(df)[0], cols].astype(float).round(1)
#   dict(zip(rounded[cols[0]].map("{:.1f}".format), rounded[cols[1]].map("{:.1f}".format)))
photocatalysis_market_share_mapping_STH_10_UGHS = {
    "173.3": "0.0",
    "168.7": "4.6",
    "164.1": "5.8",
    "155.0": "16.7",
    "145.9": "28.3",
    "136.8": "35.5",
    "127.7": "43.6",
    "120.4": "49.8",
}


# Golden-ratio sizing adapted from Jack Walton, "Embedding Matplotlib figures in
# LaTeX documents": https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(
    use=True, fraction=1, subplots=(1, 1), scale=1, width_pt=None, height_pt=None
):
    """
    Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    use: bool, optional
            Height factor switch: True uses the plain golden ratio, False adds
            a further factor of 1.3. Ignored when `height_pt` is given.
    fraction: float, optional
            Fraction of the document width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots. Ignored when
            `height_pt` is given.
    scale: float, optional
            Multiplies the golden ratio. Ignored when `height_pt` is given.
    width_pt: float, optional
            Document width in points, default 468 pt, times `fraction`
    height_pt: float, optional
            Figure height in points, replaces the golden ratio height
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
    fig_width_pt = (468 if width_pt is None else width_pt) * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    if height_pt is not None:
        # explicit height, so no golden ratio and no subplot correction
        return (fig_width_in, height_pt * inches_per_pt)

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2 * scale

    if use == True:
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = fig_width_in * golden_ratio * 1.3 * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def save_or_plot_img(fig, output_folder_path, name, save=False, export_format="svg"):
    """
    Write `fig`, or show it when `save` is False.

    `export_format` can hold several formats, e.g. "svg,png"; the
    environment variable PC_FIGURE_FORMATS replaces it (cases.figure_formats).
    """
    if save:
        for fmt in figure_formats(export_format):
            savefig_kwargs = {"format": fmt, "bbox_inches": "tight"}
            if fmt == "tiff":
                # LZW is lossless compression
                savefig_kwargs["dpi"] = 600
                savefig_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(
                os.path.join(output_folder_path, f"{name}.{fmt}"), **savefig_kwargs
            )
    else:
        # tight_layout would move hand-placed overlay axes
        if all(ax.get_subplotspec() is not None for ax in fig.axes):
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
    export_format="svg",
    indices=None,
    xlim=(117, 178),
    ylim=(-1.0, 55),
    case_scenarios=(
        "150_lv1.25_I_H_2045_PC_950Euro",
        "150_lv1.25_I_H_2045_PC_660Euro",
    ),
    pc50_text_position=(130, 49.5),
    name=None,
):
    """
    Photocatalysis market share over the annualized photocatalysis cost.
    """

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

    if indices is None:
        base_index, noUGHS_index, pct6_index, pct3_index, *_ = get_base_index(df)
    else:
        base_index, noUGHS_index, pct6_index, pct3_index = indices

    host.set(
        xlim=list(xlim),
        ylim=list(ylim),
        xlabel=r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$",
        ylabel=r"Photocatalysis market share in \%",
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
        labels=(
            [
                r"$STH=10~\mathrm{\%}$",
                r"$STH=6~\mathrm{\%}$",
                r"$STH=3~\mathrm{\%}$",
                r"$STH=10~\mathrm{\%}$," "\nno UGHS",
            ]
            if all
            else [r"$STH=10~\mathrm{\%}$"]
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
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
    pc0_scenario, pc50_scenario = case_scenarios
    el_100_x = df.loc[pc0_scenario, "pc total annual cost"]
    el_100_y = df.loc[pc0_scenario, "pc market share"]
    balanced_mix_x = df.loc[pc50_scenario, "pc total annual cost"]
    balanced_mix_y = df.loc[pc50_scenario, "pc market share"]
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
        xytext=pc50_text_position,
        arrowprops=dict(arrowstyle="->", lw=1.5),
        ha="left",
    )

    if invert_axis:
        host.invert_xaxis()
        if para_axis:
            par1.invert_xaxis()
            par2.invert_xaxis()

    if name is None:
        name = (
            "market_share_over_annualized_PC_cost"
            if all
            else "market_share_over_annualized_PC_cost_only_10"
        )

    save_or_plot_img(
        fig,
        output_folder_path,
        name,
        save=save,
        export_format=export_format,
    )


def system_development(
    df,
    output_folder_path,
    tech_colors,
    save=False,
    invert_axis=True,
    export_format="svg",
    index=None,
    xlim=(117, 178),
    capacity_ylim=(-100, 2500),
    energy_ylim=(-400, 10000),
    energy_ytick_step=2000,
    name="overall_system_development_over_annualized_PC_cost",
):
    """
    Capacities and annual energy of the base sweep over the annualized
    photocatalysis cost.

    The two y axes keep a ratio of 4 to share one grid.
    """

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

    if index is None:
        base_index, *_ = get_base_index(df)
    else:
        base_index = index

    host.set(
        xlim=list(xlim),
        ylim=list(capacity_ylim),
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
        ylim=list(energy_ylim),
        yticks=np.arange(0, energy_ylim[1], energy_ytick_step),
    )
    par3.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    # Add description for PC-0 and PC-50 case
    max_x_val = max(df.loc[base_index, "pc total annual cost"])
    min_x_val = min(df.loc[base_index, "pc total annual cost"])
    rectangle_extent = 2
    rectangle_bottom = capacity_ylim[0] + 10
    rectangle_top = capacity_ylim[1] - 20
    rectangle_heigth = rectangle_top - rectangle_bottom
    y_text_position = 1.05  # relative to axis
    for tick_value, text_label in zip([max_x_val, min_x_val], ["PC-0", "PC-50"]):
        host.add_patch(
            plt.Rectangle(
                (tick_value - (rectangle_extent / 2), rectangle_bottom),
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
            xy=(tick_value, rectangle_top),  # Arrow target position (on x-axis)
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
        name,
        save=save,
        export_format=export_format,
    )


#### ---- EL vs Mix


# font.size unscaled: the gridline labels take it and need the room
@mpl.rc_context(
    get_pgf_with_latex(
        FIGURE_FONT_SCALE,
        title_scale=TEXT_SCALE,
        axis_scale=TEXT_SCALE,
        legend_scale=TEXT_SCALE,
    )
)
def plot_electrolysis_only_vs_balanced_mix_map(
    n_el,
    n_mix,
    regions,
    tech_colors,
    output_folder_path,
    save=False,
    export_format="svg",
    titles=("PC-0 case", "PC-50 case"),
    name="el_vs_mix",
):
    """
    The two-panel hydrogen production and infrastructure map.

    The circle and line handles of the legends keep their map scale.
    """

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
        "el": {"n": n_el, "ax": axes[0], "title": titles[0]},
        "mix": {"n": n_mix, "ax": axes[1], "title": titles[1]},
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
        vals["ax"].set_title(vals["title"], fontweight="bold")
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

    add_legend_circles(
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
        bbox_to_anchor=(0.23, -0.05),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    add_legend_lines(
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
        bbox_to_anchor=(0.50, -0.05),
        ncol=2,
        frameon=False,
    )

    add_legend_patches(axes[0], colors, labels, legend_kw=legend_kw)

    fig.subplots_adjust(wspace=+0.1)

    save_or_plot_img(
        fig, output_folder_path, name, save=save, export_format=export_format
    )


def ccs_market_share_scatter(
    df,
    output_folder_path,
    base_index,
    ccs_indices,
    case_scenarios,
    save=False,
    export_format="svg",
    name="CCS_scatter",
    pc0_text_offset=(-15, 1.5),
    pc50_text_position=(135, 49),
):
    """
    Photocatalysis market share at the PC-0 and PC-50 cost points for each CO2
    sequestration limit.

    Parameters
    ----------
    base_index : pd.Index
        PC-0 and PC-50 scenario
    ccs_indices : dict
        sequestration limit -> index of its one or two scenarios
    case_scenarios : (str, str)
        PC-0 and PC-50 scenario names for the annotations
    """
    series = [
        ("600", "#3D1D2E", "x", 120),
        ("500", "#582945", "x", 120),
        ("400", "#733957", "+", 120),
        ("300", "#A3672C", "x", 120),
        ("200 (base)", "#D6D893", "o", 30),
        ("100", "#74BBCD", "x", 120),
        ("50", "#5C538B", "x", 120),
    ]
    series = [s for s in series if s[0].startswith("200") or s[0] in ccs_indices]

    # legend in the empty upper left corner (high share at high cost)
    fig, ax = plt.subplots(figsize=(4.0, 3.5))

    for label, color, marker, size in series:
        index = base_index if label.startswith("200") else ccs_indices[label]
        ax.scatter(
            df.loc[index, "pc total annual cost"],
            df.loc[index, "pc market share"],
            c=color,
            marker=marker,
            s=size,
            label=label,
        )

    pc0_scenario, pc50_scenario = case_scenarios
    el_100_x = df.loc[pc0_scenario, "pc total annual cost"]
    el_100_y = df.loc[pc0_scenario, "pc market share"]
    balanced_mix_x = df.loc[pc50_scenario, "pc total annual cost"]
    balanced_mix_y = df.loc[pc50_scenario, "pc market share"]
    ax.annotate(
        "PC-0",
        xy=(el_100_x, el_100_y),
        xytext=(el_100_x + pc0_text_offset[0], el_100_y + pc0_text_offset[1]),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        ha="center",
    )
    ax.annotate(
        "PC-50",
        xy=(balanced_mix_x, balanced_mix_y),
        xytext=pc50_text_position,
        arrowprops=dict(arrowstyle="->", lw=1.5),
        ha="left",
    )

    ax.set_xlabel("Annualized photocatalysis cost\nin € a$^{-1}$ kW$^{-1}$")
    ax.set_ylabel(r"Photocatalysis market share in \%")
    ax.legend(
        title="CO$_2$ sequestration\npotential in Mt/a",
        loc="upper left",
    )
    ax.invert_xaxis()
    ax.grid()
    ax.tick_params(
        axis="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    save_or_plot_img(
        fig, output_folder_path, name, save=save, export_format=export_format
    )


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
        capacity = parse_nodal(df.loc[ind, "nodal capacity node:el:pc"], ["EL", "PC"])
        lcoh = parse_nodal(df.loc[ind, "nodal LCOH node:el:pc"], ["EL", "PC"])
        el_cap, pc_cap = capacity["EL"].tolist(), capacity["PC"].tolist()
        el_LCOH, pc_LCOH = lcoh["EL"].tolist(), lcoh["PC"].tolist()

        if not capacity.index.equals(lcoh.index):
            raise Exception(
                "Nodes differ between LCOH and capacity data.\n"
                + "Please check indices (e.g. node-count, order)!"
            )

        capacity_df = pd.DataFrame(
            np.ones((len(el_cap) + len(pc_cap), 4)) * np.nan,
            columns=[
                r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$",
                "Capacity",
                r"$LCOH$ in € kg$^{-1}$",
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
        capacity_df[r"$LCOH$ in € kg$^{-1}$"] = capacity_df[
            r"$LCOH$ in € kg$^{-1}$"
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
        capacity_df.loc[: len(el_LCOH) - 1, r"$LCOH$ in € kg$^{-1}$"] = el_LCOH
        capacity_df.loc[: len(el_cap) - 1, "Technology"] = "Electrolysis"

        capacity_df.loc[len(el_cap) :, "Capacity"] = pc_cap
        capacity_df.loc[len(el_cap) :, r"$LCOH$ in € kg$^{-1}$"] = pc_LCOH
        capacity_df.loc[len(el_cap) :, "Technology"] = "Photocatalysis"

        if ind_iterator == 0:
            total_capacity_df = capacity_df
        else:
            total_capacity_df = pd.concat(
                [total_capacity_df, capacity_df], ignore_index=True
            )

    return total_capacity_df


def remove_legend_patch_edges(legend):
    """
    Drop the outline from the patch keys of `legend`.
    """
    if legend is None:
        return
    for handle in legend.legend_handles:
        if handle is not None and hasattr(handle, "set_edgecolor"):
            handle.set_edgecolor("none")
            handle.set_linewidth(0)


@mpl.rc_context(
    get_pgf_with_latex(
        FIGURE_FONT_SCALE,
        title_scale=TEXT_SCALE,
        annotation_scale=TEXT_SCALE,
        axis_scale=TEXT_SCALE,
        legend_scale=TEXT_SCALE,
    )
)
def plot_LCOH_violins_of_electrolysis_and_photocatalysis(
    df,
    output_folder_path,
    tech_colors,
    UGHS_line_color,
    represent_regarding_market_share,
    line_cost_or_market_share,
    violin_edge=False,
    beeswarm=False,
    save=False,
    export_format="svg",
    violin_width=None,
    case_label_positions=None,
    swarm_skip_scenarios=(4.6, 168.7),
    name=None,
    ylim=(1.95, 3.95),
):
    """
    Split violins of the nodal LCOH of electrolysis and photocatalysis.

    Parameters
    ----------
    swarm_skip_scenarios : tuple of float
        scenarios without beeswarm dots (the 4.6 % violin is too close to its
        neighbour), in both axis units
    name : str, optional
        file name, default "LCOH" or "LCOH_beeswarm"
    ylim : (float, float)
        LCOH axis limits
    case_label_positions : (float, float), optional
        positions of the "PC-0" and "PC-50" ticks in axis units, default the
        base sweep
    violin_width : float, optional
        width of one violin pair in axis units, default 7 (market share) or
        1.6 (cost)
    beeswarm : bool
        add one dot per node, sized by its installed capacity
    """
    # get nodal data
    base_index, *_ = get_base_index(df)

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

    clip_capacity = 1
    total_LCOH_df = total_LCOH_df.drop(
        total_LCOH_df[total_LCOH_df.Capacity < clip_capacity].index
    )

    scenario_column = r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$"
    LCOH_column = r"$LCOH$ in € kg$^{-1}$"
    palette = {
        "Electrolysis": tech_colors["H2 Electrolysis"],
        "Photocatalysis": tech_colors["photocatalysis"],
    }
    if violin_width is None:
        violin_width = 7 if represent_regarding_market_share else 1.6

    fig, ax = plt.subplots(figsize=set_size(use=True, fraction=1, scale=1))

    sns.violinplot(
        data=total_LCOH_df,
        x=scenario_column,
        y=LCOH_column,
        hue="Technology",
        palette=palette,
        split=True,
        gap=0.0,
        density_norm="width",
        bw_method="silverman",
        # width=1.6,
        width=violin_width,
        inner="quart",
        ax=ax,
        native_scale=True,
    )
    # the swarm below adds collections of its own, so the violin bodies are kept
    # apart here rather than looked up as ax.collections later on
    violin_bodies = list(ax.collections)

    if beeswarm:
        for violin_body in violin_bodies:
            violin_body.set_alpha(VIOLIN_ALPHA)

        # the violins grow out of the tick, so the swarm gets the same band
        swarm_width = WIDTH_FRAC * violin_half_band(
            total_LCOH_df[scenario_column], violin_width
        )
        # nodes without the technology carry no LCOH and never reach the violins
        swarm_source = total_LCOH_df.dropna(subset=["Capacity", LCOH_column]).copy()
        # one mapping for the whole figure, so a dot of a given size always
        # means the same installed capacity
        swarm_source["size"] = sizes_from_values(swarm_source["Capacity"])

        # Electrolysis is hue level 0 and lands left of the tick
        for technology, data_sign in [("Electrolysis", -1), ("Photocatalysis", +1)]:
            for scenario, group in swarm_source[
                swarm_source["Technology"] == technology
            ].groupby(scenario_column):
                if scenario in swarm_skip_scenarios:
                    continue
                add_beeswarm(
                    ax,
                    scenario,
                    group[LCOH_column],
                    [palette[technology]] * len(group),
                    group["size"].to_numpy(),
                    swarm_width,
                    data_sign=data_sign,
                )

    if represent_regarding_market_share:
        ax.set_xlabel(r"Photocatalysis market share in \%")

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

    if violin_edge == False:
        # Remove edgecolor (outline) from violin bodies
        for violin_body in violin_bodies:
            violin_body.set_edgecolor("none")

    ### Add description for PC-0 and PC-50 cases
    if case_label_positions is not None:
        label_pos = list(case_label_positions)
    elif represent_regarding_market_share:
        label_pos = [0.0, 49.8]
    else:
        label_pos = [173.3, 120.4]
    # the tick reaches from just below the top of the axis up to the label
    arrow_target = ylim[0] + 0.946 * (ylim[1] - ylim[0])
    for x_val, lab in zip(label_pos, ["PC-0", "PC-50"]):
        ax.annotate(
            lab,
            xy=(x_val, arrow_target),  # Arrow target position (on x-axis)
            xytext=(x_val, 1.05),  # Text position (above x-axis)
            textcoords=("data", "axes fraction"),  # Mix of data and fraction coords
            arrowprops=dict(arrowstyle="-", facecolor="black", linewidth=2),
            ha="center",
        )

    if represent_regarding_market_share == False:
        ax.invert_xaxis()
    ax.set_ylim(list(ylim))
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
            ax2.set_ylabel(r"Photocatalysis market share in \%")
            ax2_legend = ["Market share"]
        ax2_handles, _ = ax2.get_legend_handles_labels()
    else:
        ax2_handles = []
        ax2_legend = []

    ax_handles, _ = ax.get_legend_handles_labels()

    legend = ax.legend(
        ax_handles + ax2_handles,
        (["Electrolysis", "Photocatalysis"] + ax2_legend),
        loc="center",
        bbox_to_anchor=(0.5, -0.22),
        ncols=3,
        frameon=False,
        fancybox=False,
    )
    if violin_edge == False:
        remove_legend_patch_edges(legend)

    save_or_plot_img(
        fig,
        output_folder_path,
        name if name is not None else ("LCOH_beeswarm" if beeswarm else "LCOH"),
        save=save,
        export_format=export_format,
    )


def violin_call_widths(
    generation_df,
    demand_df,
    width_normalization="per_call",
    violin_width=1.0,
    generation_scale=1.0,
):
    """
    `width=` of the three violinplot calls of the generation/demand figure.

    With density_norm="count" every call is scaled to its own largest violin,
    so the widths of the three calls are not comparable without this.

    Parameters
    ----------
    width_normalization : {"per_call", "generation", "global"}
        "per_call" each call on its own scale, "generation" electrolysis and
        photocatalysis on one scale, "global" all three on one scale
    generation_scale : float
        extra factor on the two generation calls
    """
    max_counts = {
        case: generation_df[generation_df["case"] == case]
        .groupby("PC capex")
        .size()
        .max()
        for case in ("Electrolysis", "Photocatalysis")
    }
    max_counts["Demand"] = demand_df.groupby("PC capex").size().max()

    if width_normalization == "per_call":
        rulers = dict(max_counts)
    elif width_normalization == "generation":
        generation_ruler = max(max_counts["Electrolysis"], max_counts["Photocatalysis"])
        rulers = {
            "Electrolysis": generation_ruler,
            "Photocatalysis": generation_ruler,
            "Demand": max_counts["Demand"],
        }
    elif width_normalization == "global":
        rulers = {case: max(max_counts.values()) for case in max_counts}
    else:
        raise ValueError(
            f"Invalid width_normalization '{width_normalization}', expected "
            "'per_call', 'generation' or 'global'."
        )

    widths = {
        case: violin_width * max_counts[case] / rulers[case] for case in max_counts
    }
    for case in ("Electrolysis", "Photocatalysis"):
        widths[case] *= generation_scale
    return widths


def draw_stacked_gen_dem_violins(
    ax,
    swarm_df,
    coordinate_column,
    coordinate_lim,
    series_colors,
    generation_sign,
    vertical,
    bandwidth,
    width_factor,
    violin_width,
    violin_edge,
):
    """
    Stacked generation half (EL + PC) and demand half of the generation/demand
    violins on one width scale. Drawn by hand from volume_profiles(), since
    seaborn scales every violin to its own peak.

    Returns
    -------
    bodies : list of PolyCollection
        The drawn halves, in the order they were added.
    scale : float
        data units per TWh/a per degree
    generation_peak, demand_peak : dict
        scenario -> peak of that half in TWh/a per degree.
    positions : list
        The scenario coordinates, ascending.
    """
    import hydrogen_gen_and_dem_violins

    grid = np.linspace(*coordinate_lim, 700)
    profiles = hydrogen_gen_and_dem_violins.volume_profiles(
        swarm_df, grid, bandwidth, coordinate_column
    )
    positions = sorted(swarm_df["PC capex"].unique())

    generation_peak = {
        position: (
            profiles[(position, "Electrolysis")]
            + profiles[(position, "Photocatalysis")]
        ).max()
        for position in positions
    }
    demand_peak = {
        position: profiles[(position, "Demand")].max() for position in positions
    }
    widest = max(max(generation_peak.values()), max(demand_peak.values()))
    # at width_factor 1 the widest half fills the band between two scenarios
    scale = width_factor * violin_half_band(positions, violin_width) / widest

    edge = dict(edgecolor="0.25", linewidth=0.5) if violin_edge else dict(linewidth=0)
    fill_between = ax.fill_betweenx if vertical else ax.fill_between

    bodies = []
    for position in positions:
        electrolysis = profiles[(position, "Electrolysis")] * scale
        photocatalysis = profiles[(position, "Photocatalysis")] * scale
        demand = profiles[(position, "Demand")] * scale
        tick = np.full_like(grid, position)

        # generation on one side with photocatalysis stacked outside
        # electrolysis, demand on the other
        layers = [
            ("Electrolysis", tick, position + generation_sign * electrolysis),
            (
                "Photocatalysis",
                position + generation_sign * electrolysis,
                position + generation_sign * (electrolysis + photocatalysis),
            ),
            ("Demand", tick, position - generation_sign * demand),
        ]
        for series, inner, outer in layers:
            bodies.append(
                fill_between(
                    grid,
                    inner,
                    outer,
                    color=series_colors[series],
                    zorder=-3,
                    label=(
                        (
                            f"Hydrogen generation by {series.lower()}"
                            if series != "Demand"
                            else "Hydrogen demand"
                        )
                        if position == positions[0]
                        else None
                    ),
                    **edge,
                )
            )

            # quartile lines, each across the layer it belongs to
            group = swarm_df[
                (swarm_df["PC capex"] == position) & (swarm_df["series"] == series)
            ]
            if group.empty:
                continue
            quartiles = interpolated_weighted_quantiles(
                group[coordinate_column], group["n_rows"], [0.25, 0.5, 0.75]
            )
            for quartile, style in zip(quartiles, [":", "--", ":"]):
                span = [np.interp(quartile, grid, edge_) for edge_ in (inner, outer)]
                ax.plot(
                    span if vertical else [quartile, quartile],
                    [quartile, quartile] if vertical else span,
                    ls=style,
                    lw=0.8,
                    color="0.25",
                    zorder=-2,
                )

    return bodies, scale, generation_peak, demand_peak, positions


def add_stacked_width_reference(
    ax, scale, widest, bandwidth, vertical, coordinate_unit
):
    """
    Reference bar for the width of the stacked violins in TWh/a per degree.
    """
    reference = float(
        min([1, 2, 5, 10, 20, 25, 50, 100], key=lambda step: abs(step - widest / 3))
    )
    span = abs(np.diff(ax.get_xlim() if vertical else ax.get_ylim())[0])
    length = reference * scale / span

    anchor, thickness = 0.02, 0.012
    bar = (
        ([anchor, anchor + length], [0.965, 0.965])
        if vertical
        else ([0.035, 0.035], [anchor, anchor + length])
    )
    ax.plot(*bar, color="black", lw=1.0, solid_capstyle="butt", transform=ax.transAxes)
    for end in (0, 1):
        cap = (
            ([bar[0][end]] * 2, [0.965 - thickness, 0.965 + thickness])
            if vertical
            else ([0.035 - thickness, 0.035 + thickness], [bar[1][end]] * 2)
        )
        ax.plot(*cap, color="black", lw=1.0, transform=ax.transAxes)

    # kernel smoothed, so only approximate
    ax.annotate(
        "width of both halves: $\\approx%g~\\mathrm{TWh\\,a^{-1}}$ per %s\n"
        r"(kernel-smoothed over $\pm%.1f$°, so approximate)"
        % (reference, coordinate_unit, 2 * bandwidth),
        xy=(anchor, 0.955) if vertical else (0.06, anchor + length),
        xycoords="axes fraction",
        va="top",
        ha="left",
        linespacing=1.3,
        # not a legend, so without TEXT_SCALE: it sits inside the axes
        fontsize=get_pgf_with_latex(FIGURE_FONT_SCALE)["legend.fontsize"],
        zorder=5,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
    )


@mpl.rc_context(
    get_pgf_with_latex(
        FIGURE_FONT_SCALE,
        title_scale=TEXT_SCALE,
        annotation_scale=TEXT_SCALE,
        axis_scale=TEXT_SCALE,
        legend_scale=TEXT_SCALE,
    )
)
def hydrogen_generation_and_demand_violins(
    output_folder_path,
    represent_regarding_market_share,
    line_cost_or_market_share,
    orientation="vertical",
    violin_edge=False,
    beeswarm=False,
    stacked=True,
    stacked_bandwidth=0.4,
    stacked_width_factor=1.08,
    width_normalization="generation",
    generation_scale=1.25,
    save=False,
    export_format="svg",
    pc50_label_position=None,
    name=None,
):
    """
    Split violins of the nodal hydrogen generation and demand per scenario.

    Parameters
    ----------
    name : str, optional
        file name, default "gen_dem_violin" (+ "_horizontal", "_beeswarm")
    pc50_label_position : float, optional
        position of the "PC-50" tick in axis units, default the base sweep
    output_folder_path : str
        output folder
    represent_regarding_market_share : bool
        market share (True) or annualized cost (False) on the scenario axis
    line_cost_or_market_share : bool
        add the other quantity as a line on a twin axis
    orientation : {"vertical", "horizontal"}
        latitude (vertical) or longitude (horizontal) on the value axis
    violin_edge : bool
        keep the violin outlines
    beeswarm : bool
        add one dot per node, sized by its hydrogen volume
    stacked : bool
        stack photocatalysis on top of electrolysis so both halves share one
        width scale, False overlays them with seaborn
    stacked_bandwidth : float
        kernel bandwidth in degrees for `stacked`
    stacked_width_factor : float
        scales all stacked violins, above ~1.15 the 43.6 % and 49.8 % violins touch
    width_normalization, generation_scale
        see violin_call_widths(), ignored when `stacked`
    """
    import hydrogen_gen_and_dem_violins

    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"Invalid orientation '{orientation}', expected 'vertical' or 'horizontal'."
        )
    vertical = orientation == "vertical"

    (
        total_summary_df,
        generation_violin_split_df,
        demand_violin_df,
        node_density_and_land_density,
        capex_annualized_cost_map,
        base_index,
        colors_dict,
    ) = hydrogen_gen_and_dem_violins.prepare_data(orientation)

    # one row per node, for the swarm and the stacked violins
    swarm_df = (
        hydrogen_gen_and_dem_violins.prepare_swarm_data(
            capex_annualized_cost_map, orientation
        )
        if beeswarm or stacked
        else None
    )

    if represent_regarding_market_share:

        def to_market_share(val):
            return float(
                photocatalysis_market_share_mapping_STH_10_UGHS.get(str(val), str(val))
            )

        generation_violin_split_df[r"PC capex"] = generation_violin_split_df[
            r"PC capex"
        ].map(to_market_share)
        demand_violin_df[r"PC capex"] = demand_violin_df[r"PC capex"].map(
            to_market_share
        )
        if swarm_df is not None:
            swarm_df[r"PC capex"] = swarm_df[r"PC capex"].map(to_market_share)

    background_map = True
    violin_width = 1

    # hue_order picks the side of the tick, the cost axis is inverted
    if vertical:
        generation_first = represent_regarding_market_share
    else:
        generation_first = not represent_regarding_market_share
    generation_hue_order = [True, False] if generation_first else [False, True]
    demand_hue_order = generation_hue_order[::-1]

    # Scenario axis (PC capex resp. its market share) and geographic value axis.
    # (only the unstacked path keeps these; the stacked one follows its violins)
    scenario_lim = [2.5, 53.5] if represent_regarding_market_share else [117, 167.5]
    scenario_label = (
        r"Photocatalysis market share in \%"
        if represent_regarding_market_share
        else r"Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$"
    )

    if vertical:
        violin_x, violin_y = "PC capex", "latitude"
        violin_orient = "v"
        density_x, density_y, density_coordinate = "x", "Latitude", "Latitude"
        coordinate_lim = [35, 70]
        coordinate_label = "Latitude in °N"
        nrows, ncols = 1, 2
        fig, (ax1, ax2) = plt.subplots(
            figsize=set_size(use=True, fraction=2, subplots=(nrows, ncols)),
            ncols=ncols,
            nrows=nrows,
            width_ratios=[0.82, 0.18],
        )
    else:
        violin_x, violin_y = "longitude", "PC capex"
        violin_orient = "h"
        density_x, density_y, density_coordinate = "longitude", "y", "longitude"
        # matches get_density_data()'s horizontal binning range
        coordinate_lim = [-11, 32]
        coordinate_label = "Longitude in °E"
        nrows, ncols = 2, 1
        fig, (ax1, ax2) = plt.subplots(
            figsize=set_size(fraction=2),
            ncols=ncols,
            nrows=nrows,
            height_ratios=[0.8, 0.2],
        )

    coordinate_column = "latitude" if vertical else "longitude"
    series_colors = {
        "Electrolysis": colors_dict["tech_colors"]["H2 Electrolysis"],
        "Photocatalysis": colors_dict["tech_colors"]["photocatalysis"],
        "Demand": colors_dict["tech_colors"]["hydrogen"],
    }
    # generation sits on the hue level that lands left of / below the tick
    generation_sign = -1 if generation_first else +1

    if stacked:
        (
            ax1_violin_bodies,
            stacked_scale,
            generation_peak,
            demand_peak,
            scenario_positions,
        ) = draw_stacked_gen_dem_violins(
            ax1,
            swarm_df,
            coordinate_column,
            coordinate_lim,
            series_colors,
            generation_sign,
            vertical,
            stacked_bandwidth,
            stacked_width_factor,
            violin_width,
            violin_edge,
        )
        # axis limits follow the outermost violins plus a margin
        margin = 0.6
        scenario_lim = sorted(
            [
                scenario_positions[0]
                + generation_sign
                * (generation_peak[scenario_positions[0]] * stacked_scale + margin),
                scenario_positions[-1]
                - generation_sign
                * (demand_peak[scenario_positions[-1]] * stacked_scale + margin),
            ]
        )
    else:
        generation_ughs = generation_violin_split_df[
            generation_violin_split_df["ughs_state"] == "UGHS"
        ]
        demand_ughs = demand_violin_df[demand_violin_df["case"] == "UGHS"]
        case_widths = violin_call_widths(
            generation_ughs,
            demand_ughs,
            width_normalization,
            violin_width,
            generation_scale,
        )

        # Violins for the H2 generation (one half of the split, by technology)
        for case, zorder in [("Electrolysis", -3), ("Photocatalysis", -2)]:
            sns.violinplot(
                data=generation_ughs[generation_ughs["case"] == case],
                x=violin_x,
                y=violin_y,
                hue=True,
                hue_order=generation_hue_order,
                split=True,
                palette=[series_colors[case]] * 2,
                dodge=True,
                orient=violin_orient,
                inner="quart",  # quartiles of the data
                bw_method="silverman",
                density_norm="count",
                common_norm=True,
                fill=True,
                ax=ax1,
                label=f"Hydrogen generation by {case.lower()}",
                legend=False,
                native_scale=True,
                width=case_widths[case],
                zorder=zorder,
            )

        # Violin for the H2 demand (the other half of the split)
        sns.violinplot(
            data=demand_ughs,
            x=violin_x,
            y=violin_y,
            hue=True,
            hue_order=demand_hue_order,
            palette=[series_colors["Demand"]] * 2,
            split=True,
            # dodge=True + gap=0 instead of dodge="auto" (False for a constant
            # hue) + gap=0.5: same width, but anchored on the tick
            dodge=True,
            gap=0,
            orient=violin_orient,
            inner="quart",  # quartiles of the data
            bw_method="silverman",
            density_norm="count",
            common_norm=True,
            fill=True,
            legend=False,
            ax=ax1,
            label=r"Hydrogen demand",
            native_scale=True,
            width=case_widths["Demand"],
            zorder=-1,
        )
        # the swarm below adds collections of its own, so the violin bodies are
        # kept apart here rather than looked up as ax1.collections later on
        ax1_violin_bodies = list(ax1.collections)

    if vertical:
        ax1.set_xlim(scenario_lim)
        ax1.set_ylim(coordinate_lim)
    else:
        ax1.set_xlim(coordinate_lim)
        ax1.set_ylim(scenario_lim)

    if beeswarm:
        swarm_orient = "x" if vertical else "y"
        # one mapping for the whole figure, so a dot of a given size always
        # means the same hydrogen volume
        swarm_df["size"] = sizes_from_values(swarm_df["value_MWh"])
        # with `stacked` every half gets the band of its own violin
        figure_swarm_width = WIDTH_FRAC * violin_half_band(
            swarm_df["PC capex"], violin_width
        )

        for scenario, group in swarm_df.groupby("PC capex"):
            # electrolysis and photocatalysis share one half of the split, so
            # they are packed as a single swarm and cannot overlap each other
            generation = group[group["series"] != "Demand"]
            demand = group[group["series"] == "Demand"]
            for series, data_sign, peak in [
                (generation, generation_sign, generation_peak if stacked else None),
                (demand, -generation_sign, demand_peak if stacked else None),
            ]:
                if series.empty:
                    continue
                add_beeswarm(
                    ax1,
                    scenario,
                    series[coordinate_column],
                    [series_colors[name] for name in series["series"]],
                    series["size"].to_numpy(),
                    (
                        WIDTH_FRAC * peak[scenario] * stacked_scale
                        if stacked
                        else figure_swarm_width
                    ),
                    data_sign=data_sign,
                    orient=swarm_orient,
                )

    #  Add description for PC-50 case
    if pc50_label_position is None:
        label_position = 49.8 if represent_regarding_market_share else 120.4
    else:
        label_position = pc50_label_position
    if vertical:
        # text above the axes, line down to just inside the top
        annotation_position = dict(
            xy=(label_position, coordinate_lim[1] - 0.5),
            xytext=(label_position, 1.05),
            textcoords=("data", "axes fraction"),
            ha="center",
        )
    else:
        # text right of the axes, line left to just inside the right edge
        annotation_position = dict(
            xy=(coordinate_lim[1] - 0.5, label_position),
            xytext=(1.012, label_position),
            textcoords=("axes fraction", "data"),
            ha="left",
            va="center",
        )
    ax1.annotate(
        "PC-50",
        arrowprops=dict(arrowstyle="-", facecolor="black", linewidth=2),
        **annotation_position,
    )

    if stacked:
        add_stacked_width_reference(
            ax1,
            stacked_scale,
            max(max(generation_peak.values()), max(demand_peak.values())),
            stacked_bandwidth,
            vertical,
            "°N" if vertical else "°E",
        )

    # --- Add market share line
    if line_cost_or_market_share:
        market_share_values = total_summary_df.loc[base_index, "pc market share"]
        cost_values = total_summary_df.loc[base_index, "pc total annual cost"]

        if represent_regarding_market_share:
            scenario_values, line_values = market_share_values, cost_values
            line_label = "Annualized photocatalysis cost"
            line_axis_label = "Annualized photocatalysis cost in € a$^{-1}$ kW$^{-1}$"
            line_axis_lim = None
        else:
            scenario_values, line_values = cost_values, market_share_values
            line_label = "Market share"
            line_axis_label = r"Photocatalysis market share in $\mathrm{\%}$"
            line_axis_lim = [0, 55]

        # twinx/twiny share the scenario axis, so they follow its inversion
        ax1_y2 = ax1.twinx() if vertical else ax1.twiny()
        ax1_y2.plot(
            scenario_values if vertical else line_values,
            line_values if vertical else scenario_values,
            ls="--",
            marker="s",
            c=colors_dict["others"]["UGHS_line"],
            alpha=0.7,
            label=line_label,
        )

        if vertical:
            ax1_y2.set_ylabel(line_axis_label)
            if line_axis_lim is not None:
                ax1_y2.set_ylim(line_axis_lim)
        else:
            ax1_y2.set_xlabel(line_axis_label)
            if line_axis_lim is not None:
                ax1_y2.set_xlim(line_axis_lim)
        # --- Add market share line

    #### Create the second panel with node density and land area density
    sns.violinplot(
        data=node_density_and_land_density,
        x=density_x,
        y=density_y,
        hue="hue",
        palette=[
            colors_dict["others"]["node density"],
            colors_dict["others"]["land area density"],
        ],
        split=True,
        orient=violin_orient,
        inner="quart",  # quartiles of the data
        density_norm="count",
        bw_method="silverman",
        fill=True,
        ax=ax2,
        zorder=-2,
    )
    ax2_violin_bodies = list(ax2.collections)

    # Add descriptive annotations with arrows
    df_for_description = node_density_and_land_density[
        node_density_and_land_density["hue"] == "Node density"
    ][density_coordinate]
    q1 = df_for_description.quantile(0.25)
    median = df_for_description.median()
    q3 = df_for_description.quantile(0.75)

    if beeswarm:
        # One dot per node, centred on the split and growing to both sides.
        # Every node counts once here, so the dots are all the same size
        node_swarm_scale = 1.4
        # the scatter resets the axis limits, so they are put back afterwards
        category_lim = ax2.get_xlim() if vertical else ax2.get_ylim()
        add_beeswarm(
            ax2,
            0,  # ax2 holds a single category
            df_for_description,
            # darkened, the plain colour disappears on its own violin half
            [darken(colors_dict["others"]["node density"])] * len(df_for_description),
            np.full(len(df_for_description), CONSTANT_SIZE * node_swarm_scale),
            0.8 * node_swarm_scale,  # ax2 spans one categorical unit
            orient="x" if vertical else "y",
        )
        if vertical:
            ax2.set_xlim(category_lim)
        else:
            ax2.set_ylim(category_lim)

    if vertical:
        # (text, arrow target, text position); node density is the left half
        descriptions = [
            ("Quartile 1", (-0.1, q1), (-0.45, 37)),
            ("Median", (-0.2, median), (-0.49, 42)),
            ("Quartile 3", (-0.08, q3), (-0.49, 57)),
            ("Kernel\ndensity", (-0.065, 60), (-0.45, 62)),
        ]
        description_va = "baseline"
    else:
        # Same, rotated. seaborn inverts the categorical axis here, so node
        # density ends up on top and its offsets stay negative.
        descriptions = [
            ("Quartile 1", (q1, -0.1), (-8.0, -0.46)),
            ("Median", (median, -0.2), (0.5, -0.46)),
            ("Quartile 3", (q3, -0.08), (18.0, -0.46)),
            ("Kernel density", (22.0, -0.065), (25.0, -0.46)),
        ]
        description_va = "top"
    for text, arrow_target, text_position in descriptions:
        ax2.annotate(
            text,
            xy=arrow_target,
            xytext=text_position,
            va=description_va,
            # without TEXT_SCALE, the narrow panel has no room for more
            fontsize=get_pgf_with_latex(FIGURE_FONT_SCALE)["font.size"],
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

    if vertical:
        ax1.set_xlabel(scenario_label)
        ax1.set_ylabel(coordinate_label)
        ax2.set_xlabel("PyPSA-Eur 150 nodes")
        ax2.set_ylabel(coordinate_label)
        ax2.set_ylim(coordinate_lim)
    else:
        # ax2 sits directly underneath ax1 and shares its coordinate range, so
        # one label at the bottom serves both panels
        ax1.set_xlabel("")
        ax1.set_ylabel(scenario_label)
        ax2.set_xlabel(coordinate_label)
        ax2.set_ylabel("PyPSA-Eur\n150 nodes")
        ax2.set_xlim(coordinate_lim)

    # Gather handles and labels: all three violin labels now live on ax1
    handles, labels = ax1.get_legend_handles_labels()  # Supply EL & PC & demand
    if line_cost_or_market_share:
        handles2, labels2 = ax1_y2.get_legend_handles_labels()  # Market share
    else:
        handles2 = labels2 = []

    # Combine and deduplicate
    unique_labels = dict(zip(labels + labels2, handles + handles2))

    # Create the legends below the plots
    if vertical:
        ax1_legend = ax1.legend(
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
    else:
        # ax2 sits underneath ax1, so ax1's legend is anchored to ax2 to clear
        # it; one row each, which is what reads on a slide
        ax1_legend = ax1.legend(
            unique_labels.values(),
            unique_labels.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.55),
            bbox_transform=ax2.transAxes,
            ncol=len(unique_labels),
            frameon=False,
        )
        sns.move_legend(
            ax2,
            "upper center",
            bbox_to_anchor=(0.5, -0.95),
            ncol=2,
            title=None,
            frameon=False,
        )

    if violin_edge == False:
        # move_legend() rebuilds the legend, so ax2's keys are fetched from the
        # axes rather than kept from the call above
        remove_legend_patch_edges(ax1_legend)
        remove_legend_patch_edges(ax2.get_legend())

    # Every violin lives on ax1, so the axes grid (zorder 1.5) draws over
    # them by itself
    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)

    if represent_regarding_market_share == False:
        # annualized cost decreases towards higher market share
        if vertical:
            ax1.invert_xaxis()
        else:
            ax1.invert_yaxis()

    for ax in (ax1, ax2):
        ax.tick_params(
            axis="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )

    violin_bodies = ax1_violin_bodies + ax2_violin_bodies
    if violin_edge == False:
        # Remove edgecolor (outline) from violin bodies
        for violin_body in violin_bodies:
            violin_body.set_edgecolor("none")
    if not stacked:
        # overlaid violins need an alpha
        for violin_body in violin_bodies:
            violin_body.set_alpha(VIOLIN_ALPHA)

    if vertical:
        fig.subplots_adjust(wspace=0.1075)
    else:
        fig.subplots_adjust(hspace=0.06)

    if background_map:
        # the horizontal layout needs longitude linear
        proj_kwargs = {"name": "EqualEarth" if vertical else "PlateCarree"}

        proj_func = getattr(ccrs, proj_kwargs.pop("name"))
        proj = proj_func(**proj_kwargs)

        regions = gpd.read_file(
            hydrogen_gen_and_dem_violins.REGIONS_GEOJSON_PATH
        ).set_index("name")
        regions = regions.to_crs(proj.proj4_init)

        # Read the position again
        ax1_map = fig.add_axes(
            ax1.get_position(), projection=proj, zorder=-10
        )  # Create overlay axes

        # Plot the map onto the overlay axes
        regions.plot(
            ax=ax1_map, linewidth=1, color="lightgrey", edgecolor="white", alpha=0.4
        )
        # the map lines up with the geographic axis of the violins
        if vertical:
            ax1_map.set_extent([-9.5, 26.5, *coordinate_lim])  # extent of the map
        else:
            # latitude window from the box aspect, so the map is not distorted
            position = ax1.get_position()
            box_aspect = (position.height * fig.get_figheight()) / (
                position.width * fig.get_figwidth()
            )
            half_latitude = (coordinate_lim[1] - coordinate_lim[0]) * box_aspect / 2
            latitude_center = 52.5  # midpoint of the vertical layout's 35..70
            ax1_map.set_extent(
                [
                    *coordinate_lim,
                    latitude_center - half_latitude,
                    latitude_center + half_latitude,
                ]
            )  # extent of the map
            ax1_map.set_aspect("auto")
        ax1_map.axis("off")  # Turn off the axis to show only the map

        # Adjust ax1 to be transparent or semi-transparent to show the map underneath
        ax1.set_facecolor("none")  # Make the background of ax1

    if name is None:
        name = "gen_dem_violin" if vertical else "gen_dem_violin_horizontal"
        if beeswarm:
            name += "_beeswarm"
    save_or_plot_img(
        fig,
        output_folder_path,
        name,
        save=save,
        export_format=export_format,
    )


# --- nodal capacity bars (Fig. cap-per-cost-bar) --------------------------
CAPACITY_BARS_XMAX_EL = 70
CAPACITY_BARS_XMAX_PC = 95
CAPACITY_BARS_TICK_STEP = 20
CAPACITY_BARS_STACKING_LABEL_ADJUSTMENT = 0.05
CAPACITY_BARS_STACKING_TITLE_ADJUSTMENT = 12.5
# legends larger than the figure's legend sizes, the figure is printed small
CAPACITY_BARS_LEGEND_SCALE = 1.2
CAPACITY_BARS_FIGSIZE = (10, 15)


def plot_installed_capacities_bars(
    output_folder_path,
    represent_regarding_market_share,
    save=False,
    export_format="svg",
    name="capacity_per_cost_bar",
    ughs_style="column",
):
    """
    Nodal electrolysis/photocatalysis capacity bars with the inset map.

    represent_regarding_market_share: label the cost points by market share
    (True) or annualized cost (False).
    ughs_style: "column" or "label", see installed_capacities_bars.mark_ughs_nodes()
    """
    import installed_capacities_bars as bars

    # axis labels, tick labels and node names scaled, the legends have
    # CAPACITY_BARS_LEGEND_SCALE, the inset map its own fixed size
    with mpl.rc_context(get_pgf_with_latex(FIGURE_FONT_SCALE, axis_scale=TEXT_SCALE)):
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
        ) = bars.prepare_bar_plot_data()

        if represent_regarding_market_share:
            # the bar columns are keyed by CAPEX, so the shares are read per
            # scenario rather than through the cost -> share mapping
            total_summary_df = load_df_without_unit_row(bars.TOTAL_SUMMARY_FILE)
            label_map = {
                capex: round(
                    float(
                        total_summary_df.loc[
                            bars.BASE_SCENARIOS[int(capex)], "pc market share"
                        ]
                    ),
                    1,
                )
                for capex in capex_annualized_cost_map
            }
            legend_title = bars.MARKET_SHARE_LEGEND_TITLE
        else:
            label_map = capex_annualized_cost_map
            legend_title = bars.ANNUALIZED_COST_LEGEND_TITLE

        with open(COLORS_YAML) as stream:
            colors = yaml.safe_load(stream)

        fig = bars.plot_vertical(
            label_map,
            df_p_nom_max,
            joined_df_reordered,
            colors_pc,
            colors_el,
            regions,
            pc_regions,
            el_regions,
            proj,
            colors["tech_colors"],
            colors["others"]["h2_storage"],
            xmax_el=CAPACITY_BARS_XMAX_EL,
            xmax_pc=CAPACITY_BARS_XMAX_PC,
            legend_title=legend_title,
            tick_step=CAPACITY_BARS_TICK_STEP,
            stacking_label_adjustment=CAPACITY_BARS_STACKING_LABEL_ADJUSTMENT,
            stacking_title_adjustment=CAPACITY_BARS_STACKING_TITLE_ADJUSTMENT,
            ughs_style=ughs_style,
            legend_scale=CAPACITY_BARS_LEGEND_SCALE,
            figsize=CAPACITY_BARS_FIGSIZE,
        )
        save_or_plot_img(
            fig, output_folder_path, name, save=save, export_format=export_format
        )


# --- carbon-capture segments in the system cost difference bars ----------
# gas CHP and gas CHP CC share one bar: the CC part is stacked on top and is hatched.

CC_HATCH = "/////"
CC_EDGE = "black"
CC_TINT = 0.6
# applied by plot_system_cost_differences_bars() for the whole figure
CC_HATCH_LINEWIDTH = 0.5


def cc_tint(color, amount=CC_TINT):
    """
    Blend `color` towards white; 0 keeps it, 1 is white.
    """
    return tuple(
        channel + (1.0 - channel) * amount for channel in mcolors.to_rgb(color)
    )


def add_cc_segment(ax, x_values, values, bottom, width, color, label=None):
    """
    Stack the carbon capture part of a bar on top of its plain part.

    No outline: it would make the segment wider than the bar below it, the
    hatch keeps its color from the edge color. Zero rows are skipped. Returns
    None if no row has a carbon capture part.
    """
    values = np.asarray(values, dtype=float)
    present = values != 0
    if not present.any():
        return None
    return ax.bar(
        np.asarray(x_values)[present],
        values[present],
        width,
        bottom=np.asarray(bottom, dtype=float)[present],
        label=label,
        color=cc_tint(color),
        hatch=CC_HATCH,
        edgecolor=CC_EDGE,
        linewidth=0,
    )


def cc_legend_patch():
    """
    Legend key for the carbon capture segment, which comes in both bar colors.
    """
    return patches.Patch(
        facecolor="white", edgecolor=CC_EDGE, hatch=CC_HATCH, linewidth=0.4
    )


# --- by-product heat bar of the heating and power box --------------------
# Electrolysis is the solid part, all other sources one hatched segment: in
# most figures both move in the same direction, so the segments stack and need
# a legend key rather than a name at the end of the bar. Tinted like the carbon
# capture segment, with a hatch of its own.

BYPRODUCT_HATCH = "...."


def byproduct_heat_parts(segments):
    """
    The two drawn parts of the by-product heat bar: electrolysis, and all
    other sources netted into one segment.
    """
    electrolysis = float(segments.get("electrolysis", 0.0))
    others = float(segments.drop("electrolysis", errors="ignore").sum())
    return electrolysis, others


def byproduct_heat_ends(segments):
    """
    Lower and upper end of the by-product heat bar, zero included.
    """
    parts = np.array(byproduct_heat_parts(segments))
    return parts[parts < 0].sum(), parts[parts > 0].sum()


def byproduct_heat_legend_label(segments):
    """
    Legend text of the hatched segment, naming the sources it holds.
    """
    sources = [source for source in segments.index if source != "electrolysis"]
    return f"Of which from {' and '.join(sources)} (by-product heat)"


def add_byproduct_heat_bar(ax, segments, width, color, rotation_deg=0):
    """
    By-product heat bar in the slot at x = 0 of a box whose other bars were
    drawn at x = 1, 2, ..., with a vertical line between the two parts.

    One bar, centered in its slot and without a cost bar: the costs of its
    sources are in the other boxes. Electrolysis is the solid part from zero;
    the other sources are one hatched segment, stacked on the electrolysis
    part when both have the same sign and on the other side of zero when not.

    Returns:
        matplotlib.patches.Patch or None: legend key of the hatched segment,
        None if the bar has none
    """
    electrolysis, others = byproduct_heat_parts(segments)
    ax.bar(0, electrolysis, width, color=color, linewidth=0)

    key = None
    if others != 0:
        same_side = np.sign(others) == np.sign(electrolysis)
        hatched = {
            "facecolor": cc_tint(color),
            "hatch": BYPRODUCT_HATCH,
            "edgecolor": CC_EDGE,
        }
        # no outline, as the carbon capture segment
        ax.bar(
            0,
            others,
            width,
            bottom=electrolysis if same_side else 0.0,
            linewidth=0,
            **hatched,
        )
        key = patches.Patch(linewidth=0.4, **hatched)

    # the by-product heat on the left, the bars with costs on the right
    ax.axvline(0.5, color="black", linewidth=mpl.rcParams["axes.linewidth"], zorder=3)

    labels = [label.get_text() for label in ax.get_xticklabels()]
    ax.set_xticks(np.arange(len(labels) + 1))
    ax.set_xticklabels(["By-product\nheat", *labels], rotation=rotation_deg)
    ax.set_xlim([-0.5, len(labels) + 0.5])
    return key


# --- striped "Difference" bar of the difference explanation box ----------
# both bar colors of the boxes, as a difference is shown in either of them

DIFFERENCE_STRIPE_WIDTH = 2.0  # pt
DIFFERENCE_STRIPE_ANGLE = 26.0  # degrees from the x-axis


class StripedRectangle(patches.Rectangle):
    """
    Rectangle filled with diagonal stripes of two alternating colors.

    The stripes are built in display space at draw time, so `stripe_width`
    (points) and `stripe_angle` (degrees, counterclockwise from the x-axis)
    hold whatever the aspect of the axes. `phase` shifts the stripes by a
    fraction of `stripe_width`, measured from the top left corner. (Not
    `angle`: that is the rotation of the Rectangle itself.)
    """

    def __init__(
        self,
        xy,
        width,
        height,
        colors,
        stripe_width=DIFFERENCE_STRIPE_WIDTH,
        stripe_angle=DIFFERENCE_STRIPE_ANGLE,
        phase=0.5,
        **kwargs,
    ):
        kwargs.setdefault("linewidth", 0)
        super().__init__(xy, width, height, facecolor=colors[0], **kwargs)
        self.stripe_colors = tuple(colors)
        self.stripe_width = stripe_width
        self.stripe_angle = stripe_angle
        self.phase = phase

    def draw(self, renderer):
        if not self.get_visible():
            return
        # first color as background, second color as stripes on top
        super().draw(renderer)

        corners = self.get_transform().transform(self.get_path().vertices[:4])
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        anchor = np.array([x0, y1])
        width = renderer.points_to_pixels(self.stripe_width)
        theta = np.deg2rad(self.stripe_angle)
        along = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-np.sin(theta), np.cos(theta)])

        # offsets of the corners along the normal, the top left one is 0
        offsets = (corners - anchor) @ normal
        reach = (x1 - x0) + (y1 - y0)
        verts, codes = [], []
        start = -self.phase * width
        while start > offsets.min() - 2 * width:
            near, far = start - width, start - 2 * width
            band = [
                anchor + near * normal - reach * along,
                anchor + near * normal + reach * along,
                anchor + far * normal + reach * along,
                anchor + far * normal - reach * along,
            ]
            verts += band + [band[0]]
            codes += [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
            start = far

        stripes = patches.PathPatch(
            Path(verts, codes),
            transform=transforms.IdentityTransform(),
            facecolor=self.stripe_colors[1],
            linewidth=0,
        )
        stripes.set_figure(self.figure)
        stripes.set_clip_path(self.get_path(), self.get_transform())
        if self.get_clip_on():
            stripes.set_clip_box(self.get_clip_box())
        stripes.draw(renderer)


class HandlerStripedRectangle(legend_handler.HandlerPatch):
    """
    Legend key for a StripedRectangle, with its colors, width, angle and phase.
    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        key = StripedRectangle(
            (-xdescent, -ydescent),
            width,
            height,
            colors=orig_handle.stripe_colors,
            stripe_width=orig_handle.stripe_width,
            stripe_angle=orig_handle.stripe_angle,
            phase=orig_handle.phase,
        )
        key.set_transform(trans)
        return [key]


# --- panel labels (letter in a circle) -----------------------------------

PANEL_LABEL_RADIUS = 7.2  # pt
PANEL_LABEL_FONTSIZE = 12  # pt
PANEL_LABEL_LINEWIDTH = 0.6
PANEL_LABEL_ALPHA = 0.8


def add_panel_label(fig, corner, offset, letter):
    """
    Circled `letter` centered `offset` (pt, x right, y up) from `corner`
    (figure fraction).

    Inches from an anchor in figure fraction, not plain inches: savefig
    with bbox_inches="tight" shifts transFigure for the crop, not
    dpi_scale_trans.
    """
    trans = fig.dpi_scale_trans + transforms.ScaledTranslation(*corner, fig.transFigure)
    center = (offset[0] / 72, offset[1] / 72)
    fig.add_artist(
        patches.Circle(
            center,
            PANEL_LABEL_RADIUS / 72,
            transform=trans,
            fill=False,
            edgecolor="black",
            linewidth=PANEL_LABEL_LINEWIDTH,
            alpha=PANEL_LABEL_ALPHA,
        )
    )
    fig.text(
        *center,
        letter,
        transform=trans,
        ha="center",
        va="center",
        fontsize=PANEL_LABEL_FONTSIZE,
        alpha=PANEL_LABEL_ALPHA,
    )


# capacity axis units per cost axis unit
COST_AXIS_RATIO = 15


def stack_extent(*segments):
    """
    Lowest and highest partial sum of a stack of bar segments, zero included
    (Invest and Opex can have opposite signs).
    """
    lower = upper = 0.0
    total = None
    for segment in segments:
        if segment is None:
            continue
        values = np.nan_to_num(np.asarray(segment, dtype=float), nan=0.0)
        total = values if total is None else total + values
        lower = min(lower, float(np.min(total, initial=0.0)))
        upper = max(upper, float(np.max(total, initial=0.0)))
    return lower, upper


def bar_limits(df, pad=0.05, min_range=5, cost_axis_ratio=COST_AXIS_RATIO, extra=()):
    """
    Capacity axis limits of one box: the largest bar on each side of zero plus
    `pad` times the total span. Cost bars are converted with `cost_axis_ratio`,
    `min_range` is the smallest total span, `extra` are capacity values of
    bars drawn outside `df` (the by-product heat bar).
    """

    def column(label):
        return df[label] if label in df else None

    capacity = stack_extent(column("Installation"), column("Installation CC"))
    cost = stack_extent(column("Invest"), column("Opex"), column("Cost CC"))
    lower = min(capacity[0], cost[0] * cost_axis_ratio, *extra)
    upper = max(capacity[1], cost[1] * cost_axis_ratio, *extra)

    headroom = pad * (upper - lower)
    limit_up = upper + headroom if upper > 0 else 0.0
    limit_down = -lower + headroom if lower < 0 else 0.0
    if limit_up + limit_down < min_range:
        if limit_up >= limit_down:
            limit_up = min_range - limit_down
        else:
            limit_down = min_range - limit_up
    return np.array([-limit_down, limit_up])


# hatch line width only for this figure
@mpl.rc_context(
    {
        **get_pgf_with_latex(FIGURE_FONT_SCALE, legend_scale=TEXT_SCALE),
        "hatch.linewidth": CC_HATCH_LINEWIDTH,
    }
)
def plot_system_cost_differences_bars(
    n_el,
    n_pc,
    output_folder_path,
    scale,
    save=False,
    export_format="svg",
    name="system_cost_difference",
    ylims=None,
    legend_labels=("PC-50", "PC-0"),
    panel_labels=False,
):
    """
    Capacity, generation and cost differences between two solved networks.

    The heating and power box opens with the by-product heat (heat output
    capacity of electrolysis and fuel synthesis), without a cost bar and
    separated from the rest of the box by a vertical line.

    name : str
        file name
    ylims : dict, optional
        capacity axis limits per box ("energy", "hydrogen", "heating",
        "synfuels"), default bar_limits()
    legend_labels : (str, str)
        variant and base name for the "Difference explanation" box
    panel_labels : bool
        circled a-e on the boxes, only for the figure of the main document
    """
    from system_cost_differences import (
        calc_byproduct_heat_box_values,
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
    byproduct_heat = calc_byproduct_heat_box_values(n_el, n_pc)

    with open(COLORS_YAML) as stream:
        colors_dict = yaml.safe_load(stream)["others"]

    # Capacity-axis limits per box, each from its own data
    box_ylims = {
        "energy": bar_limits(df_energy_carriers),
        "hydrogen": bar_limits(df_hydrogen_system, min_range=1),
        "heating": bar_limits(
            df_power_and_heating_system, extra=byproduct_heat_ends(byproduct_heat)
        ),
        "synfuels": bar_limits(df_syn_fuels),
    }
    for key, value in (ylims or {}).items():
        box_ylims[key] = np.asarray(value, dtype=float)

    rotation_deg = 0

    # six mosaic columns per bar, gaps are empty columns
    # fmt: off
    layout = [
        ["A"] * 48,
        ["C"] * 48,
        ["B"] * 12 + ["."] * 9 + ["D"] * 18 + ["."] * 2 + ["E"] * 7,
    ]
    # fmt: on

    # same bar width in all boxes
    box_bars = {
        "A": len(df_energy_carriers),
        "B": len(df_hydrogen_system),
        # and the slot of the by-product heat bar
        "C": len(df_power_and_heating_system) + 1,
        "D": len(df_syn_fuels),
    }
    slot = {
        key: max(row.count(key) for row in layout) / bars
        for key, bars in box_bars.items()
    }
    bar_width = {key: 0.35 * min(slot.values()) / slot[key] for key in slot}

    def create_bar_plot(
        ax, x_values, df, name_map, title, y_label_left, ax1_ylim, width
    ):
        ax.set_title(title)

        # Plot the Installation bars on ax
        ax.bar(
            x_values - width / 2,
            df["Installation"],
            width,
            label="Installation",
            color=colors_dict["generation installation"],
        )
        if "Installation CC" in df:
            add_cc_segment(
                ax,
                x_values - width / 2,
                df["Installation CC"],
                df["Installation"],
                width,
                colors_dict["generation installation"],
                label="with carbon capture",
            )
        ax.set_ylabel(y_label_left, color=colors_dict["generation installation"])
        ax.tick_params(axis="y", labelcolor=colors_dict["generation installation"])

        # Create a secondary y-axis for the cost bars
        ax2 = ax.twinx()
        # capacity axes drawn over the cost axes, so its grid lies on both bars
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        # Invest and Opex have the same color: one bar over the extent of
        # their stack, two touching bars show a hairline seam in SVG viewers
        invest = np.nan_to_num(df["Invest"].to_numpy(dtype=float))
        stacked = invest + np.nan_to_num(df["Opex"].to_numpy(dtype=float))
        cost_bottom = np.minimum(0, np.minimum(invest, stacked))
        cost_top = np.maximum(0, np.maximum(invest, stacked))
        ax2.bar(
            x_values + width / 2,
            cost_top - cost_bottom,
            width,
            bottom=cost_bottom,
            label="Annualized costs",
            color=colors_dict["annualized costs"],
        )
        if "Cost CC" in df:
            add_cc_segment(
                ax2,
                x_values + width / 2,
                df["Cost CC"],
                df["Invest"] + df["Opex"],
                width,
                colors_dict["annualized costs"],
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
        ax2.set_ylim(ax1_ylim / COST_AXIS_RATIO)

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
        Plots bar data on the given axis, returns the three legend handles.
        """
        variant = ax.bar(
            [0.2, 3.8], [5, 5], 0.9, align="edge", color=colors_dict["pc-50"]
        )
        base = ax.bar(
            [1.2, 4.8],
            [3, 7],
            0.9,
            bottom=[2, -2],
            align="edge",
            color=colors_dict["pc-0"],
        )
        # in the colors of both bars of the boxes, a difference is either
        for x, bottom in [(2.2, 0), (5.8, -2)]:
            difference = StripedRectangle(
                (x, bottom),
                0.9,
                2,
                colors=(
                    colors_dict["annualized costs"],
                    colors_dict["generation installation"],
                ),
            )
            ax.add_patch(difference)
        return [variant[0], base[0], difference]

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

        # none on the striped difference bars
        arrows = {
            "bm-1": {"xy": (0.65, 0), "xyt": (0.65, 5), "ls": "--", "a": 0.5},
            "el-1": {"xy": (1.65, 5), "xyt": (1.65, 2), "ls": "--", "a": 0.5},
            "bm-2": {"xy": (4.25, 0), "xyt": (4.25, 5), "ls": "--", "a": 0.5},
            "el-2": {"xy": (5.25, 5), "xyt": (5.25, -2), "ls": "--", "a": 0.5},
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
        title="Hydrogen production, electricity generation and fossil fuel use",
        y_label_left="Difference in energy\ngeneration in TWh a$^{-1}$",
        ax1_ylim=box_ylims["energy"],
        width=bar_width["A"],
    )

    create_bar_plot(
        ax=ax_dict["B"],
        x_values=x_hydrogen_system,
        df=df_hydrogen_system,
        name_map=hydrogen_system_name_map,
        title="Hydrogen system",
        y_label_left="Difference in storage capacity in TWh\nand transport capacity in TWkm ",
        ax1_ylim=box_ylims["hydrogen"],
        width=bar_width["B"],
    )

    create_bar_plot(
        ax=ax_dict["C"],
        # slot 0 holds the by-product heat bar
        x_values=x_power_and_heating_system + 1,
        df=df_power_and_heating_system,
        name_map=power_and_heating_system_name_map,
        title="Heating and power system",
        y_label_left="Difference in installed capacity\nin GW and TWh (storage)",
        ax1_ylim=box_ylims["heating"],
        width=bar_width["C"],
    )
    byproduct_heat_key = add_byproduct_heat_bar(
        ax_dict["C"],
        byproduct_heat,
        bar_width["C"],
        colors_dict["generation installation"],
        rotation_deg=rotation_deg,
    )

    ax2 = create_bar_plot(
        ax=ax_dict["D"],
        x_values=x_syn_fuels,
        df=df_syn_fuels,
        name_map=name_map_syn_fuels,
        title="Synthetic fuels",
        y_label_left="Difference in generation in TWh a$^{-1}$\nand Mt CO$_2$ a$^{-1}$ (DAC)",
        ax1_ylim=box_ylims["synfuels"],
        width=bar_width["D"],
    )

    # Show the plot
    handles1, labels1 = ax_dict["A"].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    labels = ["Yearly generation / total installation", "Annualized costs per year"]
    handles = handles1 + [handles2[0]]
    # The carbon capture and by-product heat keys go last: they qualify the
    # other two entries rather than standing beside them, and they are the
    # least important.
    if (df_power_and_heating_system["Installation CC"] != 0).any():
        labels.append("Of which with carbon capture (gas CHP)")
        handles.append(cc_legend_patch())
    if byproduct_heat_key is not None:
        labels.append(byproduct_heat_legend_label(byproduct_heat))
        handles.append(byproduct_heat_key)
    # two rows at most, in one row the scaled legend is wider than the figure;
    # it is placed at the end, below the explanation frame
    key_legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        labelspacing=0.3,
        frameon=False,
        fancybox=False,
    )

    # wspace stays 0: the gaps between boxes are empty mosaic columns
    plt.subplots_adjust(left=0, bottom=0.07, right=1, top=0.9, wspace=0.0, hspace=0.28)

    ### Add explanatory legend plot
    # in the reserved mosaic columns, shifted right to make room for the cost
    # label of the synthetic fuel box. The frame top is level with the titles
    # of the third row; title, drawing and legend sit close below it, so the
    # frame and the key legend below it end near the tick labels
    reserved = ax_dict["E"].get_position(fig).bounds
    bounds = (
        reserved[0] + 0.074,
        reserved[1] + 0.055,
        0.13,
        reserved[3] - 0.088,
    )
    fig.delaxes(ax_dict["E"])

    # # Create and set up AxesZero
    ax_explain = setup_axes_zero(fig, bounds)
    # # Plot components
    explain_handles = plot_bars(ax_explain)
    plot_lines_and_arrows(ax_explain)
    ax_explain.set_title("Difference\nexplanation")

    # # Add legend
    explain_legend = ax_explain.legend(
        explain_handles,
        [legend_labels[0], legend_labels[-1], "Difference"],
        handler_map={StripedRectangle: HandlerStripedRectangle()},
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        borderaxespad=0.2,
        labelspacing=0.3,
        frameon=False,
        fancybox=False,
    )

    # Draw a rectangle on the entire figure (not inside any specific axis).
    # Long case names widen the frame to the right
    frame_left = bounds[0] - 0.015
    frame_right = bounds[0] + bounds[2] + 0.025
    legend_extent = explain_legend.get_window_extent().transformed(
        fig.transFigure.inverted()
    )
    overhang = max(0.0, frame_left - legend_extent.x0, legend_extent.x1 - frame_right)
    if overhang > 0:
        ax_explain.set_position([bounds[0] + overhang, *bounds[1:]])
    # the frame reaches below the legend, however tall its text is
    frame_bottom = min(bounds[1] - 0.071, legend_extent.y0 - 0.004)
    frame_top = bounds[1] + bounds[-1] + 0.046
    rect = plt.Rectangle(
        (frame_left, frame_bottom),
        frame_right - frame_left + 2 * overhang,
        frame_top - frame_bottom,
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="black",
        fill=False,
    )
    fig.patches.append(rect)

    if panel_labels:
        # offsets in pt from the top left corner of each box (e: of the
        # explanation frame), as placed by hand in the printed version
        def top_left(ax):
            return ax.get_position().x0, ax.get_position().y1

        for letter, corner, offset in [
            ("a", top_left(ax_dict["A"]), (-16.7, 7.2)),
            ("b", top_left(ax_dict["C"]), (-16.7, 8.5)),
            ("c", top_left(ax_dict["B"]), (-16.7, 12.8)),
            ("d", top_left(ax_dict["D"]), (-15.2, 12.8)),
            ("e", (rect.get_x(), rect.get_y() + rect.get_height()), (-13.0, 4.8)),
        ]:
            add_panel_label(fig, corner, offset, letter)

    # Key legend below the explanation frame and the tick labels of the third
    # row, centered on the saved figure rather than on the figure: the
    # explanation frame widens the figure to the right
    to_figure = fig.transFigure.inverted()
    row3_bottom = min(
        ax.get_tightbbox().transformed(to_figure).y0
        for ax in (ax_dict["B"], ax_dict["D"])
    )
    key_legend.set_in_layout(False)
    extent = fig.get_tightbbox()  # inches
    key_legend.set_in_layout(True)
    key_legend.set_bbox_to_anchor(
        (
            (extent.x0 + extent.x1) / 2 / fig.get_figwidth(),
            min(row3_bottom, frame_bottom),
        ),
        fig.transFigure,
    )

    save_or_plot_img(
        fig,
        output_folder_path,
        name,
        save=save,
        export_format=export_format,
    )


def load_hydrogen_salt_cavern_data():
    """
    Onshore regions, salt cavern potential per region and the cavern polygons.

    Salt cavern data: Caglayan et al. (2020), Technical potential of salt caverns
    for hydrogen storage in Europe, Int. J. Hydrogen Energy 45(11), 6793-6805,
    https://doi.org/10.1016/j.ijhydene.2019.12.161
    """
    data_folder = os.path.join(REPO_ROOT, "data")
    bus_regions_onshore = gpd.read_file(
        os.path.join(data_folder, "regions_onshore_elec_s_150.geojson")
    )
    bus_regions_onshore.set_index("name", inplace=True)
    caverns = gpd.read_file(
        os.path.join(data_folder, "h2_salt_caverns_GWh_per_sqkm.geojson")
    )
    cavern_potentials = pd.read_csv(
        os.path.join(data_folder, "salt_cavern_potentials_s_150.csv"),
        index_col="name",
    )
    cavern_potential_by_region_gdf = bus_regions_onshore.join(
        cavern_potentials, on="name"
    )
    cavern_potential_by_region_gdf["all"] = (
        cavern_potential_by_region_gdf[["nearshore", "offshore", "onshore"]]
        .fillna(0)
        .sum(axis=1)
    )
    return bus_regions_onshore, cavern_potential_by_region_gdf, caverns


def plot_hydrogen_salt_cavern_storage_potential(
    output_folder_path, scale, save, export_format="svg"
):
    (
        bus_regions_onshore,
        cavern_potential_by_region_gdf,
        caverns,
    ) = load_hydrogen_salt_cavern_data()

    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        figsize=set_size(use=True, subplots=(1, 1), fraction=2, scale=scale),
        subplot_kw={"projection": proj},
    )
    ax.set_extent([-10, 26.5, 34.7, 62])

    bus_regions_onshore.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.1,
    )

    cmap = "cmc.grayC_r"
    vmin = 1  # log Norm: vmin has to be > 0

    cavern_potential_by_region_gdf.plot(
        "all",
        ax=ax,
        cmap=cmap,
        linewidths=0,
        legend=True,
        vmin=vmin,
        norm=mcolors.LogNorm(
            vmin=vmin, vmax=cavern_potential_by_region_gdf["all"].max()
        ),
        # vmax=100,
        legend_kwds={
            "label": "Hydrogen storage potential in TWh",
        },
        # **map_opts
    )

    energy_density_col = "val_kwhm3"
    # get distinct values
    unique_values = sorted(caverns[energy_density_col].unique())
    color_list = cmcrameri.cm.roma(np.linspace(0, 1, len(unique_values)))
    color_map = dict(zip(unique_values, color_list))
    caverns["color"] = caverns[energy_density_col].map(color_map)
    caverns.plot(
        "val_kwhm3",
        ax=ax,
        color=caverns["color"],
    )

    # Energy density legend
    legend_handles = [
        patches.Patch(color=color_map[val], label=str(val)) for val in unique_values
    ]
    ax.legend(
        bbox_to_anchor=(0, 1),
        handles=legend_handles,
        title="Energy density\npotential\nin " r"$\mathrm{kWh~m^{-3}}$",
        loc="upper right",
    )

    ax.set_facecolor("white")

    save_or_plot_img(
        fig,
        output_folder_path,
        "hydrogen_salt_cavern_storage_potential",
        save=save,
        export_format=export_format,
    )


def main():
    """
    Draw the data figures 3-8 of the main document into cases.FIGURE_FOLDER
    (img/ by default).

    The Supplementary Material figures come from
    src/supplementary_material.py.
    """
    mpl.rcParams.update(get_pgf_with_latex(FIGURE_FONT_SCALE))
    filepath = os.path.join(REPO_ROOT, "results", "total_summary.csv")

    # Load dataframe but skip the unit column
    total_summary_df = load_df_without_unit_row(filepath)

    # the PC-0 and PC-50 cases
    from cases import PC0, PC50, find_network

    n_el = pypsa.Network(find_network(PC0))
    n_mix = pypsa.Network(find_network(PC50))
    regions = gpd.read_file(
        os.path.join(REPO_ROOT, "data", "regions_onshore_elec_s_150.geojson")
    ).set_index("name")

    with open(COLORS_YAML) as stream:
        all_colors = yaml.safe_load(stream)
        tech_colors = all_colors["tech_colors"]
        noUGHS_line_color = all_colors["others"]["no_UGHS_line"]
        UGHS_line_color = all_colors["others"]["UGHS_line"]
        colors_dict = all_colors["others"]

    output_folder_path = FIGURE_FOLDER
    # svg and pdf (vector); add "tiff" for 600 dpi LZW rasters
    export_format = "svg,pdf"

    ### Small plots (1 column)

    market_share(
        total_summary_df,
        output_folder_path,
        UGHS_line_color,
        noUGHS_line_color,
        save=True,
        all=False,  # "false" to plot only 10% case
        export_format=export_format,
        # Fig. 3 keeps its own axis range and label position; the defaults
        # are those of the annex figure with all four sweeps
        xlim=(117, 177),
        pc50_text_position=(129, 49),
    )

    plot_LCOH_violins_of_electrolysis_and_photocatalysis(
        total_summary_df,
        output_folder_path,
        tech_colors,
        UGHS_line_color,
        represent_regarding_market_share=True,
        line_cost_or_market_share=False,
        violin_edge=False,
        save=True,
        export_format=export_format,
    )

    # panel labels only here, the annex versions have none
    plot_system_cost_differences_bars(
        n_el,
        n_mix,
        output_folder_path,
        save=True,
        scale=5,
        export_format=export_format,
        panel_labels=True,
    )

    ### Large plots (2 columns)

    plot_electrolysis_only_vs_balanced_mix_map(
        n_el,
        n_mix,
        regions,
        tech_colors,
        output_folder_path,
        save=True,
        export_format=export_format,
    )

    hydrogen_generation_and_demand_violins(
        output_folder_path,
        represent_regarding_market_share=True,
        line_cost_or_market_share=False,
        orientation="vertical",
        violin_edge=False,
        # photocatalysis stacked on electrolysis, so the generation half shows
        # the same quantity as the demand half and one ruler serves both
        stacked=True,
        save=True,
        export_format=export_format,
    )

    plot_installed_capacities_bars(
        output_folder_path,
        represent_regarding_market_share=True,
        save=True,
        export_format=export_format,
    )


if __name__ == "__main__":
    main()
