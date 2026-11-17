# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Wolfram Tuschewitzki
#
# SPDX-License-Identifier: MIT
"""
Scenario names and where their solved networks are.

This module is the only place that defines where networks are searched.

- data/raw/<scenario>/ holds one solved network (.nc) and its
  costs_2040.csv per scenario; the networks are published separately and
  unpacked there. data/raw/hofmann_2025_baseline.nc is the reference network
  of Hofmann et al. (2025).
- EXTRA_RAW_FOLDERS lists further folders with the same layout, e.g. on
  another drive. It is read from the environment variable
  PC_EXTRA_RAW_FOLDERS (several folders separated by os.pathsep) and is
  empty by default.
- A scenario is looked up in data/raw/ first, then in EXTRA_RAW_FOLDERS in
  order; the first folder that has it wins, folders that do not exist are
  skipped.
- create_summary_csv() and extract_bus_info() walk only data/raw/ when no
  scenarios are named; scenarios named in only_subfolders are looked up in
  all folders.
- FIGURE_FOLDER is where the figures are written, img/ by default; the
  environment variable PC_FIGURE_FOLDER names another folder.
- figure_formats() gives the file formats a figure is written in: those in
  the environment variable PC_FIGURE_FORMATS (e.g. "pdf" or "svg,png") if
  it is set, else the defaults of the calling script.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FOLDER = os.path.join(REPO_ROOT, "data", "raw")
EXTRA_RAW_FOLDERS = [
    folder
    for folder in os.environ.get("PC_EXTRA_RAW_FOLDERS", "").split(os.pathsep)
    if folder
]
RESULT_ROOTS = [RAW_FOLDER] + EXTRA_RAW_FOLDERS

FIGURE_FOLDER = os.environ.get("PC_FIGURE_FOLDER") or os.path.join(REPO_ROOT, "img")


def figure_formats(default):
    """
    File formats to write a figure in, as a list: PC_FIGURE_FORMATS if set,
    else `default` ("svg,pdf" or ["svg", "pdf"]).
    """
    formats = os.environ.get("PC_FIGURE_FORMATS") or default
    if isinstance(formats, str):
        formats = formats.split(",")
    return [fmt.strip().lstrip(".") for fmt in formats if fmt.strip()]


SUMMARY_CSV = os.path.join(REPO_ROOT, "results", "total_summary.csv")

PC0 = "150_lv1.25_I_H_2045_PC_950Euro"
PC50 = "150_lv1.25_I_H_2045_PC_660Euro"

PC0_LABEL = "PC-0"
PC50_LABEL = "PC-50"

# base sweep, by increasing market share
SWEEP = [
    "150_lv1.25_I_H_2045_PC_950Euro",  # PC-0
    "150_lv1.25_I_H_2045_PC_925Euro",
    "150_lv1.25_I_H_2045_PC_900Euro",
    "150_lv1.25_I_H_2045_PC_850Euro",
    "150_lv1.25_I_H_2045_PC_800Euro",
    "150_lv1.25_I_H_2045_PC_750Euro",
    "150_lv1.25_I_H_2045_PC_700Euro",
    "150_lv1.25_I_H_2045_PC_660Euro",  # PC-50
]


def balances_python():
    """
    Command that starts Python in the environment of plot_balance_pairs.py,
    which needs PyPSA 1.0.

    Under pixi this is the `balances` environment of the project. Otherwise it
    is the conda environment named in the environment variable
    PC_BALANCES_ENV (default: pypsa-plots-new).
    """
    manifest = os.environ.get("PIXI_PROJECT_MANIFEST")
    if manifest:
        pixi = os.environ.get("PIXI_EXE", "pixi")
        return [pixi, "run", "--manifest-path", manifest, "-e", "balances", "python"]
    conda_env = os.environ.get("PC_BALANCES_ENV", "pypsa-plots-new")
    return ["conda", "run", "-n", conda_env, "python"]


def variation_of(scenario, variation):
    """
    Name of a variation run, e.g. ..._660Euro + noUGHS -> ..._660Euro_noUGHS.
    """
    if not variation.startswith("_"):
        variation = "_" + variation
    return scenario + variation


def scenario_folder(scenario):
    """
    Folder of a scenario in the first root that has it.
    """
    for root in RESULT_ROOTS:
        folder = os.path.join(root, scenario)
        if os.path.isdir(folder):
            return folder
    raise FileNotFoundError(
        f"No folder for {scenario!r}. Looked in:\n  "
        + "\n  ".join(RESULT_ROOTS)
        + "\nDownload the solved networks and unpack them into data/raw/."
    )


def find_network(scenario):
    """
    Path of the solved .nc file of a scenario.
    """
    folder = scenario_folder(scenario)
    nc_files = [f for f in os.listdir(folder) if f.endswith(".nc")]
    if len(nc_files) != 1:
        raise RuntimeError(f"Expected one .nc in {folder}, found {nc_files}")
    return os.path.join(folder, nc_files[0])


def find_costs(scenario):
    """
    Path of the cost file next to a scenario's network.
    """
    return os.path.join(scenario_folder(scenario), "costs_2040.csv")
