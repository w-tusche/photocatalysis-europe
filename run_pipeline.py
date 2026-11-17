# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Wolfram Tuschewitzki
#
# SPDX-License-Identifier: MIT
"""
Run the post-processing, from the solved networks to the figures and tables.

Usage: python run_pipeline.py [options]

Stages, in this order (all by default, pick with --stages):

- summary: reads every network; writes data/generation_data/ and a
  recomputed total_summary.csv / total_eval.csv into
  results/recomputed/, then compares them with the tables in results/
- bus-sizes: writes data/bus_sizes/; reads only the networks the summary
  stage of the same run did not read, their files are written there
- main: figures of the main document, src/plots.py
- supplementary: figures and tables of the Supplementary Material,
  src/supplementary_material.py; the tables are also written to
  results/recomputed/supplementary_tables.txt

The solved networks are expected in data/raw/<scenario>/, see README.md.
Scenarios that are already processed are skipped; --force reads them
again.
"""

import argparse
import os
import runpy
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO_ROOT, "results")
RECOMPUTED = os.path.join(RESULTS, "recomputed")
SUMMARY_TABLES = ["total_summary.csv", "total_eval.csv"]
# rows of total_summary.csv that are not scenarios
SUMMARY_HEAD_ROWS = ["Unit"]
STAGES = ["summary", "bus-sizes", "main", "supplementary"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=STAGES,
        help="stages to run (default: all)",
    )
    parser.add_argument(
        "--raw-folder",
        help="folder with the scenario folders, if they are not in data/raw/",
    )
    parser.add_argument(
        "--outdir", help="folder the figures are written to (default: img/)"
    )
    parser.add_argument(
        "--skip-balances",
        action="store_true",
        help="leave out the energy balance figures, which need the second "
        "environment",
    )
    parser.add_argument(
        "--update-results",
        action="store_true",
        help="copy the recomputed tables over the shipped ones in results/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="read every network again, also where the derived files exist",
    )
    return parser.parse_args()


def scenarios_of_shipped_summary():
    import pandas as pd

    index = pd.read_csv(os.path.join(RESULTS, "total_summary.csv"), index_col=0).index
    return [name for name in index if name not in SUMMARY_HEAD_ROWS]


def check_inputs(scenarios, stages):
    """
    Stop before anything is loaded if a network is missing.
    """
    import cases

    missing = []
    for scenario in scenarios:
        try:
            cases.find_network(scenario)
        except (FileNotFoundError, RuntimeError):
            missing.append(scenario)
    if missing:
        sys.exit(
            f"{len(missing)} of {len(scenarios)} scenarios have no solved network:\n  "
            + "\n  ".join(missing)
            + "\nLooked in:\n  "
            + "\n  ".join(cases.RESULT_ROOTS)
            + "\nDownload the data package and unpack it into data/raw/, or name "
            "its folder with --raw-folder (README.md, 'Setup')."
        )
    if "supplementary" in stages:
        baseline = "hofmann_2025_baseline.nc"
        if not any(
            os.path.exists(os.path.join(root, baseline)) for root in cases.RESULT_ROOTS
        ):
            sys.exit(
                f"{baseline} not found in:\n  "
                + "\n  ".join(cases.RESULT_ROOTS)
                + "\nThe supplementary stage needs the baseline network of Hofmann "
                "et al. (2025), https://doi.org/10.6084/m9.figshare.24923028. Save "
                f"it as data/raw/{baseline}, or leave the stage out with --stages."
            )


def compare_tables():
    """
    Compare the recomputed tables with the shipped ones, cell by cell.

    Numbers are equal within a relative tolerance of 1e-6, everything
    else has to match as text. Returns the number of deviations.
    """
    import numpy as np
    import pandas as pd

    deviations = 0
    for name in SUMMARY_TABLES:
        shipped, recomputed = (
            pd.read_csv(
                os.path.join(folder, name),
                index_col=0,
                dtype=str,
                keep_default_na=False,
            )
            for folder in (RESULTS, RECOMPUTED)
        )
        only_one_side = list(shipped.index.symmetric_difference(recomputed.index))
        only_one_side += list(shipped.columns.symmetric_difference(recomputed.columns))
        rows = shipped.index.intersection(recomputed.index)
        columns = shipped.columns.intersection(recomputed.columns)
        a, b = shipped.loc[rows, columns], recomputed.loc[rows, columns]
        a_number = a.apply(pd.to_numeric, errors="coerce")
        b_number = b.apply(pd.to_numeric, errors="coerce")
        numeric = (a_number.notna() & b_number.notna()).to_numpy()
        close = np.isclose(a_number, b_number, rtol=1e-6, atol=1e-9)
        differs = np.where(numeric, ~close, a.to_numpy() != b.to_numpy())
        count = int(differs.sum()) + len(only_one_side)
        deviations += count
        if count == 0:
            print(f"{name}: recomputed table equals the shipped one ({a.size} cells).")
            continue
        print(f"{name}: {count} deviations from the shipped table.")
        if only_one_side:
            print(f"  rows or columns in only one of the two: {only_one_side[:10]}")
        for row, column in list(zip(*np.nonzero(differs)))[:10]:
            print(
                f"  {rows[row]} / {columns[column]}: "
                f"shipped {a.iat[row, column][:40]!r}, "
                f"recomputed {b.iat[row, column][:40]!r}"
            )
    return deviations


def run_summary(scenarios, force, update_results, bus_sizes_written=None):
    """
    With `bus_sizes_written` (a set), the bus sizes of every network that is
    read here are written as well, and the scenario is added to the set.
    """
    from create_basic_summary import DEFAULT_GENERATION_DATA_FOLDER, create_summary_csv

    per_network = None
    if bus_sizes_written is not None:
        from extract_bus_information import DEFAULT_BUS_SIZES_FOLDER, bus_size_writer

        write = bus_size_writer()

        def per_network(n, scenario):
            path = os.path.join(DEFAULT_BUS_SIZES_FOLDER, f"bus_sizes_{scenario}.csv")
            if force or not os.path.exists(path):
                write(n, scenario)
                bus_sizes_written.add(scenario)

    # the generation data of a scenario is only written while it is processed
    without_generation_data = [
        scenario
        for scenario in scenarios
        if not os.path.exists(
            os.path.join(DEFAULT_GENERATION_DATA_FOLDER, f"generation_{scenario}.csv")
        )
    ]
    if force or without_generation_data:
        for name in SUMMARY_TABLES:
            path = os.path.join(RECOMPUTED, name)
            if os.path.exists(path):
                os.remove(path)

    create_summary_csv(
        results_folder=RECOMPUTED, only_subfolders=scenarios, per_network=per_network
    )
    deviations = compare_tables()
    if update_results:
        for name in SUMMARY_TABLES:
            shutil.copyfile(os.path.join(RECOMPUTED, name), os.path.join(RESULTS, name))
        print("Copied the recomputed tables to results/.")
    return deviations


def run_bus_sizes(scenarios, force, already_written=()):
    """
    Scenarios in `already_written` got their file in the summary stage.
    """
    from extract_bus_information import extract_bus_info

    remaining = [s for s in scenarios if s not in already_written]
    if already_written:
        print(
            f"{len(already_written)} bus-size files were written in the summary stage."
        )
    if remaining:
        extract_bus_info(only_subfolders=remaining, re_create_files=force)


def run_main_figures():
    import plots

    plots.main()


class Tee:
    """
    Write to several streams at once.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_supplementary():
    tables = os.path.join(RECOMPUTED, "supplementary_tables.txt")
    stdout = sys.stdout
    with open(tables, "w", encoding="utf-8") as stream:
        sys.stdout = Tee(stdout, stream)
        try:
            runpy.run_path(
                os.path.join("src", "supplementary_material.py"), run_name="__main__"
            )
        finally:
            sys.stdout = stdout
    print(f"Tables of the Supplementary Material written to {tables}.")


def main():
    args = parse_arguments()

    # the modules in src/ read these when they are imported
    if args.raw_folder:
        folders = [os.path.abspath(args.raw_folder)]
        folders += [os.environ.get("PC_EXTRA_RAW_FOLDERS", "")]
        os.environ["PC_EXTRA_RAW_FOLDERS"] = os.pathsep.join(filter(None, folders))
    if args.outdir:
        os.environ["PC_FIGURE_FOLDER"] = os.path.abspath(args.outdir)
    if args.skip_balances:
        os.environ["PC_SKIP_BALANCE_PAIRS"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")

    os.chdir(REPO_ROOT)
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

    scenarios = scenarios_of_shipped_summary()
    check_inputs(scenarios, args.stages)

    deviations = None
    # both stages need every network; if both run, summary writes the bus
    # sizes of the networks it reads, so that each is read only once
    bus_sizes_written = set()
    for stage in STAGES:
        if stage not in args.stages:
            continue
        print(f"\n=== {stage} ===", flush=True)
        if stage == "summary":
            deviations = run_summary(
                scenarios,
                args.force,
                args.update_results,
                bus_sizes_written if "bus-sizes" in args.stages else None,
            )
        elif stage == "bus-sizes":
            run_bus_sizes(scenarios, args.force, bus_sizes_written)
        elif stage == "main":
            run_main_figures()
        elif stage == "supplementary":
            run_supplementary()

    print("\nDone.")
    if deviations:
        print(
            f"The recomputed tables deviate from the shipped ones in {deviations} "
            "places, see the summary stage above."
        )


if __name__ == "__main__":
    main()
