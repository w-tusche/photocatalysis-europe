[![DOI](https://zenodo.org/badge/1028259081.svg)](https://doi.org/10.5281/zenodo.16567312)

# photocatalysis-europe

Post-processing code for the paper "Impacts of photocatalytic hydrogen production on the European energy system".

- Reads the solved PyPSA-Eur networks of the study and writes all data figures and tables of the paper and its Supplementary Material.
- Solved networks (data package): [Zenodo](https://doi.org/10.5281/zenodo.24293786)
- Model code: [PyPSA-Eur-PC](https://github.com/w-tusche/pypsa-eur-pc) and [atlite-pc](https://github.com/w-tusche/atlite-pc), forks of [PyPSA-Eur](https://github.com/pypsa/pypsa-eur) and [atlite](https://github.com/pypsa/atlite) with photocatalysis added.

## Folders

```text
run_pipeline.py             entry script
pixi.toml, pixi.lock        environments
src/                        code
data/
  raw/                      solved networks, from the data package (not in git)
  generation_data/          nodal hydrogen generation and demand (written by stage "summary")
  bus_sizes/                installed capacities per node (written by stage "bus-sizes")
  balance_pairs/            cache of the energy balance figures
  system_cost_breakdown/    cache of the system cost figure
  *.geojson, *.csv          model regions, salt cavern potential, bus locations
results/
  total_summary.csv         one row per scenario, pre-computed
  total_eval.csv            nodal capacities per scenario, pre-computed
  recomputed/               the same two tables, recomputed by stage "summary"
img/                        figures as .pdf (written by the stages "main" and "supplementary")
LICENSES/                   license texts of third-party code and data
```

## Requirements

| What | Why |
| --- | --- |
| [pixi](https://pixi.sh) | installs both Python environments from `pixi.lock` (Windows, Linux, macOS on Apple silicon; tested on Windows) |
| LaTeX (TeX Live or MiKTeX) with `siunitx`, `eurosym`, `mhchem`, `type1cm`, `cm-super`, `dvipng` | all figure text is typeset with LaTeX |
| Chrome or Chromium | export of the carbon flow diagram |
| about 40 GB of disk space | data package 30 GB, environments, derived files |

## Setup

1. Get the code.

   ```bash
   git clone https://github.com/w-tusche/photocatalysis-europe
   cd photocatalysis-europe
   ```

2. Install the environments.

   ```bash
   pixi install --all
   ```

3. Download the data package and unpack it into `data/raw/`, one folder per scenario. The package consists of five zip archives (base cost sweep, CO2 sequestration, solar-to-hydrogen efficiency, no underground hydrogen storage, other sensitivities); the full pipeline needs all five, the PC-0 and PC-50 cases are in the base cost sweep. Download all files at once with [zenodo_get](https://github.com/dvolgyes/zenodo_get) or "Download all" on the record page, then unpack them in one step from inside `data/raw/`:

   ```bash
   for %f in (*.zip) do tar -xf "%f"                  # Windows, Command Prompt
   Get-ChildItem *.zip | % { tar -xf $_.FullName }    # Windows, PowerShell
   unzip '*.zip'                                      # Linux, macOS
   ```

   The result:

   ```text
   data/raw/
     150_lv1.25_I_H_2045_PC_660Euro/
       elec_s_150_lv1.25__I-H_2045.nc             solved network
       costs_2040.csv                             cost assumptions of the run
       config.elec_s_150_lv1.25__I-H_2045.yaml    configuration of the run
     150_lv1.25_I_H_2045_PC_660Euro_noUGHS/
     ...
   ```

4. Download the baseline network of Hofmann et al. (2025) from [figshare](https://doi.org/10.6084/m9.figshare.24923028) and save it as `data/raw/hofmann_2025_baseline.nc`. Only the stage `supplementary` needs it.

## Run

```bash
pixi run pipeline
```

Stages run in this order. A scenario that is already processed is skipped. If `summary` and `bus-sizes` run together, each network is read once: `summary` also writes the bus sizes of the networks it reads.

| Stage | Script | Reads | Writes |
| --- | --- | --- | --- |
| `summary` | `src/create_basic_summary.py` | all networks | `data/generation_data/`, `results/recomputed/total_*.csv`; compares them with `results/total_*.csv` |
| `bus-sizes` | `src/extract_bus_information.py` | networks not read by `summary` in the same run | `data/bus_sizes/` |
| `main` | `src/plots.py` | tables, derived files, PC-0 and PC-50 networks | Figs. 3 to 8 of the paper in `img/` |
| `supplementary` | `src/supplementary_material.py` | tables, derived files, networks one or two at a time | figures of the Supplementary Material in `img/`, its tables in `results/recomputed/supplementary_tables.txt` |

Options, appended to the command (`pixi run pipeline --stages main`):

| Option | Meaning | Default |
| --- | --- | --- |
| `--stages A [B ...]` | stages to run: `summary`, `bus-sizes`, `main`, `supplementary` | all |
| `--raw-folder PATH` | folder that holds the scenario folders | `data/raw/` |
| `--outdir PATH` | folder the figures are written to | `img/` |
| `--skip-balances` | leave out the energy balance figures (the only step that uses the second environment) | off |
| `--update-results` | copy the recomputed tables over `results/total_*.csv` | off |
| `--force` | read every network again, also where the derived files exist | off |

Single scripts run the same way, from the repository root:

```bash
pixi run python src/plots.py
pixi run python src/supplementary_material.py
pixi run -e balances python src/plot_balance_pairs.py plot
```

Figures are written as `.pdf`. The variable `PC_FIGURE_FORMATS` sets the formats, comma-separated (e.g. `svg,png`); `pixi.toml` sets it to `pdf`.

Without pixi: any environment with the packages of `pixi.toml` works; set `PC_FIGURE_FORMATS=pdf` to get the same files. `src/plot_balance_pairs.py` needs PyPSA 1.0 and a second environment; name a conda environment for it in the variable `PC_BALANCES_ENV`.

## Scenarios

Folder name: `150_lv1.25_I_H_2045_PC_<CAPEX>Euro[_<variation>]`

| Part | Meaning |
| --- | --- |
| `150` | 150 model regions |
| `lv1.25` | transmission expansion up to 1.25 times today's line volume |
| `I_H` | industry and heating sectors included |
| `2045` | planning year |
| `PC_<CAPEX>Euro` | photocatalysis CAPEX in €/kW, set exogenously per run |

Base runs (10 % solar-to-hydrogen efficiency, underground hydrogen storage available):

| CAPEX [€/kW] | Annualized cost [€/a/kW] | Photocatalysis market share [%] | Case |
| ---: | ---: | ---: | --- |
| 950 | 173.3 | 0.0 | PC-0 |
| 925 | 168.7 | 4.6 | |
| 900 | 164.1 | 5.8 | |
| 850 | 155.0 | 16.7 | |
| 800 | 145.9 | 28.3 | |
| 750 | 136.8 | 35.5 | |
| 700 | 127.7 | 43.6 | |
| 660 | 120.4 | 49.8 | PC-50 |

Annualized cost = CAPEX × 0.18238 (annuity at 7 % over 10 years plus 4 % fixed operation and maintenance). The paper reports annualized costs and market shares, the folder names carry the CAPEX.

Variations:

| Suffix | Change against the base run |
| --- | --- |
| `_noUGHS` | no underground hydrogen storage |
| `_3pct_efficiency`, `_6pct_efficiency` | solar-to-hydrogen efficiency 3 % or 6 % |
| `_CCS_<n>` | CO2 sequestration limit of n Mt/a instead of 200 |
| `_wind_1_2` | wind land-use density 1.2 instead of 3.0 MW/km² |
| `_transport` | transport sector included (file names with `I-H-T`) |
| `_elheat0` | no usable heat from electrolysis |

## Pre-computed tables

- `results/total_summary.csv`: one row per scenario (capacities, generation, market share, costs, storage, pipelines). The second line holds the units; read it with `load_df_without_unit_row()` from `src/create_basic_summary.py`.
- `results/total_eval.csv`: electrolysis and photocatalysis capacity per node and scenario.

## License

- Code: MIT, see [`LICENSE.md`](LICENSE.md).
- Figures the scripts produce: CC-BY-4.0.
- Third-party code and data keep their licenses; `LICENSE.md` lists them, `LICENSES/` holds the texts.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Contributions

- Wolfram Tuschewitzki
- Jelto Lange

Parts of the code are taken from or derived from PyPSA-Eur, see `LICENSE.md`.
