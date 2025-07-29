[![DOI](https://zenodo.org/badge/1028259081.svg)](https://doi.org/10.5281/zenodo.16567312)

# photocatalysis-europe

Here you find the code to create all figures and do the required data transformations and calculations to get the aggregated data for the paper: "Impacts of photocatalytic hydrogen production on the European energy system"

In the paper photocatalysis was introduced to the energy system optimization model PyPSA-Eur to analyze the systematic impacts of rising shares of photocatalysis in the European energy system.

With the code and data in this repository all post-processing steps are published and all plots can be created with the data available on zenodo without running other software. For this, the .nc files from zenodo have to be placed in the respective folders in this repository under 'results/raw/' (see usage)

To implement photocatalysis in the optimization model the technology was added in [PyPSA-Eur](https://github.com/pypsa/pypsa-eur) and [atlite](https://github.com/pypsa/atlite). For this two forks of the software were created to directly show the changes while keeping the original history:

- [PyPSA-Eur-PC](https://github.com/w-tusche/pypsa-eur-pc)
- [atlite-pc](https://github.com/w-tusche/atlite-pc.git)

The raw results of the two main scenarios PC-0 and PC-50 are published on zenodo: [Raw data of PC-0 and PC-50 cases for paper: Impacts of photocatalytic hydrogen production on the European energy system](https://10.5281/zenodo.16360844). All other raw data will be made available on reasonable request.

A summary of all data is in `results/total_eval.csv` and `results/total_summary.csv` with the corresponding extraction script: `src/create_basic_summary.py`.

For the differenent cases presented in the paper and supplementary material the capex values and the efficiency of photocatalysis were changed through the "resources/costs_2040.csv" file. Hydrogen storage in salt caverns was turned on and off in the config files.

In the paper the photocatalysis costs are communicated as annualized costs! However in the runs the capex is changed. Thus the resulting naming convention:

- 150_lv1.25_I_H_2045_3H_PC_{CAPEX-VALUE}Euro_{ADDITIONAL_INFORMATION}
- PC-0: `results\raw\150_lv1.25_I_H_2045_3H_PC_925Euro\`
- PC-50: `results\raw\150_lv1.25_I_H_2045_3H_PC_650Euro\`
- 650 Euro photocatalysis capex and 3 % STH 150_lv1.25_I_H_2045_3H_PC_650Euro_3pct_efficiency
- Electrolysis only with net zero carbon emissions: 150_lv1.25_I_H_2045_3H_PC_925Euro_net_zero
- a case without the possibility to build underground hydrogen storage: 150_lv1.25_I_H_2045_3H_PC_875Euro_noUGHS

## Usage

To use this software go through the following steps:

```bash
git clone https://github.com/w-tusche/photocatalysis-europe
cd photocatalysis-europe
```

The package requirements are curated in the envs/environment.yaml file. Thus install the required packages using [mamba](https://mamba.readthedocs.io/en/latest/index.html):

```bash
mamba update conda

mamba env create -f envs/environment.yaml  # general 
# on a windows machine other platforms try it with the environment-fixed-windows.yaml file

mamba activate pypsa-pc-plots
```

Now go to [Raw data of PC-0 and PC-50 cases for paper: Impacts of photocatalytic hydrogen production on the European energy system](https://10.5281/zenodo.16360844), download the raw data and place the .nc files from the folders `150_lv1.25_I_H_2045_3H_PC_925Euro` and `150_lv1.25_I_H_2045_3H_PC_650Euro` in the respective folder under `results/raw/`

Once you have done all this you may run the `plots.py` script to recreate all plots, from here on you can explore the post-processing steps for each plot.

```bash
mamba activate pypsa-pc-plots
python plots.py
```

## Contributions

The majority of the code is from:

- Wolfram Tuschewitzki
- Jelto Lange

Parts of the code are extracted from PyPSA-Eur and partly modified.
Some of the data files are also from PyPSA-Eur or a direct output from running PyPSA-Eur.

## License

The code in photocatalysis-europe is released as free software under the
[MIT License](https://opensource.org/licenses/MIT), see [`LICENSE.md`](LICENSE.md).
However, different licenses and terms of use may apply to the various input data.
