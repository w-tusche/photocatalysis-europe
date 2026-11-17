Copyright (c) 2025 Wolfram Tuschewitzki, Jelto Lange

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Figures

The MIT license above covers the code. The figures of this study, including the figures the scripts produce, are licensed under the Creative Commons Attribution 4.0 International license (CC-BY-4.0), Copyright (c) 2025 Wolfram Tuschewitzki, Jelto Lange. Full text: `LICENSES/CC-BY-4.0.md`.

## Third-party code

Parts of this repository contain code taken from or derived from other projects. Those parts stay under their original licenses, whose full texts are kept in `LICENSES/`. The SPDX headers at the top of the affected files say which license applies.

| Files | Origin | License | Full text |
| --- | --- | --- | --- |
| `src/beeswarm.py` (`OneSidedBeeswarm`) | [seaborn](https://github.com/mwaskom/seaborn), Copyright (c) 2012-2023 Michael L. Waskom | BSD-3-Clause | `LICENSES/seaborn-BSD-3-Clause.md` |
| `src/extract_bus_information.py`, `src/plot_balance_pairs.py` (both modified) | [PyPSA-Eur](https://github.com/pypsa/pypsa-eur) | MIT | `LICENSES/pypsa-eur-MIT.md` |
| `src/plotting.default.yaml` | [PyPSA-Eur](https://github.com/pypsa/pypsa-eur) | CC0-1.0 | see SPDX header in the file |

## Third-party data

The MIT license above covers the code. Data files in `data/` that are taken from or derived from other sources stay under their original licenses; their sources are listed in `data/sources_info.md`.

| Files | Origin | License | Full text |
| --- | --- | --- | --- |
| `data/h2_salt_caverns_GWh_per_sqkm.geojson` (unchanged), `data/salt_cavern_potentials_s_150.csv` (derived) | Caglayan et al. (2020), [doi:10.1016/j.ijhydene.2019.12.161](https://doi.org/10.1016/j.ijhydene.2019.12.161), via the [PyPSA-Eur v0.11.0 data bundle](https://doi.org/10.5281/zenodo.3517935) | CC-BY-4.0 | `LICENSES/h2-salt-caverns-CC-BY-4.0.md` |
| `data/regions_onshore_elec_s_150.geojson` (derived) | Created with [PyPSA-Eur](https://github.com/pypsa/pypsa-eur) v0.11.0 from the GridKit ENTSO-E bus locations (CC BY 4.0, The PyPSA-Eur Authors), [Natural Earth](https://www.naturalearthdata.com) country shapes (public domain) and the [Marine Regions](https://www.marineregions.org) World EEZ v8 (CC BY) | CC-BY-4.0 | `LICENSES/regions-CC-BY-4.0.md` |
