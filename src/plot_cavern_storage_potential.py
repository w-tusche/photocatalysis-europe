# -*- coding: utf-8 -*-
import cartopy.crs as ccrs
import cmcrameri  # To register colormaps for matplotlib e.g. cmc.batlow
import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Salt cavern raw data from:
# Caglayan, D. G., Weber, N., Heinrichs, H. U., Linßen, J., Robinius, M.,
# Kukla, P. A., & Stolten, D. (2020).
# Technical potential of salt caverns for hydrogen storage in Europe.
# International Journal of Hydrogen Energy, 45(11), 6793–6805.
# https://doi.org/10.1016/j.ijhydene.2019.12.161


def load_hydrogen_salt_cavern_data():
    fn_onshore = "./data/regions_onshore_elec_s_150.geojson"
    bus_regions_onshore = gpd.read_file(fn_onshore)
    bus_regions_onshore.set_index("name", inplace=True)
    caverns = gpd.read_file("./data/h2_salt_caverns_GWh_per_sqkm.geojson")
    cavern_potentials = pd.read_csv(
        "./data/salt_cavern_potentials_s_150.csv", index_col="name"
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


(
    bus_regions_onshore,
    cavern_potential_by_region_gdf,
    caverns,
) = load_hydrogen_salt_cavern_data()

proj = ccrs.PlateCarree()

fig, ax = plt.subplots(subplot_kw={"projection": proj})
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
    norm=mcolors.LogNorm(vmin=vmin, vmax=cavern_potential_by_region_gdf["all"].max()),
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
    mpatches.Patch(color=color_map[val], label=str(val)) for val in unique_values
]
ax.legend(
    bbox_to_anchor=(0, 1),
    handles=legend_handles,
    title="Energy density\npotential\nin $\mathrm{kWh~m^{-3}}$",
    loc="upper right",
)

ax.set_facecolor("white")

plt.show()
# plt.savefig("../img/salt_cavern_potentials.pdf", bbox_inches="tight", format="pdf")
