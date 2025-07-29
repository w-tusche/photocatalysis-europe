# -*- coding: utf-8 -*-
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

fn_onshore = "./data/regions_onshore_elec_s_150.geojson"
bus_regions_onshore = gpd.read_file(fn_onshore)
bus_regions_onshore.set_index("name", inplace=True)
caverns = gpd.read_file("./data/h2_salt_caverns_GWh_per_sqkm.geojson")
cavern_potentials = pd.read_csv(
    "./data/salt_cavern_potentials_s_150.csv", index_col="name"
)
cavern_potential_by_region_gdf = bus_regions_onshore.join(cavern_potentials, on="name")
cavern_potential_by_region_gdf["all"] = (
    cavern_potential_by_region_gdf[["nearshore", "offshore", "onshore"]]
    .fillna(0)
    .sum(axis=1)
)

proj = ccrs.EqualEarth()

fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": proj})

bus_regions_onshore.boundary.plot(
    ax=ax,
    color="black",
    linewidth=0.1,
)

cavern_potential_by_region_gdf.plot(
    "all",
    ax=ax,
    cmap="Blues",
    linewidths=0,
    legend=True,
    vmax=6,
    vmin=0,
    legend_kwds={
        "label": "Hydrogen Storage [TWh]",
        "shrink": 0.7,
        "extend": "max",
    },
    # **map_opts
)

caverns.plot(
    ax=ax,
    cmap="Reds",
    # color="red",
)

ax.set_facecolor("white")

plt.show()

# plt.savefig("../img/salt_cavern_potentials.pdf", bbox_inches="tight", format="pdf")
