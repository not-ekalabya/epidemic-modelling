import folium
from folium.plugins import HeatMap

import numpy as np
import pandas as pd

import webbrowser
import os

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))

from data_population_density import get_country_population_density # ignore warning - intellisense issue

# constants

MAP_SAVE_LOCATION = "tmp/map.html"
COUNTRY = "JPN"

map_save_location = os.path.join(os.getcwd(), MAP_SAVE_LOCATION)

# generate map data
map_data = get_country_population_density(country=COUNTRY)

# clean data
map_data = map_data.dropna(subset=["latitude", "longitude", "population_density"])

# normalize density (important for heatmap visibility)
map_data["population_density"] = (
    map_data["population_density"] - map_data["population_density"].min()
) / (
    map_data["population_density"].max() - map_data["population_density"].min()
)

# convert to heatmap format
data = map_data[
    ["latitude", "longitude", "population_density"]
].values.tolist()

# center map on Japan
center_lat = map_data["latitude"].mean()
center_lon = map_data["longitude"].mean()

m = folium.Map([center_lat, center_lon], zoom_start=5)

HeatMap(
    data,
    radius=6,
    blur=4,
    max_zoom=6
).add_to(m)

# ensure directory exists
save_dir = os.path.dirname(map_save_location)
os.makedirs(save_dir, exist_ok=True)

# save map
m.save(map_save_location)

# view map
map_absolute_path = "file://" + os.path.abspath(map_save_location)
webbrowser.open(map_absolute_path)