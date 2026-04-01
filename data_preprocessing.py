import numpy as np
import pandas as pd
import folium
import os
import webbrowser

from branca.colormap import LinearColormap

from helpers.data_covid import get_covid_data
from helpers.data_mobility import get_mobility_data, MOBILITY_COLUMNS
from helpers.data_population_density import get_country_population_density


def get_spatiotemporal_covid_dataset(
    country_iso2: str,
    country_iso3: str,
    grid_km: int = 1
) -> pd.DataFrame:

    print("Loading population density...")
    pop_df = get_country_population_density(country_iso3)

    print("Loading covid data...")
    covid_df = get_covid_data(country_iso2)

    print("Loading mobility data...")
    mobility_df = get_mobility_data(country_iso2)

    grid_size = grid_km / 111.0   # km → degrees

    pop_df = pop_df.copy()

    pop_df["grid_lat"] = (
        np.floor(pop_df["latitude"] / grid_size) * grid_size
    )

    pop_df["grid_lon"] = (
        np.floor(pop_df["longitude"] / grid_size) * grid_size
    )

    # aggregate population into grid
    pop_grid = pop_df.groupby(
        ["grid_lat", "grid_lon"]
    ).agg(
        population_density=("population_density", "sum")
    ).reset_index()


    # --------------------------------
    # Prepare temporal dataset
    # --------------------------------

    mobility_df = mobility_df.copy()

    mobility_df["grid_lat"] = (
        np.floor(mobility_df["latitude"] / grid_size) * grid_size
    )

    mobility_df["grid_lon"] = (
        np.floor(mobility_df["longitude"] / grid_size) * grid_size
    )

    mobility_grid = mobility_df.groupby(
        ["date", "grid_lat", "grid_lon"]
    ).agg(
        **{
            column: (column, "mean")
            for column in MOBILITY_COLUMNS
        }
    ).reset_index()

    covid_df = covid_df.copy()

    covid_df["grid_lat"] = (
        np.floor(covid_df["latitude"] / grid_size) * grid_size
    )

    covid_df["grid_lon"] = (
        np.floor(covid_df["longitude"] / grid_size) * grid_size
    )


    # aggregate covid into grid per date

    covid_grid = covid_df.groupby(
        ["date", "grid_lat", "grid_lon"]
    ).agg(
        new_confirmed=("new_confirmed", "sum"),
        new_deceased=("new_deceased", "sum"),
        new_recovered=("new_recovered", "sum")
    ).reset_index()

    print("Merging population, mobility, and covid grids...")

    final_df = covid_grid.merge(
        pop_grid,
        on=["grid_lat", "grid_lon"],
        how="left"
    )

    final_df = final_df.merge(
        mobility_grid,
        on=["date", "grid_lat", "grid_lon"],
        how="left"
    )

    final_df["population_density"] = final_df[
        "population_density"
    ].fillna(0)

    for column in MOBILITY_COLUMNS:
        final_df[column] = final_df[column].fillna(0.0)


    return final_df


def visualize_generated_cells(
    df: pd.DataFrame,
    grid_km: int = 1,
    metric: str = "population_density",
    date: str | None = None,
    output_path: str = "tmp/spatiotemporal_covid_cells.html",
    max_cells: int = np.inf,
    open_in_browser: bool = True
) -> str:

    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in dataframe.")

    if len(df) == 0:
        raise ValueError("Cannot visualize an empty dataframe.")

    plot_df = df.copy()

    if date is not None:
        plot_df = plot_df[plot_df["date"] == date]
    elif metric != "population_density" and "date" in plot_df.columns:
        date_totals = (
            plot_df.groupby("date")[metric]
            .sum()
            .sort_index()
        )

        positive_dates = date_totals[date_totals > 0]

        if len(positive_dates) > 0:
            selected_date = positive_dates.index[-1]
        else:
            selected_date = date_totals.index[-1]

        plot_df = plot_df[plot_df["date"] == selected_date]
        date = str(selected_date)

    aggregation = {
        metric: "sum",
        "population_density": "max"
    }

    plot_df = plot_df.groupby(
        ["grid_lat", "grid_lon"],
        as_index=False
    ).agg(aggregation)

    plot_df = plot_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=["grid_lat", "grid_lon", metric]
    )

    positive_plot_df = plot_df[plot_df[metric] > 0]

    if len(positive_plot_df) > 0:
        plot_df = positive_plot_df
    elif len(plot_df) == 0:
        raise ValueError("No grid cells found for the selected view.")

    if len(plot_df) > max_cells:
        plot_df = plot_df.nlargest(max_cells, metric)

    grid_size = grid_km / 111.0
    center_lat = plot_df["grid_lat"].mean() + (grid_size / 2)
    center_lon = plot_df["grid_lon"].mean() + (grid_size / 2)

    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron"
    )

    metric_min = float(plot_df[metric].min())
    metric_max = float(plot_df[metric].max())

    if metric_min == metric_max:
        metric_max = metric_min + 1.0

    color_scale = LinearColormap(
        colors=["#fff7ec", "#fc8d59", "#7f0000"],
        vmin=metric_min,
        vmax=metric_max,
        caption=metric if date is None else f"{metric} ({date})"
    )

    color_scale.add_to(map_obj)

    for _, row in plot_df.iterrows():
        lat = float(row["grid_lat"])
        lon = float(row["grid_lon"])
        value = float(row[metric])

        folium.Rectangle(
            bounds=[
                [lat, lon],
                [lat + grid_size, lon + grid_size]
            ],
            color=color_scale(value),
            fill=True,
            fill_color=color_scale(value),
            fill_opacity=0.65,
            weight=0.4,
            popup=(
                f"metric={metric}<br>"
                f"value={value:,.2f}<br>"
                f"lat={lat:.4f}<br>"
                f"lon={lon:.4f}"
            )
        ).add_to(map_obj)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    map_obj.save(output_path)

    if open_in_browser:
        webbrowser.open(
            "file://" + os.path.abspath(output_path)
        )

    return output_path


if __name__ == "__main__":
    COUNTRY_ISO2 = "BD"
    COUNTRY_ISO3 = "BGD"
    GRID_KM = 20
    METRIC = "population_density"

    dataset = get_spatiotemporal_covid_dataset(
        country_iso2=COUNTRY_ISO2,
        country_iso3=COUNTRY_ISO3,
        grid_km=GRID_KM
    )

    save_path = visualize_generated_cells(
        df=dataset,
        grid_km=GRID_KM,
        metric=METRIC,
        output_path="tmp/spatiotemporal_covid_cells.html"
    )

    print(f"Saved cell visualization to {save_path}")
