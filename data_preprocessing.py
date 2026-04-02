import numpy as np
import pandas as pd
import folium
import os
import webbrowser

from branca.colormap import LinearColormap

from helpers.data_covid import get_covid_data, get_country_total_covid_data
from helpers.data_demographics import (
    ADDITIVE_DEMOGRAPHICS_COLUMNS,
    MEAN_DEMOGRAPHICS_COLUMNS,
    get_demographics_data,
)
from helpers.data_mobility import get_mobility_data, MOBILITY_COLUMNS
from helpers.data_population_density import get_country_population_density


def get_spatiotemporal_covid_dataset(
    country_iso2: str,
    country_iso3: str,
    grid_km: int = 1,
    data_sources: list[str] | None = None,
) -> pd.DataFrame:
    selected_sources = set(
        ["population_density", "mobility"] if data_sources is None else data_sources
    )

    pop_grid: pd.DataFrame | None = None
    if "population_density" in selected_sources:
        print("Loading population density...")
        pop_df = get_country_population_density(country_iso3)

    demographics_grid: pd.DataFrame | None = None
    if "demographics" in selected_sources:
        print("Loading demographics data...")
        demographics_df = get_demographics_data(country_iso2)

    print("Loading covid data...")
    covid_df = get_covid_data(country_iso2)

    mobility_grid: pd.DataFrame | None = None
    if "mobility" in selected_sources:
        print("Loading mobility data...")
        mobility_df = get_mobility_data(country_iso2)

    grid_size = grid_km / 111.0   # km → degrees

    if "population_density" in selected_sources:
        pop_df = pop_df.copy()
        pop_df["grid_lat"] = (
            np.floor(pop_df["latitude"] / grid_size) * grid_size
        )
        pop_df["grid_lon"] = (
            np.floor(pop_df["longitude"] / grid_size) * grid_size
        )
        pop_grid = pop_df.groupby(
            ["grid_lat", "grid_lon"]
        ).agg(
            population_density=("population_density", "sum")
        ).reset_index()

    if "demographics" in selected_sources:
        demographics_df = demographics_df.copy()
        demographics_df["grid_lat"] = (
            np.floor(demographics_df["latitude"] / grid_size) * grid_size
        )
        demographics_df["grid_lon"] = (
            np.floor(demographics_df["longitude"] / grid_size) * grid_size
        )
        demographics_grid = demographics_df.groupby(
            ["grid_lat", "grid_lon"]
        ).agg(
            **{
                column: (column, "sum")
                for column in ADDITIVE_DEMOGRAPHICS_COLUMNS
            },
            **{
                column: (column, "mean")
                for column in MEAN_DEMOGRAPHICS_COLUMNS
            },
        ).reset_index()


    # --------------------------------
    # Prepare temporal dataset
    # --------------------------------

    if "mobility" in selected_sources:
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

    country_totals = get_country_total_covid_data(country_iso2).copy()
    country_totals = country_totals[["date", "new_confirmed", "new_deceased", "new_recovered"]]
    country_totals["date"] = pd.to_datetime(country_totals["date"])
    country_totals = (
        country_totals.groupby("date", as_index=False)
        .agg(
            new_confirmed=("new_confirmed", "sum"),
            new_deceased=("new_deceased", "sum"),
            new_recovered=("new_recovered", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    covid_grid["date"] = pd.to_datetime(covid_grid["date"])

    cell_template = covid_grid[["grid_lat", "grid_lon"]].drop_duplicates().reset_index(drop=True)
    date_template = country_totals[["date"]].drop_duplicates().reset_index(drop=True)

    if mobility_grid is not None:
        mobility_dates = mobility_grid[["date"]].copy()
        mobility_dates["date"] = pd.to_datetime(mobility_dates["date"])
        date_template = pd.concat([date_template, mobility_dates], ignore_index=True)

    date_template = date_template.drop_duplicates().sort_values("date").reset_index(drop=True)
    grid_base = date_template.merge(cell_template, how="cross")

    covid_grid = grid_base.merge(
        covid_grid,
        on=["date", "grid_lat", "grid_lon"],
        how="left",
    )

    covid_grid = covid_grid.merge(
        country_totals.rename(
            columns={
                "new_confirmed": "new_confirmed_country",
                "new_deceased": "new_deceased_country",
                "new_recovered": "new_recovered_country",
            }
        ),
        on="date",
        how="left",
    )

    cell_weights = (
        covid_grid.groupby(["grid_lat", "grid_lon"], as_index=False)["new_confirmed"]
        .sum(min_count=1)
        .rename(columns={"new_confirmed": "cell_weight_raw"})
    )
    cell_weights["cell_weight_raw"] = cell_weights["cell_weight_raw"].fillna(0.0)

    total_weight = float(cell_weights["cell_weight_raw"].sum())
    if total_weight <= 0:
        cell_weights["cell_weight"] = 1.0 / max(len(cell_weights), 1)
    else:
        cell_weights["cell_weight"] = cell_weights["cell_weight_raw"] / total_weight

    covid_grid = covid_grid.merge(
        cell_weights[["grid_lat", "grid_lon", "cell_weight"]],
        on=["grid_lat", "grid_lon"],
        how="left",
    )
    covid_grid["cell_weight"] = covid_grid["cell_weight"].fillna(0.0)

    for metric in ["new_confirmed", "new_deceased", "new_recovered"]:
        country_metric = f"{metric}_country"
        covid_grid[metric] = pd.to_numeric(covid_grid[metric], errors="coerce")
        observed_sum = covid_grid.groupby("date")[metric].transform(lambda s: s.sum(min_count=1))
        observed_sum = observed_sum.fillna(0.0)
        country_total = pd.to_numeric(covid_grid[country_metric], errors="coerce").fillna(0.0)

        value = covid_grid[metric].fillna(0.0).astype(float)
        has_observed = observed_sum > 0
        has_country_total = country_total > 0

        scale = pd.Series(1.0, index=covid_grid.index, dtype=float)
        scale.loc[has_observed & has_country_total] = (
            country_total.loc[has_observed & has_country_total]
            / observed_sum.loc[has_observed & has_country_total]
        )
        value = value * scale

        fill_mask = (~has_observed) & has_country_total
        if fill_mask.any():
            date_weight_sum = covid_grid.groupby("date")["cell_weight"].transform("sum")
            safe_date_weight_sum = date_weight_sum.where(date_weight_sum > 0, other=1.0)
            normalized_weight = covid_grid["cell_weight"] / safe_date_weight_sum
            value.loc[fill_mask] = country_total.loc[fill_mask] * normalized_weight.loc[fill_mask]

        covid_grid[metric] = value

    covid_grid = covid_grid[
        ["date", "grid_lat", "grid_lon", "new_confirmed", "new_deceased", "new_recovered"]
    ]
    covid_grid["date"] = covid_grid["date"].dt.strftime("%Y-%m-%d")

    print("Merging selected data sources into covid grids...")

    final_df = covid_grid.copy()

    if pop_grid is not None:
        final_df = final_df.merge(
            pop_grid,
            on=["grid_lat", "grid_lon"],
            how="left"
        )
        final_df["population_density"] = final_df[
            "population_density"
        ].fillna(0)

    if demographics_grid is not None:
        final_df = final_df.merge(
            demographics_grid,
            on=["grid_lat", "grid_lon"],
            how="left"
        )
        for column in ADDITIVE_DEMOGRAPHICS_COLUMNS:
            final_df[column] = final_df[column].fillna(0.0)
        for column in MEAN_DEMOGRAPHICS_COLUMNS:
            final_df[column] = final_df[column].fillna(0.0)

    if mobility_grid is not None:
        final_df = final_df.merge(
            mobility_grid,
            on=["date", "grid_lat", "grid_lon"],
            how="left"
        )
        for column in MOBILITY_COLUMNS:
            final_df[column] = final_df[column].fillna(0.0)

    final_df["country_iso2"] = country_iso2
    final_df["country_iso3"] = country_iso3

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
