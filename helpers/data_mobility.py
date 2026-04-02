import os
import pandas as pd

from helpers.data_covid import (
    SAVE_DIR,
    download_file,
    get_country_location_lookup,
    get_leaf_location_lookup,
)


MOBILITY_PATH = os.path.join(SAVE_DIR, "covid_mobility.csv")
SAVE_PATH = os.path.join(SAVE_DIR, "mobility_data.csv")
MOBILITY_URL = "https://storage.googleapis.com/covid19-open-data/v3/mobility.csv"

MOBILITY_COLUMNS = [
    "mobility_retail_and_recreation",
    "mobility_grocery_and_pharmacy",
    "mobility_parks",
    "mobility_transit_stations",
    "mobility_workplaces",
    "mobility_residential",
]

REQUIRED_MOBILITY_COLUMNS = [
    "date",
    "location_key",
    "latitude",
    "longitude",
    *MOBILITY_COLUMNS,
]


def get_mobility_data(country: str) -> pd.DataFrame:
    leaf_lookup = get_leaf_location_lookup(country)
    expected_lookup = leaf_lookup

    download_file(MOBILITY_URL, MOBILITY_PATH)
    available_location_keys = set(
        pd.read_csv(MOBILITY_PATH, usecols=["location_key"])["location_key"]
        .dropna()
        .astype(str)
        .unique()
    )
    if not (set(leaf_lookup["location_key"].tolist()) & available_location_keys):
        expected_lookup = _select_deepest_available_lookup(
            get_country_location_lookup(country),
            available_location_keys,
        )

    expected_location_keys = set(expected_lookup["location_key"].tolist())

    if os.path.exists(SAVE_PATH):
        df_existing = pd.read_csv(SAVE_PATH)
        missing_columns = [
            column for column in REQUIRED_MOBILITY_COLUMNS
            if column not in df_existing.columns
        ]

        df_country = df_existing[
            df_existing["location_key"].fillna("").str.startswith(country)
        ]
        cached_location_keys = set(df_country["location_key"].dropna().astype(str).unique())
        has_expected_keys = cached_location_keys == expected_location_keys

        if len(df_country) > 0 and len(missing_columns) == 0 and has_expected_keys:
            print(f"Using cached mobility data for {country}")
            return df_country

    print("Loading mobility dataset...")

    location_lookup = expected_lookup
    location_keys = set(location_lookup["location_key"].tolist())

    mobility_df = pd.read_csv(
        MOBILITY_PATH,
        usecols=["date", "location_key", *MOBILITY_COLUMNS]
    )
    mobility_df = mobility_df[
        mobility_df["location_key"].isin(location_keys)
    ].copy()

    mobility_df = mobility_df.merge(
        location_lookup[["location_key", "latitude", "longitude"]],
        on="location_key",
        how="left"
    )

    mobility_df = mobility_df.dropna(
        subset=["latitude", "longitude"]
    )

    for column in MOBILITY_COLUMNS:
        mobility_df[column] = pd.to_numeric(
            mobility_df[column],
            errors="coerce"
        ).fillna(0.0)

    if os.path.exists(SAVE_PATH):
        df_existing = pd.read_csv(SAVE_PATH)
        df_existing = df_existing[
            ~df_existing["location_key"].fillna("").str.startswith(country)
        ]
        df_combined = pd.concat(
            [df_existing, mobility_df],
            ignore_index=True
        )
    else:
        df_combined = mobility_df

    df_combined.to_csv(
        SAVE_PATH,
        index=False
    )

    return mobility_df


def _select_deepest_available_lookup(
    location_lookup: pd.DataFrame,
    available_location_keys: set[str],
) -> pd.DataFrame:
    lookup_df = location_lookup.copy()
    lookup_df["location_key"] = lookup_df["location_key"].astype(str)
    lookup_df = lookup_df[lookup_df["location_key"].isin(available_location_keys)].copy()

    if lookup_df.empty:
        return lookup_df

    location_keys = lookup_df["location_key"].tolist()
    keep_keys: list[str] = []

    for location_key in location_keys:
        child_prefix = f"{location_key}_"
        has_children = any(
            other_key.startswith(child_prefix)
            for other_key in location_keys
            if other_key != location_key
        )
        if not has_children:
            keep_keys.append(location_key)

    if not keep_keys:
        return lookup_df

    return lookup_df[lookup_df["location_key"].isin(keep_keys)].copy()


if __name__ == "__main__":

    COUNTRY_ISO2 = "JP"

    df_mobility = get_mobility_data(COUNTRY_ISO2)
    print(df_mobility.head())
