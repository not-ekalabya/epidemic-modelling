import os
import pandas as pd

from helpers.data_covid import (
    SAVE_DIR,
    download_file,
    get_country_location_lookup,
    get_leaf_location_lookup,
)


DEMOGRAPHICS_RAW_PATH = os.path.join(SAVE_DIR, "covid_demographics.csv")
DEMOGRAPHICS_SAVE_PATH = os.path.join(SAVE_DIR, "demographics_data.csv")
DEMOGRAPHICS_URL = "https://storage.googleapis.com/covid19-open-data/v3/demographics.csv"

DEMOGRAPHICS_COLUMNS = [
    "demographics_population",
    "demographics_population_male",
    "demographics_population_female",
    "demographics_population_rural",
    "demographics_population_urban",
    "demographics_population_largest_city",
    "demographics_population_clustered",
    "demographics_population_density",
    "demographics_human_development_index",
    "demographics_population_age_00_09",
    "demographics_population_age_10_19",
    "demographics_population_age_20_29",
    "demographics_population_age_30_39",
    "demographics_population_age_40_49",
    "demographics_population_age_50_59",
    "demographics_population_age_60_69",
    "demographics_population_age_70_79",
    "demographics_population_age_80_and_older",
]

SOURCE_TO_OUTPUT_COLUMNS = {
    "population": "demographics_population",
    "population_male": "demographics_population_male",
    "population_female": "demographics_population_female",
    "population_rural": "demographics_population_rural",
    "population_urban": "demographics_population_urban",
    "population_largest_city": "demographics_population_largest_city",
    "population_clustered": "demographics_population_clustered",
    "population_density": "demographics_population_density",
    "human_development_index": "demographics_human_development_index",
    "population_age_00_09": "demographics_population_age_00_09",
    "population_age_10_19": "demographics_population_age_10_19",
    "population_age_20_29": "demographics_population_age_20_29",
    "population_age_30_39": "demographics_population_age_30_39",
    "population_age_40_49": "demographics_population_age_40_49",
    "population_age_50_59": "demographics_population_age_50_59",
    "population_age_60_69": "demographics_population_age_60_69",
    "population_age_70_79": "demographics_population_age_70_79",
    "population_age_80_and_older": "demographics_population_age_80_and_older",
}

REQUIRED_DEMOGRAPHICS_COLUMNS = [
    "location_key",
    "latitude",
    "longitude",
    *DEMOGRAPHICS_COLUMNS,
]

ADDITIVE_DEMOGRAPHICS_COLUMNS = [
    "demographics_population",
    "demographics_population_male",
    "demographics_population_female",
    "demographics_population_rural",
    "demographics_population_urban",
    "demographics_population_largest_city",
    "demographics_population_clustered",
    "demographics_population_age_00_09",
    "demographics_population_age_10_19",
    "demographics_population_age_20_29",
    "demographics_population_age_30_39",
    "demographics_population_age_40_49",
    "demographics_population_age_50_59",
    "demographics_population_age_60_69",
    "demographics_population_age_70_79",
    "demographics_population_age_80_and_older",
]

MEAN_DEMOGRAPHICS_COLUMNS = [
    "demographics_population_density",
    "demographics_human_development_index",
]


def get_demographics_data(country: str) -> pd.DataFrame:
    leaf_lookup = get_leaf_location_lookup(country)
    expected_lookup = leaf_lookup

    download_file(DEMOGRAPHICS_URL, DEMOGRAPHICS_RAW_PATH)
    available_location_keys = set(
        pd.read_csv(DEMOGRAPHICS_RAW_PATH, usecols=["location_key"])["location_key"]
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

    if os.path.exists(DEMOGRAPHICS_SAVE_PATH):
        df_existing = pd.read_csv(DEMOGRAPHICS_SAVE_PATH)
        missing_columns = [
            column for column in REQUIRED_DEMOGRAPHICS_COLUMNS
            if column not in df_existing.columns
        ]

        df_country = df_existing[
            df_existing["location_key"].fillna("").str.startswith(country)
        ]
        cached_location_keys = set(df_country["location_key"].dropna().astype(str).unique())
        has_expected_keys = cached_location_keys == expected_location_keys

        if len(df_country) > 0 and len(missing_columns) == 0 and has_expected_keys:
            print(f"Using cached demographics data for {country}")
            return df_country

    source_columns = ["location_key", *SOURCE_TO_OUTPUT_COLUMNS.keys()]
    demographics_df = pd.read_csv(
        DEMOGRAPHICS_RAW_PATH,
        usecols=source_columns,
    )

    location_lookup = expected_lookup
    location_keys = set(location_lookup["location_key"].tolist())
    demographics_df = demographics_df[
        demographics_df["location_key"].isin(location_keys)
    ].copy()

    demographics_df = demographics_df.rename(columns=SOURCE_TO_OUTPUT_COLUMNS)

    for column in DEMOGRAPHICS_COLUMNS:
        demographics_df[column] = pd.to_numeric(
            demographics_df[column],
            errors="coerce",
        ).fillna(0.0)

    demographics_df = demographics_df.merge(
        location_lookup[["location_key", "latitude", "longitude"]],
        on="location_key",
        how="left",
    )

    demographics_df = demographics_df.dropna(
        subset=["latitude", "longitude"],
    )

    if os.path.exists(DEMOGRAPHICS_SAVE_PATH):
        df_existing = pd.read_csv(DEMOGRAPHICS_SAVE_PATH)
        df_existing = df_existing[
            ~df_existing["location_key"].fillna("").str.startswith(country)
        ]
        df_combined = pd.concat([df_existing, demographics_df], ignore_index=True)
    else:
        df_combined = demographics_df

    df_combined.to_csv(DEMOGRAPHICS_SAVE_PATH, index=False)

    return demographics_df


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

    df_demographics = get_demographics_data(COUNTRY_ISO2)
    print(df_demographics.head())
