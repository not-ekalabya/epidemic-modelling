import os
import time
import requests
import pandas as pd


# ========================
# Constants
# ========================

SAVE_DIR = "data/covid"
RAW_DATA_PATH = os.path.join(SAVE_DIR, "covid_data_raw.csv")
GEOGRAPHY_PATH = os.path.join(SAVE_DIR, "covid_geography.csv")
SAVE_PATH = os.path.join(SAVE_DIR, "covid_data.csv")
LOCATION_LOOKUP_DIR = os.path.join(SAVE_DIR, "location_lookup")

EPIDEMIOLOGY_URL = "https://storage.googleapis.com/covid19-open-data/v3/epidemiology.csv"
GEOGRAPHY_URL = "https://storage.googleapis.com/covid19-open-data/v3/geography.csv"

REQUIRED_COVID_COLUMNS = [
    "date",
    "location_key",
    "new_confirmed",
    "new_deceased",
    "new_recovered",
    "latitude",
    "longitude",
]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOCATION_LOOKUP_DIR, exist_ok=True)


# ========================
# Download Utility
# ========================

def download_file(url, path):

    if os.path.exists(path):
        return

    print(f"Downloading {os.path.basename(path)}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    downloaded = 0
    start_time = time.time()

    with open(path, "wb") as f:

        for chunk in response.iter_content(chunk_size=8192):

            if chunk:

                f.write(chunk)
                downloaded += len(chunk)

                if total_size:

                    elapsed = time.time() - start_time
                    speed = (downloaded / elapsed) / (1024 * 1024)
                    progress = downloaded / total_size * 100

                    print(
                        f"\rDownload progress: {progress:.1f}% ({speed:.2f} MB/s)",
                        end=""
                    )

    print()


# ========================
# Main Function
# ========================

def get_country_location_lookup(country: str) -> pd.DataFrame:

    lookup_path = os.path.join(
        LOCATION_LOOKUP_DIR,
        f"{country}_location_lookup.csv"
    )

    if os.path.exists(lookup_path):
        return pd.read_csv(lookup_path)

    download_file(GEOGRAPHY_URL, GEOGRAPHY_PATH)

    geo = pd.read_csv(
        GEOGRAPHY_PATH,
        usecols=["location_key", "latitude", "longitude"]
    )

    lookup_df = geo[
        geo["location_key"].fillna("").str.startswith(country)
    ].dropna(
        subset=["latitude", "longitude"]
    ).drop_duplicates(
        subset=["location_key"]
    ).copy()

    lookup_df.to_csv(
        lookup_path,
        index=False
    )

    return lookup_df


def get_covid_data(country: str) -> pd.DataFrame:

    # ------------------------
    # Check cached dataset
    # ------------------------

    if os.path.exists(SAVE_PATH):

        df_existing = pd.read_csv(SAVE_PATH)
        missing_columns = [
            column for column in REQUIRED_COVID_COLUMNS
            if column not in df_existing.columns
        ]

        df_country = df_existing[
            df_existing["location_key"].fillna("").str.startswith(country)
        ]

        # If already computed → return cached
        if len(df_country) > 0 and len(missing_columns) == 0:

            print(f"Using cached coordinates for {country}")
            return df_country


    # ------------------------
    # Download datasets
    # ------------------------

    download_file(EPIDEMIOLOGY_URL, RAW_DATA_PATH)
    download_file(GEOGRAPHY_URL, GEOGRAPHY_PATH)

    print("Loading datasets...")

    location_lookup = get_country_location_lookup(country)
    location_keys = set(location_lookup["location_key"].tolist())

    df_raw = pd.read_csv(
        RAW_DATA_PATH,
        usecols=[
            "date",
            "location_key",
            "new_confirmed",
            "new_deceased",
            "new_recovered"
        ]
    )
    df_raw = df_raw[df_raw["location_key"].isin(location_keys)].copy()

    # ------------------------
    # Filter Country
    # ------------------------

    print(f"Generating coordinates for {country}")

    df = df_raw.copy()

    # ------------------------
    # Merge Coordinates
    # ------------------------

    df = df.merge(
        location_lookup[["location_key", "latitude", "longitude"]],
        on="location_key",
        how="left"
    )

    df = df.dropna(
        subset=["latitude", "longitude"]
    )

    print(f"Rows generated: {len(df)}")

    # ------------------------
    # Save Incrementally
    # ------------------------

    if os.path.exists(SAVE_PATH):

        df_existing = pd.read_csv(SAVE_PATH)

        # remove existing country rows
        df_existing = df_existing[
            ~df_existing["location_key"].fillna("").str.startswith(country)
        ]

        df_combined = pd.concat(
            [df_existing, df],
            ignore_index=True
        )

    else:

        df_combined = df

    df_combined.to_csv(
        SAVE_PATH,
        index=False
    )

    return df

if __name__ == "__main__":
    
    COUNTRY_ISO2 = "JP"

    df_covid = get_covid_data(COUNTRY_ISO2)
    print(df_covid.head())
