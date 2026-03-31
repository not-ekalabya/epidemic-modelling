import os
import time
import requests
import pandas as pd
from tqdm import tqdm


# ========================
# Constants
# ========================

cwd = os.getcwd()

SAVE_DIR = "data/covid"
RAW_DATA_PATH = os.path.join(SAVE_DIR, "covid_data_raw.csv")
GEOGRAPHY_PATH = os.path.join(SAVE_DIR, "covid_geography.csv")
SAVE_PATH = os.path.join(SAVE_DIR, "covid_data.csv")

EPIDEMIOLOGY_URL = "https://storage.googleapis.com/covid19-open-data/v3/epidemiology.csv"
GEOGRAPHY_URL = "https://storage.googleapis.com/covid19-open-data/v3/geography.csv"

os.makedirs(SAVE_DIR, exist_ok=True)


# ========================
# Download Utility
# ========================

def download_file(url, path):

    if os.path.exists(path):
        return

    print(f"Downloading {os.path.basename(path)}...")

    response = requests.get(url, stream=True)
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

def get_covid_data(country: str) -> pd.DataFrame:

    # ------------------------
    # Check cached dataset
    # ------------------------

    if os.path.exists(SAVE_PATH):

        df_existing = pd.read_csv(SAVE_PATH)

        df_country = df_existing[
            df_existing["location_key"].str.startswith(country)
        ]

        # If already computed → return cached
        if len(df_country) > 0:

            print(f"Using cached coordinates for {country}")
            return df_country


    # ------------------------
    # Download datasets
    # ------------------------

    download_file(EPIDEMIOLOGY_URL, RAW_DATA_PATH)
    download_file(GEOGRAPHY_URL, GEOGRAPHY_PATH)

    print("Loading datasets...")

    df_raw = pd.read_csv(RAW_DATA_PATH)
    geo = pd.read_csv(GEOGRAPHY_PATH)

    # ------------------------
    # Filter Country
    # ------------------------

    print(f"Generating coordinates for {country}")

    df = df_raw[
        df_raw["location_key"].str.startswith(country)
    ].copy()

    # ------------------------
    # Merge Coordinates
    # ------------------------

    df = df.merge(
        geo[["location_key", "latitude", "longitude"]],
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
            ~df_existing["location_key"].str.startswith(country)
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
    
    COUNTRY_ISO2 = "JPN"

    df_covid = get_covid_data(COUNTRY_ISO2)
    print(df_covid.head())