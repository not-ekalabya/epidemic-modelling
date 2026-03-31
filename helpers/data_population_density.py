import requests
import os
import zipfile
import shutil

import pandas as pd
import time

def get_country_population_density(country:str) -> pd.DataFrame:
        
    if os.path.exists(f"data/population_density/{country}_population_2020.csv"):
        print(f"Population density data for {country} already exists. Skipping download.")
    else:

        # ensure directory exists

        os.makedirs("data/population_density", exist_ok=True)

        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_ASCII_gridded_XYZ/{country}_ppp_1km_2020_ASCII_gridded_XYZ.zip"

        response = requests.get(url, stream=True)

        zip_path = f"data/population_density/{country}_population_2020.zip"
        extract_path = f"data/population_density/{country}_population_2020"

        # download file

        start_time = time.time()
        with open(zip_path, "wb") as f:

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        elapsed = time.time() - start_time
                        speed = (downloaded / elapsed) / (1024 * 1024)  # MB/s
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}% ({speed:.2f} MB/s)", end="")
            
            print()  # newline after progress bar

        print("Download complete")

        # decompress

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        print("Decompression complete")

        # move CSV to correct location

        src = f"{extract_path}/ppp_{country}_2020_1km_Aggregated.csv"
        dst = f"data/population_density/{country}_population_2020.csv"

        shutil.move(src, dst)

        print("File moved to correct location")

        # remove zip file

        os.remove(zip_path)
        shutil.rmtree(extract_path)

        print("Zip file removed")

    # load pandas dataframe

    population_density_df = pd.read_csv(f"data/population_density/{country}_population_2020.csv", header=None, names=["longitude", "latitude", "population_density"])

    population_density_df = pd.read_csv(
        f"data/population_density/{country}_population_2020.csv",
        header=None,
        names=["longitude", "latitude", "population_density"],
        low_memory=False
    )

    # convert to numeric
    population_density_df["longitude"] = pd.to_numeric(population_density_df["longitude"], errors="coerce")
    population_density_df["latitude"] = pd.to_numeric(population_density_df["latitude"], errors="coerce")
    population_density_df["population_density"] = pd.to_numeric(
        population_density_df["population_density"], errors="coerce"
    )
    
    # drop invalid rows
    population_density_df = population_density_df.dropna()
    
    return population_density_df
    
    # test data loading

if __name__ == "__main__":
    country = "JPN"
    population_density_df = get_country_population_density(country)
    print(population_density_df.head())