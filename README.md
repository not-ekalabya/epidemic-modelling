#  Epidemic Progression Modelling: CodeCure

A machine learning–driven system for modeling and analyzing epidemic spread using spatiotemporal data, mobility patterns, and demographic features.

---

##  Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run the FastAPI Server

```bash
uvicorn server.server:app --port 8000
```

Once running, the API will be available at:

```
http://127.0.0.1:8000
```

---

## Model Training

### Step 1: Data Preprocessing

```bash
python preprocessing.py
```

### Step 2: Train Model

```bash
python model.py
```

The model is trained using a structured spatio-temporal dataset, and a cache file is generated during preprocessing to eliminate redundant data loading and significantly improve computational efficiency in subsequent runs. This caching mechanism ensures that once the dataset has been transformed into the required grid-based temporal format, it can be reused without repeating expensive preprocessing steps such as aggregation, normalization, and feature construction.

The current implementation follows a single-country training paradigm, where the model is trained independently on data corresponding to one country at a time. This allows the model to better capture region-specific transmission dynamics, demographic patterns, and mobility trends without interference from cross-country heterogeneity. Each country’s dataset is discretized into spatial grid cells (e.g., 20 km resolution), and temporal data is aggregated into weekly intervals to form consistent time-series inputs.

During training, the model constructs multiple training episodes by sliding a temporal window of length k weeks over the dataset. For each episode, the model takes as input:

historical infection trends within a target grid cell,
infection dynamics from neighboring cells within a specified spatial radius, and
auxiliary features such as mobility indicators and population density.

The model is implemented as a multi-layer perceptron (MLP) with hidden layers that learn complex non-linear relationships between spatial and temporal features. Training is performed over multiple epochs using a gradient-based optimization process, where the objective is to minimize the difference between predicted and actual infection trajectories.

To stabilize training and handle the highly skewed distribution of case counts, the target values are transformed into log space before optimization. Additionally, input features are normalized using mean and standard deviation computed from the training data.

A key aspect of the training process is the use of a trajectory-based error metric, where the discrepancy between predicted and actual cumulative case curves is measured as the area between the two curves. This formulation encourages the model to capture not only point-wise accuracy but also the overall progression trend of the epidemic.

Overall, the training pipeline is designed to efficiently learn spatio-temporal dependencies while maintaining scalability and adaptability to different regional datasets.

## 🌍 Configuring Country Data

To run the model for a specific country, modify the following variables in `main.py`:

```python
COUNTRY_ISO2 = "BD"   # Example: Bangladesh
COUNTRY_ISO3 = "BGD"
```

You can replace these with any valid ISO country codes to fetch:

* Population density
* Mobility trends
* Epidemiological data

---

## 📁 Project Structure

```
├── server/                # FastAPI backend
├── helpers/               # Data fetching and utility scripts
├── visual_analysis/       # Analysis and plotting utilities
├── data/                  # Cached datasets
├── model.py               # Core ML model
├── preprocessing.py       # Data preparation pipeline
├── requirements.txt
└── Dockerfile
```

---

## 📊 Data Pipeline Overview

The project integrates multiple datasets:

### 1. Epidemiological Data

* Source: Google COVID-19 Open Data
* Includes: cases, deaths, recoveries

### 2. Mobility Data

* Tracks population movement trends

### 3. Demographics Data

* Includes population density and regional attributes

---

## 📦 Data Handling Notes

* `helpers/data_covid.py`
  → Downloads and caches epidemiology + geography data

* `helpers/data_mobility.py`
  → Downloads and caches mobility datasets

* `helpers/data_demographics.py`
  → Downloads and caches demographic datasets

* Cached location mappings stored in:

```
data/covid/location_lookup/
```

This improves performance by avoiding repeated joins.

---

## 📈 Output & Evaluation

The model generates:

* Infection trend predictions
* Temporal progression curves
* Region-wise spread analysis

You can extend evaluation using:

* Accuracy vs baseline
* ROC-AUC
* F1-score (recommended for imbalanced data)

---

## 🧩 Conventions

### File Naming

#### `/helpers`

* Data-related utilities:

```
data_<source>.py
```

Example:

```
data_covid.py
data_mobility.py
```

#### `/visual_analysis`

* Analysis scripts:

```
analysis_<purpose>.py
```

---

### Code Practices

* Keep functions modular and reusable
* Use type hints where possible
* Document non-trivial logic

---

## 🤝 Contribution Guidelines

Before submitting a PR:

1. Update dependencies:

```bash
requirements.txt
```

2. Ensure Docker build works:

```bash
docker build .
```

3. Follow naming conventions

4. Add clear documentation for:

   * New features
   * Data sources
   * Model changes

---

## For further details, visit our Google Docs: https://docs.google.com/document/d/1urf0ZvHUWplV77SkfjjVzyYSTYORODY5IiTjLksIypY/edit?usp=sharing

## 🔧 Future Improvements

* Add experiment tracking (e.g., MLflow)
* Improve model evaluation metrics
* Add visualization dashboard (Streamlit / React)
* Hyperparameter tuning pipeline
* Multi-country comparative analysis
