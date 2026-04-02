# 🦠 Epidemic Progression Modelling: CodeCure

A machine learning–driven system for modeling and analyzing epidemic spread using spatiotemporal data, mobility patterns, and demographic features.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run the FastAPI Server

```bash
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

Once running, the API will be available at:

```
http://127.0.0.1:8000
```

---

## 🧠 Model Training

### Step 1: Data Preprocessing

```bash
python preprocessing.py
```

### Step 2: Train Model

```bash
python model.py
```

---

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
├── main.py                # Entry point / configuration
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

## 🔧 Future Improvements

* Add experiment tracking (e.g., MLflow)
* Improve model evaluation metrics
* Add visualization dashboard (Streamlit / React)
* Hyperparameter tuning pipeline
* Multi-country comparative analysis
