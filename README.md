# Epidemic Progression Modelling: PandemicDots

PandemicDots is a spatiotemporal forecasting system for COVID progression. It combines epidemiological time series with mobility and demographic context, then predicts short-horizon future confirmed cases at grid-cell level and country-total level.

## What Is Included

- FastAPI backend for metadata, prediction, and country configuration endpoints
- Python training pipeline for a supervised spatiotemporal forecasting model
- Data helper modules for COVID, mobility, demographics, and population density sources
- Next.js client app under `client/` for interactive visualization and API interaction

## Prerequisites

- Python 3.11 recommended
- pip
- Node.js 20+ and npm (for the frontend)
- Optional for containerized run: Docker and Docker Compose plugin

## Local Setup (Backend + Model)

1. Clone and enter the repository.

```bash
git clone <your-repo-url>
cd PandemicDots
```

2. Create and activate a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install backend dependencies.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Start the API server.

```bash
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

5. Verify API health.

```bash
curl http://127.0.0.1:8000/health
```

Useful API routes:

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /config/country`

## Local Setup (Frontend)

1. Open a new terminal and install frontend dependencies.

```bash
cd client
npm install
```

2. Configure API base URL for the Next.js route handlers.

```bash
echo "PREDICTION_API_BASE_URL=http://127.0.0.1:8000" > .env.local
```

3. Start frontend development server.

```bash
npm run dev
```

4. Open `http://localhost:3000`.

## Container Setup

The repository includes a backend Docker build file at `dockerfile`.

Build the backend image:

```bash
docker build -f dockerfile -t pandemicdots-api .
```

Run the backend container:

```bash
docker run --rm -p 8000:8000 pandemicdots-api
```

If you add a Compose file later for full-stack orchestration, connect frontend and backend through internal service DNS and set `PREDICTION_API_BASE_URL` accordingly.

## Training Workflow

### Step 1: Build/refresh dataset features

The preprocessing logic is integrated via `data_preprocessing.py` and invoked by model-loading utilities.

### Step 2: Train model

```bash
python model.py
```

This script:

- loads spatiotemporal country data
- trains the forecasting model
- computes metrics
- emits a country progression report
- optionally saves model artifacts under `models/`

## Model Architecture (Technical Overview)

The core model class is `SpatioTemporalCovidRLModel` in `model.py`. Despite the historical name, the training path is supervised regression over forecast trajectories.

### Input representation

For each training episode, the state vector is formed by concatenating:

- target-cell history over `history_weeks`
- neighborhood history over `history_weeks` within radius `radius`
- optional external features based on enabled `data_sources`:
  - mobility indicators
  - demographics indicators
  - population density

The dataset is aggregated into monthly buckets (`to_period("M")`) and grouped by spatial grid coordinates.

### Network structure

- Feed-forward MLP with configurable hidden dimensions (`hidden_dims`)
- Default hidden layers: `(128, 64)`
- Output dimension: `forecast_weeks` (multi-step trajectory)
- Output clipped to non-negative bounded range via `output_clip_max`

### Optimization and stabilization

- Targets transformed with `log1p` before optimization
- Feature-wise normalization using training mean and std (`feature_mean_`, `feature_std_`)
- Learning rate decay (`learning_rate_decay`)
- L2 regularization (`l2_regularization`)
- Optional training noise scaling (`noise_scale`)

### Prediction outputs

For each reference month, the model produces:

- next-step predicted confirmed cases
- full multi-step trajectory of length `forecast_weeks`
- optional per-cell decomposition in country-level reports

### Evaluation metrics

Implemented scoring includes:

- RMSE
- MAE
- MAPE
- $R^2$
- mean reward (trajectory-based)
- mean area error between predicted and actual trajectories

The plotting utility `visualize_prediction_progress` compares monthly predictions and cumulative curves and highlights area error.

## Runtime Configuration

The API behavior can be controlled using environment variables in `server/server.py`:

- `COVID_COUNTRY_CONFIGS` (JSON list of country configs)
- `COVID_DATA_SOURCES` (comma-separated)
- `COVID_TRAIN_IF_MISSING`
- `COVID_MODEL_PREFIX`
- `COVID_MODEL_OUTPUT_DIR`
- `COVID_HISTORY_WEEKS`
- `COVID_FORECAST_WEEKS`
- `COVID_RADIUS`
- `COVID_LEARNING_RATE`
- `COVID_NOISE_SCALE`
- `COVID_EPOCHS`
- `COVID_RANDOM_SEED`
- `COVID_SHOW_PROGRESS`
- `COVID_SAVE_MODEL`

Example:

```bash
export COVID_DATA_SOURCES=mobility,demographics
export COVID_HISTORY_WEEKS=12
export COVID_FORECAST_WEEKS=4
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

## Project Structure

```text
.
├── server/                    # FastAPI backend
├── client/                    # Next.js frontend
├── helpers/                   # Source-specific data helpers
├── data/                      # Raw/cache data artifacts
├── figures/                   # Generated visual outputs
├── models/                    # Saved model artifacts (.npz + .json)
├── data_preprocessing.py      # Feature assembly pipeline
├── model.py                   # Model training/inference logic
├── requirements.txt
└── dockerfile
```

## Contribution Notes

Before opening a PR:

1. Validate backend boots and `/health` returns success.
2. Run model training path at least once for regression checks.
3. Keep new data sources isolated in `helpers/data_<source>.py`.
4. Update documentation for any new source, feature column, or model behavior change.

## Additional Reference

Google Docs: https://docs.google.com/document/d/1urf0ZvHUWplV77SkfjjVzyYSTYORODY5IiTjLksIypY/edit?usp=sharing

## Future Improvements

- Add experiment tracking (MLflow or equivalent)
- Add automated backtesting across multiple rolling windows
- Add model versioning and reproducibility manifests
- Expand evaluation dashboard in frontend
- Add multi-country comparative training benchmarks
