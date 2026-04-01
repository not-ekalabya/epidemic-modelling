from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import SpatioTemporalCovidRLModel, load_multi_country_dataset, train_rl_covid_model


DEFAULT_COUNTRY_CONFIGS = [
    {"country_iso2": "BD", "country_iso3": "BGD", "grid_km": 20},
    {"country_iso2": "ID", "country_iso3": "IDN", "grid_km": 20},
    {"country_iso2": "PH", "country_iso3": "PHL", "grid_km": 20},
    {"country_iso2": "TH", "country_iso3": "THA", "grid_km": 20},
    {"country_iso2": "VN", "country_iso3": "VNM", "grid_km": 20},
]


class PredictionRequest(BaseModel):
    cell_lat: float = Field(..., description="Grid latitude of the cell.")
    cell_lon: float = Field(..., description="Grid longitude of the cell.")
    prediction_date: str = Field(..., description="ISO date aligned to a prediction week.")
    include_actual: bool = Field(
        default=False,
        description="Include the actual future trajectory if available in the dataset.",
    )


class ServerState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._dataset: pd.DataFrame | None = None
        self._model: SpatioTemporalCovidRLModel | None = None
        self._country_configs = self._read_country_configs()
        self._train_if_missing = self._read_bool_env("COVID_TRAIN_IF_MISSING", default=True)
        self._model_prefix = os.getenv("COVID_MODEL_PREFIX")
        self._model_output_dir = os.getenv("COVID_MODEL_OUTPUT_DIR", "models")
        self._training_kwargs = {
            "history_weeks": int(os.getenv("COVID_HISTORY_WEEKS", "12")),
            "forecast_weeks": int(os.getenv("COVID_FORECAST_WEEKS", "4")),
            "radius": float(os.getenv("COVID_RADIUS", "1.5")),
            "learning_rate": float(os.getenv("COVID_LEARNING_RATE", "0.001")),
            "noise_scale": float(os.getenv("COVID_NOISE_SCALE", "0.2")),
            "epochs": int(os.getenv("COVID_EPOCHS", "50")),
            "random_seed": int(os.getenv("COVID_RANDOM_SEED", "42")),
            "show_progress": self._read_bool_env("COVID_SHOW_PROGRESS", default=False),
        }

    def get_dataset(self) -> pd.DataFrame:
        self.ensure_ready()
        if self._dataset is None:
            raise RuntimeError("Dataset failed to initialize.")
        return self._dataset

    def get_model(self) -> SpatioTemporalCovidRLModel:
        self.ensure_ready()
        if self._model is None:
            raise RuntimeError("Model failed to initialize.")
        return self._model

    def ensure_ready(self) -> None:
        if self._dataset is not None and self._model is not None:
            return

        with self._lock:
            if self._dataset is None:
                self._dataset = load_multi_country_dataset(self._country_configs)

            if self._model is not None:
                return

            model_prefix = self._resolve_model_prefix()
            if model_prefix is not None:
                self._model = SpatioTemporalCovidRLModel.load(model_prefix)
                return

            if not self._train_if_missing:
                raise FileNotFoundError(
                    "No saved model found. Set COVID_MODEL_PREFIX or enable COVID_TRAIN_IF_MISSING."
                )

            self._model = train_rl_covid_model(
                df=self._dataset,
                **self._training_kwargs,
            )

            save_model = self._read_bool_env("COVID_SAVE_MODEL", default=True)
            if save_model:
                model_prefix = self._model.save(output_dir=self._model_output_dir)
                self._model_prefix = model_prefix

    def metadata(self) -> dict[str, Any]:
        dataset = self.get_dataset()
        model = self.get_model()
        weeks = pd.to_datetime(dataset["date"]).sort_values()
        return {
            "country_configs": self._country_configs,
            "model_prefix": self._model_prefix,
            "train_if_missing": self._train_if_missing,
            "training_kwargs": self._training_kwargs,
            "dataset_rows": int(len(dataset)),
            "grid_cell_count": int(
                dataset[["grid_lat", "grid_lon"]].drop_duplicates().shape[0]
            ),
            "date_min": weeks.iloc[0].strftime("%Y-%m-%d") if len(weeks) else None,
            "date_max": weeks.iloc[-1].strftime("%Y-%m-%d") if len(weeks) else None,
            "forecast_weeks": model.forecast_weeks,
            "history_weeks": model.history_weeks,
        }

    def _resolve_model_prefix(self) -> str | None:
        if self._model_prefix:
            return self._strip_model_extension(self._model_prefix)

        model_dir = Path(self._model_output_dir)
        if not model_dir.exists():
            return None

        candidates = sorted(model_dir.glob("spatiotemporal_covid_rl_*.npz"))
        if not candidates:
            return None

        latest = candidates[-1]
        return str(latest.with_suffix(""))

    @staticmethod
    def _strip_model_extension(model_prefix: str) -> str:
        if model_prefix.endswith(".npz") or model_prefix.endswith(".json"):
            return str(Path(model_prefix).with_suffix(""))
        return model_prefix

    @staticmethod
    def _read_bool_env(name: str, default: bool) -> bool:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _read_country_configs() -> list[dict[str, Any]]:
        raw_configs = os.getenv("COVID_COUNTRY_CONFIGS")
        if not raw_configs:
            return DEFAULT_COUNTRY_CONFIGS

        configs = json.loads(raw_configs)
        if not isinstance(configs, list) or not configs:
            raise ValueError("COVID_COUNTRY_CONFIGS must be a non-empty JSON list.")
        return configs


app = FastAPI(
    title="CodeCure Prediction API",
    version="1.0.0",
    description="HTTP interface for COVID progression predictions using the local RL model.",
)
state = ServerState()


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        metadata = state.metadata()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
    return {"status": "ok", "metadata": metadata}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    try:
        return state.metadata()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        dataset = state.get_dataset()
        model = state.get_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    matching_cells = dataset[
        (dataset["grid_lat"] == request.cell_lat)
        & (dataset["grid_lon"] == request.cell_lon)
    ]
    if matching_cells.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Cell ({request.cell_lat}, {request.cell_lon}) not found in the loaded dataset."
            ),
        )

    try:
        response: dict[str, Any] = {
            "prediction": model.predict(
                df=dataset,
                cell_lat=request.cell_lat,
                cell_lon=request.cell_lon,
                prediction_date=request.prediction_date,
            )
        }
        if request.include_actual:
            response["actual"] = model.actual_trajectory(
                df=dataset,
                cell_lat=request.cell_lat,
                cell_lon=request.cell_lon,
                prediction_date=request.prediction_date,
            )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
