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

from helpers.data_covid import get_country_total_covid_data
from model import SpatioTemporalCovidRLModel, load_multi_country_dataset, train_rl_covid_model

COVID_DATA_SOURCES = ["mobility", "demographics"]


DEFAULT_COUNTRY_CONFIGS = [
    {"country_iso2": "BD", "country_iso3": "BGD", "grid_km": 20},
]


class PredictionRequest(BaseModel):
    prediction_date: str = Field(..., description="ISO date aligned to a prediction month.")
    include_actual: bool = Field(
        default=False,
        description="Include the actual future country trajectory if available in the dataset.",
    )


class CountryConfigRequest(BaseModel):
    country_iso2: str = Field(..., min_length=2, max_length=2, description="ISO2 country code.")
    country_iso3: str = Field(..., min_length=3, max_length=3, description="ISO3 country code.")
    grid_km: int = Field(default=20, ge=1, le=200, description="Grid resolution in kilometers.")


class ServerState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._dataset: pd.DataFrame | None = None
        self._model: SpatioTemporalCovidRLModel | None = None
        self._country_actuals: pd.DataFrame | None = None
        self._country_configs = self._read_country_configs()
        self._data_sources = self._read_data_sources()
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
            "data_sources": self._data_sources,
        }
        self._allow_latest_model_discovery = True

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

    def get_country_actuals(self) -> pd.DataFrame:
        self.ensure_ready()
        if self._country_actuals is None:
            raise RuntimeError("Country actuals failed to initialize.")
        return self._country_actuals

    def ensure_ready(self) -> None:
        if self._dataset is not None and self._model is not None:
            return

        with self._lock:
            if self._dataset is None:
                self._dataset = load_multi_country_dataset(
                    self._country_configs,
                    data_sources=self._data_sources,
                )
                self._country_actuals = self._load_country_actuals()

            if self._model is not None:
                return

            model_prefix = self._resolve_model_prefix()
            if model_prefix is not None:
                loaded_model = SpatioTemporalCovidRLModel.load(model_prefix)
                if loaded_model.data_sources == self._data_sources:
                    self._model = loaded_model
                    return
                if not self._train_if_missing:
                    raise ValueError(
                        "Saved model data_sources "
                        f"{loaded_model.data_sources} do not match server data_sources "
                        f"{self._data_sources}. Retraining is disabled."
                    )

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

    def country_actual_trajectory(
        self,
        prediction_date: str,
        forecast_weeks: int,
    ) -> dict[str, Any]:
        country_actuals = self.get_country_actuals().copy()
        reference_week = pd.to_datetime(prediction_date).to_period("M").start_time
        future_df = country_actuals[country_actuals["week"] > reference_week].head(forecast_weeks)
        actual = future_df["new_confirmed"].to_numpy(dtype=float)

        if len(actual) < forecast_weeks:
            actual = pd.Series(actual, dtype=float).reindex(range(forecast_weeks), fill_value=0.0).to_numpy(dtype=float)

        return {
            "reference_week": reference_week,
            "actual_trajectory": actual.tolist(),
        }

    def metadata(self) -> dict[str, Any]:
        dataset = self.get_dataset()
        model = self.get_model()
        weeks = pd.to_datetime(dataset["date"]).sort_values()
        return {
            "country_configs": self._country_configs,
            "data_sources": self._data_sources,
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

    def update_country_config(
        self,
        country_iso2: str,
        country_iso3: str,
        grid_km: int,
    ) -> None:
        next_config = [{
            "country_iso2": country_iso2,
            "country_iso3": country_iso3,
            "grid_km": grid_km,
        }]

        with self._lock:
            self._country_configs = next_config
            self._dataset = None
            self._model = None
            self._country_actuals = None
            self._model_prefix = None
            self._allow_latest_model_discovery = False

    def _load_country_actuals(self) -> pd.DataFrame:
        country_frames: list[pd.DataFrame] = []

        for config in self._country_configs:
            country_iso2 = str(config["country_iso2"])
            country_df = get_country_total_covid_data(country_iso2).copy()
            country_df["date"] = pd.to_datetime(country_df["date"])
            country_df["week"] = country_df["date"].dt.to_period("M").dt.start_time
            country_frames.append(country_df)

        combined = pd.concat(country_frames, ignore_index=True)
        return (
            combined.groupby("week", as_index=False)
            .agg(new_confirmed=("new_confirmed", "sum"))
            .sort_values("week")
            .reset_index(drop=True)
        )

    def _resolve_model_prefix(self) -> str | None:
        if self._model_prefix:
            return self._strip_model_extension(self._model_prefix)

        if not self._allow_latest_model_discovery:
            return None

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

    @staticmethod
    def _read_data_sources() -> list[str]:
        raw_sources = os.getenv("COVID_DATA_SOURCES")
        if not raw_sources:
            return COVID_DATA_SOURCES

        data_sources = [
            source.strip()
            for source in raw_sources.split(",")
            if source.strip()
        ]
        if not data_sources:
            raise ValueError("COVID_DATA_SOURCES must contain at least one source.")
        return data_sources


app = FastAPI(
    title="PandemicDots Prediction API",
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


@app.post("/config/country")
def configure_country(request: CountryConfigRequest) -> dict[str, Any]:
    country_iso2 = request.country_iso2.strip().upper()
    country_iso3 = request.country_iso3.strip().upper()

    if not country_iso2.isalpha() or len(country_iso2) != 2:
        raise HTTPException(status_code=400, detail="country_iso2 must be a 2-letter ISO code.")
    if not country_iso3.isalpha() or len(country_iso3) != 3:
        raise HTTPException(status_code=400, detail="country_iso3 must be a 3-letter ISO code.")

    previous_configs = state._country_configs.copy()
    previous_dataset = state._dataset
    previous_model = state._model
    previous_actuals = state._country_actuals
    previous_prefix = state._model_prefix
    previous_discovery = state._allow_latest_model_discovery

    state.update_country_config(
        country_iso2=country_iso2,
        country_iso3=country_iso3,
        grid_km=int(request.grid_km),
    )

    try:
        state.ensure_ready()
        return {"status": "ok", "metadata": state.metadata()}
    except Exception as exc:
        with state._lock:
            state._country_configs = previous_configs
            state._dataset = previous_dataset
            state._model = previous_model
            state._country_actuals = previous_actuals
            state._model_prefix = previous_prefix
            state._allow_latest_model_discovery = previous_discovery
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        dataset = state.get_dataset()
        model = state.get_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        country_report = model.country_progression_report(
            df=dataset,
            reference_week=request.prediction_date,
            require_nonzero_actuals=request.include_actual,
            fallback_to_latest_nonzero_actual=request.include_actual,
        )
        per_cell_progression = country_report.get("per_cell_progression", [])
        if not request.include_actual:
            per_cell_progression = [
                {
                    "grid_lat": cell["grid_lat"],
                    "grid_lon": cell["grid_lon"],
                    "predicted_next_confirmed": cell["predicted_next_confirmed"],
                    "predicted_trajectory": cell["predicted_trajectory"],
                }
                for cell in per_cell_progression
            ]

        response: dict[str, Any] = {
            "prediction": {
                "requested_reference_week": country_report.get("requested_reference_week"),
                "reference_week": country_report["reference_week"],
                "predicted_next_confirmed": float(country_report["predicted_trajectory"][0]),
                "predicted_trajectory": country_report["predicted_trajectory"],
                "cell_count": country_report["cell_count"],
                "per_cell_progression": per_cell_progression,
            }
        }
        if request.include_actual:
            response["actual"] = state.country_actual_trajectory(
                prediction_date=str(country_report["reference_week"]),
                forecast_weeks=model.forecast_weeks,
            )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
