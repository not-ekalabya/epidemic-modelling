from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_preprocessing import get_spatiotemporal_covid_dataset
from helpers.data_mobility import MOBILITY_COLUMNS


@dataclass
class EpisodeBatch:
    states: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame


class SpatioTemporalCovidRLModel:
    """
    Policy-gradient model for forecasting confirmed cases in a grid cell.

    The policy observes:
    - last `history_weeks` of target-cell progression
    - last `history_weeks` of neighborhood progression inside radius `radius`
    - optional external source features such as mobility or population density

    The action is a non-negative future trajectory of confirmed cases over
    `forecast_weeks`. Reward is inverse to the area between predicted and
    actual cumulative-confirmed curves.
    """

    def __init__(
        self,
        history_weeks: int = 4,
        forecast_weeks: int = 2,
        radius: float = 0.5,
        learning_rate: float = 0.001,
        noise_scale: float = 0.2,
        epochs: int = 250,
        pretrain_epochs: int = 100,
        random_seed: int = 42,
        show_progress: bool = True,
        data_sources: list[str] | None = None,
    ) -> None:
        if history_weeks < 1:
            raise ValueError("history_weeks must be >= 1")
        if forecast_weeks < 1:
            raise ValueError("forecast_weeks must be >= 1")
        if radius < 0:
            raise ValueError("radius must be >= 0")

        self.history_weeks = history_weeks
        self.forecast_weeks = forecast_weeks
        self.radius = radius
        self.learning_rate = learning_rate
        self.noise_scale = noise_scale
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.random_seed = random_seed
        self.show_progress = show_progress
        self.data_sources = list(data_sources or ["population_density", "mobility"])

        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.training_history_: list[dict[str, float]] = []
        self.weekly_df_: pd.DataFrame | None = None
        self.grid_spacing_: float | None = None

    def fit(self, df: pd.DataFrame) -> "SpatioTemporalCovidRLModel":
        weekly_df = self._prepare_weekly_dataframe(df)
        batch = self._build_episode_batch(weekly_df)

        if len(batch.states) == 0:
            raise ValueError(
                "Not enough spatiotemporal history to build training episodes."
            )

        states = batch.states.astype(float)
        targets = batch.targets.astype(float)

        self.feature_mean_ = states.mean(axis=0)
        self.feature_std_ = states.std(axis=0)
        self.feature_std_[self.feature_std_ == 0] = 1.0

        x = self._normalize_states(states)
        rng = np.random.default_rng(self.random_seed)

        self.weights = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(x.shape[1], self.forecast_weeks),
        )
        self.bias = self._inverse_softplus(targets.mean(axis=0))
        baseline = 0.0

        self.training_history_ = []

        if self.pretrain_epochs > 0:
            pretrain_iterator = range(self.pretrain_epochs)
            if self.show_progress:
                pretrain_iterator = tqdm(
                    pretrain_iterator,
                    total=self.pretrain_epochs,
                    desc="Supervised warm start",
                )

            for _ in pretrain_iterator:
                logits = x @ self.weights + self.bias
                predictions = self._positive_output(logits)
                grad_logits = (
                    2.0 * (predictions - targets) * self._sigmoid(logits)
                ) / len(x)

                self.weights -= self.learning_rate * (x.T @ grad_logits)
                self.bias -= self.learning_rate * grad_logits.sum(axis=0)

                if self.show_progress:
                    rmse = float(np.sqrt(np.mean(np.square(predictions - targets))))
                    pretrain_iterator.set_postfix(rmse=f"{rmse:.2f}")

        epoch_iterator = range(self.epochs)
        if self.show_progress:
            epoch_iterator = tqdm(
                epoch_iterator,
                total=self.epochs,
                desc="Training RL model",
            )

        for epoch in epoch_iterator:
            logits = x @ self.weights + self.bias
            means = self._positive_output(logits)
            noise = rng.normal(size=logits.shape)
            sampled_logits = logits + self.noise_scale * noise
            actions = self._positive_output(sampled_logits)
            rewards = self._reward(actions, targets)

            baseline = 0.9 * baseline + 0.1 * float(rewards.mean())
            advantages = rewards - baseline

            if self.noise_scale <= 0:
                raise ValueError("noise_scale must be positive for policy-gradient")

            grad_logits = (
                (sampled_logits - logits) / (self.noise_scale ** 2)
            ) * advantages[:, None] / len(x)

            self.weights += self.learning_rate * x.T @ grad_logits
            self.bias += self.learning_rate * grad_logits.sum(axis=0)

            mean_area = float(self._area(actions, targets).mean())
            self.training_history_.append(
                {
                    "epoch": float(epoch + 1),
                    "reward": float(rewards.mean()),
                    "mean_area": mean_area,
                }
            )
            if self.show_progress:
                epoch_iterator.set_postfix(
                    reward=f"{rewards.mean():.4f}",
                    area=f"{mean_area:.2f}",
                )

        self.weekly_df_ = weekly_df
        return self

    def predict(
        self,
        df: pd.DataFrame,
        cell_lat: float,
        cell_lon: float,
        prediction_date: str | pd.Timestamp,
    ) -> dict[str, object]:
        self._check_is_fitted()

        weekly_df = self._prepare_weekly_dataframe(df)
        prediction_date = pd.to_datetime(prediction_date).to_period("M").start_time
        resolved_lat, resolved_lon = self.resolve_cell_coordinates(
            cell_lat=cell_lat,
            cell_lon=cell_lon,
            weekly_df=weekly_df,
        )
        cell_df = weekly_df[
            (weekly_df["grid_lat"] == resolved_lat)
            & (weekly_df["grid_lon"] == resolved_lon)
        ].sort_values("week").reset_index(drop=True)
        cell_history_df = self._cell_history_frame(cell_df)
        neighborhood_df = self._neighborhood_frame(weekly_df, resolved_lat, resolved_lon)
        neighborhood_history_df = self._neighborhood_history_frame(neighborhood_df)
        demographic_features = self._demographic_summary(
            neighborhood_df,
            resolved_lat,
            resolved_lon,
        )
        state_vector = self._build_single_state(
            cell_history_df=cell_history_df,
            neighborhood_history_df=neighborhood_history_df,
            demographic_features=demographic_features,
            reference_week=prediction_date,
        )
        x = self._normalize_states(state_vector[None, :])
        trajectory = self._positive_output(x @ self.weights + self.bias)[0]

        return {
            "cell": {"grid_lat": resolved_lat, "grid_lon": resolved_lon},
            "reference_week": prediction_date,
            "predicted_next_confirmed": float(trajectory[0]),
            "predicted_trajectory": trajectory.tolist(),
        }

    def score(self, df: pd.DataFrame) -> dict[str, float]:
        self._check_is_fitted()
        weekly_df = self._prepare_weekly_dataframe(df)
        batch = self._build_episode_batch(weekly_df)

        if len(batch.states) == 0:
            raise ValueError("No evaluation samples available.")

        x = self._normalize_states(batch.states)
        predictions = self._positive_output(x @ self.weights + self.bias)
        rewards = self._reward(predictions, batch.targets)
        area = self._area(predictions, batch.targets)
        errors = predictions - batch.targets

        rmse = float(np.sqrt(np.mean(np.square(errors))))
        mae = float(np.mean(np.abs(errors)))
        target_mean = float(np.mean(batch.targets))
        ss_res = float(np.sum(np.square(errors)))
        ss_tot = float(
            np.sum(np.square(batch.targets - np.mean(batch.targets)))
        )
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        mape_denominator = np.where(batch.targets == 0, np.nan, batch.targets)
        mape = float(
            np.nanmean(np.abs(errors) / np.abs(mape_denominator)) * 100.0
        )

        return {
            "mean_reward": float(rewards.mean()),
            "mean_area": float(area.mean()),
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": float(r2),
            "target_mean": target_mean,
            "sample_count": float(len(batch.states)),
        }

    def evaluation_batch(self, df: pd.DataFrame) -> EpisodeBatch:
        weekly_df = self._prepare_weekly_dataframe(df)
        return self._build_episode_batch(weekly_df)

    def actual_trajectory(
        self,
        df: pd.DataFrame,
        cell_lat: float,
        cell_lon: float,
        prediction_date: str | pd.Timestamp,
    ) -> dict[str, object]:
        weekly_df = self._prepare_weekly_dataframe(df)
        prediction_date = pd.to_datetime(prediction_date).to_period("M").start_time
        resolved_lat, resolved_lon = self.resolve_cell_coordinates(
            cell_lat=cell_lat,
            cell_lon=cell_lon,
            weekly_df=weekly_df,
        )

        cell_df = weekly_df[
            (weekly_df["grid_lat"] == resolved_lat)
            & (weekly_df["grid_lon"] == resolved_lon)
        ].sort_values("week")

        future_df = cell_df[cell_df["week"] > prediction_date].head(self.forecast_weeks)
        actual = future_df["new_confirmed"].to_numpy(dtype=float)

        if len(actual) < self.forecast_weeks:
            actual = np.pad(
                actual,
                (0, self.forecast_weeks - len(actual)),
                mode="constant",
                constant_values=0.0,
            )

        return {
            "cell": {"grid_lat": resolved_lat, "grid_lon": resolved_lon},
            "reference_week": prediction_date,
            "actual_trajectory": actual.tolist(),
        }

    def save(self, output_dir: str = "models", version: str | None = None) -> str:
        self._check_is_fitted()
        os.makedirs(output_dir, exist_ok=True)

        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_prefix = os.path.join(output_dir, f"spatiotemporal_covid_rl_{version}")
        np.savez(
            f"{model_prefix}.npz",
            weights=self.weights,
            bias=self.bias,
            feature_mean=self.feature_mean_,
            feature_std=self.feature_std_,
        )

        metadata = {
            "history_weeks": self.history_weeks,
            "forecast_weeks": self.forecast_weeks,
            "radius": self.radius,
            "learning_rate": self.learning_rate,
            "noise_scale": self.noise_scale,
            "epochs": self.epochs,
            "random_seed": self.random_seed,
            "data_sources": self.data_sources,
            "training_history": self.training_history_,
        }
        with open(f"{model_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return model_prefix

    @classmethod
    def load(cls, model_prefix: str) -> "SpatioTemporalCovidRLModel":
        metadata_path = f"{model_prefix}.json"
        weights_path = f"{model_prefix}.npz"

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model = cls(
            history_weeks=int(metadata["history_weeks"]),
            forecast_weeks=int(metadata["forecast_weeks"]),
            radius=float(metadata["radius"]),
            learning_rate=float(metadata["learning_rate"]),
            noise_scale=float(metadata["noise_scale"]),
            epochs=int(metadata["epochs"]),
            random_seed=int(metadata["random_seed"]),
            show_progress=False,
            data_sources=metadata.get("data_sources"),
        )

        weights = np.load(weights_path)
        model.weights = weights["weights"]
        model.bias = weights["bias"]
        model.feature_mean_ = weights["feature_mean"]
        model.feature_std_ = weights["feature_std"]
        model.training_history_ = metadata.get("training_history", [])

        return model

    def country_progression_report(
        self,
        df: pd.DataFrame,
        reference_week: str | pd.Timestamp | None = None,
    ) -> dict[str, object]:
        batch = self.evaluation_batch(df)
        if len(batch.metadata) == 0:
            raise ValueError("No valid evaluation window available for country reporting.")

        metadata = batch.metadata.copy()
        metadata["reference_week"] = pd.to_datetime(metadata["reference_week"])

        if reference_week is None:
            selected_week = metadata["reference_week"].max()
        else:
            selected_week = pd.to_datetime(reference_week).to_period("M").start_time

        selected_mask = metadata["reference_week"] == selected_week
        if not selected_mask.any():
            raise ValueError(f"No evaluation episodes found for reference week {selected_week}.")

        states = batch.states[selected_mask.to_numpy()]
        actuals = batch.targets[selected_mask.to_numpy()]
        predictions = self._positive_output(
            self._normalize_states(states) @ self.weights + self.bias
        )

        return {
            "reference_week": selected_week,
            "cell_count": int(selected_mask.sum()),
            "predicted_trajectory": predictions.sum(axis=0).tolist(),
            "actual_trajectory": actuals.sum(axis=0).tolist(),
        }

    def _prepare_weekly_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {
            "date",
            "grid_lat",
            "grid_lon",
            "new_confirmed",
            "new_deceased",
            "new_recovered",
        }
        if "population_density" in self.data_sources:
            required_columns.add("population_density")
        if "mobility" in self.data_sources:
            required_columns.update(MOBILITY_COLUMNS)
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        weekly_df = df.copy()
        weekly_df["date"] = pd.to_datetime(weekly_df["date"])
        weekly_df["week"] = weekly_df["date"].dt.to_period("M").dt.start_time
        cell_key_columns = self._cell_key_columns(weekly_df)

        weekly_df = (
            weekly_df.groupby(["week", *cell_key_columns], as_index=False)
            .agg(
                new_confirmed=("new_confirmed", "sum"),
                new_deceased=("new_deceased", "sum"),
                new_recovered=("new_recovered", "sum"),
                **(
                    {"population_density": ("population_density", "max")}
                    if "population_density" in self.data_sources
                    else {}
                ),
                **(
                    {
                        column: (column, "mean")
                        for column in MOBILITY_COLUMNS
                    }
                    if "mobility" in self.data_sources
                    else {}
                ),
            )
            .sort_values([*cell_key_columns, "week"])
        )

        weekly_df["cumulative_confirmed"] = (
            weekly_df.groupby(cell_key_columns)["new_confirmed"].cumsum()
        )

        unique_lats = np.sort(weekly_df["grid_lat"].unique())
        if len(unique_lats) > 1:
            self.grid_spacing_ = float(np.min(np.diff(unique_lats)))
        else:
            self.grid_spacing_ = 1.0

        return weekly_df

    def resolve_cell_coordinates(
        self,
        cell_lat: float,
        cell_lon: float,
        df: pd.DataFrame | None = None,
        weekly_df: pd.DataFrame | None = None,
    ) -> tuple[float, float]:
        if weekly_df is None:
            if df is None:
                raise ValueError("Either df or weekly_df must be provided.")
            weekly_df = self._prepare_weekly_dataframe(df)

        available_cells = weekly_df[["grid_lat", "grid_lon"]].drop_duplicates()

        exact_match = available_cells[
            (available_cells["grid_lat"] == cell_lat)
            & (available_cells["grid_lon"] == cell_lon)
        ]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return float(row["grid_lat"]), float(row["grid_lon"])

        grid_spacing = float(self.grid_spacing_ or 1.0)
        resolved_lat = float(np.floor(cell_lat / grid_spacing) * grid_spacing)
        resolved_lon = float(np.floor(cell_lon / grid_spacing) * grid_spacing)

        snapped_match = available_cells[
            np.isclose(available_cells["grid_lat"], resolved_lat)
            & np.isclose(available_cells["grid_lon"], resolved_lon)
        ]
        if not snapped_match.empty:
            row = snapped_match.iloc[0]
            return float(row["grid_lat"]), float(row["grid_lon"])

        cell_coordinates = available_cells.to_numpy(dtype=float)
        distances = np.sqrt(
            (cell_coordinates[:, 0] - cell_lat) ** 2
            + (cell_coordinates[:, 1] - cell_lon) ** 2
        )
        nearest_index = int(np.argmin(distances))
        row = available_cells.iloc[nearest_index]
        return float(row["grid_lat"]), float(row["grid_lon"])

    @staticmethod
    def _cell_key_columns(df: pd.DataFrame) -> list[str]:
        key_columns = ["grid_lat", "grid_lon"]
        for column in ("country_iso2", "country_iso3"):
            if column in df.columns:
                key_columns.insert(0, column)
        return key_columns

    def _build_episode_batch(self, weekly_df: pd.DataFrame) -> EpisodeBatch:
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        metadata_rows: list[dict[str, object]] = []
        cell_key_columns = self._cell_key_columns(weekly_df)
        coordinate_columns = ["grid_lat", "grid_lon"]

        cell_groups = list(weekly_df.groupby(cell_key_columns, sort=False))
        cell_frames = {
            self._normalize_cell_group_key(cell_key): cell_df.sort_values("week").reset_index(drop=True)
            for cell_key, cell_df in cell_groups
        }
        neighborhood_map = self._neighborhood_coordinate_map(
            list(cell_frames.keys())
        )

        group_iterator = list(cell_frames.items())
        if self.show_progress:
            group_iterator = tqdm(
                group_iterator,
                total=len(cell_frames),
                desc="Building episodes",
            )

        for cell_key, cell_df in group_iterator:
            if len(cell_df) < self.history_weeks + self.forecast_weeks:
                continue

            cell_lat = float(cell_df["grid_lat"].iloc[0])
            cell_lon = float(cell_df["grid_lon"].iloc[0])
            cell_history_df = self._cell_history_frame(cell_df)
            neighborhood_df = pd.concat(
                [
                    cell_frames[neighbor_coordinate]
                    for neighbor_coordinate in neighborhood_map[cell_key]
                ],
                ignore_index=True,
            )
            neighborhood_history_df = self._neighborhood_history_frame(neighborhood_df)
            demographic_features = self._demographic_summary(
                neighborhood_df,
                cell_lat,
                cell_lon,
            )

            for idx in range(self.history_weeks, len(cell_df) - self.forecast_weeks + 1):
                reference_week = cell_df.loc[idx - 1, "week"]
                state = self._build_single_state(
                    cell_history_df=cell_history_df,
                    neighborhood_history_df=neighborhood_history_df,
                    demographic_features=demographic_features,
                    reference_week=reference_week,
                )
                future = (
                    cell_df.loc[idx: idx + self.forecast_weeks - 1, "new_confirmed"]
                    .to_numpy(dtype=float)
                )
                rows.append(state)
                targets.append(future)
                metadata_rows.append(
                    {
                        "grid_lat": cell_lat,
                        "grid_lon": cell_lon,
                        "reference_week": reference_week,
                        **{
                            column: cell_df[column].iloc[0]
                            for column in cell_key_columns
                            if column not in coordinate_columns
                        },
                    }
                )

        if not rows:
            return EpisodeBatch(
                states=np.empty((0, 0)),
                targets=np.empty((0, self.forecast_weeks)),
                metadata=pd.DataFrame(columns=["grid_lat", "grid_lon", "reference_week"]),
            )

        return EpisodeBatch(
            states=np.vstack(rows),
            targets=np.vstack(targets),
            metadata=pd.DataFrame(metadata_rows),
        )

    def _build_single_state(
        self,
        cell_history_df: pd.DataFrame,
        neighborhood_history_df: pd.DataFrame,
        demographic_features: np.ndarray,
        reference_week: pd.Timestamp,
    ) -> np.ndarray:
        weeks = pd.date_range(
            end=reference_week,
            periods=self.history_weeks,
            freq="MS",
        )
        weeks = pd.to_datetime(weeks).normalize()

        cell_history = self._cell_history(cell_history_df, weeks)
        neighborhood_history = self._neighborhood_history(
            neighborhood_history_df,
            weeks,
        )

        return np.concatenate(
            [cell_history, neighborhood_history, demographic_features]
        ).astype(float)

    def _cell_history_frame(self, cell_df: pd.DataFrame) -> pd.DataFrame:
        mobility_columns = MOBILITY_COLUMNS if "mobility" in self.data_sources else []
        return cell_df[
            [
                "week",
                "new_confirmed",
                "new_deceased",
                "new_recovered",
                "cumulative_confirmed",
                *mobility_columns,
            ]
        ].set_index("week")

    def _cell_history(
        self,
        cell_history_df: pd.DataFrame,
        weeks: pd.DatetimeIndex,
    ) -> np.ndarray:
        return (
            cell_history_df.reindex(weeks, fill_value=0.0)
            .to_numpy(dtype=float)
            .reshape(-1)
        )

    def _neighborhood_frame(
        self,
        weekly_df: pd.DataFrame,
        cell_lat: float,
        cell_lon: float,
    ) -> pd.DataFrame:
        distances = np.sqrt(
            (weekly_df["grid_lat"] - cell_lat) ** 2
            + (weekly_df["grid_lon"] - cell_lon) ** 2
        )
        radius_in_degrees = self.radius * float(self.grid_spacing_ or 1.0)
        return weekly_df[distances <= radius_in_degrees].copy()

    def _neighborhood_coordinate_map(
        self,
        cell_keys: list[tuple[object, ...]],
    ) -> dict[tuple[object, ...], list[tuple[object, ...]]]:
        if not cell_keys:
            return {}

        coordinate_values = np.asarray(
            [[float(cell_key[-2]), float(cell_key[-1])] for cell_key in cell_keys],
            dtype=float,
        )
        radius_in_degrees = self.radius * float(self.grid_spacing_ or 1.0)
        mapping: dict[tuple[object, ...], list[tuple[object, ...]]] = {}

        for idx, cell_key in enumerate(cell_keys):
            same_partition_indices = [
                index
                for index, other_key in enumerate(cell_keys)
                if other_key[:-2] == cell_key[:-2]
            ]
            partition_coordinates = coordinate_values[same_partition_indices]
            cell_lat, cell_lon = coordinate_values[idx]
            distances = np.sqrt(
                (partition_coordinates[:, 0] - cell_lat) ** 2
                + (partition_coordinates[:, 1] - cell_lon) ** 2
            )
            neighbor_indices = np.flatnonzero(distances <= radius_in_degrees)
            mapping[cell_key] = [
                cell_keys[same_partition_indices[neighbor_index]]
                for neighbor_index in neighbor_indices
            ]

        return mapping

    @staticmethod
    def _normalize_cell_group_key(cell_key: object) -> tuple[object, ...]:
        if isinstance(cell_key, tuple):
            normalized = list(cell_key)
        else:
            normalized = [cell_key]

        normalized[-2] = float(normalized[-2])
        normalized[-1] = float(normalized[-1])
        return tuple(normalized)

    def _neighborhood_history_frame(
        self,
        neighborhood_df: pd.DataFrame,
    ) -> pd.DataFrame:
        mobility_aggregations = (
            {
                f"{column}_mean": (column, "mean")
                for column in MOBILITY_COLUMNS
            }
            if "mobility" in self.data_sources
            else {}
        )
        return (
            neighborhood_df.groupby("week", as_index=True)
            .agg(
                new_confirmed_sum=("new_confirmed", "sum"),
                new_confirmed_mean=("new_confirmed", "mean"),
                cumulative_confirmed_sum=("cumulative_confirmed", "sum"),
                new_deceased_sum=("new_deceased", "sum"),
                new_recovered_sum=("new_recovered", "sum"),
                **mobility_aggregations,
            )
        )

    def _neighborhood_history(
        self,
        neighborhood_history_df: pd.DataFrame,
        weeks: pd.DatetimeIndex,
    ) -> np.ndarray:
        return (
            neighborhood_history_df.reindex(weeks, fill_value=0.0)
            .to_numpy(dtype=float)
            .reshape(-1)
        )

    def _demographic_summary(
        self,
        neighborhood_df: pd.DataFrame,
        cell_lat: float,
        cell_lon: float,
    ) -> np.ndarray:
        if "population_density" not in self.data_sources:
            return np.empty(0, dtype=float)

        cell_density = neighborhood_df[
            (neighborhood_df["grid_lat"] == cell_lat)
            & (neighborhood_df["grid_lon"] == cell_lon)
        ]["population_density"]
        cell_density_value = float(cell_density.max()) if len(cell_density) else 0.0

        densities = (
            neighborhood_df[["grid_lat", "grid_lon", "population_density"]]
            .drop_duplicates(subset=["grid_lat", "grid_lon"])
            ["population_density"]
            .to_numpy(dtype=float)
        )

        if len(densities) == 0:
            return np.zeros(5, dtype=float)

        return np.array(
            [
                cell_density_value,
                float(densities.sum()),
                float(densities.mean()),
                float(densities.max()),
                float(len(densities)),
            ],
            dtype=float,
        )

    def _normalize_states(self, states: np.ndarray) -> np.ndarray:
        self._check_is_fitted(allow_pre_fit_stats=True)
        return (states - self.feature_mean_) / self.feature_std_

    def _reward(self, predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        area = self._area(predictions, actuals)
        return 1.0 / (1.0 + area)

    def _area(self, predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        predicted_curve = np.cumsum(predictions, axis=1)
        actual_curve = np.cumsum(actuals, axis=1)
        return np.trapezoid(np.abs(predicted_curve - actual_curve), axis=1)

    @staticmethod
    def _positive_output(values: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _inverse_softplus(values: np.ndarray) -> np.ndarray:
        clipped = np.maximum(values, 1e-6)
        return np.where(
            clipped > 20.0,
            clipped,
            np.log(np.expm1(clipped)),
        )

    def _check_is_fitted(self, allow_pre_fit_stats: bool = False) -> None:
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError("Model has not been fitted yet.")
        if not allow_pre_fit_stats and (self.weights is None or self.bias is None):
            raise ValueError("Model has not been fitted yet.")

def train_rl_covid_model(
    df: pd.DataFrame,
    history_weeks: int = 4,
    forecast_weeks: int = 2,
    radius: float = 1.0,
    learning_rate: float = 0.001,
    noise_scale: float = 0.2,
    epochs: int = 250,
    random_seed: int = 42,
    show_progress: bool = True,
    data_sources: list[str] | None = None,
) -> SpatioTemporalCovidRLModel:
    model = SpatioTemporalCovidRLModel(
        history_weeks=history_weeks,
        forecast_weeks=forecast_weeks,
        radius=radius,
        learning_rate=learning_rate,
        noise_scale=noise_scale,
        epochs=epochs,
        random_seed=random_seed,
        show_progress=show_progress,
        data_sources=data_sources,
    )
    return model.fit(df)


def load_multi_country_dataset(
    country_configs: list[dict[str, object]],
    data_sources: list[str] | None = None,
) -> pd.DataFrame:
    datasets: list[pd.DataFrame] = []

    for config in country_configs:
        dataset = get_spatiotemporal_covid_dataset(
            country_iso2=str(config["country_iso2"]),
            country_iso3=str(config["country_iso3"]),
            grid_km=int(config.get("grid_km", 20)),
            data_sources=data_sources,
        ).copy()
        dataset["country_iso2"] = str(config["country_iso2"])
        dataset["country_iso3"] = str(config["country_iso3"])
        datasets.append(dataset)

    if not datasets:
        raise ValueError("country_configs must contain at least one country")

    return pd.concat(datasets, ignore_index=True)


def visualize_prediction_progress(
    predicted_trajectory: list[float],
    actual_trajectory: list[float],
    output_path: str,
    title: str = "Predicted vs Actual Confirmed Cases",
) -> str:
    predicted = np.asarray(predicted_trajectory, dtype=float)
    actual = np.asarray(actual_trajectory, dtype=float)

    if len(predicted) != len(actual):
        raise ValueError("Predicted and actual trajectories must have the same length")

    predicted_curve = np.cumsum(predicted)
    actual_curve = np.cumsum(actual)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    horizon = np.arange(1, len(predicted) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bar_width = 0.35
    axes[0].bar(horizon - bar_width / 2, predicted, width=bar_width, color="#c0392b", label="Predicted")
    axes[0].bar(horizon + bar_width / 2, actual, width=bar_width, color="#1f618d", label="Actual")
    axes[0].set_title("Monthly confirmed cases")
    axes[0].set_xlabel("Forecast month")
    axes[0].set_ylabel("Confirmed cases")
    axes[0].set_xticks(horizon)
    axes[0].set_ylim(0, max(float(predicted.max(initial=0.0)), float(actual.max(initial=0.0)), 1.0) * 1.15)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    for x_pos, pred_value, actual_value in zip(horizon, predicted, actual):
        axes[0].text(x_pos - bar_width / 2, pred_value, f"{pred_value:.0f}", ha="center", va="bottom", fontsize=8)
        axes[0].text(x_pos + bar_width / 2, actual_value, f"{actual_value:.0f}", ha="center", va="bottom", fontsize=8)

    axes[1].plot(horizon, predicted_curve, marker="o", linewidth=2.5, color="#c0392b", label="Predicted cumulative")
    axes[1].plot(horizon, actual_curve, marker="o", linewidth=2.5, color="#1f618d", label="Actual cumulative")
    axes[1].fill_between(horizon, predicted_curve, actual_curve, color="#d4ac0d", alpha=0.2, label="Area error")
    axes[1].set_title("Cumulative progression")
    axes[1].set_xlabel("Forecast month")
    axes[1].set_ylabel("Cumulative confirmed cases")
    axes[1].set_xticks(horizon)
    axes[1].set_ylim(0, max(float(predicted_curve.max(initial=0.0)), float(actual_curve.max(initial=0.0)), 1.0) * 1.15)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    COUNTRY_CONFIGS = [
        {"country_iso2": "BD", "country_iso3": "BGD", "grid_km": 20}, # Bangladesh
        {"country_iso2": "ID", "country_iso3": "IDN", "grid_km": 20}, # Indonesia
        {"country_iso2": "PH", "country_iso3": "PHL", "grid_km": 20}, # Philippines
        {"country_iso2": "TH", "country_iso3": "THA", "grid_km": 20}, # Thailand
        {"country_iso2": "VN", "country_iso3": "VNM", "grid_km": 20}, # Vietnam
    ]
    HISTORY_WEEKS = 12
    FORECAST_WEEKS = 4
    RADIUS = 1.5
    EPOCHS = 50
    SHOW_PROGRESS = True
    SAVE_MODEL = True
    VISUALIZE_PROGRESS = True
    VISUALIZATION_OUTPUT_PATH = "figures/latest_prediction_vs_actual.png"

    dataset = load_multi_country_dataset(COUNTRY_CONFIGS)

    model = train_rl_covid_model(
        df=dataset,
        history_weeks=HISTORY_WEEKS,
        forecast_weeks=FORECAST_WEEKS,
        radius=RADIUS,
        epochs=EPOCHS,
        show_progress=SHOW_PROGRESS,
    )

    metrics = model.score(dataset)
    print("Training complete.")
    print("Training metrics:", metrics)

    country_report = model.country_progression_report(dataset)
    print("Country progression report:", country_report)

    if SAVE_MODEL:
        saved_model_prefix = model.save(output_dir="models")
        print(f"Saved model to {saved_model_prefix}.npz/.json")

    if VISUALIZE_PROGRESS:
        visualization_path = visualize_prediction_progress(
            predicted_trajectory=country_report["predicted_trajectory"],
            actual_trajectory=country_report["actual_trajectory"],
            output_path=VISUALIZATION_OUTPUT_PATH,
            title="Predicted vs Actual COVID Progression (Country Total)",
        )
        print(f"Saved visualization to {visualization_path}")
