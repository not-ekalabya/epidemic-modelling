"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

const PredictionMap = dynamic(() => import("../components/Map"), {
  ssr: false,
});

type CellProgression = {
  grid_lat: number;
  grid_lon: number;
  predicted_next_confirmed: number;
  predicted_trajectory: number[];
  actual_trajectory?: number[];
};

type PredictionResponse = {
  prediction: {
    requested_reference_week?: string;
    reference_week: string;
    predicted_next_confirmed: number;
    predicted_trajectory: number[];
    cell_count: number;
    per_cell_progression?: CellProgression[];
  };
  actual?: {
    reference_week: string;
    actual_trajectory: number[];
  };
  detail?: string;
};

type MetadataResponse = {
  date_min?: string;
  date_max?: string;
  forecast_weeks?: number;
  history_weeks?: number;
  dataset_rows?: number;
  grid_cell_count?: number;
  country_configs: Array<{
    country_iso2: string;
    country_iso3: string;
    grid_km?: number;
  }>;
};

type ConfigureCountryResponse = {
  status?: string;
  metadata?: MetadataResponse;
  detail?: string;
};

const defaultValues = {
  predictionDate: "2021-08-02",
  includeActual: true,
};

export default function Home() {
  const [predictionDate, setPredictionDate] = useState(defaultValues.predictionDate);
  const [includeActual, setIncludeActual] = useState(defaultValues.includeActual);
  const [countryIso2, setCountryIso2] = useState("BD");
  const [countryIso3, setCountryIso3] = useState("BGD");
  const [gridKm, setGridKm] = useState("20");
  const [isLoading, setIsLoading] = useState(false);
  const [isConfiguringCountry, setIsConfiguringCountry] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [countryConfigError, setCountryConfigError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [selectedCellIndex, setSelectedCellIndex] = useState<number>(0);
  const [sortMode, setSortMode] = useState<"predicted" | "actual">("predicted");

  useEffect(() => {
    let active = true;

    async function loadMetadata() {
      try {
        const response = await fetch("/api/metadata", { cache: "no-store" });
        if (!response.ok) {
          return;
        }

        const payload = (await response.json()) as MetadataResponse;
        if (active) {
          setMetadata(payload);
          const firstCountry = payload.country_configs?.[0];
          if (firstCountry) {
            setCountryIso2(firstCountry.country_iso2);
            setCountryIso3(firstCountry.country_iso3);
            setGridKm(firstCountry.grid_km ? String(firstCountry.grid_km) : "20");
          }
        }
      } catch {
        // Ignore metadata loading failures in the UI.
      }
    }

    void loadMetadata();

    return () => {
      active = false;
    };
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prediction_date: predictionDate,
          include_actual: includeActual,
        }),
      });

      const payload = (await response.json()) as PredictionResponse;

      if (!response.ok) {
        setResult(null);
        setError(payload.detail ?? "Prediction request failed.");
        return;
      }

      setResult(payload);
      setSelectedCellIndex(0);
      setSortMode("predicted");
    } catch {
      setResult(null);
      setError("Could not reach the prediction server.");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleConfigureCountry(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsConfiguringCountry(true);
    setCountryConfigError(null);
    setError(null);

    const parsedGridKm = Number.parseInt(gridKm, 10);
    if (!Number.isFinite(parsedGridKm) || parsedGridKm < 1 || parsedGridKm > 200) {
      setCountryConfigError("Grid size must be between 1 and 200 km.");
      setIsConfiguringCountry(false);
      return;
    }

    try {
      const response = await fetch("/api/configure-country", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          country_iso2: countryIso2,
          country_iso3: countryIso3,
          grid_km: parsedGridKm,
        }),
      });

      const payload = (await response.json()) as ConfigureCountryResponse;

      if (!response.ok || payload.status !== "ok" || !payload.metadata) {
        setCountryConfigError(payload.detail ?? "Could not update country settings.");
        return;
      }

      setMetadata(payload.metadata);
      const firstCountry = payload.metadata.country_configs?.[0];
      if (firstCountry) {
        setCountryIso2(firstCountry.country_iso2);
        setCountryIso3(firstCountry.country_iso3);
        setGridKm(firstCountry.grid_km ? String(firstCountry.grid_km) : "20");
      }
      setResult(null);
      setSelectedCellIndex(0);
      setSortMode("predicted");
    } catch {
      setCountryConfigError("Could not reach the prediction server.");
    } finally {
      setIsConfiguringCountry(false);
    }
  }

  const cells = result?.prediction.per_cell_progression ?? [];
  const selectedCell = cells[selectedCellIndex] ?? null;
  const forecastWeeks = result?.prediction.predicted_trajectory.length ?? metadata?.forecast_weeks ?? 0;
  const sortedCells = useMemo(
    () => sortCells(cells, sortMode),
    [cells, sortMode],
  );
  const totalPredicted = result?.prediction.predicted_trajectory.reduce((sum, value) => sum + value, 0) ?? 0;
  const totalActual = result?.actual?.actual_trajectory.reduce((sum, value) => sum + value, 0) ?? null;

  return (
    <main className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.18),_transparent_32%),radial-gradient(circle_at_top_right,_rgba(251,191,36,0.16),_transparent_24%),linear-gradient(180deg,_#07111f_0%,_#0a1324_40%,_#0d1728_100%)] text-slate-100">
      <div className="absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.08)_1px,transparent_1px)] bg-[size:48px_48px] opacity-30" />
      <div className="relative mx-auto flex min-h-screen w-full max-w-[1800px] flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
        <header className="rounded-[28px] border border-white/10 bg-white/7 px-5 py-5 shadow-[0_30px_120px_rgba(2,6,23,0.45)] backdrop-blur-xl sm:px-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl space-y-3">
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs font-medium uppercase tracking-[0.24em] text-cyan-200">
                Interactive prediction dashboard
              </div>
              <div>
                <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl lg:text-5xl">
                  Spatial COVID forecast, mapped cell by cell.
                </h1>
                <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300 sm:text-base">
                  Choose a month, run the backend model, and inspect the predicted
                  progression on a live map with cell-level ranking and trajectory details.
                </p>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3 lg:min-w-[460px]">
              <StatCard label="Grid cells" value={metadata?.grid_cell_count?.toString() ?? "61"} />
              <StatCard label="Forecast weeks" value={forecastWeeks.toString()} />
              <StatCard
                label="Dataset rows"
                value={metadata?.dataset_rows ? compactNumber(metadata.dataset_rows) : "60k+"}
              />
            </div>
          </div>
        </header>

        <section className="grid flex-1 gap-6 xl:grid-cols-[420px_minmax(0,1.55fr)_390px]">
          <aside className="space-y-6 rounded-[28px] border border-white/10 bg-slate-950/45 p-5 shadow-[0_30px_120px_rgba(2,6,23,0.35)] backdrop-blur-xl">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold text-white">Controls</h2>
                <p className="mt-1 text-sm text-slate-400">Run a fresh forecast from the API.</p>
              </div>
              {metadata?.country_configs?.length ? (
                <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                  {metadata.country_configs[0].country_iso2} / {metadata.country_configs[0].country_iso3}
                </div>
              ) : null}
            </div>

            <form className="space-y-4" onSubmit={handleSubmit}>
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-200" htmlFor="prediction-date">
                  Prediction date
                </label>
                <input
                  id="prediction-date"
                  className="dashboard-input"
                  value={predictionDate}
                  onChange={(event) => setPredictionDate(event.target.value)}
                  type="date"
                />
                <p className="mt-2 text-xs text-slate-400">
                  Available window: {formatDateValue(metadata?.date_min ?? "")}
                  {metadata?.date_max ? ` to ${formatDateValue(metadata.date_max)}` : ""}
                </p>
              </div>

              <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200">
                <input
                  checked={includeActual}
                  className="accent-cyan-400"
                  onChange={(event) => setIncludeActual(event.target.checked)}
                  type="checkbox"
                />
                Include actual trajectory
              </label>

              <button className="dashboard-button w-full" disabled={isLoading} type="submit">
                {isLoading ? "Running prediction..." : "Predict now"}
              </button>
            </form>

            <form className="space-y-4 rounded-3xl border border-white/10 bg-white/5 p-4" onSubmit={handleConfigureCountry}>
              <div>
                <h3 className="font-semibold text-white">Country configuration</h3>
                <p className="mt-1 text-sm text-yellow-400">
                  Warning: This might take 2-3 mins as it involves retraining the model.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-2 block text-xs font-medium uppercase tracking-[0.16em] text-slate-300" htmlFor="country-iso2">
                    ISO2
                  </label>
                  <input
                    id="country-iso2"
                    className="dashboard-input uppercase"
                    maxLength={2}
                    onChange={(event) => setCountryIso2(event.target.value.toUpperCase())}
                    value={countryIso2}
                  />
                </div>

                <div>
                  <label className="mb-2 block text-xs font-medium uppercase tracking-[0.16em] text-slate-300" htmlFor="country-iso3">
                    ISO3
                  </label>
                  <input
                    id="country-iso3"
                    className="dashboard-input uppercase"
                    maxLength={3}
                    onChange={(event) => setCountryIso3(event.target.value.toUpperCase())}
                    value={countryIso3}
                  />
                </div>
              </div>

              <div>
                <label className="mb-2 block text-xs font-medium uppercase tracking-[0.16em] text-slate-300" htmlFor="country-grid-km">
                  Grid size (km)
                </label>
                <input
                  id="country-grid-km"
                  className="dashboard-input"
                  min={1}
                  max={200}
                  onChange={(event) => setGridKm(event.target.value)}
                  type="number"
                  value={gridKm}
                />
              </div>

              <button className="dashboard-button w-full" disabled={isConfiguringCountry} type="submit">
                {isConfiguringCountry ? "Applying country..." : "Apply country config"}
              </button>

              {countryConfigError ? (
                <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                  {countryConfigError}
                </div>
              ) : null}
            </form>

            {error ? (
              <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                {error}
              </div>
            ) : null}

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
              <InfoTile label="Model week" value={result ? formatDateValue(result.prediction.reference_week) : "Waiting"} />
              <InfoTile label="Cells in view" value={result?.prediction.cell_count.toString() ?? "0"} />
              <InfoTile
                label="Predicted total"
                value={result ? compactNumber(totalPredicted) : "0"}
              />
              <InfoTile
                label="Actual total"
                value={totalActual !== null ? compactNumber(totalActual) : "N/A"}
              />
            </div>

            {metadata?.country_configs?.length ? (
              <section className="rounded-3xl border border-white/10 bg-white/5 p-4">
                <h3 className="text-sm font-semibold text-white">Configured regions</h3>
                <div className="mt-3 space-y-2 text-sm text-slate-300">
                  {metadata.country_configs.map((country) => (
                    <div
                      className="flex items-center justify-between gap-3 rounded-2xl border border-white/5 bg-slate-950/50 px-3 py-2"
                      key={`${country.country_iso2}-${country.country_iso3}`}
                    >
                      <span>{country.country_iso2} / {country.country_iso3}</span>
                      <span className="text-slate-500">{country.grid_km ? `${country.grid_km} km grid` : "grid"}</span>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}
          </aside>

          <section className="grid gap-6 xl:grid-rows-[minmax(0,1fr)_auto]">
            <div className="rounded-[28px] border border-white/10 bg-slate-950/45 p-4 shadow-[0_30px_120px_rgba(2,6,23,0.35)] backdrop-blur-xl sm:p-5">
              <div className="flex flex-col gap-4 border-b border-white/10 pb-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-white">Prediction map</h2>
                  <p className="mt-1 text-sm text-slate-400">
                    Circle size and color track predicted next-week confirmed cases. Click a cell to inspect its trajectory.
                  </p>
                </div>

                {result?.prediction.per_cell_progression?.length ? (
                  <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
                    <button
                      className={sortButtonClass(sortMode === "predicted")}
                      onClick={() => setSortMode("predicted")}
                      type="button"
                    >
                      Sort by predicted
                    </button>
                    <button
                      className={sortButtonClass(sortMode === "actual")}
                      onClick={() => setSortMode("actual")}
                      type="button"
                    >
                      Sort by actual
                    </button>
                  </div>
                ) : null}
              </div>

              <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
                <div className="rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(15,23,42,0.92),rgba(15,23,42,0.68))] p-3">
                  <div className="relative overflow-hidden rounded-[22px] border border-white/10 bg-slate-900/70">
                    {cells.length ? (
                      <PredictionMap
                        cells={cells}
                        onSelectCell={setSelectedCellIndex}
                        selectedCellIndex={selectedCellIndex}
                      />
                    ) : (
                      <EmptyState
                        title="Run a prediction to populate the map"
                        description="The backend will return per-cell projections and the dashboard will draw them on the map here."
                      />
                    )}
                  </div>
                </div>

                <div className="space-y-3 rounded-[24px] border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-semibold text-white">Top cells</h3>
                      <p className="text-xs text-slate-400">Ranked by {sortMode}.</p>
                    </div>
                    <div className="rounded-full border border-white/10 bg-slate-950/60 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      {cells.length} cells
                    </div>
                  </div>

                  <div className="max-h-[520px] space-y-2 overflow-auto pr-1">
                    {sortedCells.slice(0, 12).map((cell, index) => {
                      const originalIndex = cells.findIndex(
                        (candidate) => candidate.grid_lat === cell.grid_lat && candidate.grid_lon === cell.grid_lon,
                      );
                      const isSelected = originalIndex === selectedCellIndex;
                      return (
                        <button
                          className={`w-full rounded-2xl border px-3 py-3 text-left transition ${isSelected ? "border-cyan-300/50 bg-cyan-400/12" : "border-white/10 bg-slate-950/40 hover:border-white/20 hover:bg-white/8"}`}
                          key={`${cell.grid_lat}-${cell.grid_lon}-${index}`}
                          onClick={() => setSelectedCellIndex(originalIndex)}
                          type="button"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div>
                              <div className="text-sm font-medium text-white">
                                {cell.grid_lat.toFixed(3)}, {cell.grid_lon.toFixed(3)}
                              </div>
                              <div className="mt-1 text-xs text-slate-400">
                                Next {cell.predicted_next_confirmed.toFixed(2)}
                              </div>
                            </div>
                            <div className="text-right text-xs text-slate-300">
                              <div>{formatTrajectoryMini(cell.predicted_trajectory)}</div>
                              {cell.actual_trajectory ? (
                                <div className="mt-1 text-slate-500">A {formatTrajectoryMini(cell.actual_trajectory)}</div>
                              ) : null}
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
              <div className="rounded-[28px] border border-white/10 bg-slate-950/45 p-5 shadow-[0_30px_120px_rgba(2,6,23,0.35)] backdrop-blur-xl">
                <div className="flex items-end justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Country trajectories</h2>
                    <p className="mt-1 text-sm text-slate-400">
                      View the aggregate forecast and actual series side by side.
                    </p>
                  </div>
                </div>

                {result?.prediction ? (
                  <div className="mt-5 grid gap-4 lg:grid-cols-2">
                    <TrajectoryCard title="Predicted trajectory" values={result.prediction.predicted_trajectory} tone="cyan" />
                    {result.actual ? (
                      <TrajectoryCard title="Actual trajectory" values={result.actual.actual_trajectory} tone="amber" />
                    ) : (
                      <EmptyCard title="Actual trajectory" description="Enable the comparison toggle to fetch actual future values." />
                    )}
                  </div>
                ) : (
                  <EmptyCard
                    title="No prediction loaded"
                    description="Submit the form to visualize the country forecast and the per-cell map."
                  />
                )}
              </div>

              <div className="rounded-[28px] border border-white/10 bg-slate-950/45 p-5 shadow-[0_30px_120px_rgba(2,6,23,0.35)] backdrop-blur-xl">
                <div>
                  <h2 className="text-lg font-semibold text-white">Selected cell</h2>
                  <p className="mt-1 text-sm text-slate-400">A focused readout updates as you hover or click the map.</p>
                </div>

                {selectedCell ? (
                  <div className="mt-5 space-y-4">
                    <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
                      <div className="text-sm text-slate-400">Cell coordinates</div>
                      <div className="mt-1 text-2xl font-semibold text-white">
                        {selectedCell.grid_lat.toFixed(4)}, {selectedCell.grid_lon.toFixed(4)}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <MetricPill label="Next" value={selectedCell.predicted_next_confirmed.toFixed(2)} />
                      <MetricPill label="Rank" value={`#${sortedCellRank(cells, selectedCell, sortMode)}`} />
                    </div>

                    <TrajectoryCard title="Predicted path" values={selectedCell.predicted_trajectory} tone="cyan" compact />
                    {selectedCell.actual_trajectory ? (
                      <TrajectoryCard title="Actual path" values={selectedCell.actual_trajectory} tone="amber" compact />
                    ) : null}
                  </div>
                ) : (
                  <EmptyCard
                    title="No cell selected"
                    description="Click any point on the map to inspect its prediction details."
                  />
                )}
              </div>
            </div>
          </section>
        </section>
      </div>
    </main>
  );
}

function formatDateValue(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toISOString().slice(0, 10);
}

function compactNumber(value: number) {
  return new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatTrajectoryMini(values: number[]) {
  if (!values.length) {
    return "-";
  }

  return values
    .slice(0, 3)
    .map((value) => value.toFixed(0))
    .join(" · ");
}

function sortCells(cells: CellProgression[], sortMode: "predicted" | "actual") {
  return [...cells].sort((left, right) => {
    const leftValue = sortMode === "actual"
      ? left.actual_trajectory?.[0] ?? -Infinity
      : left.predicted_next_confirmed;
    const rightValue = sortMode === "actual"
      ? right.actual_trajectory?.[0] ?? -Infinity
      : right.predicted_next_confirmed;

    if (rightValue === leftValue) {
      return left.grid_lat - right.grid_lat || left.grid_lon - right.grid_lon;
    }

    return rightValue - leftValue;
  });
}

function sortedCellRank(cells: CellProgression[], selectedCell: CellProgression, sortMode: "predicted" | "actual") {
  const ranked = sortCells(cells, sortMode);
  return Math.max(
    1,
    ranked.findIndex(
      (cell) => cell.grid_lat === selectedCell.grid_lat && cell.grid_lon === selectedCell.grid_lon,
    ) + 1,
  );
}

function sortButtonClass(active: boolean) {
  return [
    "rounded-full border px-3 py-1.5 transition",
    active
      ? "border-cyan-300/40 bg-cyan-400/15 text-cyan-100"
      : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:bg-white/10",
  ].join(" ");
}

function TrajectoryCard({
  title,
  values,
  tone,
  compact = false,
}: {
  title: string;
  values: number[];
  tone: "cyan" | "amber";
  compact?: boolean;
}) {
  const maxValue = Math.max(...values, 1);
  const minValue = Math.min(...values, 0);
  const lineColor = tone === "cyan" ? "#67e8f9" : "#fbbf24";
  const areaColor = tone === "cyan" ? "rgba(34,211,238,0.16)" : "rgba(251,191,36,0.14)";
  const width = 460;
  const height = compact ? 140 : 180;
  const points = values.length
    ? values.map((value, index) => {
        const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * (width - 24) + 12;
        const yScale = maxValue === minValue ? 0.5 : (value - minValue) / (maxValue - minValue);
        const y = height - 18 - yScale * (height - 40);
        return { x, y, value };
      })
    : [];

  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
  const areaPath = points.length
    ? `${path} L ${points[points.length - 1].x.toFixed(2)} ${height - 12} L ${points[0].x.toFixed(2)} ${height - 12} Z`
    : "";

  return (
    <div className="rounded-3xl border border-white/10 bg-slate-950/55 p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        <span className={`rounded-full px-2.5 py-1 text-[11px] uppercase tracking-[0.2em] ${tone === "cyan" ? "bg-cyan-400/10 text-cyan-200" : "bg-amber-400/10 text-amber-100"}`}>
          {compact ? "Cell" : "Aggregate"}
        </span>
      </div>

      <div className={compact ? "mt-3" : "mt-4"}>
        {points.length ? (
          <svg viewBox={`0 0 ${width} ${height}`} className={compact ? "h-36 w-full" : "h-44 w-full"} role="img" aria-label={title}>
            <rect x="0" y="0" width={width} height={height} rx="18" fill="rgba(15,23,42,0.7)" />
            <path d={areaPath} fill={areaColor} />
            <path d={path} fill="none" stroke={lineColor} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
            {points.map((point, index) => (
              <g key={`${title}-${index}`}>
                <circle cx={point.x} cy={point.y} r="4.5" fill={lineColor} />
              </g>
            ))}
            <g fill="rgba(148,163,184,0.72)" fontSize="11">
              <text x="12" y="18">{compact ? "0" : "Month 1"}</text>
              <text x={width - 42} y="18">{compact ? `${values.length}` : `Month ${values.length}`}</text>
            </g>
          </svg>
        ) : (
          <div className="rounded-2xl border border-dashed border-white/10 bg-white/5 px-4 py-8 text-center text-sm text-slate-400">
            No trajectory values available.
          </div>
        )}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-slate-400">
        <div className="rounded-2xl border border-white/8 bg-white/5 px-3 py-2">
          <div className="uppercase tracking-[0.18em] text-slate-500">Start</div>
          <div className="mt-1 text-sm text-slate-100">{values[0]?.toFixed(2) ?? "-"}</div>
        </div>
        <div className="rounded-2xl border border-white/8 bg-white/5 px-3 py-2">
          <div className="uppercase tracking-[0.18em] text-slate-500">Peak</div>
          <div className="mt-1 text-sm text-slate-100">{Math.max(...values, 0).toFixed(2)}</div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/6 px-4 py-4 shadow-lg shadow-slate-950/20 backdrop-blur-sm">
      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
    </div>
  );
}

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 px-4 py-4">
      <div className="text-[11px] uppercase tracking-[0.24em] text-slate-500">{label}</div>
      <div className="mt-2 text-base font-semibold text-white">{value}</div>
    </div>
  );
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
      <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">{label}</div>
      <div className="mt-1 text-lg font-semibold text-white">{value}</div>
    </div>
  );
}

function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex h-[520px] flex-col items-center justify-center px-8 text-center">
      <div className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-4 py-2 text-xs uppercase tracking-[0.24em] text-cyan-100">
        Map waiting for data
      </div>
      <h3 className="mt-5 text-2xl font-semibold text-white">{title}</h3>
      <p className="mt-3 max-w-md text-sm leading-6 text-slate-400">{description}</p>
    </div>
  );
}

function EmptyCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="mt-5 rounded-3xl border border-dashed border-white/10 bg-white/4 p-5 text-center">
      <h3 className="text-base font-semibold text-white">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
    </div>
  );
}
