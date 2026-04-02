"use client";

import { FormEvent, useEffect, useState } from "react";

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
  country_configs: Array<{
    country_iso2: string;
    country_iso3: string;
    grid_km?: number;
  }>;
};

const defaultValues = {
  predictionDate: "2021-08-02",
  includeActual: true,
};

export default function Home() {
  const [predictionDate, setPredictionDate] = useState(defaultValues.predictionDate);
  const [includeActual, setIncludeActual] = useState(defaultValues.includeActual);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [showAllCells, setShowAllCells] = useState(false);

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
      setShowAllCells(false);
    } catch {
      setResult(null);
      setError("Could not reach the prediction server.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="mx-auto min-h-screen max-w-3xl px-4 py-10">
      <h1 className="text-2xl font-semibold">COVID Prediction</h1>
      <p className="mt-2 text-sm text-neutral-600">
        Submit a date to request a country-level prediction from the backend.
      </p>

      {metadata?.country_configs?.length ? (
        <section className="mt-6">
          <h2 className="text-lg font-semibold">Available countries</h2>
          <ul className="mt-2 list-disc pl-5 text-sm text-neutral-700">
            {metadata.country_configs.map((country) => (
              <li key={`${country.country_iso2}-${country.country_iso3}`}>
                {country.country_iso2} / {country.country_iso3}
                {country.grid_km ? `, ${country.grid_km} km grid` : ""}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <form className="mt-8 space-y-4" onSubmit={handleSubmit}>
        <div>
          <label className="mb-1 block text-sm font-medium" htmlFor="prediction-date">
            Prediction date
          </label>
          <input
            id="prediction-date"
            className="w-full rounded border px-3 py-2"
            value={predictionDate}
            onChange={(event) => setPredictionDate(event.target.value)}
            type="date"
          />
        </div>

        <label className="flex items-center gap-2 text-sm">
          <input
            checked={includeActual}
            onChange={(event) => setIncludeActual(event.target.checked)}
            type="checkbox"
          />
          Include actual trajectory
        </label>

        <button
          className="rounded border bg-black px-4 py-2 text-white disabled:opacity-60"
          disabled={isLoading}
          type="submit"
        >
          {isLoading ? "Loading..." : "Predict"}
        </button>
      </form>

      {error ? (
        <div className="mt-8 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      ) : null}

      {result?.prediction ? (
        <section className="mt-10 space-y-6">
          <div>
            <h2 className="text-lg font-semibold">Summary</h2>
            <div className="mt-3 overflow-hidden rounded border">
              <table className="w-full border-collapse text-sm">
                <tbody>
                  <TableRow
                    label="Requested month"
                    value={formatDateValue(
                      result.prediction.requested_reference_week ?? result.prediction.reference_week,
                    )}
                  />
                  <TableRow
                    label="Reference month"
                    value={formatDateValue(result.prediction.reference_week)}
                  />
                  <TableRow
                    label="Cells aggregated"
                    value={result.prediction.cell_count.toString()}
                  />
                  <TableRow
                    label="Predicted next confirmed"
                    value={result.prediction.predicted_next_confirmed.toFixed(2)}
                  />
                </tbody>
              </table>
            </div>
          </div>

          <TrajectoryTable
            title="Predicted trajectory"
            values={result.prediction.predicted_trajectory}
          />

          <CellProgressionTable
            rows={result.prediction.per_cell_progression ?? []}
            showAll={showAllCells}
            onToggleShowAll={() => setShowAllCells((current) => !current)}
          />

          {result.actual ? (
            <TrajectoryTable
              title="Actual trajectory"
              values={result.actual.actual_trajectory}
            />
          ) : null}
        </section>
      ) : null}
    </main>
  );
}

function TableRow({ label, value }: { label: string; value: string }) {
  return (
    <tr className="border-t first:border-t-0">
      <th className="w-52 bg-neutral-50 px-3 py-2 text-left font-medium">{label}</th>
      <td className="px-3 py-2">{value}</td>
    </tr>
  );
}

function formatDateValue(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toISOString().slice(0, 10);
}

function TrajectoryTable({
  title,
  values,
}: {
  title: string;
  values: number[];
}) {
  return (
    <div>
      <h2 className="text-lg font-semibold">{title}</h2>
      <div className="mt-3 overflow-hidden rounded border">
        <table className="w-full border-collapse text-sm">
          <thead className="bg-neutral-50">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Month</th>
              <th className="px-3 py-2 text-left font-medium">Value</th>
            </tr>
          </thead>
          <tbody>
            {values.map((value, index) => (
              <tr className="border-t" key={`${title}-${index}`}>
                <td className="px-3 py-2">{index + 1}</td>
                <td className="px-3 py-2">{value.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CellProgressionTable({
  rows,
  showAll,
  onToggleShowAll,
}: {
  rows: CellProgression[];
  showAll: boolean;
  onToggleShowAll: () => void;
}) {
  if (!rows.length) {
    return null;
  }

  const maxRows = 100;
  const visibleRows = showAll ? rows : rows.slice(0, maxRows);

  return (
    <div>
      <div className="flex items-end justify-between gap-3">
        <h2 className="text-lg font-semibold">Per-cell progression</h2>
        {rows.length > maxRows ? (
          <button
            className="rounded border px-3 py-1 text-sm"
            onClick={onToggleShowAll}
            type="button"
          >
            {showAll ? "Show top 100" : `Show all ${rows.length}`}
          </button>
        ) : null}
      </div>

      <div className="mt-3 overflow-auto rounded border">
        <table className="w-full border-collapse text-sm">
          <thead className="bg-neutral-50">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Cell (lat, lon)</th>
              <th className="px-3 py-2 text-left font-medium">Predicted next</th>
              <th className="px-3 py-2 text-left font-medium">Predicted trajectory</th>
              <th className="px-3 py-2 text-left font-medium">Actual trajectory</th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, index) => (
              <tr className="border-t" key={`${row.grid_lat}-${row.grid_lon}-${index}`}>
                <td className="whitespace-nowrap px-3 py-2">
                  {row.grid_lat.toFixed(4)}, {row.grid_lon.toFixed(4)}
                </td>
                <td className="px-3 py-2">{row.predicted_next_confirmed.toFixed(2)}</td>
                <td className="px-3 py-2">{formatTrajectory(row.predicted_trajectory)}</td>
                <td className="px-3 py-2">
                  {row.actual_trajectory ? formatTrajectory(row.actual_trajectory) : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatTrajectory(values: number[]): string {
  return values.map((value) => value.toFixed(2)).join(" -> ");
}
