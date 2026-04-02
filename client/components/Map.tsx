"use client";

import { useEffect, useMemo, useRef } from "react";
import Map, { Marker, type MapRef } from "react-map-gl/maplibre";

type MapCell = {
  grid_lat: number;
  grid_lon: number;
  predicted_next_confirmed: number;
  predicted_trajectory: number[];
  actual_trajectory?: number[];
};

type MapStats = {
  minLat: number;
  maxLat: number;
  minLon: number;
  maxLon: number;
  minValue: number;
  maxValue: number;
};

type MapProps = {
  cells: MapCell[];
  selectedCellIndex: number;
  onSelectCell: (index: number) => void;
};

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export default function PredictionMap({ cells, selectedCellIndex, onSelectCell }: MapProps) {
  const mapRef = useRef<MapRef | null>(null);
  const stats = useMemo(() => computeStats(cells), [cells]);
  const selectedCell = cells[selectedCellIndex] ?? null;
  const selectedKey = selectedCell ? `${selectedCell.grid_lat}-${selectedCell.grid_lon}` : null;

  useEffect(() => {
    if (!stats || !mapRef.current) {
      return;
    }

    mapRef.current.fitBounds(
      [
        [stats.minLon, stats.minLat],
        [stats.maxLon, stats.maxLat],
      ],
      {
        padding: 72,
        duration: 900,
        essential: true,
      },
    );
  }, [stats, cells.length]);

  if (!cells.length || !stats) {
    return <MapLoadingState />;
  }

  const centerLatitude = (stats.minLat + stats.maxLat) / 2;
  const centerLongitude = (stats.minLon + stats.maxLon) / 2;
  const zoom = estimateZoom(stats);

  return (
    <div className="relative h-[520px] w-full overflow-hidden rounded-[22px] bg-slate-950">
      <Map
        ref={mapRef}
        initialViewState={{
          latitude: centerLatitude,
          longitude: centerLongitude,
          zoom,
        }}
        attributionControl={false}
        boxZoom={false}
        doubleClickZoom={false}
        dragPan={false}
        dragRotate={false}
        interactive={false}
        keyboard={false}
        mapStyle={MAP_STYLE}
        logoPosition="bottom-right"
        pitchWithRotate={false}
        reuseMaps
        scrollZoom={false}
        style={{ width: "100%", height: "100%" }}
        touchPitch={false}
        touchZoomRotate={false}
      >
        <MapChrome />
        {cells
          .map((cell, index) => ({ cell, index }))
          .filter(({ index }) => index !== selectedCellIndex)
          .map(({ cell, index }) => (
            <CellMarker
              cell={cell}
              index={index}
              isSelected={false}
              key={`${cell.grid_lat}-${cell.grid_lon}`}
              onSelectCell={onSelectCell}
              stats={stats}
            />
          ))}
        {selectedCell ? (
          <CellMarker
            cell={selectedCell}
            index={selectedCellIndex}
            isSelected
            key={selectedKey ?? "selected-cell"}
            onSelectCell={onSelectCell}
            stats={stats}
          />
        ) : null}
      </Map>
    </div>
  );
}

function CellMarker({
  cell,
  index,
  isSelected,
  onSelectCell,
  stats,
}: {
  cell: MapCell;
  index: number;
  isSelected: boolean;
  onSelectCell: (index: number) => void;
  stats: MapStats;
}) {
  const intensity = normalizedValue(cell.predicted_next_confirmed, stats);
  const coreSize = 7 + intensity * 10;
  const ringSize = coreSize + (isSelected ? 14 : 8);
  const color = fillColor(intensity);
  const ringColor = isSelected ? "rgba(125, 211, 252, 0.86)" : "rgba(148, 163, 184, 0.42)";

  return (
    <Marker anchor="center" latitude={cell.grid_lat} longitude={cell.grid_lon}>
      <button
        aria-label={`Select cell ${cell.grid_lat.toFixed(4)}, ${cell.grid_lon.toFixed(4)}`}
        className="relative block rounded-full outline-none"
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onSelectCell(index);
        }}
        style={{ height: ringSize, width: ringSize }}
        type="button"
      >
        <span
          className="absolute inset-0 rounded-full border"
          style={{
            borderColor: ringColor,
            boxShadow: isSelected
              ? "0 0 0 4px rgba(56, 189, 248, 0.14), 0 0 20px rgba(56, 189, 248, 0.28)"
              : "0 0 0 2px rgba(15, 23, 42, 0.2)",
          }}
        />
        <span
          className="absolute left-1/2 top-1/2 rounded-full border border-white/40"
          style={{
            background: `radial-gradient(circle at 35% 35%, rgba(255,255,255,0.85), ${color} 58%, rgba(2,6,23,0.9) 100%)`,
            height: coreSize,
            transform: "translate(-50%, -50%)",
            width: coreSize,
          }}
        />
      </button>
    </Marker>
  );
}

function MapChrome() {
  return (
    <div className="pointer-events-none absolute inset-0">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(2,6,23,0.16)_58%,rgba(2,6,23,0.38)_100%)]" />
      <div className="absolute left-4 top-4 rounded-full border border-white/15 bg-slate-950/65 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.24em] text-slate-300 backdrop-blur-md">
        MapLibre / Dark Matter
      </div>
    </div>
  );
}

function MapLoadingState() {
  return (
    <div className="flex h-[520px] items-center justify-center bg-[linear-gradient(180deg,rgba(148,163,184,0.04),rgba(15,23,42,0.55))] text-sm text-slate-400">
      Loading map...
    </div>
  );
}

function computeStats(cells: MapCell[]): MapStats | null {
  if (!cells.length) {
    return null;
  }

  const lats = cells.map((cell) => cell.grid_lat);
  const lons = cells.map((cell) => cell.grid_lon);
  const values = cells.map((cell) => cell.predicted_next_confirmed);

  return {
    minLat: Math.min(...lats),
    maxLat: Math.max(...lats),
    minLon: Math.min(...lons),
    maxLon: Math.max(...lons),
    minValue: Math.min(...values),
    maxValue: Math.max(...values),
  };
}

function normalizedValue(value: number, stats: MapStats) {
  const span = Math.max(stats.maxValue - stats.minValue, 0.0001);
  return Math.min(1, Math.max(0, (value - stats.minValue) / span));
}

function fillColor(t: number) {
  const hue = 194 - t * 36;
  const lightness = 58 - t * 20;
  return `hsl(${hue} 88% ${lightness}%)`;
}

function estimateZoom(stats: MapStats) {
  const span = Math.max(stats.maxLat - stats.minLat, stats.maxLon - stats.minLon);
  if (span > 4) {
    return 6;
  }
  if (span > 2) {
    return 7;
  }
  if (span > 1) {
    return 8;
  }
  return 9;
}
