import React from "react";

interface GridVisualizationProps {
  floorWidth: number;  // in feet
  floorLength: number; // in feet
  floorHeight: number; // in feet (not used for positioning here)
  uniformFlux: number; // computed uniform luminous flux for the grid
  gridArrangement: { rows: number; cols: number }; // e.g., { rows: 8, cols: 8 }
  gridPPFD: number;   // average PPFD from grid simulation
}

type CobPosition = { x: number; y: number };

const buildGridPositions = (W_m: number, L_m: number, rows: number, cols: number): CobPosition[] => {
  const positions: CobPosition[] = [];
  const marginX = 0.05 * W_m;
  const marginY = 0.05 * L_m;
  const effectiveW = W_m - 2 * marginX;
  const effectiveL = L_m - 2 * marginY;
  for (let i = 0; i < rows; i++) {
    const y = marginY + (i + 0.5) * (effectiveL / rows);
    for (let j = 0; j < cols; j++) {
      const x = marginX + (j + 0.5) * (effectiveW / cols);
      positions.push({ x, y });
    }
  }
  return positions;
};

// Re-use the same jet color mapping as in ModularVisualization.tsx.
const jetColorPoints: { t: number; color: [number, number, number] }[] = [
  { t: 0.0, color: [0, 0, 255] },
  { t: 0.16, color: [0, 255, 255] },
  { t: 0.33, color: [0, 255, 0] },
  { t: 0.5, color: [255, 255, 0] },
  { t: 0.66, color: [255, 165, 0] },
  { t: 0.83, color: [255, 0, 0] },
  { t: 1.0, color: [227, 25, 55] },
];
const getJetColor = (t: number): string => {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < jetColorPoints.length - 1; i++) {
    const c0 = jetColorPoints[i];
    const c1 = jetColorPoints[i + 1];
    if (t >= c0.t && t <= c1.t) {
      const localT = (t - c0.t) / (c1.t - c0.t);
      const r = Math.round(c0.color[0] + (c1.color[0] - c0.color[0]) * localT);
      const g = Math.round(c0.color[1] + (c1.color[1] - c0.color[1]) * localT);
      const b = Math.round(c0.color[2] + (c1.color[2] - c0.color[2]) * localT);
      return `rgb(${r},${g},${b})`;
    }
  }
  return `rgb(${jetColorPoints[jetColorPoints.length - 1].color.join(",")})`;
};

interface Props extends GridVisualizationProps {}

const GridVisualization: React.FC<Props> = ({
  floorWidth,
  floorLength,
  uniformFlux,
  gridArrangement,
  gridPPFD,
}) => {
  const W_m = floorWidth * 0.3048;
  const L_m = floorLength * 0.3048;
  const positions = buildGridPositions(W_m, L_m, gridArrangement.rows, gridArrangement.cols);

  // For a uniform simulation, all COBs share the same flux.
  // Use the same normalization as ModularVisualization.tsx.
  const minFlux = 1000;
  const maxFlux = 2000;
  const norm = (uniformFlux - minFlux) / (maxFlux - minFlux);
  const ledColor = getJetColor(norm);

  // Compute the viewBox to cover all COB positions.
  const xs = positions.map((p) => p.x);
  const ys = positions.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const extra = (maxX - minX) * 0.1;
  const viewBox = `${minX - extra} ${minY - extra} ${maxX - minX + 2 * extra} ${maxY - minY + 2 * extra}`;

  return (
    <div className="grid-visualization-container" style={{ textAlign: "center" }}>
      <h3>Uniform Grid COB Arrangement</h3>
      <div style={{ width: "700px", height: "700px", border: "1px solid black", margin: "0 auto" }}>
        <svg
          width="100%"
          height="100%"
          viewBox={viewBox}
          preserveAspectRatio="xMidYMid meet"
          style={{ shapeRendering: "geometricPrecision" }}
        >
          <defs>
            <radialGradient id="gridLEDGradient" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="white" stopOpacity="1" />
              <stop offset="10%" stopColor="white" stopOpacity="1" />
              <stop offset="30%" stopColor={ledColor} stopOpacity="1" />
              <stop offset="100%" stopColor={ledColor} stopOpacity="0" />
            </radialGradient>
          </defs>
          {positions.map((pos, index) => (
            <circle
              key={index}
              cx={pos.x}
              cy={pos.y}
              r={0.04 * W_m}
              fill="url(#gridLEDGradient)"
              shapeRendering="geometricPrecision"
              stroke="black"
              strokeWidth={0.002 * W_m}
            />
          ))}
        </svg>
      </div>
    </div>
  );
};

export default GridVisualization;
