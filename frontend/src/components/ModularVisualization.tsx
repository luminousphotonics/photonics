import React from "react";

interface ModularVisualizationProps {
  mad: number;
  optimized_ppfd: number;
  floorWidth: number; // in feet
  floorLength: number; // in feet
  floorHeight: number; // in feet
  optimizedLumensByLayer: number[]; // array for layers 0-5 (center and 5 outer layers)
  simulationResult: any;
}

// --- Color Mapping Functions (Jet Colormap) ---
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

const createCustomJetColormap = (numShades: number): string[] => {
  const jetColors: string[] = [];
  for (let i = 0; i < numShades; i++) {
    const ratio = i / (numShades - 1);
    jetColors.push(getJetColor(ratio));
  }
  return jetColors;
};

const jetColors: string[] = createCustomJetColormap(512);

const getColorForLumens = (lumens: number, minLumens: number, maxLumens: number): string => {
  if (minLumens === maxLumens) {
    return jetColors[jetColors.length - 1];
  }
  const ratio = (lumens - minLumens) / (maxLumens - minLumens);
  const index = Math.round(ratio * (jetColors.length - 1));
  return jetColors[index];
};

// --- COB Positioning: Replicates Python's build_cob_positions ---
// Each COB will have an x, y position along with its layer index (0 = center, 1-5 = outer layers)
type CobPosition = { x: number; y: number; layer: number };

const buildCobPositions = (W: number, L: number, H: number): CobPosition[] => {
  // Define the diamond pattern layers (layer 0 is center)
  const layersCoords: [number, number][][] = [
    [[0, 0]], // Layer 0 (center)
    [[-1, 0], [1, 0], [0, -1], [0, 1]], // Layer 1
    [
      [-1, -1], [1, -1], [-1, 1], [1, 1],
      [-2, 0], [2, 0], [0, -2], [0, 2]
    ], // Layer 2
    [
      [-2, -1], [2, -1], [-2, 1], [2, 1],
      [-1, -2], [1, -2], [-1, 2], [1, 2],
      [-3, 0], [3, 0], [0, -3], [0, 3]
    ], // Layer 3
    [
      [-2, -2], [2, -2], [-2, 2], [2, 2],
      [-3, -1], [3, -1], [-3, 1], [3, 1],
      [-1, -3], [1, -3], [-1, 3], [1, 3],
      [-4, 0], [4, 0], [0, -4], [0, 4]
    ], // Layer 4
    [
      [-3, -2], [3, -2], [-3, 2], [3, 2],
      [-2, -3], [2, -3], [-2, 3], [2, 3],
      [-4, -1], [4, -1], [-4, 1], [4, 1],
      [-1, -4], [1, -4], [-1, 4], [1, 4],
      [-5, 0], [5, 0], [0, -5], [0, 5]
    ] // Layer 5
  ];
  const positions: CobPosition[] = [];
  const centerX = W / 2;
  const centerY = L / 2;
  // Add center (layer 0)
  positions.push({ x: centerX, y: centerY, layer: 0 });
  const factor = W / 7.2;
  const theta = Math.PI / 4; // 45° in radians
  for (let i = 1; i < layersCoords.length; i++) {
    const layer = layersCoords[i];
    for (const [dx, dy] of layer) {
      const rx = dx * Math.cos(theta) - dy * Math.sin(theta);
      const ry = dx * Math.sin(theta) + dy * Math.cos(theta);
      const px = centerX + rx * factor;
      const py = centerY + ry * factor;
      positions.push({ x: px, y: py, layer: i });
    }
  }
  return positions;
};

const ModularVisualization: React.FC<ModularVisualizationProps> = ({
  mad,
  optimized_ppfd,
  floorWidth,
  floorLength,
  floorHeight,
  optimizedLumensByLayer,
  simulationResult,
}) => {
  // Convert floor dimensions (and height) from feet to meters (1 ft = 0.3048 m)
  const floorWidthMeters = floorWidth * 0.3048;
  const floorLengthMeters = floorLength * 0.3048;
  const floorHeightMeters = floorHeight * 0.3048; // For example, 3 ft ~0.9144 m

  // Compute COB positions using the same algorithm as in ml_simulation.py
  const cobPositions = buildCobPositions(floorWidthMeters, floorLengthMeters, floorHeightMeters);

  // Determine the min and max luminous flux from the provided layers (assumes 6 layers)
  const minLumens = Math.min(...optimizedLumensByLayer);
  const maxLumens = Math.max(...optimizedLumensByLayer);

  // Compute bounding box of COB positions and add a margin for the SVG viewBox
  const xs = cobPositions.map(p => p.x);
  const ys = cobPositions.map(p => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const margin = (maxX - minX) * 0.1; // 10% margin
  const viewBox = `${minX - margin} ${minY - margin} ${maxX - minX + 2 * margin} ${maxY - minY + 2 * margin}`;

  return (
    <div className="modular-visualization-container">
      <div className="visualization-content">
        {/* SVG container for COB points */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            width: "700px",
            height: "700px",
            border: "1px solid black",
          }}
        >
          <svg width="100%" height="100%" viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
            {cobPositions.map((pos, index) => {
              // Use the layer index to get the corresponding luminous flux
              const color = getColorForLumens(optimizedLumensByLayer[pos.layer], minLumens, maxLumens);
              return (
                <circle
                  key={index}
                  cx={pos.x}
                  cy={pos.y}
                  r={0.03 * floorWidthMeters} // Reduced circle radius multiplier
                  fill={color}
                  stroke="#000"
                  strokeWidth={0.002 * floorWidthMeters} // Adjusted stroke width
                />
              );
            })}
          </svg>
          
        </div>
        {/* Color Legend */}
        <div className="color-legend" style={{ marginTop: "20px", textAlign: "center" }}>
          <div
            style={{
              background: `linear-gradient(to top, rgb(0,0,255), rgb(0,255,255), rgb(0,255,0), rgb(255,255,0), rgb(255,165,0), rgb(255,0,0), rgb(227,25,55))`,
              width: "20px",
              height: "300px",
              border: "1px solid #000",
              borderRadius: "4px",
              margin: "0 auto",
            }}
          ></div>
          <div style={{ display: "flex", justifyContent: "space-between", width: "100px", margin: "10px auto" }}>
            <span>Low</span>
            <span>High</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModularVisualization;
