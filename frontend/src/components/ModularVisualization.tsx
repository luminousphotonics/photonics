import React from "react";

interface ModularVisualizationProps {
  mad: number;
  optimized_ppfd: number;
  floorWidth: number; // in feet
  floorLength: number; // in feet
  floorHeight: number; // in feet
  optimizedLumensByLayer: number[]; // one per layer: index 0 = Center COB, index 1 = first ring, etc.
  simulationResult: any;
}

type CobPosition = { x: number; y: number; layer: number };

const buildCobPositionsDynamic = (W: number, L: number, H: number, floorWidth_ft: number, floorLength_ft: number): CobPosition[] => {
  // Use the larger dimension to determine the number of layers
  const maxDim_ft = Math.max(floorWidth_ft, floorLength_ft);
  const n = Math.max(1, Math.floor(maxDim_ft / 2) - 1);

  // Generate initial positions in a diamond pattern
  const positions: { x: number; y: number; layer: number }[] = [];
  positions.push({ x: 0, y: 0, layer: 0 }); // Center COB
  for (let i = 1; i <= n; i++) {
    for (let x = -i; x <= i; x++) {
      const yAbs = i - Math.abs(x);
      if (yAbs === 0) {
        positions.push({ x: x, y: 0, layer: i });
      } else {
        positions.push({ x: x, y: yAbs, layer: i });
        positions.push({ x: x, y: -yAbs, layer: i });
      }
    }
  }

  // Rotate positions by 45 degrees
  const theta = Math.PI / 4;
  const cos_t = Math.cos(theta);
  const sin_t = Math.sin(theta);
  const centerX = W / 2;
  const centerY = L / 2;

  // Define separate scaling factors for x and y
  const scaleX = (W / 2 * 0.95 * Math.SQRT2) / n;
  const scaleY = (L / 2 * 0.95 * Math.SQRT2) / n;

  // Transform positions with separate scaling
  const transformed = positions.map(pos => {
    const rx = pos.x * cos_t - pos.y * sin_t;
    const ry = pos.x * sin_t + pos.y * cos_t;
    return {
      x: centerX + rx * scaleX,
      y: centerY + ry * scaleY,
      layer: pos.layer
    };
  });

  return transformed;
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
  const floorWidthMeters = floorWidth * 0.3048;
  const floorLengthMeters = floorLength * 0.3048;
  const floorHeightMeters = floorHeight * 0.3048;
  
  const cobPositions = buildCobPositionsDynamic(floorWidthMeters, floorLengthMeters, floorHeightMeters, floorWidth, floorLength);
    
  const minLumens = Math.min(...optimizedLumensByLayer);
  const maxLumens = Math.max(...optimizedLumensByLayer);
  
  const xs = cobPositions.map(p => p.x);
  const ys = cobPositions.map(p => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const margin = (maxX - minX) * 0.1;
  const viewBox = `${minX - margin} ${minY - margin} ${maxX - minX + 2 * margin} ${maxY - minY + 2 * margin}`;
  
  // Color mapping functions (unchanged)
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
  const jetColors = createCustomJetColormap(512);
  const getColorForLumens = (lumens: number): string => {
    if (minLumens === maxLumens) return jetColors[jetColors.length - 1];
    const ratio = (lumens - minLumens) / (maxLumens - minLumens);
    const index = Math.round(ratio * (jetColors.length - 1));
    return jetColors[index];
  };

  return (
    <div className="modular-visualization-container">
      <h3>Staggered COB Arrangement</h3>
      <div className="visualization-wrapper" style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
        {/* SVG Visualization */}
        <div
          style={{
            width: "700px",
            height: "700px",
            border: "1px solid black",
          }}
        >
          <svg
            width="100%"
            height="100%"
            viewBox={viewBox}
            preserveAspectRatio="xMidYMid meet"
            style={{ shapeRendering: "geometricPrecision" }}
          >
            <defs>
              {cobPositions.map((pos, index) => {
                const flux =
                  optimizedLumensByLayer[pos.layer] ||
                  optimizedLumensByLayer[optimizedLumensByLayer.length - 1];
                const ledColor = getColorForLumens(flux);
                return (
                  <radialGradient
                    key={index}
                    id={`ledGradient-${index}`}
                    cx="50%"
                    cy="50%"
                    r="50%"
                  >
                    {/* Smaller white emitting center */}
                    <stop offset="0%" stopColor="white" stopOpacity="1" />
                    <stop offset="10%" stopColor="white" stopOpacity="1" />
                    {/* Transition quickly to the LED color */}
                    <stop offset="30%" stopColor={ledColor} stopOpacity="1" />
                    <stop offset="100%" stopColor={ledColor} stopOpacity="0" />
                  </radialGradient>
                );
              })}
            </defs>
            {cobPositions.map((pos, index) => (
              <circle
                key={index}
                cx={pos.x}
                cy={pos.y}
                r={0.04 * floorWidthMeters}
                fill={`url(#ledGradient-${index})`}
                shapeRendering="geometricPrecision"
                stroke="black"
                strokeWidth={0.002 * floorWidthMeters} // adjust as needed for thinness
              />
            ))}
          </svg>
        </div>
        {/* Vertical Color Legend */}
        <div style={{ marginLeft: "20px", display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div
            style={{
              background: `linear-gradient(to bottom, rgb(227,25,55), rgb(255,0,0), rgb(255,165,0), rgb(255,255,0), rgb(0,255,0), rgb(0,255,255), rgb(0,0,255))`,
              width: "20px",
              height: "300px",
              border: "1px solid #000",
              borderRadius: "4px",
            }}
          ></div>
          <div style={{ height: "300px", display: "flex", flexDirection: "column", justifyContent: "space-between", marginTop: "5px" }}>
            <span style={{ fontSize: "0.9em" }}>High</span>
            <span style={{ fontSize: "0.9em" }}>Low</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModularVisualization;