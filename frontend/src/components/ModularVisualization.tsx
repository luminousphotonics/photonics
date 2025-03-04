import React, { useState, useEffect, useRef } from "react";
import CenterUnit from "./fixtures/CenterUnit";
import LinearUnit from "./fixtures/LinearUnit";
import LShapedUnit from "./fixtures/LShapedUnit";
import CustomTopLeftUnit from "./fixtures/CustomTopLeftUnit";

interface ModularVisualizationProps {
  mad: number;
  optimized_ppfd: number;
  floorWidth: number;
  floorLength: number;
  floorHeight: number;
  optimizedLumensByLayer: number[];
  simulationResult: any;
}

const ModularVisualization: React.FC<ModularVisualizationProps> = ({
  mad,
  optimized_ppfd,
  floorWidth,
  floorLength,
  floorHeight,
  optimizedLumensByLayer,
  simulationResult,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Convert floor dimensions from feet to meters for internal calculations
  const floorWidthMeters = floorWidth * 0.3048;
  const floorLengthMeters = floorLength * 0.3048;

  // Determine min and max lumens for color scaling
  const minLumens = Math.min(...optimizedLumensByLayer);
  const maxLumens = Math.max(...optimizedLumensByLayer);

  // State for dynamically calculated layers
  const [selectedLayers, setSelectedLayers] = useState(0);

  // Fixed unit size (adjust as needed)
  const fixedUnitSize = 50;

  // Spacing between layers - adjusted for meters
  const layerSpacing = fixedUnitSize * 0.6; // Reduced spacing for meters

  // Set a large canvas size for the viewBox
  const viewBoxWidth = 11000;
  const viewBoxHeight = 11000;

  // Calculate center based on the viewBox
  const centerX = viewBoxWidth / 2;
  const centerY = viewBoxHeight / 2;

  // Introduce offsetY for upward shift
  const offsetY = 0; // Adjust this value as needed

  /**
   * Defines key color points for the custom jet colormap.
   */
  const jetColorPoints: { t: number; color: [number, number, number] }[] = [
    { t: 0.0, color: [0, 0, 255] }, // Blue
    { t: 0.16, color: [0, 255, 255] }, // Cyan
    { t: 0.33, color: [0, 255, 0] }, // Green
    { t: 0.5, color: [255, 255, 0] }, // Yellow
    { t: 0.66, color: [255, 165, 0] }, // Orange
    { t: 0.83, color: [255, 0, 0] }, // Red
    { t: 1.0, color: [227, 25, 55] }, // Custom Red (#E31937)
  ];

  /**
   * Interpolates between defined key color points to get a jet-like color.
   * Ensures the highest ratio maps to #E31937.
   * @param {number} t - Normalized ratio between 0 and 1.
   * @returns {string} - The interpolated RGB color string.
   */
  const getJetColor = (t: number): string => {
    // Clamp t to [0,1]
    t = Math.max(0, Math.min(1, t));

    // Iterate through color points to find the appropriate segment
    for (let i = 0; i < jetColorPoints.length - 1; i++) {
      const c0 = jetColorPoints[i];
      const c1 = jetColorPoints[i + 1];
      if (t >= c0.t && t <= c1.t) {
        const localT = (t - c0.t) / (c1.t - c0.t); // Normalize within the segment
        const r = Math.round(c0.color[0] + (c1.color[0] - c0.color[0]) * localT);
        const g = Math.round(c0.color[1] + (c1.color[1] - c0.color[1]) * localT);
        const b = Math.round(c0.color[2] + (c1.color[2] - c0.color[2]) * localT);
        return `rgb(${r},${g},${b})`;
      }
    }
    // Fallback to the highest color if t is exactly 1
    return `rgb(${jetColorPoints[jetColorPoints.length - 1].color.join(",")})`;
  };

  /**
   * Generates a custom jet colormap array with a specified number of shades.
   * @param {number} numShades - The number of color shades to generate.
   * @returns {string[]} - An array of RGB color strings.
   */
  const createCustomJetColormap = (numShades: number): string[] => {
    const jetColors: string[] = [];
    for (let i = 0; i < numShades; i++) {
      const ratio = i / (numShades - 1); // Normalize ratio between 0 and 1
      jetColors.push(getJetColor(ratio));
    }
    return jetColors;
  };

  // Generate the custom jet colormap with increased shades for better sensitivity
  const jetColors: string[] = createCustomJetColormap(512); // Increase to 512 shades

  // Utility function to map lumens to custom jet color
  const getColorForLumens = (lumens: number): string => {
    if (minLumens === maxLumens) {
      // Avoid division by zero if all lumens are equal
      return jetColors[jetColors.length - 1]; // Default to the highest color (#E31937)
    }
    const ratio = (lumens - minLumens) / (maxLumens - minLumens);
    const index = Math.round(ratio * (jetColors.length - 1));
    return jetColors[index];
  };

  // Function to determine the number of layers based on floor dimensions in meters
  const calculateLayers = (width: number, length: number): number => {
    const layerDimensions = [
      { width: 0, length: 0 }, // Layer 0 (center)
      { width: 0.9144, length: 0.9144 }, // Layer 1 (3 ft in meters)
      { width: 1.524, length: 1.524 }, // Layer 2 (5 ft in meters)
      { width: 2.4384, length: 2.4384 }, // Layer 3 (8 ft in meters)
      { width: 3.048, length: 3.048 }, // Layer 4 (10 ft in meters)
      { width: 3.6576, length: 3.6576 }, // Layer 5 (12 ft in meters)
    ];

    let activeLayers = 0;
    for (let i = 0; i < layerDimensions.length; i++) {
      if (width >= layerDimensions[i].width && length >= layerDimensions[i].length) {
        activeLayers = i + 1; // Add one to include the central element as Layer 0
      } else {
        break; // Stop when dimensions no longer fit
      }
    }
    return activeLayers;
  };

  // Update selectedLayers based on floor dimensions
  useEffect(() => {
    const calculatedLayers = calculateLayers(floorWidthMeters, floorLengthMeters);
    setSelectedLayers(calculatedLayers);
  }, [floorWidthMeters, floorLengthMeters]);

  // Function to determine the number of fixtures in a layer (No changes)
  const determineFixturesInLayer = (layer: number): number => {
    switch (layer) {
      case 0:
        return 1; // Center unit (Layer 0)
      case 1:
        return 4; // 4 fixtures in Layer 1
      case 2:
        return 4; // 4 fixtures in Layer 2
      case 3:
        return 4; // 4 fixtures in Layer 3
      case 4:
        return 4; // 4 fixtures in Layer 4
      case 5:
        return 4; // 4 fixtures in Layer 5
      case 6:
        return 4; // 4 fixtures in Layer 6
      default:
        return 0;
    }
  };

  // Function to determine fixture placement and return corresponding JSX Element (CORRECTED for meters)
  const determineFixturePlacement = (
    layer: number,
    fixtureIndex: number
  ): JSX.Element | null => {
    if (layer >= optimizedLumensByLayer.length) return null;

    // Determine color based on layer's lumens
    const layerLumens = optimizedLumensByLayer[layer];
    const fixtureColor = getColorForLumens(layerLumens);

    // Calculate offset based on layer (in meters)
    const offsetValue = layer * layerSpacing;

    // Define rotation angles for fixtures in each layer
    const rotationAngles = [0, -45, 0, 45, 0, 0];

    // Define the spacing for each layer (CORRECTED for meters)
    const layerSpacings = [
      fixedUnitSize * 11, // Layer 0 (center)
      fixedUnitSize * 1.7, // Layer 1
      fixedUnitSize * 1.6, // Layer 2
      fixedUnitSize * 1.5, // Layer 3
      fixedUnitSize * 1.4, // Layer 4
      fixedUnitSize * 1.3, // Layer 5
    ];

    // Center Unit
    if (layer === 0) {
      return (
        <CenterUnit
          key={`center`}
          x={centerX}
          y={centerY}
          size={fixedUnitSize}
          spacing={layerSpacings[layer]}
          color={fixtureColor}
        />
      );
    }

    // Layer 1: 4 L-Shaped Units with -45 degree rotation
    if (layer === 1) {
      const positions = [
        { x: centerX - offsetValue, y: centerY - offsetValue, rotation: rotationAngles[layer] },
        { x: centerX + offsetValue, y: centerY - offsetValue, rotation: rotationAngles[layer] },
        { x: centerX - offsetValue, y: centerY + offsetValue, rotation: rotationAngles[layer] },
        { x: centerX + offsetValue, y: centerY + offsetValue, rotation: rotationAngles[layer] },
      ];
      const position = positions[fixtureIndex];
      return (
        <LShapedUnit
          key={`L-${layer}-${fixtureIndex}`}
          x={position.x}
          y={position.y}
          numElementsLong={3}
          numElementsShort={2}
          rotation={position.rotation}
          spacing={layerSpacings[layer]}
          color={fixtureColor}
        />
      );
    }

    // Layer 2: 4 Linear Units
    if (layer === 2) {
      const positions = [
        { x: centerX, y: centerY - offsetValue, rotation: rotationAngles[layer] },
        { x: centerX + offsetValue, y: centerY, rotation: rotationAngles[layer] + 90 },
        { x: centerX, y: centerY + offsetValue, rotation: rotationAngles[layer] + 180 },
        { x: centerX - offsetValue, y: centerY, rotation: rotationAngles[layer] + 270 },
      ];
      const position = positions[fixtureIndex];
      return (
        <LinearUnit
          key={`linear-${layer}-${fixtureIndex}`}
          x={position.x}
          y={position.y}
          numElements={3}
          rotation={position.rotation}
          spacing={layerSpacings[layer]}
          color={fixtureColor}
        />
      );
    }

    // Layer 3: 4 L-Shaped Units
    if (layer === 3) {
      const positions = [
        { x: centerX - offsetValue, y: centerY - offsetValue, rotation: rotationAngles[layer] },
        { x: centerX + offsetValue, y: centerY - offsetValue, rotation: rotationAngles[layer] },
        { x: centerX - offsetValue, y: centerY + offsetValue, rotation: rotationAngles[layer] },
        { x: centerX + offsetValue, y: centerY + offsetValue, rotation: rotationAngles[layer] },
      ];
      const position = positions[fixtureIndex];
      return (
        <LShapedUnit
          key={`L-${layer}-${fixtureIndex}`}
          x={position.x}
          y={position.y}
          numElementsLong={3}
          numElementsShort={2}
          rotation={position.rotation}
          spacing={layerSpacings[layer]}
          color={fixtureColor}
        />
      );
    }

    // Layer 4: 4 Custom Top Left Units
    if (layer === 4) {
      const positions = [
        { x: centerX - offsetValue, y: centerY - offsetValue, rotation: rotationAngles[layer] },
        { x: centerX + offsetValue, y: centerY - offsetValue, rotation: rotationAngles[layer] },
        { x: centerX - offsetValue, y: centerY + offsetValue, rotation: rotationAngles[layer] },
        { x: centerX + offsetValue, y: centerY + offsetValue, rotation: rotationAngles[layer] },
      ];
      const position = positions[fixtureIndex];
      return (
        <CustomTopLeftUnit
          key={`CTL-${layer}-${fixtureIndex}`}
          x={position.x}
          y={position.y}
          numElementsLong={3}
          numElementsShort={2}
          rotation={position.rotation}
          spacing={layerSpacings[layer]}
          color={fixtureColor}
        />
      );
    }

    // Layer 5: 4 Custom Top Left Units
    if (layer === 5) {
      const positions = [
        { x: centerX - offsetValue, y: centerY - offsetValue, rotation: rotationAngles[layer] },
        { x: centerX + offsetValue, y: centerY - offsetValue, rotation: rotationAngles[layer] },
        { x: centerX - offsetValue, y: centerY + offsetValue, rotation: rotationAngles[layer] },
        { x: centerX + offsetValue, y: centerY + offsetValue, rotation: rotationAngles[layer] },
      ];
      const position = positions[fixtureIndex];
      return (
        <CustomTopLeftUnit
          key={`CTL-${layer}-${fixtureIndex}`}
          x={position.x}
          y={position.y}
          numElementsLong={3}
          numElementsShort={2}
          rotation={position.rotation}
          spacing={layerSpacings[layer]}
          color={fixtureColor}
        />
      );
    }

    return null; // For layers beyond 5 (shouldn't happen with current calculateLayers)
  };

  // Generate fixtures based on layers
  const fixtures: JSX.Element[] = [];
  for (let layer = 0; layer < selectedLayers; layer++) {
    const numFixturesInLayer = determineFixturesInLayer(layer);
    for (let fixtureIndex = 0; fixtureIndex < numFixturesInLayer; fixtureIndex++) {
      const fixture = determineFixturePlacement(layer, fixtureIndex);
      if (fixture) {
        fixtures.push(fixture);
      }
    }
  }

  // Calculate viewBox dimensions to fit the visualization and use the declared variable
  const viewBox = `0 ${offsetY} ${viewBoxWidth} ${viewBoxHeight}`;

  return (
    <div className="modular-visualization-container">
      <div className="visualization-content">
        {/* Container for Centering SVG */}
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
          {/* SVG Visualization */}
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            viewBox={viewBox}
            preserveAspectRatio="xMidYMid meet"
          >
            {fixtures}
          </svg>
        </div>

        {/* Color Legend to the right of the SVG */}
        <div className="color-legend">
          <div
            className="legend-gradient"
            style={{
              background: `linear-gradient(to top, rgb(0,0,255), rgb(0,255,255), rgb(0,255,0), rgb(255,255,0), rgb(255,165,0), rgb(255,0,0), rgb(227,25,55))`,
              width: "20px",
              height: "300px",
              border: "1px solid #000",
              borderRadius: "4px",
            }}
          ></div>
          <div className="legend-labels-container">
            <div
              className="legend-labels"
              style={{
                display: "flex",
                flexDirection: "column-reverse",
                justifyContent: "space-between",
                height: "300px",
                marginLeft: "5px",
              }}
            >
              <span>Low</span>
              <span>Mid-Low</span>
              <span>Mid</span>
              <span>Mid-High</span>
              <span>High</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModularVisualization;
