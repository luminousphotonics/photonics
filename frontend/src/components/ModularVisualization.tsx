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

  // Determine min and max lumens for color scaling
  const minLumens = Math.min(...optimizedLumensByLayer);
  const maxLumens = Math.max(...optimizedLumensByLayer);

  // State for dynamically calculated layers
  const [selectedLayers, setSelectedLayers] = useState(0);

  // Fixed unit size (adjust as needed)
  const fixedUnitSize = 50;

  // Spacing between layers
  const layerSpacing = fixedUnitSize * 22;

  // **Special spacing between Layer 0 and Layer 1**
  const layer0To1Spacing = fixedUnitSize * 30; // Increase as needed

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
    return `rgb(${jetColorPoints[jetColorPoints.length - 1].color.join(
      ","
    )})`;
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

  // Function to determine the number of layers based on floor dimensions (from previous version)
  const calculateLayers = (width: number, length: number): number => {
    const layerDimensions = [
      { width: 0, length: 0 },
      { width: 3, length: 3 },
      { width: 5, length: 5 },
      { width: 8, length: 8 },
      { width: 11, length: 11 },
      { width: 14, length: 14 },
    ];

    let fittedLayers = 0;
    for (let i = 1; i < layerDimensions.length; i++) {
      if (
        width >= layerDimensions[i].width &&
        length >= layerDimensions[i].length
      ) {
        fittedLayers = i;
      } else {
        break;
      }
    }
    return fittedLayers;
  };

  // Update selectedLayers based on floor dimensions
  useEffect(() => {
    if (floorWidth > 0 && floorLength > 0) {
      const calculatedLayers = calculateLayers(floorWidth, floorLength);
      setSelectedLayers(calculatedLayers);
    }
  }, [floorWidth, floorLength]);

  // Function to determine the number of fixtures in a layer
  const determineFixturesInLayer = (layer: number): number => {
    if (layer >= selectedLayers) return 0; // Only render up to selectedLayers

    switch (layer) {
      case 0:
        return 1;
      case 1:
        return 2;
      case 2:
        return 4;
      case 3:
        return 4;
      case 4:
        return 6;
      default:
        return 0;
    }
  };

  // Function to determine fixture placement and return corresponding JSX Element
  const determineFixturePlacement = (
    layer: number,
    fixtureIndex: number
  ): JSX.Element | null => {
    if (layer >= optimizedLumensByLayer.length) return null;

    // Determine color based on layer's lumens
    const layerLumens = optimizedLumensByLayer[layer];
    const fixtureColor = getColorForLumens(layerLumens);
    // Calculate offset based on layer using the special spacing for Layer 0 to 1
    let offsetValue = layer * layerSpacing;
    if (layer > 1) {
      offsetValue = layer0To1Spacing + (layer - 1) * layerSpacing;
    } else if (layer === 1) {
      offsetValue = layer0To1Spacing;
    }
    // Calculate offset based on layer
    //const offsetValue = layer * layerSpacing;

    // Center Unit
    if (layer === 0) {
      return (
        <CenterUnit
          key={`center`}
          x={centerX}
          y={centerY}
          size={fixedUnitSize}
          spacing={fixedUnitSize * 11}
          color={fixtureColor}
        />
      );
    }

    // Layer 1: 2 Reverse L-Shaped Units
    if (layer === 1) {
      const positions = [
        {
          type: "ReverseL",
          x: centerX - offsetValue,
          y: centerY + offsetValue,
          rotation: 180,
        },
        {
          type: "ReverseL",
          x: centerX + offsetValue,
          y: centerY - offsetValue,
          rotation: 0,
        },
      ];

      const position = positions[fixtureIndex];
      if (position.type === "ReverseL") {
        return (
          <CustomTopLeftUnit
            key={`CTL-${layer}-${fixtureIndex}`}
            x={position.x}
            y={position.y}
            numElementsLong={3}
            numElementsShort={2}
            rotation={position.rotation}
            spacing={fixedUnitSize * 30}
            color={fixtureColor}
          />
        );
      }
    }

    // Layer 2: 4 Linear Units
    if (layer === 2) {
      const positions = [
        { x: centerX / 1.2, y: centerY - offsetValue, rotation: 0 }, // Top
        { x: centerX + offsetValue, y: centerY / 1.2, rotation: 90 }, // Right
        { x: centerX * 1.16, y: centerY + offsetValue, rotation: 180 }, // Bottom
        { x: centerX - offsetValue, y: centerY * 1.120, rotation: 270 }, // Left
      ];

      const position = positions[fixtureIndex];
      return (
        <LinearUnit
          key={`linear-${layer}-${fixtureIndex}`}
          x={position.x}
          y={position.y}
          numElements={3}
          rotation={position.rotation}
          spacing={fixedUnitSize * 35}
          color={fixtureColor}
        />
      );
    }

    // Layer 3: 4 Reverse L-Shaped Units
    if (layer === 3) {
      const positions = [
        {
          type: "ReverseL",
          x: centerX - offsetValue,
          y: centerY - offsetValue,
          rotation: 270,
        },
        {
          type: "ReverseL",
          x: centerX - offsetValue,
          y: centerY + offsetValue,
          rotation: 180,
        },
        {
          type: "ReverseL",
          x: centerX + offsetValue,
          y: centerY + offsetValue,
          rotation: 90,
        },
        {
          type: "ReverseL",
          x: centerX + offsetValue,
          y: centerY - offsetValue,
          rotation: 0,
        },
      ];

      const position = positions[fixtureIndex];
      if (position.type === "ReverseL") {
        return (
          <CustomTopLeftUnit
            key={`CTL-${layer}-${fixtureIndex}`}
            x={position.x}
            y={position.y}
            numElementsLong={3}
            numElementsShort={2}
            rotation={position.rotation}
            spacing={fixedUnitSize * 35}
            color={fixtureColor}
          />
        );
      }
    }

    // Layer 4: 6 Fixtures (1 L-Shaped, 1 Reverse L-Shaped, 4 Linear Units)
    if (layer === 4) {
      const positions = [
        {
          type: "ReverseL",
          x: centerX - offsetValue,
          y: centerY - offsetValue,
          rotation: 270,
        },
        { type: "Linear", x: centerX * 1.165, y: centerY - offsetValue, rotation: 0 },
        {
          type: "Linear",
          x: centerX + offsetValue,
          y: centerY - offsetValue / 1.59,
          rotation: 90,
        }, // Adjust upward by 10 units
        {
          type: "L",
          x: centerX - offsetValue,
          y: centerY + offsetValue,
          rotation: 270,
        },
        { type: "Linear", x: centerX * 1.160, y: centerY + offsetValue, rotation: 180 },
        {
          type: "Linear",
          x: centerX + offsetValue,
          y: centerY + offsetValue / 1.59,
          rotation: 90,
        }, // Adjust downward by 10 units
      ];

      const position = positions[fixtureIndex];
      if (position.type === "L") {
        return (
          <LShapedUnit
            key={`L-${layer}-${fixtureIndex}`}
            x={position.x}
            y={position.y}
            numElementsLong={3}
            numElementsShort={2}
            rotation={position.rotation}
            spacing={fixedUnitSize * 35}
            color={fixtureColor}
          />
        );
      } else if (position.type === "ReverseL") {
        return (
          <CustomTopLeftUnit
            key={`CTL-${layer}-${fixtureIndex}`}
            x={position.x}
            y={position.y}
            numElementsLong={3}
            numElementsShort={2}
            rotation={position.rotation}
            spacing={fixedUnitSize * 35}
            color={fixtureColor}
          />
        );
      } else if (position.type === "Linear") {
        return (
          <LinearUnit
            key={`linear-${layer}-${fixtureIndex}`}
            x={position.x}
            y={position.y}
            numElements={3}
            rotation={position.rotation}
            spacing={fixedUnitSize * 35}
            color={fixtureColor}
          />
        );
      }
    }

    return null;
  };

  // Generate fixtures based on layers
  const fixtures: JSX.Element[] = [];
  for (let layer = 0; layer < selectedLayers; layer++) {
    const numFixturesInLayer = determineFixturesInLayer(layer);
    for (let fixtureIndex = 0; fixtureIndex < numFixturesInLayer; fixtureIndex++) {
      const fixture = determineFixturePlacement(layer, fixtureIndex);
      if (fixture) {
        fixtures.push(fixture);
        console.log("selectedLayers:", selectedLayers);
      console.log("optimizedLumensByLayer length:", optimizedLumensByLayer.length);

      }
    }
  }

  // Calculate viewBox dimensions to fit the visualization
  const viewBox = `0 0 ${viewBoxWidth} ${viewBoxHeight}`;

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
            viewBox={`${0} ${offsetY} ${viewBoxWidth} ${viewBoxHeight}`}
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