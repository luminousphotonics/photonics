// src/components/fixtures/CustomTopLeftUnit.tsx

import React from 'react';

interface CustomTopLeftUnitProps {
  x: number;
  y: number;
  numElementsLong: number;
  numElementsShort: number;
  spacing: number;
  color: string;
  rotation: number; // Add rotation prop
}

const getComplementaryColor = (color: string): string => {
  const colorMap: { [key: string]: string } = {
    '#00FF00': '#FF00FF',
    '#FFA500': '#0000FF',
    '#800080': '#FFFF00',
    '#00FFFF': '#FF0000',
  };
  return colorMap[color.toUpperCase()] || '#808080';
};

const CustomTopLeftUnit: React.FC<CustomTopLeftUnitProps> = ({
  x,
  y,
  numElementsLong,
  numElementsShort,
  spacing,
  color,
  rotation, // Include rotation
}) => {
  const complementaryColor = getComplementaryColor(color);
  const coBRadius = 60;

  // Linear CoBs pointing left
  const elementsLong = Array.from({ length: numElementsLong }, (_, i) => (
    <circle
      key={`long-${i}`}
      cx={-i * spacing} // Points leftward
      cy={0}
      r={coBRadius}
      fill={color}
      stroke={color}
      strokeWidth="30"
    />
  ));

  // Single CoBs pointing downward
  const elementsShort = Array.from({ length: numElementsShort }, (_, i) => (
    <circle
      key={`short-${i}`}
      cx={0}
      cy={i * spacing} // Points downward
      r={coBRadius}
      fill={color}
      stroke={color}
      strokeWidth="30"
    />
  ));

  return (
    <g transform={`translate(${x}, ${y}) rotate(${rotation})`}> {/* Apply rotation */}
      {/* Line for Linear CoBs */}
      <line
        x1={0}
        y1={0}
        x2={-(numElementsLong - 1) * spacing} // Line pointing leftward
        y2={0}
        stroke={color}
        strokeWidth="30"
      />
      {/* Line for Single CoBs */}
      <line
        x1={0}
        y1={0}
        x2={0}
        y2={(numElementsShort - 1) * spacing} // Line pointing downward
        stroke={color}
        strokeWidth="30"
      />
      {elementsLong}
      {elementsShort}
      {/* Base CoB */}
      <circle cx={0} cy={0} r={coBRadius} fill={complementaryColor} stroke="black" strokeWidth="1">
        <title>Base CoB</title>
      </circle>
      <title>Custom Top Left Unit</title>
    </g>
  );
};

export default CustomTopLeftUnit;
