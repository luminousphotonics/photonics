import React from 'react';

interface LinearUnitProps {
  x: number;
  y: number;
  numElements: number; // Number of CoBs in the line
  rotation: number; // Rotation in degrees
  spacing: number; // Spacing between CoBs
  color: string; // Fixture color
}

const LinearUnit: React.FC<LinearUnitProps> = ({ x, y, numElements, rotation, spacing, color }) => {
  const coBRadius = 60;

  // Generate positions for the CoBs in a straight line
  const coBs = Array.from({ length: numElements }, (_, i) => {
    const offset = (i - (numElements - 1) / 2) * spacing;
    return (
      <circle
        key={i}
        cx={offset}
        cy={0}
        r={coBRadius}
        fill={color}
        stroke={color}
        strokeWidth="30"
      />
    );
  });

  return (
    <g transform={`translate(${x}, ${y}) rotate(${rotation})`}>
      {/* Render the line */}
      <line
        x1={-((numElements - 1) / 2) * spacing}
        y1={0}
        x2={((numElements - 1) / 2) * spacing}
        y2={0}
        stroke={color}
        strokeWidth="30"
      />
      {/* Render the CoBs */}
      {coBs}
    </g>
  );
};

export default LinearUnit;