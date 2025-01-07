import React from 'react';

interface LShapedUnitProps {
  x: number;
  y: number;
  numElementsLong: number; // CoBs along the long edge
  numElementsShort: number; // CoBs along the short edge
  rotation: number; // Rotation in degrees
  spacing: number; // Spacing between CoBs
  color: string; // Fixture color
}

const LShapedUnit: React.FC<LShapedUnitProps> = ({
  x,
  y,
  numElementsLong,
  numElementsShort,
  rotation,
  spacing,
  color,
}) => {
  const coBRadius = 60;

  // Generate positions for the long edge CoBs
  const longCoBs = Array.from({ length: numElementsLong }, (_, i) => (
    <circle
      key={`long-${i}`}
      cx={i * spacing}
      cy={0}
      r={coBRadius}
      fill={color}
      stroke={color}
      strokeWidth="30"
    />
  ));

  // Generate positions for the short edge CoBs
  const shortCoBs = Array.from({ length: numElementsShort }, (_, i) => (
    <circle
      key={`short-${i}`}
      cx={0}
      cy={i * spacing}
      r={coBRadius}
      fill={color}
      stroke={color}
      strokeWidth="30"
    />
  ));

  return (
    <g transform={`translate(${x}, ${y}) rotate(${rotation})`}>
      {/* Render the long edge */}
      <line
        x1={0}
        y1={0}
        x2={(numElementsLong - 1) * spacing}
        y2={0}
        stroke={color}
        strokeWidth="30"
      />
      {/* Render the short edge */}
      <line
        x1={0}
        y1={0}
        x2={0}
        y2={(numElementsShort - 1) * spacing}
        stroke={color}
        strokeWidth="30"
      />
      {/* Render CoBs */}
      {longCoBs}
      {shortCoBs}
    </g>
  );
};

export default LShapedUnit;