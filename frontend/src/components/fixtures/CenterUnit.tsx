import React from 'react';

interface CenterUnitProps {
  x: number;
  y: number;
  size: number;
  color: string;
  spacing: number;
}

const getComplementaryColor = (color: string): string => {
  const colorMap: { [key: string]: string } = {
    '#00FF00': '#FF00FF', // Green -> Magenta
    '#FFA500': '#0000FF', // Orange -> Blue
    '#800080': '#FFFF00', // Purple -> Yellow
    '#00FFFF': '#FF0000', // Cyan -> Red
    // Add more mappings as needed
  };
  return colorMap[color.toUpperCase()] || '#808080'; // Default to gray if not found
};

const CenterUnit: React.FC<CenterUnitProps> = ({ x, y, size, color, spacing }) => {
  const dynamicSize = spacing * 2; // Example: Make size relative to spacing
  const half = dynamicSize / 2;
  const complementaryColor = getComplementaryColor(color);
  const coBRadius = 60;

  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Outer square */}
      <rect
        x={-half}
        y={-half}
        width={dynamicSize}
        height={dynamicSize}
        fill='none'
        stroke={color}
        strokeWidth='10'
      />
      {/* Diagonal lines forming "X" */}
      <line x1={-half} y1={-half} x2={half} y2={half} stroke={color} strokeWidth='2' />
      <line x1={-half} y1={half} x2={half} y2={-half} stroke={color} strokeWidth='2' />
      {/* Corner LEDs */}
      <circle cx={-half} cy={-half} r={coBRadius} fill={color} />
      <circle cx={half} cy={-half} r={coBRadius} fill={color} />
      <circle cx={half} cy={half} r={coBRadius} fill={color} />
      <circle cx={-half} cy={half} r={coBRadius} fill={color} />
      {/* Central LED with Complementary Color */}
      <circle cx={0} cy={0} r={coBRadius} fill={complementaryColor} />
      <title>Center Unit</title>
    </g>
  );
};

export default CenterUnit;