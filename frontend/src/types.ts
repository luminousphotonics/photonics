export interface SimulationData {
    optimized_lumens_by_layer: number[];
    mad: number;
    optimized_ppfd: number;
    floor_width: number;
    floor_length: number;
    floor_height: number;
    target_ppfd: number;
    fixtures: FixtureData[];
    //HEIGHT_FROM_FLOOR: number; // Add floor_height
  }

  export interface FixtureData {
    type: 'CenterUnit' | 'LinearUnit' | 'LShapedUnit' | 'ReverseLShapedUnit';
    x: number;
    y: number;
    rotation: number;
    color: string;
    key: string;
    numElementsLong?: number; // Optional
    numElementsShort?: number; // Optional
    size?: number; // Optional
    spacing?: number; // Optional
    layer: number;
    numElements?: number; // Optional, for LinearUnit
  }