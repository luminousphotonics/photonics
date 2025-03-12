export interface SimulationData {
  optimized_lumens_by_layer: number[];
  mad: number;
  optimized_ppfd: number;
  floor_width: number;
  floor_length: number;
  floor_height: number;
  target_ppfd: number;
  fixtures: FixtureData[];
  surface_graph?: string;
  heatmap?: string;
  grid_surface_graph?: string;
  grid_heatmap?: string;
  grid_cob_arrangement?: { rows: number; cols: number };
  grid_uniform_flux?: number;
  grid_ppfd?: number;
  grid_mad?: number;
}

export interface FixtureData {
  type: 'CenterUnit' | 'LinearUnit' | 'LShapedUnit' | 'ReverseLShapedUnit';
  x: number;
  y: number;
  rotation: number;
  color: string;
  key: string;
  numElementsLong?: number;
  numElementsShort?: number;
  size?: number;
  spacing?: number;
  layer: number;
  numElements?: number;
}
