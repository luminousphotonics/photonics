#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import CubicSpline

# ---------------------------
# Calibration Data for a 20-ft floor (6.096 m)
# Normalized layer positions for 10 calibration points (0 = center, 1 = outermost)
x_cal = np.linspace(0, 1, 10)
flux_cal = np.array([
    4036.684,   # Center COB 
    2018.342,   # Layer 2
    8074.0665,  # Layer 3
    5045.855,   # Layer 4
    6055.0856,  # Layer 5
    12110.052,  # Layer 6
    15137.565,  # Layer 7
    2018.342,   # Layer 8
    20183.42,   # Layer 9
    25000.0    # Outermost layer forced to 25,000 lumens
], dtype=np.float64)

# ---------------------------
# Target overall luminous flux (A_target)
# According to your calibration, ~1250 µmol/m²/s PPFD corresponds roughly to 9074 lumens.
A_target = 9074

# Desired number of layers (center COB + additional layers)
NUM_LAYERS = 15

# ---------------------------
def generate_flux_assignments(num_layers):
    """
    Generate predicted luminous flux assignments.
    
    Steps:
      1. Build a cubic spline from the calibration data.
      2. Evaluate the spline at normalized positions for the desired number of layers.
      3. Force the outermost layer (x = 1) to exactly equal 25,000 lumens.
      4. Compute a normalization factor S so that:
             (S * sum(inner layers) + 25000) / num_layers = A_target.
      5. Multiply the inner layers by S while keeping the outermost layer fixed.
    Returns:
      A numpy array of flux assignments.
    """
    # Build the cubic spline (natural boundary conditions)
    cs = CubicSpline(x_cal, flux_cal, bc_type='natural')
    
    # Evaluate at evenly spaced normalized positions for num_layers
    x_new = np.linspace(0, 1, num_layers)
    f_raw = cs(x_new)
    
    # Force the outermost layer to exactly 25,000 lumens
    f_raw[-1] = 25000.0
    
    # Compute normalization factor S for layers 0 to N-2
    S = (A_target * num_layers - 25000.0) / np.sum(f_raw[:-1])
    
    # Scale inner layers by S, keep outermost fixed
    f_scaled = np.copy(f_raw)
    f_scaled[:-1] = S * f_raw[:-1]
    f_scaled[-1] = 25000.0
    
    return f_scaled

def main():
    flux_assignments = generate_flux_assignments(NUM_LAYERS)
    
    print("Predicted luminous flux assignments:")
    for i, flux in enumerate(flux_assignments):
        if i == 0:
            print(f"    {flux:.4f},     # Center COB")
        else:
            print(f"    {flux:.4f},     # Layer {i}")

if __name__ == "__main__":
    main()
