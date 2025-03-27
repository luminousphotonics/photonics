#!/usr/bin/env python3
"""
Closed-form style flux assignment for concentric-square COB layers.

By default, this script has polynomial fits for N=10, N=15, and N=20.
If your desired N is between 10 and 20, it linearly interpolates those polynomial
coefficients. If N is outside that range, it simply picks the nearest set
of coefficients (you can modify if needed).

Usage:
  python closed_form_flux.py
"""

import numpy as np

def build_cob_positions(W, L, H, N):
    """
    Builds COB positions for N layers over a W x L floor at height H.
    Same approach as your original script.
    Returns list of (px, py, pz, layer_index).
    """
    positions = []
    positions.append((0, 0, H, 0))  # Center COB
    for i in range(1, N):
        for x in range(-i, i + 1):
            y_abs = i - abs(x)
            if y_abs == 0:
                positions.append((x, 0, H, i))
            else:
                positions.append((x, y_abs, H, i))
                positions.append((x, -y_abs, H, i))

    # Rotate by 45 degrees, then scale and center
    theta = np.radians(45)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    centerX, centerY = W / 2, L / 2

    # If only 1 layer, no scaling needed
    scale = (W / 2 * np.sqrt(2)) / (N - 1) if N > 1 else 0

    transformed = []
    for (xx, yy, hh, layer) in positions:
        rx = xx * cos_t - yy * sin_t
        ry = xx * sin_t + yy * cos_t
        px = centerX + rx * scale
        py = centerY + ry * scale
        transformed.append((px, py, H, layer))
    return transformed

###############################################################################
# Polynomial-based "closed-form" approach
#   g(x) = sum_{k=0 to 6} p[k] * x^k
# We define sets of polynomial coefficients p for N=10, 15, 20, etc.
# Then we'll interpolate or pick the nearest.
###############################################################################

# Example polynomial coefficients for N=10, N=15, N=20, each a 7-element array
# representing a 6th-degree polynomial p(x) = p[0] + p[1]*x + ... + p[6]*x^6.
# Below are placeholders. You would typically do a polynomial fit for each N
# that yields good uniformity in your test environment.
POLY_COEFFS = {
    10: [1.0, 0.8, 3.2, 2.6, -1.0, 0.5, 0.1],    # dummy example
    15: [1.0, 1.25, 2.25, 4.25, 4.0, 3.0, 1.0],  # from your original "gN15" but extended
    20: [1.0, 1.8, 2.1, 5.0, 3.5, 2.0, 2.0],     # dummy example
}

def interpolate_poly_coeffs(N):
    """
    If N exactly in POLY_COEFFS, return that.
    If 10 < N < 15, or 15 < N < 20, linearly interpolate the polynomial
    coefficients. If outside [10,20], pick the nearest known set.
    """
    known_N = sorted(POLY_COEFFS.keys())  # e.g. [10,15,20]
    if N in POLY_COEFFS:
        return POLY_COEFFS[N]

    minN, midN, maxN = known_N[0], known_N[1], known_N[2]  # 10, 15, 20
    if N < minN:
        # Just pick the lowest
        return POLY_COEFFS[minN]
    if N > maxN:
        # Pick the highest
        return POLY_COEFFS[maxN]

    # Otherwise N is between 10 and 20, but not exactly 10,15,or 20
    if minN < N < midN:
        # Interpolate between N=10 and N=15
        alpha = (N - minN) / (midN - minN)
        c0 = POLY_COEFFS[minN]
        c1 = POLY_COEFFS[midN]
    elif midN < N < maxN:
        # Interpolate between N=15 and N=20
        alpha = (N - midN) / (maxN - midN)
        c0 = POLY_COEFFS[midN]
        c1 = POLY_COEFFS[maxN]
    else:
        # If N==15 just above we caught that, so logically this is unreachable,
        # but let's be safe:
        return POLY_COEFFS[midN]

    # Linear interpolation of each coefficient
    cinterp = [ (1-alpha)*c0[i] + alpha*c1[i] for i in range(len(c0)) ]
    return cinterp

def g_of_x(x, coeffs):
    """Evaluate the polynomial in `coeffs` at x."""
    return sum( coeffs[k]*(x**k) for k in range(len(coeffs)) )

def compute_flux_assignments(N, W, L, target_PPFD):
    """
    Computes per-layer luminous flux (lumens per COB) using:
      1) A polynomial "shape" that depends on N.
      2) A global scale factor to meet total flux -> target_PPFD over floor area.

    Returns a list of length N: flux[i] = lumens/COB for layer i.
    """
    # 1) Pick or interpolate polynomial coefficients for the given N
    p = interpolate_poly_coeffs(N)

    # 2) We'll produce shape[i] = g_of_x(i/(N-1)) for i=0..N-1
    #    Then the final flux per COB in layer i is flux[i] = scale_factor * shape[i]
    shapes = []
    for i in range(N):
        if N > 1:
            x = i / (N - 1)
        else:
            x = 0
        shapes.append( g_of_x(x, p) )

    # 3) Total flux needed to achieve target_PPFD
    #    same logic as your original polynomial approach
    c = 63000  # lumens/m2 for PPFD=1250
    area = W * L
    total_flux_target = c * area * (target_PPFD / 1250)

    # 4) Sum of shape * #COBs
    #    #COBs in layer i is 1 if i=0 else 4*i
    sum_g = 0.0
    for i in range(N):
        num_cobs = 1 if i == 0 else 4*i
        sum_g += num_cobs * shapes[i]

    # 5) Solve for scale factor
    if sum_g < 1e-9:
        scale_factor = 0.0
    else:
        scale_factor = total_flux_target / sum_g

    # 6) Final flux, capped at 24k if desired
    fluxes = []
    for i in range(N):
        f_i = scale_factor * shapes[i]
        f_i = min(f_i, 24000.0)
        fluxes.append(f_i)

    return fluxes

def main():
    # Example usage:
    N = 22              # number of layers
    W = 13.4112           # floor width in meters
    L = 13.4112           # floor length in meters
    H = 0.9144          # height in meters
    target_PPFD = 1250  # desired PPFD

    # 1) Get flux distribution
    flux_values = compute_flux_assignments(N, W, L, target_PPFD)

    # 2) Print results
    print("Closed-Form Polynomial Approach\n")
    print("Layer-by-Layer Luminous Flux (lm/COB):")
    for i, flux in enumerate(flux_values):
        print(f"  Layer {i}: {flux:0.2f} lumens/COB")

    # 3) Check total flux and approximate average PPFD
    nu = [1 if i==0 else 4*i for i in range(N)]
    total_flux = sum(nu[i]*flux_values[i] for i in range(N))
    # same check as your original
    calc_ppfd = (total_flux / (63000.0 * (W * L))) * 1250.0

    print(f"\nCalculated Total Flux: {total_flux:0.2f} lumens")
    print(f"Approx. Average PPFD: {calc_ppfd:0.2f} µmol/m²/s")

    # 4) COB positions
    cob_positions = build_cob_positions(W, L, H, N)
    print(f"Number of COBs: {len(cob_positions)}")

    print("\nSample of COB Positions & Assigned Flux:")
    for i, (px, py, pz, layer) in enumerate(cob_positions[:10]):  # just show a few
        flux = flux_values[layer]
        print(f"  COB#{i:2d} at (x={px:.2f}, y={py:.2f}, z={pz:.2f}), Layer={layer}, Flux={flux:.2f}")


if __name__ == "__main__":
    main()
