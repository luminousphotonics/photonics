import numpy as np

def build_cob_positions(W, L, H, N):
    """
    Builds COB positions for N layers over a W x L floor at height H.
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
    theta = np.radians(45)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    centerX, centerY = W / 2, L / 2
    scale = (W / 2 * np.sqrt(2)) / (N - 1) if N > 1 else 0
    transformed = []
    for (xx, yy, hh, layer) in positions:
        rx = xx * cos_t - yy * sin_t
        ry = xx * sin_t + yy * cos_t
        px = centerX + rx * scale
        py = centerY + ry * scale
        transformed.append((px, py, H, layer))
    return transformed

def g(x, p):
    """
    Normalized flux distribution function.
    """
    return np.polyval(p, x)

def compute_flux_assignments(N, W, target_PPFD):
    """
    Computes layer-wise luminous flux assignments.
    """
    # Fit polynomial to N=15 normalized flux data
    x_N15 = np.arange(0, 15) / 14
    g_N15 = np.array([1, 1.25, 2.250345, 4.25, 4.00003, 4.0, 3.5, 6.5, 8.5, 5.0, 2.0, 3.0, 4.0, 12.0, 12.0])
    p = np.polyfit(x_N15, g_N15, 6)

    # Total flux target
    c = 63000  # lumens/m² for PPFD=1250
    area = W * W
    total_flux_target = c * area * (target_PPFD / 1250)

    # Compute scaling factor f0
    sum_g = 0
    for i in range(N):
        num_cobs = 1 if i == 0 else 4 * i
        x = i / (N - 1) if N > 1 else 0
        sum_g += num_cobs * g(x, p)
    f0 = total_flux_target / sum_g

    # Assign fluxes
    params = []
    for i in range(N):
        x = i / (N - 1) if N > 1 else 0
        f_i = f0 * g(x, p)
        # Cap flux at 24,000 lumens to match observed maximum
        f_i = min(f_i, 24000)
        params.append(f_i)
    return params

def main():
    N = 20              # Number of layers
    W = 12.192         # Floor width in meters
    L = W               # Floor length in meters
    H = 0.9144          # Height in meters
    target_PPFD = 1250  # Target PPFD in µmol/m²/s

    cob_positions = build_cob_positions(W, L, H, N)
    print(f"Total number of COBs: {len(cob_positions)}")

    params = compute_flux_assignments(N, W, target_PPFD)
    print("\nLayer-wise luminous flux assignments (lumens per COB):")
    for i, flux in enumerate(params):
        num_cobs = 1 if i == 0 else 4 * i
        #print(f"Layer {i}: {flux:.2f} lm/COB, {num_cobs} COBs, Total {flux * num_cobs:.2f} lm")
        print(f"Layer {i}: {flux:.2f} lm/COB")

if __name__ == "__main__":
    main()