# Improved spline-based interpolation with edge compensation for high-uniformity datasets

import numpy as np
import csv
from scipy.interpolate import make_interp_spline
from backups.lighting_simulation_data22 import prepare_geometry, simulate_lighting

# --- Configurable Parameters ---
HEIGHT_M = 0.9144
TARGET_PPFD = 1250.0
MIN_LAYERS = 10
MAX_LAYERS = 20
OUTPUT_CSV = "layer_flux_dataset.csv"
REFERENCE_LAYER = 10
REFERENCE_FLOOR_M = 6.096
EDGE_BOOST_FACTOR = 1.25  # Boost for newly added outer layers

# --- High-uniformity reference datasets ---
REFERENCE_FLUX = {
    10: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 20000, 20000],
    11: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000 ,20000, 20000],
    12: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 8000, 20000, 20000],
    13: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 8000, 6750, 20000, 20000],
    14: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 9000, 10000, 9000, 20000, 20000],
    15: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 9000, 10000, 9000, 9500, 20000, 20000],
    16: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 9000, 10000, 9000, 8500, 11000, 20000, 20000],
    17: [1500, 2000, 3500, 8000, 14000, 8000, 8000, 13500, 11000, 9000, 10000, 9000, 8500, 9000, 10000, 20000, 20000],
    18: [1500, 2000, 3500, 8000, 14000, 8000, 8000, 13500, 11000, 12000, 10000, 9000, 8500, 9000, 10000, 11000, 20000, 20000],
}

# --- Centered square number formula ---
def cob_count(n):
    return 2 * n * (n + 1) + 1

def get_reference_density():
    ref_cobs = cob_count(REFERENCE_LAYER - 1)
    ref_area = REFERENCE_FLOOR_M ** 2
    return ref_cobs / ref_area

def floor_size_from_layers(n_layers, density):
    cobs = cob_count(n_layers - 1)
    area = cobs / density
    return np.sqrt(area)

# --- Spline-based flux generation from multiple profiles ---
def generate_flux_via_spline(n_layers):
    max_len = max(len(v) for v in REFERENCE_FLUX.values())
    x_data = list(REFERENCE_FLUX.keys())
    y_matrix = []
    for i in range(max_len):
        row = [REFERENCE_FLUX[k][i] if i < len(REFERENCE_FLUX[k]) else REFERENCE_FLUX[k][-1] for k in x_data]
        y_matrix.append(row)

    splines = [make_interp_spline(x_data, y_vals, k=3, bc_type='natural') for y_vals in y_matrix]
    flux = [float(s(n_layers)) for s in splines]

    if len(flux) < n_layers:
        last_val = flux[-1]
        flux.extend([last_val] * (n_layers - len(flux)))
    else:
        flux = flux[:n_layers]  # Trim to layer count

    # Boost outermost perimeter layers
    if n_layers > max(REFERENCE_FLUX.keys()):
        flux[0] *= EDGE_BOOST_FACTOR
        flux[-1] = min(flux[-1] * EDGE_BOOST_FACTOR, 20000)

    flux[-1] = min(flux[-1], 20000)  # Final clamp
    return flux

# --- Evaluation Function ---
def evaluate_dou(floor_ppfd):
    mean_ppfd = np.mean(floor_ppfd)
    rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd) ** 2))
    dou = 100 * (1 - rmse / mean_ppfd)
    return dou, mean_ppfd

# --- Dataset Generation ---
def generate_dataset():
    records = []
    density = get_reference_density()

    for n_layers in range(MIN_LAYERS, MAX_LAYERS + 1):
        flux = generate_flux_via_spline(n_layers)

        floor_size = floor_size_from_layers(n_layers, density)
        geo = prepare_geometry(floor_size, floor_size, HEIGHT_M, n_layers)
        floor_ppfd, _, _, _ = simulate_lighting(np.array(flux, dtype=np.float64), geo)
        dou, mean_ppfd = evaluate_dou(floor_ppfd)

        print(f"Layers: {n_layers} | Floor: {floor_size:.2f} m | PPFD: {mean_ppfd:.1f} | DOU: {dou:.2f}%")

        record = {
            "layers": n_layers,
            "floor_size_m": floor_size,
            "height_m": HEIGHT_M,
            "target_ppfd": TARGET_PPFD,
            "mean_ppfd": mean_ppfd,
            "dou": dou,
        }
        for i in range(MAX_LAYERS):
            record[f"layer_{i}" ] = flux[i] if i < len(flux) else ""

        records.append(record)

    # Write to CSV
    keys = ["layers", "floor_size_m", "height_m", "target_ppfd", "mean_ppfd", "dou"]
    keys += [f"layer_{i}" for i in range(MAX_LAYERS)]

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    print(f"Dataset written to {OUTPUT_CSV}")

# --- Execute ---
if __name__ == "__main__":
    generate_dataset()