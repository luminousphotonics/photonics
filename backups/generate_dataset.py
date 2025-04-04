import numpy as np
import csv
from backups.lighting_simulation_data22 import prepare_geometry, simulate_lighting

# --- Configurable Parameters ---
HEIGHT_M = 0.9144
TARGET_PPFD = 1250.0
TARGET_LAYERS = 11
OUTPUT_CSV = "tuned_11_layer_dataset.csv"

# --- 10-layer high-DOU baseline ---
BASE_FLUX = [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 20000, 20000]

# --- Calculate COB count using centered square numbers ---
def cob_count(n):
    return 2 * n * (n + 1) + 1

# --- Calculate floor size from COB count and density ---
def floor_size_from_layers(n_layers, density):
    cobs = cob_count(n_layers - 1)
    area = cobs / density
    return np.sqrt(area)

# --- Evaluate uniformity ---
def evaluate_dou(floor_ppfd):
    mean_ppfd = np.mean(floor_ppfd)
    rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd) ** 2))
    dou = 100 * (1 - rmse / mean_ppfd)
    return dou, mean_ppfd

# --- Get reference density from 10-layer config ---
def get_reference_density():
    cobs = cob_count(10 - 1)
    area = 6.096 ** 2
    return cobs / area

# --- Brute-force flux tuning ---
def tune_11th_layer():
    density = get_reference_density()
    flux_base = BASE_FLUX[:]
    best_result = None

    sweep_range = np.arange(5000, 20001, 500)

    for inserted_flux in sweep_range:
        flux = flux_base[:-1] + [inserted_flux] + [20000]
        floor_size = floor_size_from_layers(TARGET_LAYERS, density)
        geo = prepare_geometry(floor_size, floor_size, HEIGHT_M, TARGET_LAYERS)
        floor_ppfd, _, _, _ = simulate_lighting(np.array(flux, dtype=np.float64), geo)
        dou, mean_ppfd = evaluate_dou(floor_ppfd)

        result = {
            "flux": flux,
            "mean_ppfd": mean_ppfd,
            "dou": dou,
            "inserted": inserted_flux,
            "floor_size": floor_size
        }

        if best_result is None or (dou > best_result["dou"] and abs(mean_ppfd - TARGET_PPFD) < 50):
            best_result = result

    # Save result to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["layer_{}".format(i) for i in range(TARGET_LAYERS)] + ["mean_ppfd", "dou", "inserted_flux", "floor_size_m"])
        writer.writerow(best_result["flux"] + [best_result["mean_ppfd"], best_result["dou"], best_result["inserted"], best_result["floor_size"]])

    print(f"Best inserted flux: {best_result['inserted']} | PPFD: {best_result['mean_ppfd']:.2f} | DOU: {best_result['dou']:.2f}%")
    print(f"Dataset written to {OUTPUT_CSV}")

# --- Execute ---
if __name__ == "__main__":
    tune_11th_layer()