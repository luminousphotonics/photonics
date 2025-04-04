import numpy as np
from backups.lighting_simulation_data22 import prepare_geometry, simulate_lighting

# --- Configurable Parameters ---
HEIGHT_M = 0.9144
TARGET_PPFD = 1250.0
MAX_LUMENS = 20000.0

# --- COB flux profile to scale ---
INPUT_FLUX = [1676.6, 2193.79, 3745.36, 8400.08, 14606.37, 8400.08, 5814.12, 14089.18, 11503.22, 8400.08, 7107.1, 5814.12, 20000.0, 20000.0]  # Replace with any starting profile

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

# --- Scale flux profile to target PPFD with 20,000 lm cap ---
def scale_flux_to_target(flux, target_ppfd):
    density = get_reference_density()
    n_layers = len(flux)
    floor_size = floor_size_from_layers(n_layers, density)
    geo = prepare_geometry(floor_size, floor_size, HEIGHT_M, n_layers)

    floor_ppfd, _, _, _ = simulate_lighting(np.array(flux, dtype=np.float64), geo)
    _, current_ppfd = evaluate_dou(floor_ppfd)

    scale_factor = target_ppfd / current_ppfd
    scaled_flux = np.array(flux) * scale_factor

    # Clamp and redistribute
    clamped_flux = np.minimum(scaled_flux, MAX_LUMENS)
    excess = np.sum(scaled_flux - clamped_flux)

    unclamped_indices = [i for i, val in enumerate(clamped_flux) if val < MAX_LUMENS]
    if unclamped_indices:
        redistribute = excess / len(unclamped_indices)
        for i in unclamped_indices:
            clamped_flux[i] += redistribute
            clamped_flux[i] = min(clamped_flux[i], MAX_LUMENS)

    clamped_flux = [round(val, 2) for val in clamped_flux.tolist()]

    floor_ppfd_scaled, _, _, _ = simulate_lighting(np.array(clamped_flux, dtype=np.float64), geo)
    dou_scaled, mean_ppfd_scaled = evaluate_dou(floor_ppfd_scaled)

    print(f"Scaled Flux: {', '.join(str(val) for val in clamped_flux)}")
    print(f"PPFD: {mean_ppfd_scaled:.2f} | DOU: {dou_scaled:.2f}% | Scale factor: {scale_factor:.4f}")

# --- Execute ---
if __name__ == "__main__":
    scale_flux_to_target(INPUT_FLUX, TARGET_PPFD)
