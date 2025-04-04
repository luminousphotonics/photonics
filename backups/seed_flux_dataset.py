import numpy as np

MAX_LAYERS = 20
HEIGHT_M = 0.9144  # 3 ft
PPFD = 1250

# Your known-good flux vectors
known_flux_data = {
    10: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 20000, 20000],
    11: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 20000, 20000],
    12: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 8000, 20000, 20000],
    13: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 8000, 10000, 20000, 20000],
    14: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 9000, 10000, 9000, 20000, 20000],
    15: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 9000, 10000, 9000, 9500, 20000, 20000],
    16: [1500, 2000, 3500, 8000, 14000, 8000, 5500, 13500, 11000, 9000, 10000, 9000, 8500, 11000, 20000, 20000],
    17: [1500, 2000, 3500, 8000, 14000, 8000, 8000, 13500, 11000, 9000, 10000, 9000, 8500, 9000, 10000, 20000, 20000],
    18: [1500, 2000, 3500, 8000, 14000, 8000, 8000, 13500, 11000, 12000, 10000, 9000, 8500, 9000, 10000, 11000, 20000, 20000]
}

X = []
Y = []

for layer_count, flux_vector in known_flux_data.items():
    flux = np.array(flux_vector[:layer_count])
    flux_normalized = flux / np.sum(flux)
    padded_flux = np.pad(flux_normalized, (0, MAX_LAYERS - layer_count))

    # Normalize input features
    input_vector = [layer_count / 20, HEIGHT_M / 3.048, PPFD / 1500]

    X.append(input_vector)
    Y.append(padded_flux)

# Save dataset to .npz
np.savez("flux_learning_dataset.npz", X=np.array(X), Y=np.array(Y))
print(f"✅ Seeded dataset with {len(X)} flux profiles.")
