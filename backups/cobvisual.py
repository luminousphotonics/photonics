import numpy as np
import matplotlib.pyplot as plt

def build_cob_positions(W, L, H, num_layers):
    """
    Builds COB positions (centered square pattern - FINALLY CORRECT).
    """
    light_positions = []
    center_x = W / 2.0
    center_y = L / 2.0

    if num_layers > 0:
        spacing_x = W / (2 * num_layers)
        spacing_y = L / (2 * num_layers)
    else:
        spacing_x = 0
        spacing_y = 0

    for layer in range(num_layers + 1):
        if layer == 0:
            light_positions.append((center_x, center_y, H))
            continue

        # --- CORRECTED SIDE GENERATION (FINALLY!) ---
        # Top side:
        for x in range(-layer, layer + 1):
            light_positions.append((center_x + x * spacing_x, center_y + layer * spacing_y, H))
        # Bottom side:
        for x in range(-layer, layer + 1):
            light_positions.append((center_x + x * spacing_x, center_y - layer * spacing_y, H))
        # Left side (CORRECTED INDICES):
        for y in range(-layer + 1, layer):  #  <--  +1 and no +1
            light_positions.append((center_x - layer * spacing_x, center_y + y * spacing_y, H))
        # Right side (CORRECTED INDICES):
        for y in range(-layer + 1, layer):  # <-- +1 and no +1
            light_positions.append((center_x + layer * spacing_x, center_y + y * spacing_y, H))

    return np.array(light_positions)

def generate_unscaled_positions(num_layers):
    """Generates unscaled integer coordinates (for verification)."""
    unscaled_positions = []
    for layer in range(num_layers + 1):
        layer_coords = []
        if layer == 0:
            layer_coords.append((0, 0))
            unscaled_positions.append(layer_coords)
            continue
        #Corrected Indices
        for x in range(-layer, layer + 1):
            layer_coords.append((x, layer))
        for x in range(-layer, layer + 1):
            layer_coords.append((x, -layer))
        for y in range(-layer + 1, layer):
            layer_coords.append((-layer, y))
        for y in range(-layer + 1, layer):
            layer_coords.append((layer, y))
        unscaled_positions.append(layer_coords)

    return unscaled_positions

def pack_luminous_flux(params, cob_positions):
    """Packs luminous flux parameters."""
    MIN_LUMENS = 100
    num_cobs = len(cob_positions)
    led_intensities = np.full(num_cobs, MIN_LUMENS)
    for i in range(min(len(params), num_cobs)):
        led_intensities[i] = params[i]
    return led_intensities

# --- Example Usage and Verification ---

W = 10
L = 10
H = 2
num_layers = 3

cob_positions = build_cob_positions(W, L, H, num_layers)
params = [200, 300, 400, 250, 350, 450, 280, 380, 150, 500, 600, 700, 800, 100, 200, 300, 400, 500, 123, 456]
led_intensities = pack_luminous_flux(params, cob_positions)

# --- Verification (Unscaled Coordinates) ---
reference_coords = generate_unscaled_positions(num_layers)
unscaled_calculated_coords = generate_unscaled_positions(num_layers)

reference_set = set()
for layer in reference_coords:
    reference_set.update(set(layer))

calculated_set = set()
for layer in unscaled_calculated_coords:
   calculated_set.update(set(layer))

if reference_set == calculated_set:
    print("\n--- Verification PASSED: Calculated coordinates match reference coordinates. ---")
else:
    print("\n--- Verification FAILED: Coordinate mismatch. ---")
    print("  Elements in reference but not in calculated:", reference_set - calculated_set)
    print("  Elements in calculated but not in reference:", calculated_set - reference_set)

# --- Plotting ---
plt.figure(figsize=(6, 6))
plt.scatter(cob_positions[:, 0], cob_positions[:, 1], s=50)
plt.title("COB Positions (Centered Square Pattern)")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(0, W)
plt.ylim(0, L)
plt.grid(True)
plt.show()

# --- Regular Grid (for comparison) ---
def build_regular_grid(W, L, H, num_points_x, num_points_y):
    x_coords = np.linspace(0, W, num_points_x)
    y_coords = np.linspace(0, L, num_points_y)
    xv, yv = np.meshgrid(x_coords, y_coords)
    positions = np.stack([xv.flatten(), yv.flatten(), np.full(xv.size, H)], axis=-1)
    return positions

num_points_x = 5
num_points_y = 5
cob_positions_grid = build_regular_grid(W, L, H, num_points_x, num_points_y)
led_intensities_grid = pack_luminous_flux(params, cob_positions_grid)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(cob_positions[:, 0], cob_positions[:, 1], label="Centered Square", s=50)
for i, (x, y, _) in enumerate(cob_positions):
    plt.text(x + 0.1, y + 0.1, f"{led_intensities[i]:.0f}", fontsize=8)
plt.title("Correct Centered Square")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(0, W)
plt.ylim(0, L)
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(cob_positions_grid[:, 0], cob_positions_grid[:, 1], label="Regular Grid", s=50, marker="s")
for i, (x, y, _) in enumerate(cob_positions_grid):
    plt.text(x + 0.1, y + 0.1, f"{led_intensities_grid[i]:.0f}", fontsize=8)
plt.title("Regular Grid (for comparison)")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(0, W)
plt.ylim(0, L)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()