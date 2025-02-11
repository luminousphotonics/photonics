import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import ttk
import json
import os

# Number of LEDs per layer
LAYER_COUNTS = [1, 4, 8, 12, 16, 20]
SETTINGS_FILE = "settings.json"

def run_simulation():
    # --------------------------------------------------
    # Read GUI Parameters
    # --------------------------------------------------
    try:
        layer_intensities = [float(entry.get().strip()) for entry in layer_intensity_entries]
    except Exception as e:
        print("Error reading layer intensities:", e)
        return

    led_intensities = []
    for intensity, count in zip(layer_intensities, LAYER_COUNTS):
        led_intensities.extend([intensity] * count)

    try:
        corner_values = [float(entry.get().strip()) for entry in corner_intensity_entries]
    except Exception as e:
        print("Error reading corner intensities:", e)
        return
    # In layer 6 starting at index 41, override the four corner COBs.
    corner_indices = [41 + 16, 41 + 17, 41 + 18, 41 + 19]
    for idx, cob_idx in enumerate(corner_indices):
        if cob_idx < len(led_intensities):
            led_intensities[cob_idx] = corner_values[idx]

    try:
        led_strip_fluxes = [float(entry.get().strip()) for entry in led_strip_entries]
    except Exception as e:
        print("Error reading LED strip luminous flux values:", e)
        return

    # --------------------------------------------------
    # Luminous-to-Radiant Conversion Parameters
    # --------------------------------------------------
    luminous_efficacy = 182.0  # lm/W

    try:
        spd = np.loadtxt('/Users/austinrouse/photonics/backups/spd_data.csv', delimiter=' ', skiprows=1)
    except Exception as e:
        print("Error loading SPD data:", e)
        return
    wavelengths = spd[:, 0]  # in nm
    intensities = spd[:, 1]

    mask_par = (wavelengths >= 400) & (wavelengths <= 700)
    integral_total = np.trapz(intensities, wavelengths)
    integral_par = np.trapz(intensities[mask_par], wavelengths[mask_par])
    PAR_fraction = integral_par / integral_total

    wavelengths_m = wavelengths * 1e-9
    lambda_eff = np.trapz(wavelengths_m[mask_par] * intensities[mask_par],
                          wavelengths_m[mask_par]) / np.trapz(intensities[mask_par], wavelengths_m[mask_par])
    h = 6.626e-34  # J s
    c = 3.0e8      # m/s
    E_photon = h * c / lambda_eff
    N_A = 6.022e23
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction

    print(f"Effective wavelength = {lambda_eff*1e9:.1f} nm")
    print(f"PAR fraction = {PAR_fraction:.3f}")
    print(f"Conversion factor = {conversion_factor:.3f} µmol/J")

    # --------------------------------------------------
    # Room and Simulation Parameters (in meters)
    # --------------------------------------------------
    W_m, L_m, H_m = 12.0, 12.0, 3.0
    meters_to_feet = 3.28084
    W = W_m / meters_to_feet
    L = L_m / meters_to_feet
    H = H_m / meters_to_feet

    # --------------------------------------------------
    # Build the Ceiling Lighting Arrangement (Diamond: 61)
    # --------------------------------------------------
    light_sources = []
    light_luminous_fluxes = []

    layers = [
        [(0, 0)],  # Layer 1 (center)
        [(-1, 0), (1, 0), (0, -1), (0, 1)],  # Layer 2
        [(-1, -1), (1, -1), (-1, 1), (1, 1),
         (-2, 0), (2, 0), (0, -2), (0, 2)],  # Layer 3
        [(-2, -1), (2, -1), (-2, 1), (2, 1),
         (-1, -2), (1, -2), (-1, 2), (1, 2),
         (-3, 0), (3, 0), (0, -3), (0, 3)],  # Layer 4
        [(-2, -2), (2, -2), (-2, 2), (2, 2),
         (-3, -1), (3, -1), (-3, 1), (3, 1),
         (-1, -3), (1, -3), (-1, 3), (1, 3),
         (-4, 0), (4, 0), (0, -4), (0, 4)],  # Layer 5
        [(-3, -2), (3, -2), (-3, 2), (3, 2),
         (-2, -3), (2, -3), (-2, 3), (2, 3),
         (-4, -1), (4, -1), (-4, 1), (4, 1),
         (-1, -4), (1, -4), (-1, 4), (1, 4),
         (-5, 0), (5, 0), (0, -5), (0, 5)]  # Layer 6
    ]
    spacing_x = W / 7.2
    spacing_y = L / 7.2

    center_source = (W/2, L/2, H)
    light_sources.append(center_source)
    light_luminous_fluxes.append(0.0)  # Placeholder

    for layer_index, layer in enumerate(layers):
        if layer_index == 0:
            continue
        for dot in layer:
            x_offset = spacing_x * dot[0]
            y_offset = spacing_y * dot[1]
            theta = math.radians(45)
            rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
            rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
            pos = (center_source[0] + rotated_x,
                   center_source[1] + rotated_y,
                   H)
            light_sources.append(pos)
            light_luminous_fluxes.append(0.0)

    N_cob_leds = len(light_sources)
    print(f"Total number of COB LEDs: {N_cob_leds}")

    if len(led_intensities) != N_cob_leds:
        print(f"Error: Expected {N_cob_leds} intensities, got {len(led_intensities)}")
        return
    for i in range(N_cob_leds):
        light_luminous_fluxes[i] = led_intensities[i]

    view_angles = [0.0] * N_cob_leds

    # --------------------------------------------------
    # Add LED Strips to the Simulation
    # --------------------------------------------------
    led_strips = {
        1: [48, 56, 61, 57],
        2: [49, 45, 53],
        3: [59, 51, 43],
        4: [47, 55, 60],
        5: [54, 46, 42],
        6: [50, 58, 52, 44],
        7: [28, 36, 41, 37],
        8: [29, 33, 39, 31],
        9: [27, 35, 40, 34],
        10: [26, 30, 38, 32],
        11: [25, 21, 17],
        12: [23, 15, 19],
        13: [24, 18, 14],
        14: [22, 16, 20],
        15: [8, 13, 9, 11],
        16: [7, 12, 6, 10],
        17: [2, 4],
        18: [4, 3],
        19: [3, 5],
        20: [5, 2]
    }

    num_points_per_segment = 5
    for strip_num, led_ids in led_strips.items():
        points = []
        for i in range(len(led_ids) - 1):
            start_idx = led_ids[i] - 1
            end_idx   = led_ids[i+1] - 1
            start_pos = light_sources[start_idx]
            end_pos   = light_sources[end_idx]
            for t in np.linspace(0, 1, num_points_per_segment, endpoint=False):
                pos = (start_pos[0] + t * (end_pos[0] - start_pos[0]),
                       start_pos[1] + t * (end_pos[1] - start_pos[1]),
                       start_pos[2] + t * (end_pos[2] - start_pos[2]))
                points.append(pos)
        points.append(light_sources[led_ids[-1] - 1])
        flux_this_strip = led_strip_fluxes[strip_num - 1]
        num_points = len(points)
        flux_per_point = flux_this_strip / num_points if num_points > 0 else 0.0

        for pos in points:
            light_sources.append(pos)
            light_luminous_fluxes.append(flux_per_point)
            view_angles.append(0.0)
        print(f"Added LED Strip #{strip_num} with {num_points} points.")

    # --------------------------------------------------
    # Compute PPFD on Floor (Grid-based)
    # --------------------------------------------------
    grid_res = 0.01  # meters
    x_coords = np.arange(0, W, grid_res)
    y_coords = np.arange(0, L, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    direct_irradiance = np.zeros_like(X, dtype=np.float64)

    for i, (src, lum) in enumerate(zip(light_sources, light_luminous_fluxes)):
        I = lum / luminous_efficacy  # Radiant flux (W)
        x0, y0, _ = src
        r = np.sqrt((X - x0)**2 + (Y - y0)**2 + H**2)
        r_horizontal = np.sqrt((X - x0)**2 + (Y - y0)**2)
        theta_pt = np.arctan2(r_horizontal, H)
        theta_led = math.radians(view_angles[i])
        if theta_led <= 0:
            contribution = (I / math.pi) * (H**2) / (r**4)
        else:
            sigma = theta_led / 2.0
            gaussian_weight = np.exp(-0.5 * (theta_pt / sigma)**2)
            contribution = (I / (2 * math.pi * sigma**2)) * gaussian_weight * (H**2) / (r**4)
        direct_irradiance += contribution

    rho_wall = 0.9
    rho_ceiling = 0.1
    rho_floor = 0.0
    area_wall = 2 * (W * H + L * H)
    area_ceiling = W * L
    area_floor = W * L
    total_area = area_wall + area_ceiling + area_floor
    R_eff = (rho_wall * area_wall + rho_ceiling * area_ceiling + rho_floor * area_floor) / total_area

    irradiance_direct = direct_irradiance
    irradiance_refl1 = direct_irradiance * rho_wall * (area_wall / total_area)
    irradiance_refl2 = irradiance_refl1 * R_eff
    total_irradiance = irradiance_direct + irradiance_refl1 + irradiance_refl2
    total_PPFD = total_irradiance * conversion_factor

    avg_PPFD = np.mean(total_PPFD)
    rmse = np.sqrt(np.mean((total_PPFD - avg_PPFD)**2))
    dou = 100.0 * (1.0 - (rmse / avg_PPFD))
    print(f"Average PPFD = {avg_PPFD:.1f} µmol/m²/s")
    print(f"RMSE = {rmse:.1f} µmol/m²/s")
    print(f"Degree of Uniformity = {dou:.1f} %")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(total_PPFD, origin='lower', extent=[0, W, 0, L], cmap='viridis')
    plt.colorbar(im, label='PPFD (µmol m⁻² s⁻¹)')
    plt.xlabel('Room Width (m)')
    plt.ylabel('Room Length (m)')
    plt.title('Simulated PPFD on Floor\n(SPD Conversion, Lensed Emission)')
    stats_text = f"Avg PPFD: {avg_PPFD:.1f}\nRMSE: {rmse:.1f}\nDOU: {dou:.1f} %"
    plt.text(0.05 * W, 0.95 * L, stats_text, color='white', ha='left', va='top',
             fontsize=8, bbox=dict(facecolor='black', alpha=0.7, pad=5))

    num_measure = 10
    x_measure = np.linspace(0, W, num_measure)
    y_measure = np.linspace(0, L, num_measure)
    for xm in x_measure:
        for ym in y_measure:
            ix = min(int(xm / grid_res), total_PPFD.shape[1]-1)
            iy = min(int(ym / grid_res), total_PPFD.shape[0]-1)
            ppfd_val = total_PPFD[iy, ix]
            plt.text(xm, ym, f'{ppfd_val:.1f}', color='white', ha='center', va='center',
                     fontsize=6, bbox=dict(facecolor='black', alpha=0.5, pad=1))

    for i, src in enumerate(light_sources[:N_cob_leds]):
        plt.plot(src[0], src[1], marker='x', color='red', markersize=8, markeredgewidth=2)
        plt.text(src[0] + 0.03, src[1] + 0.03, f"{i+1}", color='blue', fontsize=12, fontweight='bold')
    
    strip_midpoints = {}
    for strip_num, led_ids in led_strips.items():
        ref_positions = [light_sources[id - 1] for id in led_ids if (id - 1) < N_cob_leds]
        if ref_positions:
            x_line = [pos[0] for pos in ref_positions]
            y_line = [pos[1] for pos in ref_positions]
            plt.plot(x_line, y_line, 'm--', linewidth=2)
            mid_idx = len(ref_positions) // 2
            if mid_idx < len(ref_positions):
                strip_midpoints[strip_num] = ref_positions[mid_idx]

    #for strip_num, midpoint in strip_midpoints.items():
        #plt.text(midpoint[0], midpoint[1], f"S{strip_num}", color='cyan', fontsize=10, fontweight='bold',
                 #ha='center', va='center')

    plt.show()

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print("Error loading settings:", e)
    return {}

def save_settings():
    settings = {
        "layer_intensities": [entry.get() for entry in layer_intensity_entries],
        "corner_intensities": [entry.get() for entry in corner_intensity_entries],
        "led_strip_fluxes": [entry.get() for entry in led_strip_entries]
    }
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        print("Error saving settings:", e)

def on_closing():
    save_settings()
    root.destroy()

# ========================
# Create Tkinter GUI for Simulation
# ========================
root = tk.Tk()
root.title("PPFD Simulation Parameters")
root.protocol("WM_DELETE_WINDOW", on_closing)

# --- Frame: Layer Settings ---
layer_frame = ttk.LabelFrame(root, text="Layer Settings")
layer_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

layer_intensity_frame = ttk.LabelFrame(layer_frame, text="COB Intensities (lm)")
layer_intensity_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
layer_intensity_entries = []
default_layer_intensities = [6000.0, 8000.0, 12000.0, 8000.0, 10000.0, 18000.0]
for i in range(len(LAYER_COUNTS)):
    label = ttk.Label(layer_intensity_frame, text=f"Layer {i+1}:")
    label.grid(row=i, column=0, padx=2, pady=2, sticky="e")
    entry = ttk.Entry(layer_intensity_frame, width=7)
    entry.insert(0, str(default_layer_intensities[i]))
    entry.grid(row=i, column=1, padx=2, pady=2, sticky="w")
    layer_intensity_entries.append(entry)

# --- Frame: Corner COB Settings ---
corner_frame = ttk.LabelFrame(root, text="Corner COB Intensities (lm)")
corner_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
corner_intensity_entries = []
corner_labels = ["Left:", "Right:", "Bottom:", "Top:"]
default_corner_intensities = [18000.0, 18000.0, 18000.0, 18000.0]
for i in range(4):
    label = ttk.Label(corner_frame, text=corner_labels[i])
    label.grid(row=i, column=0, padx=2, pady=2, sticky="e")
    entry = ttk.Entry(corner_frame, width=7)
    entry.insert(0, str(default_corner_intensities[i]))
    entry.grid(row=i, column=1, padx=2, pady=2, sticky="w")
    corner_intensity_entries.append(entry)

# --- Frame: LED Strip Settings ---
led_strip_frame = ttk.LabelFrame(root, text="LED Strip Luminous Flux (lm)")
led_strip_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
led_strip_entries = []
num_strips = 20
num_strip_cols = 2
for i in range(num_strips):
    row_num = i // num_strip_cols
    col_num = i % num_strip_cols
    label = ttk.Label(led_strip_frame, text=f"Strip #{i+1}:")
    label.grid(row=row_num, column=2*col_num, padx=2, pady=2, sticky="e")
    entry = ttk.Entry(led_strip_frame, width=10)
    entry.insert(0, "3000.0")
    entry.grid(row=row_num, column=2*col_num+1, padx=2, pady=2, sticky="w")
    led_strip_entries.append(entry)

# Load saved settings if available
settings = load_settings()
if settings:
    if "layer_intensities" in settings:
        for entry, value in zip(layer_intensity_entries, settings["layer_intensities"]):
            entry.delete(0, tk.END)
            entry.insert(0, value)
    if "corner_intensities" in settings:
        for entry, value in zip(corner_intensity_entries, settings["corner_intensities"]):
            entry.delete(0, tk.END)
            entry.insert(0, value)
    if "led_strip_fluxes" in settings:
        for entry, value in zip(led_strip_entries, settings["led_strip_fluxes"]):
            entry.delete(0, tk.END)
            entry.insert(0, value)

# --- Run Button ---
run_button = ttk.Button(root, text="Run Simulation", command=run_simulation)
run_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
