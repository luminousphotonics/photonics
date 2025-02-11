import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

# --- Geometry functions (unchanged) ---
def build_diamond_61_pattern(width_ft, length_ft):
    layers_raw = [
        [(0, 0)],
        [(-1, 0), (1, 0), (0, -1), (0, 1)],
        [(-1, -1), (1, -1), (-1, 1), (1, 1),
         (-2, 0), (2, 0), (0, -2), (0, 2)],
        [(-2, -1), (2, -1), (-2, 1), (2, 1),
         (-1, -2), (1, -2), (-1, 2), (1, 2),
         (-3, 0), (3, 0), (0, -3), (0, 3)],
        [(-2, -2), (2, -2), (-2, 2), (2, 2),
         (-3, -1), (3, -1), (-3, 1), (3, 1),
         (-1, -3), (1, -3), (-1, 3), (1, 3),
         (-4, 0), (4, 0), (0, -4), (0, 4)],
        [(-3, -2), (3, -2), (-3, 2), (3, 2),
         (-2, -3), (2, -3), (-2, 3), (2, 3),
         (-4, -1), (4, -1), (-4, 1), (4, 1),
         (-1, -4), (1, -4), (-1, 4), (1, 4),
         (-5, 0), (5, 0), (0, -5), (0, 5)]
    ]
    spacing_x = width_ft / 7.2
    spacing_y = length_ft / 7.2
    theta = math.radians(45)
    layers_scaled = []
    for layer in layers_raw:
        coords_layer = []
        for (ix, iy) in layer:
            x_offset = ix * spacing_x
            y_offset = iy * spacing_y
            rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
            rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
            coords_layer.append((rotated_x, rotated_y))
        layers_scaled.append(coords_layer)
    return layers_scaled

def generate_rectangular_cob_arrangement(floor_width_ft, floor_length_ft):
    diamond_layers = build_diamond_61_pattern(floor_width_ft, floor_length_ft)
    all_points = [pt for layer in diamond_layers for pt in layer]
    xs = [pt[0] for pt in all_points]
    width_arr = max(xs) - min(xs)
    spacing_x = floor_width_ft / 3.5
    gap = spacing_x
    # Build second diamond shifted to the right.
    second_diamond_layers = []
    for layer in diamond_layers:
        shifted_layer = [(x + width_arr + gap, y) for (x, y) in layer]
        second_diamond_layers.append(shifted_layer)
    # Merge layers (here we merge layers 1 & 2 and then combine the rest).
    merged_first = diamond_layers[0] + diamond_layers[1] + second_diamond_layers[0] + second_diamond_layers[1]
    remaining_layers = []
    for i in range(2, len(diamond_layers)):
        remaining_layers.append(diamond_layers[i] + second_diamond_layers[i])
    combined_layers = [merged_first] + remaining_layers
    return combined_layers

# --- Simulation function without corner overrides ---
# Now the parameter vector has 5 values – one intensity per COB group.
# Room dimensions (in feet) are passed in so you can set a rectangle (e.g., 18' x 12').
SPD_FILE = '/Users/austinrouse/photonics/backups/spd_data.csv'
LUMINOUS_EFFICACY = 182.0
FIXED_VIEW_ANGLE = 0.0

def simulate_lighting_rectangular(params, base_height, room_width_ft, room_length_ft, plot_result=False):
    # Load SPD data and compute conversion factor
    try:
        spd = np.loadtxt(SPD_FILE, delimiter=' ', skiprows=1)
    except Exception as e:
        raise RuntimeError(f"Error loading SPD data: {e}")
    wavelengths = spd[:, 0]
    intensities = spd[:, 1]
    mask_par = (wavelengths >= 400) & (wavelengths <= 700)
    integral_total = np.trapz(intensities, wavelengths)
    integral_par = np.trapz(intensities[mask_par], wavelengths[mask_par])
    PAR_fraction = integral_par / integral_total
    wavelengths_m = wavelengths * 1e-9
    lambda_eff = (np.trapz(wavelengths_m[mask_par] * intensities[mask_par],
                           wavelengths_m[mask_par]) /
                  np.trapz(intensities[mask_par], wavelengths_m[mask_par]))
    h_const = 6.626e-34
    c = 3.0e8
    E_photon = h_const * c / lambda_eff
    N_A = 6.022e23
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction

    # Convert room dimensions to meters
    ft_to_m = 0.3048
    W_m = room_width_ft * ft_to_m
    L_m = room_length_ft * ft_to_m

    # Build rectangular COB arrangement (each group gets a uniform intensity)
    layers = generate_rectangular_cob_arrangement(room_width_ft, room_length_ft)
    light_sources = []
    light_luminous_fluxes = []
    # 'params' has 5 intensity values, one per group.
    for group_index, layer in enumerate(layers):
        intensity = params[group_index]
        for (x_ft, y_ft) in layer:
            x_m = x_ft * ft_to_m
            y_m = y_ft * ft_to_m
            light_sources.append((x_m, y_m, base_height))
            light_luminous_fluxes.append(intensity)

    # Floor grid for PPFD calculation
    grid_res = 0.01
    x_coords = np.arange(0, W_m, grid_res)
    y_coords = np.arange(0, L_m, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    direct_irradiance = np.zeros_like(X, dtype=np.float64)

    for src, lum in zip(light_sources, light_luminous_fluxes):
        I = lum / LUMINOUS_EFFICACY  # convert lm to W
        x0, y0, _ = src
        r = np.sqrt((X - x0)**2 + (Y - y0)**2 + base_height**2)
        r_horizontal = np.sqrt((X - x0)**2 + (Y - y0)**2)
        theta_pt = np.arctan2(r_horizontal, base_height)
        theta_led = math.radians(FIXED_VIEW_ANGLE)
        if theta_led <= 0:
            contribution = (I / math.pi) * (base_height**2) / (r**4)
        else:
            sigma = theta_led / 2.0
            gaussian_weight = np.exp(-0.5 * (theta_pt / sigma)**2)
            contribution = (I / (2 * math.pi * sigma**2)) * gaussian_weight * (base_height**2) / (r**4)
        direct_irradiance += contribution

    # Simple interreflection model (same as before)
    rho_wall    = 0.9        
    rho_ceiling = 0.0     
    rho_floor   = 0.0
    area_wall = 2 * (W_m * base_height + L_m * base_height)
    area_ceiling = W_m * L_m
    area_floor = W_m * L_m
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
    print(f"Simulated Avg PPFD = {avg_PPFD:.1f} µmol/m²/s, RMSE = {rmse:.1f}, DOU = {dou:.1f}%")
    
    if plot_result:
        plt.figure(figsize=(6, 5))
        im = plt.imshow(total_PPFD, origin='lower', extent=[0, W_m, 0, L_m], cmap='viridis')
        plt.colorbar(im, label='PPFD (µmol m⁻² s⁻¹)')
        plt.xlabel('Room Width (m)')
        plt.ylabel('Room Length (m)')
        plt.title('Simulated PPFD on Floor')
        stats_text = f"Avg PPFD: {avg_PPFD:.1f}\nRMSE: {rmse:.1f}\nDOU: {dou:.1f}%"
        plt.text(0.05 * W_m, 0.95 * L_m, stats_text, color='white', ha='left', va='top',
                 fontsize=8, bbox=dict(facecolor='black', alpha=0.7, pad=5))
        plt.show()
    
    return avg_PPFD, rmse, dou

# --- Objective function ---
def objective_function_rectangular(x, target_ppfd, fixed_base_height, room_width_ft, room_length_ft,
                                   rmse_weight=0.6, ppfd_weight=0.2):
    avg_ppfd, rmse, _ = simulate_lighting_rectangular(x, fixed_base_height, room_width_ft, room_length_ft, plot_result=False)
    if avg_ppfd < target_ppfd:
        ppfd_penalty = (target_ppfd - avg_ppfd)**2
    else:
        overshoot_multiplier = 300
        ppfd_penalty = overshoot_multiplier * (avg_ppfd - target_ppfd)**2
    obj_val = rmse_weight * rmse + ppfd_weight * ppfd_penalty
    print(f"Obj: {obj_val:.3f} | RMSE: {rmse:.3f}, PPFD penalty: {ppfd_penalty:.3f}")
    return obj_val

# --- Optimization routine (COB-only: 5 parameters) ---
DEFAULT_RECT_COB_FLUX = [6200.0, 2789.0, 1368.0, 15044.0, 28716.0]

def optimize_lighting_rectangular(target_ppfd, fixed_base_height, room_width_ft, room_length_ft, verbose=False):
    x0 = np.array(DEFAULT_RECT_COB_FLUX)
    bounds = [(1000.0, 35000.0)] * len(x0)
    result = minimize(
        objective_function_rectangular,
        x0,
        args=(target_ppfd, fixed_base_height, room_width_ft, room_length_ft),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 2000, 'disp': verbose, 'ftol': 1e-6}
    )
    return result

# --- Main example ---
if __name__ == '__main__':
    # For a rectangular layout, you might use a wider room:
    room_width_ft = 26.0  # e.g., 18 feet wide
    room_length_ft = 12.0  # 12 feet long
    target_ppfd = 1250   # µmol/m²/s
    fixed_base_height = 0.9144  # ~3 feet
    
    print("Starting optimization for rectangular COB intensities (no corner overrides)...")
    opt_result = optimize_lighting_rectangular(target_ppfd, fixed_base_height, room_width_ft, room_length_ft, verbose=True)
    
    print("\nOptimization Result:")
    print(opt_result)
    
    opt_params = opt_result.x
    print("\nOptimized COB Group Intensities (lm):")
    for i, intensity in enumerate(opt_params, start=1):
        print(f"  Group {i}: {intensity:.1f} lm")
    
    # Run final simulation with optimized parameters and plot the PPFD heatmap.
    simulate_lighting_rectangular(opt_params, fixed_base_height, room_width_ft, room_length_ft, plot_result=True)
