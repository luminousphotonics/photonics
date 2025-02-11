import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

# -------------------------------
# Global constants & defaults
# -------------------------------
NUM_LEDS = 61  # COB (diamond pattern) gives 61 sources

# COB layer counts:
# Layer 1: 1 LED, Layer 2: 4 LEDs, Layer 3: 8 LEDs,
# Layer 4: 12 LEDs, Layer 5: 16 LEDs, Layer 6: 20 LEDs
COB_LAYER_COUNTS = [1, 4, 8, 12, 16, 20]

# Default COB intensities (lm) per layer:
DEFAULT_COB_FLUX = [6242.0, 6201.0, 2789.0, 1368.0, 15044.0, 28716.0]

# For LED strips we have 5 groups (layers 2-6). Set default values (lm):
DEFAULT_LED_STRIP_FLUX = [11543.0, 9588.0, 1713.0, 1000.0, 17320.0]

#for COB Only mode
#DEFAULT_LED_STRIP_FLUX = [0.0, 0.0, 0.0, 0.0, 0.0]

# Default corner COB intensities (lm):
DEFAULT_CORNER_FLUX = [35000.0, 35000.0, 35000.0, 35000.0]

# Overall, our parameter vector (x) will be 15 elements:
#   x[0:6]   = COB layer intensities (layers 1-6)
#   x[6:11]  = LED strip group intensities (for layers 6,5,4,3,2)
#   x[11:15] = Corner COB intensities (order: Left, Right, Bottom, Top)

# Fixed viewing angle (we use 0° to enforce Lambertian behavior)
FIXED_VIEW_ANGLE = 0.0

# SPD and conversion constants
SPD_FILE = '/Users/austinrouse/photonics/backups/spd_data.csv'
LUMINOUS_EFFICACY = 182.0   # lm/W

# -------------------------------
# Simulation function
# -------------------------------
def simulate_lighting(params, base_height, plot_result=False):
    """
    params: vector of 15 values.
      params[0:6]: COB layer intensities (lm) for layers 1–6.
      params[6:11]: LED strip group intensities.
      params[11:15]: Corner COB intensities.
    base_height: ceiling height in meters.
    plot_result: if True, plot the PPFD heatmap.
    """
    # ---- SPD & Conversion Factor ----
    try:
        spd = np.loadtxt(SPD_FILE, delimiter=' ', skiprows=1)
    except Exception as e:
        raise RuntimeError(f"Error loading SPD data: {e}")
    wavelengths = spd[:, 0]  # nm
    intensities = spd[:, 1]
    
    mask_par = (wavelengths >= 400) & (wavelengths <= 700)
    integral_total = np.trapz(intensities, wavelengths)
    integral_par = np.trapz(intensities[mask_par], wavelengths[mask_par])
    PAR_fraction = integral_par / integral_total
    
    wavelengths_m = wavelengths * 1e-9
    lambda_eff = (np.trapz(wavelengths_m[mask_par] * intensities[mask_par],
                           wavelengths_m[mask_par]) /
                  np.trapz(intensities[mask_par], wavelengths_m[mask_par]))
    h_const = 6.626e-34  # J s
    c = 3.0e8          # m/s
    E_photon = h_const * c / lambda_eff
    N_A = 6.022e23
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction

    # ---- Room & Simulation Parameters ----
    W_m, L_m, H_m = 12.0, 12.0, 3.0  # room dimensions in meters
    meters_to_feet = 3.28084
    W = W_m / meters_to_feet
    L = L_m / meters_to_feet

    # ---- Build the COB (diamond) arrangement ----
    light_sources = []
    light_luminous_fluxes = []  # for COBs

    # Define the diamond layers
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
         (-5, 0), (5, 0), (0, -5), (0, 5)]   # Layer 6
    ]
    spacing_x = W / 7.2
    spacing_y = L / 7.2

    # Center COB (Layer 1)
    center_source = (W/2, L/2, base_height)
    light_sources.append(center_source)
    light_luminous_fluxes.append(params[0])  # COB intensity for layer 1

    # Build COB positions for layers 2–6.
    for layer_index, layer in enumerate(layers[1:], start=2):
        intensity = params[layer_index - 1]  # layer 2 -> params[1], etc.
        for dot in layer:
            x_offset = spacing_x * dot[0]
            y_offset = spacing_y * dot[1]
            theta = math.radians(45)
            rotated_x = x_offset * math.cos(theta) - y_offset * math.sin(theta)
            rotated_y = x_offset * math.sin(theta) + y_offset * math.cos(theta)
            pos = (center_source[0] + rotated_x,
                   center_source[1] + rotated_y,
                   base_height)
            light_sources.append(pos)
            light_luminous_fluxes.append(intensity)
    N_cob = len(light_sources)
    if N_cob != sum(COB_LAYER_COUNTS):
        raise RuntimeError(f"Expected {sum(COB_LAYER_COUNTS)} COB positions, got {N_cob}")

    # ---- Override the 4 corner COB intensities in layer 6 ----
    # The corner positions in the diamond are at indices 57, 58, 59, 60.
    corner_indices = [57, 58, 59, 60]
    corner_params = params[11:15]
    for i, idx in enumerate(corner_indices):
        light_luminous_fluxes[idx] = corner_params[i]

    # ---- Add LED Strip Sources ----
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
    # Group mapping for LED strips:
    #   Strips 1–6   (keys 1 to 6): group for layer 6 --> params[6]
    #   Strips 7–10  (keys 7 to 10): group for layer 5 --> params[7]
    #   Strips 11–14 (keys 11 to 14): group for layer 4 --> params[8]
    #   Strips 15–16 (keys 15 to 16): group for layer 3 --> params[9]
    #   Strips 17–20 (keys 17 to 20): group for layer 2 --> params[10]
    led_strip_group = {}
    for strip_num in range(1, 21):
        if 1 <= strip_num <= 6:
            led_strip_group[strip_num] = params[6]
        elif 7 <= strip_num <= 10:
            led_strip_group[strip_num] = params[7]
        elif 11 <= strip_num <= 14:
            led_strip_group[strip_num] = params[8]
        elif 15 <= strip_num <= 16:
            led_strip_group[strip_num] = params[9]
        elif 17 <= strip_num <= 20:
            led_strip_group[strip_num] = params[10]
    
    # Generate intermediate points for LED strips.
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
        flux_this_strip = led_strip_group[strip_num]
        num_points = len(points)
        flux_per_point = flux_this_strip / num_points if num_points > 0 else 0.0
        for pos in points:
            light_sources.append(pos)
            light_luminous_fluxes.append(flux_per_point)
    
    # ---- Floor Grid for PPFD accumulation ----
    grid_res = 0.01  # meters
    x_coords = np.arange(0, W, grid_res)
    y_coords = np.arange(0, L, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    direct_irradiance = np.zeros_like(X, dtype=np.float64)
    
    # ---- Compute Direct Irradiance from Each LED Source ----
    for src, lum in zip(light_sources, light_luminous_fluxes):
        I = lum / LUMINOUS_EFFICACY  # convert lm to W
        x0, y0, _ = src
        r = np.sqrt((X - x0)**2 + (Y - y0)**2 + (base_height)**2)
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

    # ---- Interreflection Model ----
    rho_wall    = 0.9        
    rho_ceiling = 0.0     
    rho_floor   = 0.0
    area_wall = 2 * (W * base_height + L * base_height)
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
    print(f"Simulated Avg PPFD = {avg_PPFD:.1f} µmol/m²/s, RMSE = {rmse:.1f}, DOU = {dou:.1f}%")
    
    if plot_result:
        plt.figure(figsize=(6, 5))
        im = plt.imshow(total_PPFD, origin='lower', extent=[0, W, 0, L], cmap='viridis')
        plt.colorbar(im, label='PPFD (µmol m⁻² s⁻¹)')
        plt.xlabel('Room Width (m)')
        plt.ylabel('Room Length (m)')
        plt.title('Simulated PPFD on Floor')
        stats_text = f"Avg PPFD: {avg_PPFD:.1f}\nRMSE: {rmse:.1f}\nDOU: {dou:.1f} %"
        plt.text(0.05 * W, 0.95 * L, stats_text, color='white', ha='left', va='top',
                 fontsize=8, bbox=dict(facecolor='black', alpha=0.7, pad=5))
        plt.show()
    
    return avg_PPFD, rmse, dou

# -------------------------------
# Objective Function for Optimization
# -------------------------------
def objective_function(x, target_ppfd, fixed_base_height, rmse_weight=0.6, ppfd_weight=0.2):
    """
    x: vector of 15 parameters.
         x[0:6]   = COB layer intensities.
         x[6:11]  = LED strip group intensities.
         x[11:15] = Corner COB intensities.
    """
    avg_ppfd, rmse, _ = simulate_lighting(x, fixed_base_height, plot_result=False)
    if avg_ppfd < target_ppfd:
        ppfd_penalty = (target_ppfd - avg_ppfd)**2
    else:
        overshoot_multiplier = 300
        ppfd_penalty = overshoot_multiplier * (avg_ppfd - target_ppfd)**2
    obj_val = rmse_weight * rmse + ppfd_weight * ppfd_penalty
    print(f"Obj: {obj_val:.3f} | RMSE: {rmse:.3f}, PPFD penalty: {ppfd_penalty:.3f}")
    return obj_val

# -------------------------------
# Wrapper Objective for COB-Only Optimization
# -------------------------------
def objective_function_cob_only(x_partial, target_ppfd, fixed_base_height, rmse_weight=0.6, ppfd_weight=0.2):
    """
    x_partial: vector of 10 values (first 6 for COB layers and last 4 for corner COBs).
    LED strip intensities are fixed to their defaults.
    """
    # Rebuild full 15-element parameter vector.
    full_params = np.concatenate([x_partial[0:6],
                                  np.array(DEFAULT_LED_STRIP_FLUX),
                                  x_partial[6:10]])
    return objective_function(full_params, target_ppfd, fixed_base_height, rmse_weight, ppfd_weight)

# -------------------------------
# Optimization Routine
# -------------------------------
def optimize_lighting(target_ppfd, fixed_base_height, verbose=False, cob_only=False):
    """
    cob_only: if True, optimize only for COB and corner COB intensities
              (LED strip intensities are fixed to defaults).
    """
    if cob_only:
        # Optimize 10 parameters: 6 COB layers + 4 corner COBs.
        x0_partial = np.array(DEFAULT_COB_FLUX + DEFAULT_CORNER_FLUX)
        bounds = [(1000.0, 35000.0)] * 10
        result = minimize(
            objective_function_cob_only,
            x0_partial,
            args=(target_ppfd, fixed_base_height),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 2000, 'disp': verbose, 'ftol': 1e-6}
        )
        # Reconstruct the full parameter vector.
        opt_params = np.concatenate([result.x[0:6],
                                     np.array(DEFAULT_LED_STRIP_FLUX),
                                     result.x[6:10]])
        result.x = opt_params  # Replace the solution with full parameters.
        return result
    else:
        # Optimize full 15-parameter vector.
        x0 = np.array(DEFAULT_COB_FLUX + DEFAULT_LED_STRIP_FLUX + DEFAULT_CORNER_FLUX)
        bounds = [(1000.0, 35000.0)] * len(x0)
        result = minimize(
            objective_function,
            x0,
            args=(target_ppfd, fixed_base_height),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 2000, 'disp': verbose, 'ftol': 1e-6}
        )
        return result

# -------------------------------
# Main: Run Optimization and Show Results
# -------------------------------
if __name__ == '__main__':
    target_ppfd = 1250  # µmol/m²/s
    fixed_base_height = 0.9144  # meters
    
    # Set to True to optimize only for COB (and corner COB) intensities.
    # LED strip intensities will remain at their default values.
    cob_only = True  
    
    if cob_only:
        print("Starting optimization for COB and corner COB intensities (LED strips fixed)...")
        opt_result = optimize_lighting(target_ppfd, fixed_base_height, verbose=True, cob_only=True)
    else:
        print("Starting optimization for COB, LED strip, and corner COB intensities...")
        opt_result = optimize_lighting(target_ppfd, fixed_base_height, verbose=True, cob_only=False)
    
    print("\nOptimization Result:")
    print(opt_result)
    
    opt_params = opt_result.x
    print("\nOptimized COB Intensities (lm):")
    for i, intensity in enumerate(opt_params[0:6], start=1):
        print(f"  Layer {i}: {intensity:.1f} lm")
    if not cob_only:
        print("\nOptimized LED Strip Group Intensities (lm):")
        groups = ["Layer 6 (Strips #1-6)", "Layer 5 (Strips #7-10)", "Layer 4 (Strips #11-14)",
                  "Layer 3 (Strips #15-16)", "Layer 2 (Strips #17-20)"]
        for group, intensity in zip(groups, opt_params[6:11]):
            print(f"  {group}: {intensity:.1f} lm")
    else:
        print("\nLED Strip Group Intensities fixed at default values:")
        for i, intensity in enumerate(DEFAULT_LED_STRIP_FLUX, start=1):
            print(f"  Group {i}: {intensity:.1f} lm")
    print("\nOptimized Corner COB Intensities (lm):")
    corners = ["Left (-5,0)", "Right (5,0)", "Bottom (0,-5)", "Top (0,5)"]
    for corner, intensity in zip(corners, opt_params[11:15]):
        print(f"  {corner}: {intensity:.1f} lm")
    
    # Run final simulation with optimized parameters and plot the PPFD heatmap.
    simulate_lighting(opt_params, fixed_base_height, plot_result=True)
