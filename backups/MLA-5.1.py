import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import differential_evolution, NonlinearConstraint

# SPD and conversion constants
SPD_FILE = r"/Users/austinrouse/photonics/backups/spd_data.csv"
LUMINOUS_EFFICACY = 182.0   # lm/W
FIXED_VIEW_ANGLE = 0.0      # degrees

# ----------------------------------------------------------------------
# Rectangular (Mirrored Diamond) COB Arrangement Functions
# ----------------------------------------------------------------------
def build_diamond_61_pattern(width_ft, length_ft):
    """
    Builds a mirrored diamond COB pattern with gap COBs.
    
    The base pattern is generated for a single square module (width_ft x length_ft).
    Then a mirrored (shifted) copy is produced to represent a second module.
    Returns a list of six groups:
      - Group 1: Merged Layers 1 & 2
      - Group 2: Layer 3
      - Group 3: Layer 4
      - Group 4: Layer 5
      - Group 5: Layer 6
      - Group 6: Extra gap COBs (from Layer 5)
    """
    layers_raw = [
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
    spacing_x = width_ft / 3.5
    spacing_y = length_ft / 3.5
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
    # Build the right diamond by shifting the left one.
    all_points = [pt for layer in layers_scaled for pt in layer]
    xs = [pt[0] for pt in all_points]
    min_x = min(xs)
    max_x = max(xs)
    width_arr = max_x - min_x
    gap = spacing_x  # gap between modules (in feet)
    offset = width_arr + gap
    second_diamond_layers = []
    for layer in layers_scaled:
        shifted_layer = [(x + offset, y) for (x, y) in layer]
        second_diamond_layers.append(shifted_layer)
    # Form groups:
    group1 = layers_scaled[0] + layers_scaled[1] + second_diamond_layers[0] + second_diamond_layers[1]
    group2 = layers_scaled[2] + second_diamond_layers[2]
    group3 = layers_scaled[3] + second_diamond_layers[3]
    group4 = layers_scaled[4] + second_diamond_layers[4]
    group5 = layers_scaled[5] + second_diamond_layers[5]
    # Compute extra gap COBs from Layer 5.
    left_group4 = layers_scaled[4]
    right_group4 = second_diamond_layers[4]
    tol = 1e-3
    unique_y = sorted({round(y, 3) for (x, y) in left_group4})
    if len(unique_y) >= 5:
        start = (len(unique_y) - 5) // 2
        central_rows = unique_y[start:start+5]
    else:
        central_rows = unique_y
    gap_group = []
    for target_y in central_rows:
        left_candidates = [x for (x, y) in left_group4 if abs(y - target_y) < tol]
        right_candidates = [x for (x, y) in right_group4 if abs(y - target_y) < tol]
        if left_candidates and right_candidates:
            x_left = max(left_candidates)
            x_right = min(right_candidates)
            mid_x = (x_left + x_right) / 2
            gap_group.append((mid_x, target_y))
    return [group1, group2, group3, group4, group5, gap_group]

# ----------------------------------------------------------------------
# Simulation & Optimization Functions
# ----------------------------------------------------------------------
def simulate_lighting(params, base_height, plot_result=False):
    """
    Simulate PPFD on the cultivation area using the rectangular COB arrangement based on a mirrored diamond pattern.
    
    params: 6-element vector.
      - params[0]: Intensity for Group 1 (Merged Layers 1 & 2)
      - params[1]: Intensity for Group 2 (Layer 3)
      - params[2]: Intensity for Group 3 (Layer 4)
      - params[3]: Intensity for Group 4 (Layer 5)
      - params[4]: Intensity for Group 5 (Layer 6)
      - params[5]: Intensity for Group 6 (Gap COBs)
    base_height: Ceiling height in meters.
    plot_result: If True, display the PPFD heatmap.
    """
    # ---- SPD & Conversion Factor ----
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
    lambda_eff = np.trapz(wavelengths_m[mask_par] * intensities[mask_par],
                            wavelengths_m[mask_par]) / np.trapz(intensities[mask_par], wavelengths_m[mask_par])
    h_const = 6.626e-34
    c = 3.0e8
    E_photon = h_const * c / lambda_eff
    N_A = 6.022e23
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction

    # ---- Generate COB Light Sources from the Pattern ----
    # Use a base pattern size of 12 ft x 12 ft for each module.
    groups = build_diamond_61_pattern(12.0, 12.0)
    
    # Convert COB positions (in feet) to meters.
    raw_sources = []
    raw_fluxes = []
    for i, group in enumerate(groups):
        intensity = params[i]
        for (x_ft, y_ft) in group:
            # Convert from feet to meters.
            x_m = x_ft * 0.3048
            y_m = y_ft * 0.3048
            raw_sources.append((x_m, y_m, base_height))
            raw_fluxes.append(intensity)
    
    # ---- Map the COB Arrangement to the Cultivation Area ----
    # We want the arrangement to span a fixed rectangular area of 26 ft x 12 ft.
    SIM_WIDTH_FT = 26.0  # horizontal span (26 ft)
    SIM_HEIGHT_FT = 12.0  # vertical span (12 ft)
    sim_width_m = SIM_WIDTH_FT * 0.3048
    sim_height_m = SIM_HEIGHT_FT * 0.3048

    # Compute the bounding box of the raw light sources.
    xs = [pt[0] for pt in raw_sources]
    ys = [pt[1] for pt in raw_sources]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Apply a non-uniform scaling so that the raw sources span exactly the desired area.
    scale_x = sim_width_m / (max_x - min_x)
    scale_y = sim_height_m / (max_y - min_y)
    # Transform all sources: translate to (0,0) and then scale.
    light_sources = [((x - min_x) * scale_x, (y - min_y) * scale_y, z) for (x, y, z) in raw_sources]
    light_luminous_fluxes = raw_fluxes  # intensities remain unchanged

    # ---- Define Fixed Simulation Grid Over the Cultivation Area ----
    x_min = 0.0
    x_max = sim_width_m
    y_min = 0.0
    y_max = sim_height_m
    grid_res = 0.01  # meters
    x_coords = np.arange(x_min, x_max, grid_res)
    y_coords = np.arange(y_min, y_max, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    direct_irradiance = np.zeros_like(X, dtype=np.float64)
   
    # ---- Compute Direct Irradiance from All Sources ----
    for src, lum in zip(light_sources, light_luminous_fluxes):
        I = lum / LUMINOUS_EFFICACY  # W
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

    # ---- Advanced Reflection (Radiosity) Model ----
    def reflectance_wall(wavelength):
        return 0.85 + 0.05 * np.sin((wavelength - 400) / 300 * math.pi)
    def reflectance_ceiling(wavelength):
        return 0.1
    rho_wall_eff = np.trapz(intensities * reflectance_wall(wavelengths), wavelengths) / np.trapz(intensities, wavelengths)
    rho_ceiling_eff = np.trapz(intensities * reflectance_ceiling(wavelengths), wavelengths) / np.trapz(intensities, wavelengths)
    area_ceiling = (x_max - x_min) * (y_max - y_min)
    area_walls = 2 * ((x_max - x_min) * base_height + (y_max - y_min) * base_height)
    R_eff = (rho_ceiling_eff * area_ceiling + rho_wall_eff * area_walls) / (area_ceiling + area_walls)
    irradiance_reflected = direct_irradiance * R_eff / (1 - R_eff)
    total_irradiance = direct_irradiance + irradiance_reflected
    total_PPFD = total_irradiance * conversion_factor

    avg_PPFD = np.mean(total_PPFD)
    rmse = np.sqrt(np.mean((total_PPFD - avg_PPFD)**2))
    dou = 100.0 * (1.0 - (rmse / avg_PPFD))
    print(f"Simulated Avg PPFD = {avg_PPFD:.1f} µmol/m²/s, RMSE = {rmse:.1f}, DOU = {dou:.1f}%")
   
    if plot_result:
        plt.figure(figsize=(8, 4))
        im = plt.imshow(total_PPFD, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis')
        plt.colorbar(im, label='PPFD (µmol m⁻² s⁻¹)')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Simulated PPFD on Cultivation Area (26ft x 12ft)')
        plt.show()
   
    return avg_PPFD, rmse, dou

# ----------------------------------------------------------------------
# Nonlinear Constraint and Objective Functions
# ----------------------------------------------------------------------
def constraint_lower(x, target_ppfd, base_height):
    avg_ppfd, _, _ = simulate_lighting(x, base_height, plot_result=False)
    return avg_ppfd - target_ppfd  # Must be >= 0

def constraint_upper(x, target_ppfd, base_height):
    avg_ppfd, _, _ = simulate_lighting(x, base_height, plot_result=False)
    return target_ppfd + 250 - avg_ppfd  # Must be >= 0

def objective_function(x, target_ppfd, base_height, rmse_weight=1.0, dou_penalty_weight=10.0, ppfd_penalty_weight=100):
    """
    x: 6-parameter vector.
         x[0]: Group 1 intensity (Merged Layers 1 & 2)
         x[1]: Group 2 intensity (Layer 3)
         x[2]: Group 3 intensity (Layer 4)
         x[3]: Group 4 intensity (Layer 5)
         x[4]: Group 5 intensity (Layer 6)
         x[5]: Group 6 intensity (Gap COBs)
    The objective minimizes RMSE, and adds:
      - a penalty if DOU falls below 95%
      - a penalty proportional to (avg_PPFD - target)^2.
    """
    avg_ppfd, rmse, dou = simulate_lighting(x, base_height, plot_result=False)
    ppfd_deviation_penalty = ppfd_penalty_weight * (avg_ppfd - target_ppfd)**2
    penalty = 0.0
    if dou < 95:
        penalty = dou_penalty_weight * (80 - dou)**2
    obj_val = rmse_weight * rmse + penalty + ppfd_deviation_penalty
    print(f"Obj: {obj_val:.3f} | RMSE: {rmse:.3f}, Avg PPFD: {avg_ppfd:.1f}, DOU: {dou:.3f}, PPFD Penalty: {ppfd_deviation_penalty:.3f}, DOU Penalty: {penalty:.3f}")
    return obj_val

def optimize_lighting_global(target_ppfd, base_height):
    """
    Use differential evolution for global optimization with nonlinear constraints so that
    the simulated average PPFD is always between the target and target+250.
    All 6 parameters (group intensities for the rectangular formation) are optimized.
    """
    bounds = [(1000.0, 18000.0)] * 6
    nc_lower = NonlinearConstraint(lambda x: constraint_lower(x, target_ppfd, base_height), 0, np.inf)
    nc_upper = NonlinearConstraint(lambda x: constraint_upper(x, target_ppfd, base_height), 0, np.inf)
    constraints = [nc_lower, nc_upper]
    result = differential_evolution(
        lambda x: objective_function(x, target_ppfd, base_height),
        bounds,
        constraints=constraints,
        maxiter=50,
        popsize=15,
        recombination=0.7,
        disp=True
    )
    return result

# ----------------------------------------------------------------------
# Main: Run Global Optimization and Show Results
# ----------------------------------------------------------------------
if __name__ == '__main__':
    target_ppfd = 1450  # µmol/m²/s
    base_height = 0.9144  # meters (~3 ft)
    print("Starting global optimization using differential evolution with nonlinear constraints...")
    opt_result = optimize_lighting_global(target_ppfd, base_height)
    
    print("\nOptimization Result:")
    print(opt_result)
    
    opt_params = opt_result.x
    group_labels = ["Group 1 (Merged Layers 1&2)", "Group 2 (Layer 3)", "Group 3 (Layer 4)",
                    "Group 4 (Layer 5)", "Group 5 (Layer 6)", "Group 6 (Gap COBs)"]
    print("\nOptimized Intensities (lm):")
    for label, intensity in zip(group_labels, opt_params):
        print(f"  {label}: {intensity:.1f} lm")
    
    # Run final simulation with optimized parameters and display PPFD heatmap.
    simulate_lighting(opt_params, base_height, plot_result=True)
