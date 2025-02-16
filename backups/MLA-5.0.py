import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import differential_evolution, NonlinearConstraint

# -------------------------------
# Global Constants & Defaults
# -------------------------------
# COB (diamond pattern) layer counts:
COB_LAYER_COUNTS = [1, 4, 8, 12, 16, 20]

# Default intensities (lm)
DEFAULT_COB_FLUX      = [6000.0, 8000.0, 12000.0, 8000.0, 10000.0, 18000.0]
DEFAULT_LED_STRIP_FLUX = [3000.0] * 20   # 20 LED strips (now optimized, not fixed)
DEFAULT_CORNER_FLUX   = [18000.0, 18000.0, 18000.0, 18000.0]

# Parameter vector (15 elements):
#   x[0:6]    = COB layer intensities (layers 1-6)
#   x[6:11]   = LED strip group intensities (for groups corresponding to layers 6,5,4,3,2)
#   x[11:15]  = Corner COB intensities (order: Left, Right, Bottom, Top)

FIXED_VIEW_ANGLE = 0.0  # degrees

# SPD and conversion constants
SPD_FILE = r"/Users/austinrouse/photonics/backups/spd_data.csv"
LUMINOUS_EFFICACY = 182.0   # lm/W

# -------------------------------
# Simulation Function
# -------------------------------
def simulate_lighting(params, base_height, plot_result=False):
    """
    Simulate PPFD on the floor using the diamond pattern arrangement.
    
    params: 15-element vector:
      - params[0:6]: COB layer intensities (lm)
      - params[6:11]: LED strip group intensities (for groups corresponding to layers 6,5,4,3,2)
      - params[11:15]: Corner COB intensities (order: Left, Right, Bottom, Top)
    base_height: ceiling height in meters.
    plot_result: if True, display the PPFD heatmap.
    """
    # ---- SPD & Conversion Factor ----
    try:
        spd = np.loadtxt(SPD_FILE, delimiter=' ', skiprows=1)
    except Exception as e:
        raise RuntimeError(f"Error loading SPD data: {e}")
    wavelengths = spd[:, 0]  # nm
    intensities = spd[:, 1]
    
    mask_par = (wavelengths >= 400) & (wavelengths <= 700)
    integral_total = np.trapz(intensities, wavelengths)  # Corrected: np.trapz
    integral_par = np.trapz(intensities[mask_par], wavelengths[mask_par])  # Corrected: np.trapz
    PAR_fraction = integral_par / integral_total
    
    wavelengths_m = wavelengths * 1e-9
    lambda_eff = np.trapz(wavelengths_m[mask_par] * intensities[mask_par],
                              wavelengths_m[mask_par]) / np.trapz(intensities[mask_par], wavelengths_m[mask_par])  # Corrected: np.trapz
    h_const = 6.626e-34  # J s
    c = 3.0e8          # m/s
    E_photon = h_const * c / lambda_eff
    N_A = 6.022e23
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction

    # ---- Room & Simulation Parameters ----
    W_m, L_m, H_m = 12.0, 12.0, 3.0   # room dimensions in meters
    meters_to_feet = 3.28084
    W = W_m / meters_to_feet
    L = L_m / meters_to_feet

    # ---- Build Diamond Pattern COB Arrangement ----
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
         (-5, 0), (5, 0), (0, -5), (0, 5)]   # Layer 6
    ]
    spacing_x = W / 7.2
    spacing_y = L / 7.2

    # Center COB (Layer 1)
    center_source = (W/2, L/2, base_height)
    light_sources.append(center_source)
    light_luminous_fluxes.append(params[0])
    # Layers 2-6
    for layer_index, layer in enumerate(layers[1:], start=2):
        intensity = params[layer_index - 1]
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

    # Override the 4 Corner COB Intensities in Layer 6 (indices 57-60)
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
    # Map LED strip groups to parameters:
    #   Strips 1-6   -> group for layer 6: params[6]
    #   Strips 7-10  -> group for layer 5: params[7]
    #   Strips 11-14 -> group for layer 4: params[8]
    #   Strips 15-16 -> group for layer 3: params[9]
    #   Strips 17-20 -> group for layer 2: params[10]
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
   
    # ---- Compute Floor Grid & Direct Irradiance ----
    grid_res = 0.01  # meters
    x_coords = np.arange(0, W, grid_res)
    y_coords = np.arange(0, L, grid_res)
    X, Y = np.meshgrid(x_coords, y_coords)
    direct_irradiance = np.zeros_like(X, dtype=np.float64)
   
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
        return 0.85 + 0.05 * np.sin((wavelength - 400) / 300 * np.pi)
    def reflectance_ceiling(wavelength):
        return 0.1
    rho_wall_eff = np.trapz(intensities * reflectance_wall(wavelengths), wavelengths) / np.trapz(intensities, wavelengths) # Corrected: np.trapz
    rho_ceiling_eff = np.trapz(intensities * reflectance_ceiling(wavelengths), wavelengths) / np.trapz(intensities, wavelengths) # Corrected: np.trapz
    area_ceiling = W * L
    area_walls = 2 * (W * base_height + L * base_height)
    R_eff = (rho_ceiling_eff * area_ceiling + rho_wall_eff * area_walls) / (area_ceiling + area_walls)
    irradiance_reflected = direct_irradiance * R_eff / (1 - R_eff)
    total_irradiance = direct_irradiance + irradiance_reflected
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
        plt.show()
   
    return avg_PPFD, rmse, dou

# -------------------------------
# Nonlinear Constraint Functions
# -------------------------------
def constraint_lower(x, target_ppfd, fixed_base_height):
    avg_ppfd, _, _ = simulate_lighting(x, fixed_base_height, plot_result=False)
    return avg_ppfd - target_ppfd  # Must be >= 0

def constraint_upper(x, target_ppfd, fixed_base_height):
    avg_ppfd, _, _ = simulate_lighting(x, fixed_base_height, plot_result=False)
    return target_ppfd + 250 - avg_ppfd  # Must be >= 0

# -------------------------------
# Modified Objective Function
# -------------------------------
def objective_function(x, target_ppfd, fixed_base_height, rmse_weight=1.0, dou_penalty_weight=50.0):
    """
    x: 15-parameter vector.
         x[0:6]   = COB layer intensities.
         x[6:11]  = LED strip group intensities.
         x[11:15] = Corner COB intensities.
    Objective minimizes RMSE and adds a penalty if DOU is below 96%.
    The PPFD constraints are handled via nonlinear constraints.
    """
    avg_ppfd, rmse, dou = simulate_lighting(x, fixed_base_height, plot_result=False)
    penalty = 0.0
    if dou < 96:
        penalty = dou_penalty_weight * (96 - dou)**2
    obj_val = rmse_weight * rmse + penalty
    print(f"Obj: {obj_val:.3f} | RMSE: {rmse:.3f}, DOU: {dou:.3f}, Penalty: {penalty:.3f}")
    return obj_val

# -------------------------------
# Global Optimization via Differential Evolution
# -------------------------------
def optimize_lighting_global(target_ppfd, fixed_base_height):
    """
    Use differential evolution for global optimization with nonlinear constraints so that
    the simulated average PPFD is always between the target and target+250.
    All 15 parameters are optimized.
    """
    bounds = [(1000.0, 15000.0)] * 15
    nc_lower = NonlinearConstraint(lambda x: constraint_lower(x, target_ppfd, fixed_base_height), 0, np.inf)
    nc_upper = NonlinearConstraint(lambda x: constraint_upper(x, target_ppfd, fixed_base_height), 0, np.inf)
    constraints = [nc_lower, nc_upper]
    result = differential_evolution(
        lambda x: objective_function(x, target_ppfd, fixed_base_height),
        bounds,
        constraints=constraints,
        maxiter=1000,
        popsize=15,
        recombination=0.7,
        disp=True
    )
    return result

# -------------------------------
# Main: Run Global Optimization and Show Results
# -------------------------------
if __name__ == '__main__':
    target_ppfd = 1250  # µmol/m²/s
    fixed_base_height = 0.9144  # meters (~3 ft)
    # Now we optimize all intensities (COB, LED strips, and corner intensities)
    print("Starting global optimization using differential evolution with nonlinear constraints...")
    opt_result = optimize_lighting_global(target_ppfd, fixed_base_height)
    
    print("\nOptimization Result:")
    print(opt_result)
    
    opt_params = opt_result.x
    print("\nOptimized COB Intensities (lm):")
    for i, intensity in enumerate(opt_params[0:6], start=1):
        print(f"  Layer {i}: {intensity:.1f} lm")
    groups = ["Layer 6 (Strips #1-6)", "Layer 5 (Strips #7-10)", "Layer 4 (Strips #11-14)",
              "Layer 3 (Strips #15-16)", "Layer 2 (Strips #17-20)"]
    print("\nOptimized LED Strip Group Intensities (lm):")
    for group, intensity in zip(groups, opt_params[6:11]):
        print(f"  {group}: {intensity:.1f} lm")
    corners = ["Left", "Right", "Bottom", "Top"]
    print("\nOptimized Corner COB Intensities (lm):")
    for corner, intensity in zip(corners, opt_params[11:15]):
        print(f"  {corner}: {intensity:.1f} lm")
    
    # Run final simulation with optimized parameters and display PPFD heatmap.
    simulate_lighting(opt_params, fixed_base_height, plot_result=True)