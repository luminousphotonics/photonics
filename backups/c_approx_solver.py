#!/usr/bin/env python3
import numpy as np
import scipy.optimize
import subprocess
import csv
import os
import argparse
import time
import math
from numba import njit, prange
from functools import lru_cache

# --- Configuration ---
# !! IMPORTANT: Set the correct path to your simulation script !!
SIMULATION_SCRIPT_PATH = "/Users/austinrouse/photonics/backups/lighting_simulation_data.py"

CACHE_DIR = "lcm_approx_cache"
L_NOMINAL = 1000.0  # Nominal flux (lumens) for C_approx calculation
EPSILON = 1e-9      # Small number for comparisons

# --- Constants based on your simulation script ---
LUMINOUS_EFFICACY = 182.0  # lumens/W (MUST MATCH YOUR SIM SCRIPT)
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv" # Used to compute conversion factor

# --- COB Datasheet Angular Data (MUST MATCH YOUR SIM SCRIPT) ---
COB_ANGLE_DATA = np.array([
    [  0, 1.00], [ 10, 0.98], [ 20, 0.95], [ 30, 0.88], [ 40, 0.78],
    [ 50, 0.65], [ 60, 0.50], [ 70, 0.30], [ 80, 0.10], [ 90, 0.00],
], dtype=np.float64)
COB_ANGLES_DEG = COB_ANGLE_DATA[:, 0]
COB_SHAPE_REL = COB_ANGLE_DATA[:, 1]

# ------------------------------------
# Helper Functions (Physics & Geometry)
# ------------------------------------

@lru_cache(maxsize=1) # Cache the result as it's expensive
def compute_spd_conversion_factor(spd_file):
    """ Computes PPFD conversion factor from SPD data (simplified from your script). """
    try:
        # Basic calculation assuming pre-processed data or simplified approach
        # In a real scenario, reuse the exact function from your script if possible
        # This is a placeholder calculation - replace with actual if needed
        raw_spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
        wl = raw_spd[:,0]
        intens = raw_spd[:,1]
        mask_par = (wl >= 400) & (wl <= 700)
        wl_m = wl * 1e-9; h, c, N_A = 6.626e-34, 3.0e8, 6.022e23;
        
        # Simplified effective wavelength calculation
        par_intens = intens[mask_par]
        par_wl_m = wl_m[mask_par]
        if len(par_intens) < 2 or np.sum(par_intens) < EPSILON:
             print("[WARN] Insufficient PAR data for conversion factor, using fallback.")
             # A reasonable fallback might be based on ~550nm depending on LED type
             lambda_eff = 550e-9 
        else:
             lambda_eff = np.trapz(par_wl_m * par_intens, par_wl_m) / np.trapz(par_intens, par_wl_m)
             if lambda_eff < EPSILON: lambda_eff = 550e-9 # Fallback if denominator is zero

        # Calculate total energy fraction in PAR
        par_fraction = 1.0
        if len(wl) >= 2:
             total_energy = np.trapz(intens, wl_m)
             par_energy = np.trapz(par_intens, par_wl_m)
             if total_energy > EPSILON:
                  par_fraction = par_energy / total_energy

        E_photon = (h * c / lambda_eff) if lambda_eff > EPSILON else (h*c/550e-9)
        conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * par_fraction
        print(f"[INFO] Approx SPD Conv Factor: PAR frac={par_fraction:.3f}, factor={conversion_factor:.5f} µmol/J")
        return conversion_factor
    except Exception as e:
        print(f"Error computing conversion factor: {e}. Using fallback.")
        return 4.57 # Typical value for white LEDs as fallback µmol/J per W_radiant -> ~0.025 µmol/s per Lumen

# Pre-calculate factor needed to convert Lux to PPFD
CONVERSION_FACTOR = compute_spd_conversion_factor(SPD_FILE)
LUX_TO_PPFD_FACTOR = CONVERSION_FACTOR / LUMINOUS_EFFICACY

@njit
def integrate_shape_for_flux_numba(angles_deg, shape):
    """ Numba implementation of shape integration """
    rad_angles = np.radians(angles_deg)
    G = 0.0
    for i in range(len(rad_angles) - 1):
        th0 = rad_angles[i]; th1 = rad_angles[i+1]
        s0 = shape[i]; s1 = shape[i+1]
        s_mean = 0.5*(s0 + s1)
        dtheta = (th1 - th0)
        th_mid = 0.5*(th0 + th1)
        sin_mid = math.sin(th_mid)
        G_seg = s_mean * 2.0 * math.pi * sin_mid * dtheta
        G += G_seg
    # Add contribution from the last segment towards 90 degrees if needed?
    # Original seems to stop before last point. If last point is angle 90, shape 0, it's fine.
    return G

# Pre-calculate shape integral
SHAPE_INTEGRAL = integrate_shape_for_flux_numba(COB_ANGLES_DEG, COB_SHAPE_REL)
if abs(SHAPE_INTEGRAL) < EPSILON:
    print("[ERROR] COB Shape Integral is zero. Check COB data.")
    # Use a fallback like Pi if shape is Lambertian-like but data is bad
    SHAPE_INTEGRAL = np.pi

@njit
def luminous_intensity_rel_numba(angle_deg, angles_deg_data, shape_rel_data):
    """ Numba implementation for relative intensity lookup """
    if angle_deg <= angles_deg_data[0]:
        return shape_rel_data[0]
    elif angle_deg >= angles_deg_data[-1]:
        return shape_rel_data[-1]
    else:
        # Numba needs explicit loop for interpolation
        for i in range(len(angles_deg_data) - 1):
            if angles_deg_data[i] <= angle_deg <= angles_deg_data[i+1]:
                x0, x1 = angles_deg_data[i], angles_deg_data[i+1]
                y0, y1 = shape_rel_data[i], shape_rel_data[i+1]
                if abs(x1 - x0) < EPSILON: return y0 # Avoid division by zero
                return y0 + (y1 - y0) * (angle_deg - x0) / (x1 - x0)
        return shape_rel_data[-1] # Fallback if somehow outside loop range

def calculate_dimensions(num_layers):
    """ Calculates room width/length based on number of layers. """
    base_n = 10
    base_floor = 6.10
    scale_factor = 0.64

    if num_layers < base_n:
        print(f"Warning: num_layers ({num_layers}) < base {base_n}. Using base floor size.")
        W = base_floor
    else:
        W = base_floor + (num_layers - base_n) * scale_factor
    L = W # Assuming square room
    return W, L

def build_cob_positions(N, W, L, H):
    """ Builds COB positions array with layer index. """
    n = N - 1 # Max layer index
    positions = []
    # Layer 0
    if n >= 0:
        positions.append((0, 0, H, 0))
    # Layers 1 to n
    for i in range(1, n + 1):
        # Corners
        positions.append(( i,  i, H, i))
        positions.append((-i,  i, H, i))
        positions.append(( i, -i, H, i))
        positions.append((-i, -i, H, i))
        # Edges (excluding corners)
        for k in range(1, i):
            positions.append(( k,  i, H, i))
            positions.append((-k,  i, H, i))
            positions.append(( k, -i, H, i))
            positions.append((-k, -i, H, i))
            positions.append(( i,  k, H, i))
            positions.append((-i,  k, H, i))
            positions.append(( i, -k, H, i))
            positions.append((-i, -k, H, i))
            
    # Convert to numpy array before scaling/rotation
    raw_positions = np.array(positions, dtype=np.float64)
    if raw_positions.shape[0] == 0: return np.empty((0,4), dtype=np.float64)

    # Scaling and Rotation (as per original description)
    theta = math.radians(45)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    centerX, centerY = W / 2.0, L / 2.0
    
    # Scale based on max index 'n' to fit within W/2, L/2 after 45deg rotation
    # A point (n, 0) rotates to (n*c, n*s). Max distance needed is sqrt((W/2)^2+(L/2)^2)
    # Simpler: max coordinate after rotation is n*(cos45+sin45)/sqrt(2) = n? Let's use original scaling logic.
    # Scale so that layer 'n' points reach near the edge.
    # Max distance from center for layer n seems to be 'n' in the integer grid.
    # After rotation, max extent might be complex. Let's assume scaling keeps layer 'n' near boundary.
    scale_x = (W / 2.0) / n if n > 0 else 1.0
    scale_y = (L / 2.0) / n if n > 0 else 1.0
    
    transformed = np.zeros_like(raw_positions)
    raw_x = raw_positions[:, 0]
    raw_y = raw_positions[:, 1]
    
    # Apply rotation
    rotated_x = raw_x * cos_t - raw_y * sin_t
    rotated_y = raw_x * sin_t + raw_y * cos_t

    # Apply scaling and shift to center
    transformed[:, 0] = centerX + rotated_x * scale_x
    transformed[:, 1] = centerY + rotated_y * scale_y
    transformed[:, 2] = H # Mounting height
    transformed[:, 3] = raw_positions[:, 3] # Layer index

    # Filter out positions outside the room (optional sanity check)
    # transformed = transformed[ (transformed[:,0]>=0) & (transformed[:,0]<=W) & \
    #                            (transformed[:,1]>=0) & (transformed[:,1]<=L) ]

    return transformed

def get_zone_info(N, W, L, cob_positions):
    """ Defines zones based on layer radii and finds representative points/areas. """
    center_x, center_y = W / 2.0, L / 2.0
    zone_boundaries = np.zeros(N + 1)
    zone_areas = np.zeros(N)
    zone_rep_points = [] # List of arrays, each array holds points for a zone

    # 1. Calculate max radius for COBs in each layer
    layer_radii = np.zeros(N)
    if cob_positions.shape[0] > 0:
        for i in range(N):
            layer_indices = np.where(cob_positions[:, 3] == i)[0]
            if len(layer_indices) > 0:
                layer_cobs = cob_positions[layer_indices, :2]
                distances = np.sqrt( (layer_cobs[:, 0] - center_x)**2 + \
                                     (layer_cobs[:, 1] - center_y)**2 )
                if len(distances) > 0:
                    layer_radii[i] = np.max(distances)
                else: # Layer exists but no COBs? Use previous radius.
                     layer_radii[i] = layer_radii[i-1] if i > 0 else 0.0
            else: # Layer index doesn't exist in COBs? Use previous.
                 layer_radii[i] = layer_radii[i-1] if i > 0 else 0.0
                 
    # 2. Define zone boundaries (midpoints between layer radii)
    zone_boundaries[0] = 0.0
    for i in range(N - 1):
        midpoint = (layer_radii[i] + layer_radii[i+1]) / 2.0
        # Ensure boundaries increase
        zone_boundaries[i+1] = max(midpoint, zone_boundaries[i] + EPSILON)

    # Final outer boundary (a bit beyond the last layer radius)
    if N > 0 :
       step = zone_boundaries[N-1] - zone_boundaries[N-2] if N > 1 else layer_radii[N-1]
       zone_boundaries[N] = max(layer_radii[N-1] + step * 0.5, zone_boundaries[N-1] + EPSILON)
    else: # Case N=0 or N=1
         zone_boundaries[N] = max(layer_radii[N-1]*1.1 if N>0 else W/2 , zone_boundaries[N-1] + EPSILON)


    # 3. Calculate zone areas and representative points
    angle_rad = math.radians(45) # Use 4 points per zone on diagonals
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    num_rep_points_per_zone = 4 # Using 4 diagonal points

    for j in range(N): # For each zone j
        r_inner = zone_boundaries[j]
        r_outer = zone_boundaries[j+1]

        # Area based on squares defined by max distance 'r' in 45deg orientation
        # Side length = 2*r, Area = (2*r)^2 = 4*r^2
        zone_areas[j] = max(0.0, 4.0 * (r_outer**2 - r_inner**2))

        # Representative points at mid-radius of the zone
        r_mid = (r_inner + r_outer) / 2.0
        points = []
        if r_mid > EPSILON:
            # Points on diagonals
            points.append((center_x + r_mid * cos_a, center_y + r_mid * sin_a, 0.0))
            points.append((center_x - r_mid * cos_a, center_y + r_mid * sin_a, 0.0))
            points.append((center_x + r_mid * cos_a, center_y - r_mid * sin_a, 0.0))
            points.append((center_x - r_mid * cos_a, center_y - r_mid * sin_a, 0.0))
        else: # Zone 0 (center)
            points.append((center_x, center_y, 0.0))
            # Adjust area for center zone if needed? Assume it's annulus formula for now.

        zone_rep_points.append(np.array(points))

    return zone_boundaries, zone_areas, zone_rep_points


@njit(parallel=True)
def calculate_C_approx_numba(N, cob_positions, zone_rep_points, H,
                             angles_deg_data, shape_rel_data, shape_integral,
                             lux_to_ppfd_factor):
    """ Calculates the approximate Layer Contribution Matrix using Numba. """
    C_approx = np.zeros((N, N), dtype=np.float64) # C[zone_j, layer_i]
    peak_intensity_per_lumen = 1.0 / shape_integral if shape_integral > EPSILON else 1.0/np.pi

    for i in prange(N): # Parallelize over source layers
        layer_cob_indices = np.where(cob_positions[:, 3] == i)[0]
        num_cobs_in_layer = len(layer_cob_indices)
        if num_cobs_in_layer == 0:
            continue # No contribution if no COBs

        layer_cobs_pos = cob_positions[layer_cob_indices, :3] # Get xyz

        for j in range(N): # Iterate through target zones
            zone_points = zone_rep_points[j] # Get rep points for this zone
            num_zone_points = zone_points.shape[0]
            if num_zone_points == 0:
                continue

            total_ppfd_contribution = 0.0
            for p_idx in range(num_zone_points): # For each point in zone j
                zp_x, zp_y, zp_z = zone_points[p_idx, 0], zone_points[p_idx, 1], 0.0 # Point on floor

                point_ppfd_sum = 0.0
                for k_idx in range(num_cobs_in_layer): # For each COB k in layer i
                    lc_x, lc_y, lc_z = layer_cobs_pos[k_idx, 0], layer_cobs_pos[k_idx, 1], layer_cobs_pos[k_idx, 2]

                    # Vector from COB to zone point
                    dx = zp_x - lc_x
                    dy = zp_y - lc_y
                    dz = zp_z - lc_z # Should be negative (zp_z=0)

                    dist_sq = dx*dx + dy*dy + dz*dz
                    if dist_sq < EPSILON:
                        dist_sq = EPSILON # Avoid division by zero if point is exactly below COB

                    dist = math.sqrt(dist_sq)

                    # Cosine with COB's downward axis (0,0,-1) and floor normal (0,0,1)
                    cos_theta = -dz / dist # = cos_phi
                    if cos_theta < EPSILON: # Light going sideways or up
                        continue

                    angle_deg = math.degrees(math.acos(cos_theta))

                    # Intensity calculation
                    I_rel = luminous_intensity_rel_numba(angle_deg, angles_deg_data, shape_rel_data)
                    I_theta_per_lumen = peak_intensity_per_lumen * I_rel

                    # Illuminance (Lux-like per lumen) = I * cos / d^2
                    illuminance_per_lumen = (I_theta_per_lumen / dist_sq) * cos_theta

                    # Convert to PPFD-like per lumen
                    ppfd_per_lumen = illuminance_per_lumen * lux_to_ppfd_factor
                    point_ppfd_sum += ppfd_per_lumen

                # Accumulate the average PPFD at this point (from all COBs in layer i)
                total_ppfd_contribution += point_ppfd_sum

            # Average PPFD contribution across all representative points in the zone
            avg_zone_ppfd_contribution = total_ppfd_contribution / num_zone_points
            C_approx[j, i] = avg_zone_ppfd_contribution

    return C_approx


# ------------------------------------
# Optimization Function
# ------------------------------------
def optimize_fluxes(C_approx, zone_areas, N, P_target, L_max):
    """ Solves the linear programming problem using C_approx. """
    print("Optimizing fluxes using C_approx...")
    if C_approx is None or zone_areas is None:
        print("Error: Cannot optimize without valid C_approx matrix and zone areas.")
        return None

    num_vars = N + 1 # [L_0, ..., L_{N-1}, P_min]
    c = np.zeros(num_vars); c[N] = -1.0 # Minimize -P_min

    # Constraints
    # 1. Average PPFD: sum(P_j * Area_j) / sum(Area_j) = P_target
    #    sum_j ( sum_i (C[j, i] * L_i) * Area_j ) = P_target * sum(Area_j)
    #    sum_i ( L_i * sum_j ( C[j, i] * Area_j ) ) = P_target * TotalArea
    total_area = np.sum(zone_areas)
    if total_area < EPSILON:
        print("Error: Total zone area is zero.")
        return None

    A_eq = np.zeros((1, num_vars))
    for i in range(N): # Sum over L_i
        A_eq[0, i] = np.sum(C_approx[:, i] * zone_areas) # Sum over zones j: C[j,i]*Area[j]
    b_eq = np.array([P_target * total_area])

    # 2. Uniformity: P_j >= P_min => P_min - P_j <= 0
    #    P_min - sum(C[j, i] * L_i) <= 0 for each zone j
    A_ub = np.zeros((N, num_vars))
    for j in range(N): # For each zone j (row)
        A_ub[j, 0:N] = -C_approx[j, :] # Coefficients for L_0..L_{N-1}
        A_ub[j, N] = 1.0               # Coefficient for P_min
    b_ub = np.zeros(N)

    # Bounds
    bounds = [(0, L_max) for _ in range(N)] + [(0, None)] # L_i >= 0, P_min >= 0

    # Solve LP
    try:
        result = scipy.optimize.linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            method='highs', options={'disp': False}
        )
    except ValueError as e:
         print(f"Error during optimization setup: {e}")
         print("Check if constraints are feasible (e.g., L_max too low for P_target).")
         return None
    except Exception as e:
         print(f"An unexpected error occurred during optimization: {e}")
         return None

    if result.success:
        print("Optimization successful.")
        L_optimal_approx = result.x[0:N]
        P_min_optimal = result.x[N]
        # Clamp results to bounds just in case solver slightly exceeds them
        L_optimal_approx = np.clip(L_optimal_approx, 0, L_max)
        print(f"  Predicted minimum zonal PPFD (P_min): {P_min_optimal:.2f}")
        estimated_dou = 100 * P_min_optimal / P_target if P_target > EPSILON else 0
        print(f"  Estimated DOU based on P_min: >= {estimated_dou:.2f}%")
        return L_optimal_approx
    else:
        print(f"Optimization failed: {result.message}")
        return None

# ------------------------------------
# Validation Function
# ------------------------------------
def run_validation_simulation(L_optimal_approx, N, H, W, L, sim_script_path):
    """ Runs the full simulation script once with the calculated fluxes. """
    print("\n--- Running Full Validation Simulation ---")
    if not os.path.exists(sim_script_path):
        print(f"Error: Simulation script not found at '{sim_script_path}'")
        return

    flux_str = [f"{f:.4f}" for f in L_optimal_approx]
    output_csv = "ppfd_data_optimized_approx.csv" # Use a distinct name

    cmd = [
        "python", sim_script_path,
        "--num_layers", str(N),
        "--height", str(H),
        # If your script needs W, L uncomment below
        # "--width", str(W),
        # "--length", str(L),
        "--no-plot",
        "--fluxes"
    ] + flux_str + ["--output_csv", output_csv]

    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=900) # 15 min timeout
        print("\nValidation simulation script output:")
        print(result.stdout)
        if result.stderr:
            print("\nValidation simulation script stderr:")
            print(result.stderr)
        print(f"\nFull simulation results saved to: {output_csv}")

    except subprocess.CalledProcessError as e:
        print("\nError running validation simulation script:")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return Code: {e.returncode}")
        print(f"  Stderr: {e.stderr}")
        print(f"  Stdout: {e.stdout}")
    except subprocess.TimeoutExpired as e:
        print(f"\nError: Validation simulation timed out after {e.timeout}s.")
        print(f"  Command: {' '.join(e.cmd)}")
    except FileNotFoundError:
         print(f"\nError: Could not execute command. Is 'python' in your PATH and script path correct?")
         print(f"  Script Path: {sim_script_path}")
    except Exception as e:
        print(f"\nAn unexpected error occurred running validation subprocess: {e}")


# ------------------------------------
# Main Execution Logic
# ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Layer Fluxes using Analytical Approximation")
    parser.add_argument('-N', '--num_layers', type=int, default=11, help="Number of COB layers")
    parser.add_argument('-H', '--height', type=float, default=0.9144, help="Mounting height (m), e.g., 0.9144 for 3ft")
    parser.add_argument('-P', '--ppfd_target', type=float, default=1250, help="Target average PPFD (µmol/m²/s)")
    parser.add_argument('--L_max', type=float, default=20000.0, help="Maximum luminous flux per layer (lumens)")
    parser.add_argument('--force_regenerate_c', action='store_true', help="Force regeneration of C_approx cache")
    parser.add_argument('--skip_validation', action='store_true', help="Skip the final validation simulation run")
    parser.add_argument('--sim_script_path', type=str, default=SIMULATION_SCRIPT_PATH, help="Path to the full simulation script")

    args = parser.parse_args()

    # Update script path if provided
    SIMULATION_SCRIPT_PATH = args.sim_script_path

    # --- 1. Geometry Setup ---
    start_geom = time.time()
    W, L = calculate_dimensions(args.num_layers)
    print(f"Calculated Dimensions: W={W:.4f}m, L={L:.4f}m for N={args.num_layers} layers")
    cob_positions = build_cob_positions(args.num_layers, W, L, args.height)
    if cob_positions.shape[0] == 0 and args.num_layers > 0:
         print(f"Error: No COB positions generated for N={args.num_layers}.")
         exit(1)
    print(f"Generated {cob_positions.shape[0]} COB positions.")
    zone_boundaries, zone_areas, zone_rep_points = get_zone_info(args.num_layers, W, L, cob_positions)
    print(f"Defined {len(zone_rep_points)} zones.")
    print(f"Geometry setup time: {time.time() - start_geom:.2f}s")

    # --- 2. Calculate or Load C_approx ---
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_filename = f"c_approx_N{args.num_layers}_H{args.height:.4f}_W{W:.4f}.npz"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    C_approx = None
    if not args.force_regenerate_c and os.path.exists(cache_path):
        print(f"Loading C_approx from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            # TODO: Add checks here to ensure cached data matches current geometry settings if needed
            if 'C_approx' in data:
                 C_approx = data['C_approx']
                 # Also load zone_areas if stored, or recalculate
                 if 'zone_areas' in data: zone_areas = data['zone_areas']

            else: print("Cache file incomplete. Regenerating.")
        except Exception as e:
            print(f"Error loading cache file {cache_path}: {e}. Regenerating.")

    if C_approx is None:
        print("Calculating Approximate Layer Contribution Matrix (C_approx)...")
        start_calc = time.time()
        C_approx = calculate_C_approx_numba(
            args.num_layers, cob_positions, zone_rep_points, args.height,
            COB_ANGLES_DEG, COB_SHAPE_REL, SHAPE_INTEGRAL,
            LUX_TO_PPFD_FACTOR
        )
        print(f"C_approx calculation time: {time.time() - start_calc:.2f}s")
        # Save to cache
        try:
            np.savez_compressed(cache_path, C_approx=C_approx, zone_areas=zone_areas)
            print(f"C_approx saved to cache: {cache_path}")
        except Exception as e:
            print(f"Error saving C_approx to cache {cache_path}: {e}")

    if C_approx is None or C_approx.shape != (args.num_layers, args.num_layers):
         print("Failed to calculate or load C_approx. Exiting.")
         exit(1)

    # --- 3. Optimize Fluxes ---
    start_opt = time.time()
    L_optimal_approx = optimize_fluxes(
        C_approx, zone_areas, args.num_layers,
        args.ppfd_target, args.L_max
    )
    print(f"Optimization time: {time.time() - start_opt:.2f}s")

    if L_optimal_approx is None:
        print("Failed to find optimal fluxes. Exiting.")
        exit(1)

    print("\n--- Optimal Luminous Flux per Layer (Approximate) ---")
    for i, flux in enumerate(L_optimal_approx):
        print(f"Layer {i}: {flux:.4f}")

    # --- 4. Run Validation Simulation ---
    if not args.skip_validation:
        start_val = time.time()
        run_validation_simulation(
            L_optimal_approx, args.num_layers, args.height, W, L,
            SIMULATION_SCRIPT_PATH
        )
        print(f"Validation simulation call time: {time.time() - start_val:.2f}s")
    else:
        print("\nSkipping validation simulation.")

    print("\nApproximate optimization process complete.")