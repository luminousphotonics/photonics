#!/usr/bin/env python3
import csv
import math
import numpy as np
from numba import njit, prange
from functools import lru_cache
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse # <--- Added
import sys      # <--- Added for error exit
import os       # <--- Added for path checks
import time

# ------------------------------------
# 1) Basic Config & Reflectances (Constants)
# ------------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0  # lumens/W
# Default SPD file path, can be overridden by argument later if needed
DEFAULT_SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv"

MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3

WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

FLOOR_GRID_RES = 0.08  # m
# FIXED_NUM_LAYERS = 19 # <--- REMOVED - Now dynamic via args
MC_SAMPLES = 16

# --- Epsilon for numerical stability ---
EPSILON = 1e-9

# ------------------------------------
# 2) COB Datasheet Angular Data (Constants)
# ------------------------------------
COB_ANGLE_DATA = np.array([
    [  0, 1.00], [ 10, 0.98], [ 20, 0.95], [ 30, 0.88], [ 40, 0.78],
    [ 50, 0.65], [ 60, 0.50], [ 70, 0.30], [ 80, 0.10], [ 90, 0.00],
], dtype=np.float64)
COB_angles_deg = COB_ANGLE_DATA[:, 0]
COB_shape = COB_ANGLE_DATA[:, 1]

# ------------------------------------
# 3) Compute SPD-based µmol/J Factor (Function)
# ------------------------------------
@lru_cache(maxsize=1) # Cache this calculation
def compute_conversion_factor(spd_file):
    """ Computes PPFD conversion factor from SPD data (in µmol/J_rad). """
    if not os.path.exists(spd_file):
         print(f"[ERROR] SPD file not found: {spd_file}", file=sys.stderr)
         # Using a typical PAR factor for white LEDs ~4.57 umol/J_rad as fallback
         return 4.57
    try:
        raw_spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
        if raw_spd.ndim == 1: # Handle case with only one data line after header
             raw_spd = raw_spd.reshape(1, -1)
        if raw_spd.shape[1] != 2:
             raise ValueError("SPD file should have 2 columns: wavelength intensity")
        unique_wl, indices, counts = np.unique(raw_spd[:, 0], return_inverse=True, return_counts=True)
        counts_nonzero = np.maximum(counts, 1)
        avg_intens = np.bincount(indices, weights=raw_spd[:, 1]) / counts_nonzero
        spd = np.column_stack((unique_wl, avg_intens))
    except Exception as e:
        print(f"[ERROR] Error loading/processing SPD file {spd_file}: {e}", file=sys.stderr)
        return 4.57 # Fallback value

    wl = spd[:, 0]; intens = spd[:, 1]; sort_idx = np.argsort(wl); wl = wl[sort_idx]; intens = intens[sort_idx]
    mask_par = (wl >= 400) & (wl <= 700); PAR_fraction = 1.0

    if len(wl) >= 2:
        tot_rad = np.trapz(intens, wl) # Integral W/nm ? Assume relative intensity unit
        if tot_rad > EPSILON:
             if np.count_nonzero(mask_par) >= 2:
                 tot_par_rad = np.trapz(intens[mask_par], wl[mask_par])
                 PAR_fraction = tot_par_rad / tot_rad
             else: print("[SPD Warning] Not enough PAR data points for PAR fraction.", file=sys.stderr)
        else: print("[SPD Warning] Zero total SPD intensity found.", file=sys.stderr)
    else: print("[SPD Warning] Not enough SPD points for integration.", file=sys.stderr)

    wl_m = wl * 1e-9; h, c, N_A = 6.626e-34, 3.0e8, 6.022e23; lambda_eff = 0.0
    if np.count_nonzero(mask_par) >= 2:
        numerator = np.trapz(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
        denominator = np.trapz(intens[mask_par], wl_m[mask_par])
        if denominator > EPSILON: lambda_eff = numerator / denominator

    if lambda_eff <= EPSILON:
        print("[SPD Warning] Could not calculate effective PAR wavelength.", file=sys.stderr)
        if np.count_nonzero(mask_par) > 0:
            lambda_eff = np.mean(wl_m[mask_par])
            print(f"[SPD Info] Using mean PAR wavelength as fallback: {lambda_eff*1e9:.1f} nm", file=sys.stderr)
        else:
            lambda_eff = 550e-9
            print("[SPD Info] Using 550nm as fallback wavelength.", file=sys.stderr)

    E_photon = (h * c / lambda_eff) if lambda_eff > EPSILON else (h*c/550e-9)
    # Conversion factor: (photons/J_rad) * (mol/photons) * (umol/mol) * PAR_energy_fraction
    conversion_factor_umol_J = (1.0 / E_photon) * (1.0 / N_A) * 1e6 * PAR_fraction
    print(f"[INFO] SPD: PAR fraction={PAR_fraction:.3f}, conv_factor={conversion_factor_umol_J:.5f} µmol/J_rad")
    return conversion_factor_umol_J

# Compute factor globally or ensure it's done once in main
CONVERSION_FACTOR_UMOL_J = compute_conversion_factor(DEFAULT_SPD_FILE)
# Factor to convert Lux (lm/m^2) to PPFD (umol/m^2/s)
# PPFD = Lux * (W_rad / lm) * (umol / J_rad) = Lux / LumEfficacy * ConvFactor_umol_J
LUX_TO_PPFD_FACTOR = CONVERSION_FACTOR_UMOL_J / LUMINOUS_EFFICACY

# ------------------------------------
# 4) Pre-Compute Normalization Factor (Function)
# ------------------------------------
# This function remains the same
def integrate_shape_for_flux(angles_deg, shape):
    """ Integrates the relative angular shape function over the sphere. """
    rad_angles = np.radians(angles_deg)
    G = 0.0
    # Ensure data is sorted by angle
    sort_indices = np.argsort(rad_angles)
    rad_angles = rad_angles[sort_indices]
    shape = shape[sort_indices]

    for i in range(len(rad_angles) - 1):
        th0 = rad_angles[i]
        th1 = rad_angles[i+1]
        s0 = shape[i]
        s1 = shape[i+1]
        # Linear interpolation for mean shape in the segment
        s_mean = 0.5*(s0 + s1)
        dtheta = (th1 - th0)
        # Use midpoint angle for sin(theta) weighting in spherical integral segment
        th_mid = 0.5*(th0 + th1)
        sin_mid = math.sin(th_mid)
        # Integral segment: I(theta) * 2*pi*sin(theta)*dtheta
        G_seg = s_mean * 2.0*math.pi * sin_mid * dtheta
        G += G_seg
    # Note: This assumes the shape goes to zero at or before 90 degrees.
    # If shape is non-zero at 90, the last segment might need different handling.
    return G

# Pre-calculate shape integral globally
SHAPE_INTEGRAL = integrate_shape_for_flux(COB_angles_deg, COB_shape)
if abs(SHAPE_INTEGRAL) < EPSILON:
    print("[ERROR] COB Shape Integral is zero. Check COB data. Using fallback Pi.", file=sys.stderr)
    SHAPE_INTEGRAL = np.pi # Fallback for Lambertian-like

@njit
def luminous_intensity(angle_deg, total_lumens):
    """ Calculates luminous intensity (candela) at an angle given total lumens. """
    # Use Numba-compatible interpolation
    if angle_deg <= COB_angles_deg[0]:
        rel = COB_shape[0]
    elif angle_deg >= COB_angles_deg[-1]:
        rel = COB_shape[-1]
    else:
        # Manual linear interpolation for Numba
        rel = np.interp(angle_deg, COB_angles_deg, COB_shape) # Numba interp works!

    # Intensity = TotalFlux * RelativeIntensity / IntegralOfRelativeIntensity
    peak_intensity = total_lumens / SHAPE_INTEGRAL
    return peak_intensity * rel


# ------------------------------------
# 5) Geometry Building (Functions accept num_layers)
# ------------------------------------
def calculate_dimensions(num_layers):
    """ Calculates room width/length based on number of layers. """
    # Based on user-provided scaling rule
    base_n = 10
    base_floor = 6.10 # meters
    scale_factor = 0.64 # meters per layer increase from base

    if num_layers < base_n:
         print(f"[Info] num_layers ({num_layers}) < base {base_n}. Using base floor size {base_floor}m.")
         W = base_floor
    else:
         W = base_floor + (num_layers - base_n) * scale_factor
    L = W # Assuming square room
    return W, L

def prepare_geometry(W, L, H, num_layers):
    """ Builds all geometric components needed for simulation. """
    # Pass num_layers to build_cob_positions
    cob_positions = build_cob_positions(W, L, H, num_layers)
    X, Y = build_floor_grid(W, L)
    patches = build_patches(W, L, H)
    return (cob_positions, X, Y, patches)

# REPLACE the existing build_cob_positions function in lighting_simulation_data.py

def build_cob_positions(W, L, H, num_layers):
    """
    Builds the COB positions array including the layer index, using the
    centered square pattern definition where layer 'i' contains points (x,y)
    such that max(|x|, |y|) = i.
    Corrected based on sequence 1, 5, 13, 25...
    """
    n = num_layers - 1 # Max layer index (0 to N-1)
    if n < 0: # Handle case of 0 layers
        return np.empty((0,4), dtype=np.float64)

    # Generate integer grid positions based on max(|x|,|y|) = layer_index
    positions_int_maxabs = []
    expected_count = 0
    for i in range(n + 1): # Layer index i = 0..n
        layer_points_count = 0
        if i == 0:
            positions_int_maxabs.append((0, 0, i)) # Add layer index 'i'
            layer_points_count = 1
        else:
            # Layer i > 0 has max(|x|, |y|) == i
            # Iterate through the square perimeter defined by i
            # Top edge (y=i), x from -i to i
            for x in range(-i, i + 1):
                 positions_int_maxabs.append((x, i, i))
                 layer_points_count += 1
            # Bottom edge (y=-i), x from -i to i
            for x in range(-i, i + 1):
                 # Avoid double counting corners if handled by top edge already?
                 # Let's add all then remove duplicates later. Seems safer.
                 positions_int_maxabs.append((x, -i, i))
                 layer_points_count +=1 # Counts potential duplicates here
            # Left edge (x=-i), y from -i+1 to i-1 (avoid corners)
            for y in range(-i + 1, i):
                 positions_int_maxabs.append((-i, y, i))
                 layer_points_count += 1
            # Right edge (x=i), y from -i+1 to i-1 (avoid corners)
            for y in range(-i + 1, i):
                 positions_int_maxabs.append((i, y, i))
                 layer_points_count += 1
        # print(f"[Debug] Raw points added for layer {i}: {layer_points_count}") # Debug

    # Remove duplicates (important!) and add H coordinate
    # Store as (x, y, layer_index) before making unique
    unique_positions_set = set([(p[0], p[1], p[2]) for p in positions_int_maxabs])
    raw_positions_list = [[p[0], p[1], H, p[2]] for p in unique_positions_set] # Add H and layer index p[2]

    # Sort primarily by layer, then x, then y for consistent ordering
    raw_positions_list.sort(key=lambda p: (p[3], p[0], p[1]))
    raw_positions = np.array(raw_positions_list, dtype=np.float64)

    # --- Verify Count ---
    # Formula for total points: 2n(n+1)+1 where n = num_layers - 1
    expected_total_count = 2*n*(n+1)+1 if n >= 0 else 0
    actual_count = raw_positions.shape[0]
    if actual_count != expected_total_count:
         print(f"[ERROR] COB count mismatch! Expected {expected_total_count}, Got {actual_count} for N={num_layers} (n={n}). Check generation logic.", file=sys.stderr)
         # This indicates the generation logic above is still not perfectly matching the formula.
         # However, the user's original description generation logic *did* match the count sequence. Let's retry THAT one.

    # --- Revert to User's original description logic which matched the sequence counts ---
    print("[Info] Using original COB generation logic (diamond shape) which matched sequence counts.")
    positions_original_logic = []
    if n >= 0:
        positions_original_logic.append((0, 0, H, 0)) # Layer 0
        for i in range(1, n + 1): # Layer i
            for x in range(-i, i + 1): # x coord
                y_abs = i - abs(x) # y magnitude for diamond shape
                if y_abs == 0: # Point is on x-axis (x,0)
                    if x!=0 : positions_original_logic.append((x, 0, H, i))
                    # else: pass # Center (0,0) handled by layer 0
                else: # Points are (x, y_abs) and (x, -y_abs)
                    positions_original_logic.append((x, y_abs, H, i))
                    positions_original_logic.append((x, -y_abs, H, i))
    # Convert to numpy array (no duplicates expected with this logic)
    raw_positions = np.array(positions_original_logic, dtype=np.float64)

    # Verify count again with this logic
    actual_count = raw_positions.shape[0]
    if actual_count != expected_total_count and num_layers > 0 : # Check only if N>0
         print(f"[ERROR] Original Logic COB count mismatch! Expected {expected_total_count}, Got {actual_count} for N={num_layers} (n={n}).", file=sys.stderr)


    # Apply transformation (Rotation/Scaling) - This part remains the same
    theta = math.radians(45)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    centerX, centerY = W / 2.0, L / 2.0

    if n > 0:
         # Use the scaling factor derivation from user's code
         scale_x = (W / 2.0 * math.sqrt(2)) / n
         scale_y = (L / 2.0 * math.sqrt(2)) / n
    else: # Only layer 0 (N=1)
         scale_x = 1.0
         scale_y = 1.0

    transformed = np.zeros_like(raw_positions)
    # Ensure raw_positions isn't empty before accessing columns
    if actual_count > 0:
        int_x = raw_positions[:, 0]
        int_y = raw_positions[:, 1]
        # Apply rotation
        rotated_x = int_x * cos_t - int_y * sin_t
        rotated_y = int_x * sin_t + int_y * cos_t
        # Apply scaling and shift to center
        transformed[:, 0] = centerX + rotated_x * scale_x
        transformed[:, 1] = centerY + rotated_y * scale_y
        transformed[:, 2] = H # Actual COB height
        transformed[:, 3] = raw_positions[:, 3] # Layer index
    else: # Handle N=0 case if needed, although n<0 check should catch this
         transformed = np.empty((0,4), dtype=np.float64)


    print(f"[Info] Generated {transformed.shape[0]} COB positions (Expected: {expected_total_count}).")
    return transformed # Return the final transformed positions

def pack_luminous_flux_dynamic(layer_flux_params, cob_positions):
    """ Assigns flux to each COB based on its layer index. """
    num_cobs = cob_positions.shape[0]
    num_layers = len(layer_flux_params)
    led_intensities = np.zeros(num_cobs, dtype=np.float64)

    if num_cobs == 0: return led_intensities # Return empty if no COBs

    for k in range(num_cobs):
        layer_index = int(cob_positions[k, 3])
        if 0 <= layer_index < num_layers:
            led_intensities[k] = layer_flux_params[layer_index]
        else:
            # This shouldn't happen if build_cob_positions is correct
            print(f"[Warning] COB {k} has invalid layer index {layer_index} (max: {num_layers-1}). Assigning 0 flux.", file=sys.stderr)
            led_intensities[k] = 0.0
    return led_intensities

@lru_cache(maxsize=4) # Cache based on W, L
def build_floor_grid(W: float, L: float):
    """ Builds the floor grid coordinates. """
    num_x = int(round(W / FLOOR_GRID_RES)) + 1
    num_y = int(round(L / FLOOR_GRID_RES)) + 1
    xs = np.linspace(0, W, num_x)
    ys = np.linspace(0, L, num_y)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

@lru_cache(maxsize=4) # Cache based on W, L, H
def build_patches(W: float, L: float, H: float):
    """ Builds radiosity patches for walls, ceiling, and floor. """
    # Floor patch (single patch for simplicity, index 0)
    patch_centers = [(W/2, L/2, 0.0)]
    patch_areas = [W * L]
    patch_normals = [(0.0, 0.0, 1.0)] # Normal points up
    patch_refl = [REFL_FLOOR]

    # Ceiling patches
    xs_ceiling = np.linspace(0, W, CEIL_SUBDIVS_X + 1)
    ys_ceiling = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_ceiling[i] + xs_ceiling[i+1]) / 2
            cy = (ys_ceiling[j] + ys_ceiling[j+1]) / 2
            area = (xs_ceiling[i+1] - xs_ceiling[i]) * (ys_ceiling[j+1] - ys_ceiling[j])
            patch_centers.append((cx, cy, H)) # Assuming ceiling is at height H
            patch_areas.append(area)
            patch_normals.append((0.0, 0.0, -1.0)) # Normal points down
            patch_refl.append(REFL_CEIL)

    # Wall patches (4 walls)
    # Wall Y=0 (Normal +Y)
    xs_wall = np.linspace(0, W, WALL_SUBDIVS_X + 1)
    zs_wall = np.linspace(0, H, WALL_SUBDIVS_Y + 1)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, 0.0, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, 1.0, 0.0)) # Normal points into room (+Y)
            patch_refl.append(REFL_WALL)

    # Wall Y=L (Normal -Y)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, L, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, -1.0, 0.0)) # Normal points into room (-Y)
            patch_refl.append(REFL_WALL)

    # Wall X=0 (Normal +X)
    ys_wall = np.linspace(0, L, WALL_SUBDIVS_X + 1) # Re-use subdivisions number
    for i in range(WALL_SUBDIVS_X): # Use same subdivision count for consistency
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((0.0, cy, cz))
            patch_areas.append(area)
            patch_normals.append((1.0, 0.0, 0.0)) # Normal points into room (+X)
            patch_refl.append(REFL_WALL)

    # Wall X=W (Normal -X)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((W, cy, cz))
            patch_areas.append(area)
            patch_normals.append((-1.0, 0.0, 0.0)) # Normal points into room (-X)
            patch_refl.append(REFL_WALL)

    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

# ------------------------------------
# 6) The Numba-JIT Computations (Unchanged in core logic)
# ------------------------------------
@njit(parallel=True)
def compute_direct_floor(cob_positions, cob_lumens, X, Y, H_cobs):
    """ Computes direct illuminance (lux) on the floor grid. """
    # Use H_cobs which is the actual height of COBs passed down
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2 # Use smaller minimum distance
    rows, cols = X.shape
    out = np.zeros_like(X, dtype=np.float64)

    for r in prange(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            fz = 0.0 # Floor is at z=0
            val = 0.0
            for k in range(cob_positions.shape[0]):
                lx = cob_positions[k, 0]
                ly = cob_positions[k, 1]
                lz = H_cobs # Use the actual COB height
                lumens_k = cob_lumens[k]

                if lumens_k < EPSILON: continue # Skip COBs with no flux

                dx = fx - lx
                dy = fy - ly
                dz = fz - lz # Vector from COB to floor point (dz will be negative)

                d2 = dx*dx + dy*dy + dz*dz
                if d2 < min_dist2: d2 = min_dist2
                dist = math.sqrt(d2)

                # Cosine between COB emission direction (downward, 0,0,-1) and vector to point
                cos_th_emission = -dz / dist
                # Cosine between floor normal (upward, 0,0,1) and vector from point to COB (-dx,-dy,-dz)
                cos_th_incidence = -dz / dist # Same value for horizontal floor

                if cos_th_emission <= EPSILON: continue # Light going sideways or up

                angle_deg = math.degrees(math.acos(cos_th_emission))
                I_theta = luminous_intensity(angle_deg, lumens_k) # Intensity in candela

                # Illuminance = I * cos(incidence) / distance^2
                E_local = (I_theta / d2) * cos_th_incidence
                val += E_local
            out[r, c] = val
    return out

@njit
def compute_patch_direct(cob_positions, cob_lumens, patch_centers, patch_normals, H_cobs):
    """ Computes direct illuminance on radiosity patches. """
    min_dist2 = (FLOOR_GRID_RES / 4.0) ** 2
    Np = patch_centers.shape[0]
    out = np.zeros(Np, dtype=np.float64)

    for ip in range(Np): # For each patch
        pcx, pcy, pcz = patch_centers[ip, 0], patch_centers[ip, 1], patch_centers[ip, 2]
        nx, ny, nz = patch_normals[ip, 0], patch_normals[ip, 1], patch_normals[ip, 2]
        norm_n = math.sqrt(nx*nx + ny*ny + nz*nz) # Should be 1 if normals are normalized
        if norm_n < EPSILON: continue # Skip invalid patch normal

        accum = 0.0
        for k in range(cob_positions.shape[0]): # For each COB
            lx, ly, lz = cob_positions[k, 0], cob_positions[k, 1], H_cobs
            lumens_k = cob_lumens[k]

            if lumens_k < EPSILON: continue

            # Vector from COB to patch center
            dx = pcx - lx
            dy = pcy - ly
            dz = pcz - lz
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_dist2: d2 = min_dist2
            dist = math.sqrt(d2)

            # Cosine for COB emission angle (relative to COB's down axis 0,0,-1)
            # Use vector from COB to patch (dx, dy, dz)
            # COB axis is (0,0,-1). Dot product is -dz.
            cos_th_emission = -dz / dist
            if cos_th_emission <= EPSILON: continue # Light going wrong way from COB

            angle_deg = math.degrees(math.acos(cos_th_emission))
            I_theta = luminous_intensity(angle_deg, lumens_k)

            # Cosine for incidence angle on patch
            # Dot product of vector from COB (dx,dy,dz) and patch normal (nx,ny,nz)
            dot_patch = dx*nx + dy*ny + dz*nz
            # Ensure light hits front face (dot product > 0 if normal points outward)
            # Or dot product < 0 if normal points inward (as defined here)? Check normal directions.
            # Normals point INTO the room volume. Vector COB->Patch points INTO room.
            # Dot product should be POSITIVE for light hitting the patch face.
            if dot_patch <= EPSILON: continue # Hits back face or edge-on

            cos_in_patch = dot_patch / (dist * norm_n)

            E_local = (I_theta / d2) * cos_in_patch
            accum += E_local
        out[ip] = accum
    return out

@njit
def form_factor(pi, ni, area_i, pj, nj, area_j):
    """ Calculates the form factor FROM patch i TO patch j using Numba. """
    dx = pj[0] - pi[0]
    dy = pj[1] - pi[1]
    dz = pj[2] - pi[2]
    dist_sq = dx * dx + dy * dy + dz * dz

    if dist_sq < EPSILON: return 0.0
    dist = math.sqrt(dist_sq)

    # Cosine term for patch i (outgoing)
    dot_i = ni[0] * dx + ni[1] * dy + ni[2] * dz
    cos_i = dot_i / dist # Assume normals are normalized
    if cos_i <= EPSILON: return 0.0 # Facing away or edge-on

    # Cosine term for patch j (incoming)
    dot_j = nj[0] * (-dx) + nj[1] * (-dy) + nj[2] * (-dz) # Vector points from j to i
    cos_j = dot_j / dist # Assume normals are normalized
    if cos_j <= EPSILON: return 0.0 # Facing away or edge-on

    # Form Factor formula: (cos_i * cos_j * Area_j) / (pi * dist_sq)
    # We need F_ij (i to j). Formula is (cos_theta_i * cos_theta_j) / (pi * r^2) * dA_j
    # Integrate over dA_j -> * Area_j IF patches are small relative to distance
    ff = (cos_i * cos_j) / (math.pi * dist_sq) * area_j # Approximated FF

    # Visibility check (crude: assume visible if cosines > 0) - Can add simple obstruction checks if needed

    return max(0.0, ff) # Ensure non-negative


@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct_lux, patch_areas, patch_refl,
                             max_bounces, convergence_threshold):
    """ Performs the iterative radiosity calculation. Inputs/Outputs in Lux. """
    Np = patch_direct_lux.shape[0]
    # Initial radiosity B = Emitted (0 for patches) + Reflected = Rho * Incident
    # Incident = Direct E_direct. So initial B = Rho * E_direct ? No.
    # Radiosity B is total flux leaving per unit area (W/m^2 or lm/m^2)
    # B = E_emitted + Rho * E_incident_total
    # E_incident_total = E_direct + E_reflected
    # For non-emitting patches, B = Rho * (E_direct + E_reflected)

    # We track total incident flux (Exitance E in Lux) on each patch
    patch_E_total = patch_direct_lux.copy() # Start with direct illumination
    patch_B = np.zeros(Np, dtype=np.float64) # Radiosity (lm/m^2 leaving the surface)

    print_interval = 1 # Print progress every N bounces

    for bounce in range(max_bounces):
        E_incident_reflected = np.zeros(Np, dtype=np.float64) # Incident flux for this bounce

        # Calculate radiosity B for all patches based on previous E_total
        for k in range(Np):
            # Radiosity B = Reflectivity * Total Incident Exitance E
            patch_B[k] = patch_refl[k] * patch_E_total[k]

        # Distribute radiosity from each patch j to all other patches i
        for j in range(Np): # Source patch
            if patch_B[j] < EPSILON or patch_areas[j] < EPSILON: continue

            # Total flux leaving patch j = B_j * Area_j
            flux_leaving_j = patch_B[j] * patch_areas[j]

            pj = patch_centers[j]; nj = patch_normals[j]; area_j = patch_areas[j]

            for i in range(Np): # Receiving patch
                if i == j: continue

                pi = patch_centers[i]; ni = patch_normals[i]; area_i = patch_areas[i]

                # Calculate form factor F_ji (from j to i)
                F_ji = form_factor(pj, nj, area_j, pi, ni, area_i) # Note: FF approx uses Area_i at end

                # Flux arriving at patch i from patch j = Flux_leaving_j * F_ji
                flux_arriving_i = flux_leaving_j * F_ji

                # Add to incident flux for this bounce, convert to Exitance E (Lux)
                if area_i > EPSILON:
                     E_incident_reflected[i] += flux_arriving_i / area_i

        # Update total incident Exitance E and check convergence
        new_patch_E_total = patch_direct_lux + E_incident_reflected
        max_rel_change = 0.0
        if bounce > 0: # Check convergence after first bounce
             delta_E = np.abs(new_patch_E_total - patch_E_total)
             # Avoid division by zero if E_total is zero
             max_rel_change = np.max(delta_E / (np.abs(patch_E_total) + EPSILON))

        patch_E_total = new_patch_E_total # Update for next iteration

        #if bounce % print_interval == 0 or bounce == max_bounces - 1:
        #    print(f"  Radiosity Bounce {bounce+1}, Max Rel Change: {max_rel_change:.6f}")

        if max_rel_change < convergence_threshold and bounce > 0:
            print(f"  Radiosity converged after {bounce+1} bounces.")
            break
    else: # Loop finished without break
         print(f"  Radiosity did not converge within {max_bounces} bounces (Max Change: {max_rel_change:.6f}).")

    # Final Radiosity B based on the converged total incident E
    final_patch_B = patch_refl * patch_E_total
    return final_patch_B # Return final radiosity B (lm/m^2 leaving surface)


# Using joblib as in the original script - Check if joblib is truly needed/effective here
# For MC samples on floor, might be better to use Numba parallel loop if possible
# Numba parallel loop for reflection calculation
@njit(parallel=True)
def compute_reflection_on_floor_numba(X, Y, patch_centers, patch_normals, patch_areas, final_patch_B,
                                     mc_samples=MC_SAMPLES):
    """ Calculates reflected illuminance (Lux) on the floor grid using Numba. """
    rows, cols = X.shape
    out = np.zeros_like(X, dtype=np.float64)
    Np = patch_centers.shape[0]

    # Floor patch is usually index 0, reflections come from patches 1 to Np-1
    # Iterate over floor points
    for r in prange(rows):
        for c in range(cols):
            fx, fy, fz = X[r, c], Y[r, c], 0.0 # Floor point
            n_floor = np.array([0.0, 0.0, 1.0]) # Floor normal (upward)
            accum_E_reflected = 0.0

            # Sum contributions from all patches (excluding floor itself if desired, but B_floor should be low anyway)
            for j in range(Np): # Source patch j
                if final_patch_B[j] < EPSILON or patch_areas[j] < EPSILON: continue
                # Skip floor patch itself as source? B should be low due to low Rho_floor
                # if j == 0: continue

                pj = patch_centers[j]
                nj = patch_normals[j] # Normal points into room volume
                area_j = patch_areas[j]
                Bj = final_patch_B[j] # Radiosity (lm/m^2)

                # Vector from patch center j to floor point (fx,fy,0)
                dx = fx - pj[0]
                dy = fy - pj[1]
                dz = fz - pj[2] # = -pj[2]
                dist_sq = dx*dx + dy*dy + dz*dz
                if dist_sq < EPSILON: continue
                dist = math.sqrt(dist_sq)

                # Cosine at source patch j (normal nj) and vector to floor point
                cos_j = (nj[0]*dx + nj[1]*dy + nj[2]*dz) / dist # Assume nj normalized
                if cos_j <= EPSILON: continue # Patch j facing away

                # Cosine at floor point (normal n_floor) and vector from floor to patch (-dx,-dy,-dz)
                cos_floor = (n_floor[0]*(-dx) + n_floor[1]*(-dy) + n_floor[2]*(-dz)) / dist
                if cos_floor <= EPSILON: continue # Floor point facing away

                # Differential form factor dF_j->floor_point = (cos_j * cos_floor) / (pi * dist^2) * dA_floor
                # Irradiance E at floor point from patch j = Integral( Bj * dF_j->dA_floor )
                # Approximated as: Bj * (cos_j * cos_floor * Area_j) / (pi * dist^2) ? No.
                # Irradiance E = Bj * F_dA_j -> dA_floor * Area_j? No.
                # Irradiance E at point P from Area A = Integral L cos_p cos_a / r^2 dA
                # If source is Lambertian (Radiosity B = pi * Luminance L), L = B/pi
                # E = Integral (B/pi) cos_p cos_a / r^2 dA
                # For small patch j relative to distance: E approx Bj * (cos_j * cos_floor / (pi * dist^2)) * Area_j

                # Contribution to floor illuminance E = Radiosity_j * FormFactor_dAj->dA_floor
                # Using the approximation for small patch j:
                E_contribution = Bj * (cos_j * cos_floor / (math.pi * dist_sq)) * area_j

                # Monte Carlo approach (if needed for larger patches / more accuracy)
                # Would sample points on patch j, calculate individual contributions, average
                # For now, using the center-to-point approximation:
                accum_E_reflected += max(0.0, E_contribution)

            out[r, c] = accum_E_reflected
    return out

# --- Joblib version (kept for reference, but Numba is likely faster) ---
# def compute_reflection_on_floor_joblib(...)
# ... Implementation using Parallel(n_jobs=-1)(delayed(compute_row_reflection)(...)) ...


# ------------------------------------
# 7) Heatmap Plotting Function
# ------------------------------------
def plot_heatmap(floor_ppfd, X, Y, W, L, cob_positions, title="Floor PPFD Heatmap"):
    """ Plots the PPFD heatmap with COB positions. """
    fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted size
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    # Choose appropriate color limits (e.g., slightly below min, above max, or fixed range)
    vmin = np.percentile(floor_ppfd, 1) if np.size(floor_ppfd)>0 else 0
    vmax = np.percentile(floor_ppfd, 99) if np.size(floor_ppfd)>0 else 1500
    if vmax < vmin + EPSILON : vmax = vmin + 1.0 # Ensure range is valid

    heatmap = ax.imshow(floor_ppfd, cmap='viridis', interpolation='nearest', # 'hot' is okay too
                        origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(heatmap, ax=ax, label='PPFD (µmol/m²/s)')

    # Optional: Add contour lines for better visualization of levels
    # levels = np.linspace(vmin, vmax, 10)
    # try:
    #    ax.contour(X, Y, floor_ppfd, levels=levels, colors='white', alpha=0.5, linewidths=0.5)
    # except ValueError as e:
    #    print(f"[Plot Warning] Could not draw contours: {e}", file=sys.stderr)


    # Plot COB positions if available
    if cob_positions is not None and cob_positions.shape[0] > 0:
        ax.scatter(cob_positions[:, 0], cob_positions[:, 1], marker='+',
                   color='red', s=20, label="COB positions", alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(0, W)
    ax.set_ylim(0, L)
    ax.set_aspect('equal', adjustable='box') # Keep aspect ratio
    # ax.legend() # Legend might obscure data, COB markers usually clear
    plt.tight_layout()
    # plt.show(block=False) # Use block=True if running interactively and want plot to stay
    # plt.pause(0.1) # Needed if block=False
    plt.show() # Default to blocking show


# ------------------------------------
# 8) CSV Output Function (Accepts num_layers)
# ------------------------------------
def write_ppfd_to_csv(filename, floor_ppfd, X, Y, cob_positions, W, L, num_layers):
    """
    Writes PPFD data to CSV, organized by distance-based rings (layers/zones),
    using boundaries defined between COB layer radii. Accepts num_layers.
    """
    layer_data = {} # Using dict for flexibility, key is layer index
    # Initialize lists for all expected layers based on num_layers
    for i in range(num_layers):
        layer_data[i] = []

    center_x, center_y = W / 2.0, L / 2.0

    # --- Calculate Layer Radii (Max distance of COBs in that layer) ---
    layer_radii = np.zeros(num_layers)
    if cob_positions.shape[0] > 0:
        # Ensure cob layer indices are integers
        cob_layer_indices = cob_positions[:, 3].astype(int)
        max_cob_layer_index = np.max(cob_layer_indices) if len(cob_layer_indices) > 0 else -1

        if max_cob_layer_index >= num_layers:
             print(f"[ERROR] COB data contains layer index {max_cob_layer_index} "
                   f"which is >= specified num_layers {num_layers}. Check COB generation.", file=sys.stderr)
             # Depending on severity, either exit or try to proceed carefully
             # For now, proceed but results might be wrong.
             # return # Or sys.exit(1)

        for i in range(num_layers):
            # Find COBs belonging to the current layer index i
            indices_in_layer = np.where(cob_layer_indices == i)[0]
            if len(indices_in_layer) > 0:
                layer_cobs_xy = cob_positions[indices_in_layer, :2]
                distances = np.sqrt((layer_cobs_xy[:, 0] - center_x)**2 +
                                    (layer_cobs_xy[:, 1] - center_y)**2)
                if len(distances) > 0:
                    layer_radii[i] = np.max(distances)
                else: # Should not happen if indices_in_layer is not empty
                     layer_radii[i] = layer_radii[i-1] if i > 0 else 0.0
            else:
                # Layer i might be intentionally empty, or indexing error
                # Use previous layer's radius as a fallback assumption
                layer_radii[i] = layer_radii[i-1] if i > 0 else 0.0
                # print(f"[Info] No COBs found for layer {i}. Using radius {layer_radii[i]:.3f}m from previous.")

    # --- Define Sampling Boundaries (midpoints between layer radii) ---
    sampling_boundaries = np.zeros(num_layers + 1)
    sampling_boundaries[0] = 0.0 # Zone 0 starts at center
    for i in range(num_layers - 1):
        midpoint = (layer_radii[i] + layer_radii[i+1]) / 2.0
        # Ensure boundaries strictly increase
        sampling_boundaries[i+1] = max(midpoint, sampling_boundaries[i] + EPSILON)

    # Define the final outer boundary
    if num_layers > 0:
        # Use a radius slightly larger than the outermost COB radius
        # Estimate step size based on last two boundaries
        step = sampling_boundaries[num_layers-1] - sampling_boundaries[num_layers-2] if num_layers > 1 else layer_radii[num_layers-1]*0.1
        final_boundary = layer_radii[num_layers-1] + max(step * 0.5, FLOOR_GRID_RES) # Add half step or grid res
        sampling_boundaries[num_layers] = max(final_boundary, sampling_boundaries[num_layers-1] + EPSILON)
    else: # No layers
        sampling_boundaries[num_layers] = max(W/2, L/2) # Boundary is room edge

    # print("[Debug] Sampling Boundaries:", sampling_boundaries) # Debugging print

    # --- Assign Floor Points to Layers/Zones based on Boundaries ---
    rows, cols = floor_ppfd.shape
    points_assigned = 0
    points_unassigned = 0
    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            dist_to_center = math.sqrt((fx - center_x)**2 + (fy - center_y)**2)

            assigned_layer = -1
            # Find which zone the point falls into
            for i in range(num_layers):
                inner_b = sampling_boundaries[i]
                outer_b = sampling_boundaries[i+1]

                # Check if distance is within [inner_b, outer_b)
                # Handle center point exactly at 0 for layer 0
                is_match = False
                if i == 0 and abs(dist_to_center) < EPSILON:
                    is_match = True
                elif inner_b <= dist_to_center < outer_b:
                    is_match = True
                # Include points exactly on the outermost boundary in the last layer
                elif i == num_layers - 1 and abs(dist_to_center - outer_b) < EPSILON:
                     is_match = True


                if is_match:
                    assigned_layer = i
                    break

            if assigned_layer != -1:
                # Check if key exists (it should due to initialization)
                if assigned_layer in layer_data:
                     layer_data[assigned_layer].append(floor_ppfd[r, c])
                     points_assigned += 1
                else: # Should not happen
                     print(f"[Warning] Layer key {assigned_layer} not found in layer_data dict.", file=sys.stderr)
                     points_unassigned += 1
            else:
                 # Point might be outside the last boundary?
                 # print(f"[Info] Point ({fx:.2f},{fy:.2f}) dist {dist_to_center:.2f} not assigned to any layer.")
                 points_unassigned += 1

    print(f"[Info] Assigned {points_assigned} floor points to zones; {points_unassigned} unassigned.")

    # --- Write Data to CSV ---
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer', 'PPFD']) # Header
            # Iterate through layers 0 to num_layers-1
            for layer_idx in range(num_layers):
                if layer_idx in layer_data and layer_data[layer_idx]: # Check if list not empty
                    for ppfd_value in layer_data[layer_idx]:
                        writer.writerow([layer_idx, f"{ppfd_value:.8f}"]) # Write with more precision
                # else: # Optional: Write a row indicating empty layer?
                #    writer.writerow([layer, 0.0]) # Or skip empty layers
    except IOError as e:
        print(f"[ERROR] Could not write to CSV file '{filename}': {e}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during CSV writing: {e}", file=sys.stderr)


# ------------------------------------
# 9) Simulation Wrapper Function
# ------------------------------------
def simulate_lighting(layer_flux_params, geo, H_cobs):
    """ Runs the core simulation steps. """
    cob_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geo

    # Pack fluxes onto individual COBs
    lumens_per_cob = pack_luminous_flux_dynamic(layer_flux_params, cob_positions)

    # --- Direct Calculation ---
    print("[Info] Calculating Direct Illumination...")
    start_direct = time.time()
    floor_lux_direct = compute_direct_floor(cob_positions, lumens_per_cob, X, Y, H_cobs)
    print(f"[Info] Direct floor calculation took: {time.time() - start_direct:.2f}s")

    print("[Info] Calculating Direct Patch Illumination...")
    start_patch_direct = time.time()
    patch_direct_lux = compute_patch_direct(cob_positions, lumens_per_cob, p_centers, p_normals, H_cobs)
    print(f"[Info] Direct patch calculation took: {time.time() - start_patch_direct:.2f}s")

    # --- Indirect Calculation (Radiosity) ---
    print("[Info] Starting Radiosity Calculation...")
    start_radiosity = time.time()
    final_patch_B = iterative_radiosity_loop(p_centers, p_normals, patch_direct_lux, p_areas, p_refl,
                                            MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD)
    print(f"[Info] Radiosity calculation took: {time.time() - start_radiosity:.2f}s")

    print("[Info] Calculating Reflected Illumination on Floor...")
    start_reflect = time.time()
    # Use the faster Numba version
    reflect_floor_lux = compute_reflection_on_floor_numba(X, Y, p_centers, p_normals, p_areas, final_patch_B, MC_SAMPLES)
    # reflect_floor_lux = compute_reflection_on_floor_joblib(...) # If using joblib
    print(f"[Info] Reflection calculation took: {time.time() - start_reflect:.2f}s")

    # --- Combine and Convert ---
    total_luminous_lux = floor_lux_direct + reflect_floor_lux

    # Convert total Lux to PPFD
    floor_ppfd = total_luminous_lux * LUX_TO_PPFD_FACTOR

    # Debugging: check ranges
    # print(f"[Debug] Min/Max Direct Lux: {np.min(floor_lux_direct):.2f}/{np.max(floor_lux_direct):.2f}")
    # print(f"[Debug] Min/Max Reflected Lux: {np.min(reflect_floor_lux):.2f}/{np.max(reflect_floor_lux):.2f}")
    # print(f"[Debug] Min/Max Total PPFD: {np.min(floor_ppfd):.2f}/{np.max(floor_ppfd):.2f}")


    return floor_ppfd, X, Y, cob_positions


# ------------------------------------
# 10) Main Execution Block
# ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lighting Simulation Script")
    parser.add_argument('--num_layers', type=int, required=True, help='Number of COB layers (N)')
    parser.add_argument('--height', type=float, default=0.9144, help='Mounting height (H) of COBs in meters (e.g., 0.9144 for 3ft)')
    # Removed width/length args, now calculated from num_layers
    parser.add_argument('--fluxes', nargs='+', type=float, required=True, help='List of luminous flux values per layer (space separated)')
    parser.add_argument('--no-plot', action='store_true', help='Disable the heatmap plot')
    parser.add_argument('--output_csv', type=str, default="ppfd_data.csv", help='Output CSV filename')
    # Optional: Add argument for SPD file path if needed
    # parser.add_argument('--spd_file', type=str, default=DEFAULT_SPD_FILE, help='Path to SPD data file')

    args = parser.parse_args()

    # --- Get Parameters ---
    num_layers_from_args = args.num_layers
    layer_fluxes_from_args = np.array(args.fluxes, dtype=np.float64)
    H_cobs = args.height # Use height directly as COB mounting height
    output_csv_filename = args.output_csv

    print(f"--- Running Simulation ---")
    print(f"Num Layers (N): {num_layers_from_args}")
    print(f"COB Height (H): {H_cobs:.4f} m")
    print(f"Output CSV: {output_csv_filename}")
    # print(f"Input Fluxes: {layer_fluxes_from_args}") # Can be very long

    # --- Validate Input ---
    if num_layers_from_args < 0:
         print(f"[ERROR] Number of layers cannot be negative ({num_layers_from_args})", file=sys.stderr)
         sys.exit(1)
    if len(layer_fluxes_from_args) != num_layers_from_args:
        print(f"[ERROR] Number of fluxes provided ({len(layer_fluxes_from_args)}) does not match number of layers ({num_layers_from_args})", file=sys.stderr)
        sys.exit(1)
    if H_cobs <= 0:
         print(f"[ERROR] COB Height must be positive ({H_cobs})", file=sys.stderr)
         sys.exit(1)

    # --- Calculate Geometry ---
    W, L = calculate_dimensions(num_layers_from_args)
    print(f"Calculated Dimensions: W={W:.4f}m, L={L:.4f}m")

    # --- Prepare Geometry ---
    print("[Info] Preparing Geometry...")
    start_geom = time.time()
    try:
        # Pass num_layers down
        geo = prepare_geometry(W, L, H_cobs, num_layers_from_args)
    except Exception as e:
        print(f"[ERROR] Failed during geometry preparation: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[Info] Geometry preparation took: {time.time() - start_geom:.2f}s")

    # --- Run Simulation ---
    print("[Info] Starting core simulation...")
    start_sim = time.time()
    try:
        # Pass H_cobs explicitly if needed by core functions
        floor_ppfd, X, Y, cob_positions = simulate_lighting(layer_fluxes_from_args, geo, H_cobs)
    except Exception as e:
        print(f"[ERROR] Failed during simulation lighting calculation: {e}", file=sys.stderr)
        # Add more specific error handling if needed
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sim_duration = time.time() - start_sim
    print(f"[Info] Core simulation finished in: {sim_duration:.2f}s")

    # --- Calculate Statistics ---
    if floor_ppfd is None or floor_ppfd.size == 0:
        print("[ERROR] Simulation returned empty PPFD results.", file=sys.stderr)
        sys.exit(1)

    mean_ppfd = np.mean(floor_ppfd)
    std_dev_ppfd = np.std(floor_ppfd)
    min_ppfd = np.min(floor_ppfd)
    max_ppfd = np.max(floor_ppfd)

    if abs(mean_ppfd) < EPSILON:
        print("[Warning] Average PPFD is near zero. Statistics might be unreliable.", file=sys.stderr)
        mad = rmse = dou = mdou = cv = 0.0
        # Avoid division by zero
    else:
        mad = np.mean(np.abs(floor_ppfd - mean_ppfd)) # Mean Absolute Deviation
        rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2)) # Root Mean Square Error
        dou_min_avg = 100 * min_ppfd / mean_ppfd # Distribution Uniformity (Min/Avg)
        dou_std_based = 100 * (1 - std_dev_ppfd / mean_ppfd) # Uniformity (1 - CV) - Sometimes used
        mdou = 100 * (1 - mad / mean_ppfd) # Mean Absolute Deviation Uniformity
        cv = 100 * (std_dev_ppfd / mean_ppfd) # Coefficient of Variation

    # --- Print Statistics to Standard Output ---
    # This is important so the calling script can potentially capture it,
    # although parsing CSV is more robust.
    print("\n--- Simulation Results ---")
    print(f"Average PPFD: {mean_ppfd:.2f} µmol/m²/s")
    print(f"Std Dev PPFD: {std_dev_ppfd:.2f}")
    print(f"Min PPFD: {min_ppfd:.2f}")
    print(f"Max PPFD: {max_ppfd:.2f}")
    # print(f"MAD: {mad:.2f}") # Less common stat
    # print(f"RMSE: {rmse:.2f}") # Less common stat
    print(f"DOU (Min/Avg) (%): {dou_min_avg:.2f}")
    print(f"CV (%): {cv:.2f}")
    print(f"Uniformity (1-CV) (%): {dou_std_based:.2f}") # Provide this common definition
    # print(f"M-DOU (%): {mdou:.2f}") # Less common stat

    # --- CSV Output ---
    print(f"[Info] Writing detailed PPFD data to {output_csv_filename}...")
    # Pass num_layers to ensure correct zone assignment
    write_ppfd_to_csv(output_csv_filename, floor_ppfd, X, Y, cob_positions, W, L, num_layers_from_args)
    print("[Info] CSV writing complete.")

    # --- Heatmap Plot (Optional) ---
    if not args.no_plot:
        print("[Info] Generating heatmap plot...")
        try:
             plot_heatmap(floor_ppfd, X, Y, W, L, cob_positions, title=f"PPFD (Avg: {mean_ppfd:.1f} µmol/m²/s)")
        except Exception as e:
             print(f"[ERROR] Failed to generate plot: {e}", file=sys.stderr)
             # Don't exit, just report error

    print("--- Simulation Complete ---")