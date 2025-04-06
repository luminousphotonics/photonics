#!/usr/bin/env python3
import csv
import math
import numpy as np
from numba import njit, prange
from functools import lru_cache
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse
import time

# ------------------------------------
# 1) Basic Config & Reflectances
# ------------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0  # lumens/W
SPD_FILE = "./backups/spd_data.csv" # Make sure this path is correct

MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3

WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

FLOOR_GRID_RES = 0.08  # m
MC_SAMPLES = 16 # Samples for reflection calculation

# REMOVED: FIXED_NUM_LAYERS = 9

# --- Epsilon for numerical stability ---
EPSILON = 1e-9
HIGH_LOSS = 1e12 # Or any suitably large number

# ------------------------------------
# 2) COB Datasheet Angular Data (Unchanged)
# ------------------------------------
COB_ANGLE_DATA = np.array([
    [  0, 1.00], [ 10, 0.98], [ 20, 0.95], [ 30, 0.88], [ 40, 0.78],
    [ 50, 0.65], [ 60, 0.50], [ 70, 0.30], [ 80, 0.10], [ 90, 0.00],
], dtype=np.float64)
COB_angles_deg = COB_ANGLE_DATA[:, 0]
COB_shape = COB_ANGLE_DATA[:, 1]

# ------------------------------------
# 3) Compute SPD-based µmol/J Factor (Unchanged, added print for clarity)
# ------------------------------------
def compute_conversion_factor(spd_file):
    """ Computes PPFD conversion factor from SPD data. """
    try:
        print(f"[INFO] Loading SPD data from: {spd_file}")
        raw_spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
        unique_wl, indices, counts = np.unique(raw_spd[:, 0], return_inverse=True, return_counts=True)
        # Avoid division by zero if counts are zero (though unique should prevent this)
        counts_nonzero = np.maximum(counts, 1)
        avg_intens = np.bincount(indices, weights=raw_spd[:, 1]) / counts_nonzero
        spd = np.column_stack((unique_wl, avg_intens))
        print(f"[INFO] SPD loaded successfully: {len(unique_wl)} unique wavelengths.")
    except FileNotFoundError:
        print(f"[ERROR] SPD file not found: {spd_file}. Using fallback conversion factor.")
        return 0.0138 # Fallback value
    except Exception as e:
        print(f"[ERROR] Error loading SPD file {spd_file}: {e}. Using fallback conversion factor.")
        return 0.0138 # Fallback value

    wl = spd[:, 0]; intens = spd[:, 1]; sort_idx = np.argsort(wl); wl = wl[sort_idx]; intens = intens[sort_idx]
    mask_par = (wl >= 400) & (wl <= 700); PAR_fraction = 1.0
    if len(wl) >= 2:
        tot = np.trapz(intens, wl)
        if tot > EPSILON:
             if np.count_nonzero(mask_par) >= 2:
                 tot_par = np.trapz(intens[mask_par], wl[mask_par]); PAR_fraction = tot_par / tot
             else: print("[SPD Warning] Not enough PAR data points (>=2) for PAR fraction calculation.")
        else: print("[SPD Warning] Zero total SPD intensity.")
    else: print("[SPD Warning] Not enough SPD points (>=2) for integration.")

    wl_m = wl * 1e-9; h, c, N_A = 6.626e-34, 3.0e8, 6.022e23; lambda_eff = 0.0
    if np.count_nonzero(mask_par) >= 2:
        numerator = np.trapz(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
        denominator = np.trapz(intens[mask_par], wl_m[mask_par])
        if denominator > EPSILON: lambda_eff = numerator / denominator
        else: print("[SPD Warning] Zero PAR intensity for effective wavelength calculation.")
    else:
         print("[SPD Warning] Not enough PAR data points (>=2) for effective wavelength calculation.")


    if lambda_eff <= EPSILON:
        print("[SPD Warning] Could not calculate effective PAR wavelength.")
        if np.count_nonzero(mask_par) > 0: lambda_eff = np.mean(wl_m[mask_par]) # Use simple mean as fallback
        else: lambda_eff = 550e-9 # Fallback to ~green if no PAR data at all
        print(f"[SPD Info] Using fallback effective wavelength: {lambda_eff*1e9:.1f} nm")

    E_photon = (h * c / lambda_eff) if lambda_eff > EPSILON else 1.0 # Avoid division by zero
    # Ensure PAR_fraction is valid
    PAR_fraction = np.clip(PAR_fraction, 0.0, 1.0)
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction

    print(f"[INFO] SPD Results: PAR fraction={PAR_fraction:.3f}, Effective λ={lambda_eff*1e9:.1f} nm, Conv Factor={conversion_factor:.5f} µmol/J")
    return conversion_factor
CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)

# ------------------------------------
# 4) Pre-Compute Normalization Factor & Intensity Fn (Unchanged)
# ------------------------------------
def integrate_shape_for_flux(angles_deg, shape):
    # Ensure angles and shape match
    if len(angles_deg) != len(shape):
        raise ValueError("Angle and shape arrays must have the same length.")
    if len(angles_deg) < 2:
        return 0.0 # Cannot integrate with less than 2 points

    rad_angles = np.radians(angles_deg)
    G = 0.0
    # Integrate using trapezoidal rule on segments
    for i in range(len(rad_angles) - 1):
        th0, th1 = rad_angles[i], rad_angles[i+1]
        s0, s1 = shape[i], shape[i+1]
        # Intensity integral over solid angle segment: I(theta) * 2*pi*sin(theta) dtheta
        # Approximate segment: avg_intensity * 2*pi*sin(mid_angle) * delta_theta
        s_mean = 0.5*(s0 + s1)
        dtheta = th1 - th0
        th_mid = 0.5*(th0 + th1)
        sin_mid = math.sin(th_mid)
        # Ensure positive contribution
        G_seg = max(0.0, s_mean * 2.0*math.pi * sin_mid * dtheta)
        G += G_seg
    return G

SHAPE_INTEGRAL = integrate_shape_for_flux(COB_angles_deg, COB_shape)
if SHAPE_INTEGRAL <= EPSILON:
    print("[ERROR] COB shape integral is zero or negative. Check COB_ANGLE_DATA. Setting integral to 1.0.")
    SHAPE_INTEGRAL = 1.0 # Avoid division by zero

@njit
def luminous_intensity(angle_deg, total_lumens):
    """Calculates luminous intensity (lumens/sr) at a given angle."""
    # Interpolate relative intensity from COB data
    # np.interp needs angles to be increasing
    rel_intensity = np.interp(angle_deg, COB_angles_deg, COB_shape)
    # Intensity = Total Lumens * Relative Intensity / Shape Integral
    # Shape integral normalizes the relative intensity distribution
    intensity = (total_lumens * rel_intensity) / SHAPE_INTEGRAL
    return max(0.0, intensity) # Ensure non-negative

# ------------------------------------
# 5) Geometry Building (MODIFIED)
# ------------------------------------
# MODIFIED: Added num_layers argument
def prepare_geometry(W, L, H, num_layers):
    """Prepares geometry components: COB positions, floor grid, patches."""
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"Invalid num_layers ({num_layers}) passed to prepare_geometry. Must be int > 0.")
    print(f"[INFO] Preparing geometry for {num_layers} layers (W={W:.2f}, L={L:.2f}, H={H:.2f})")
    # MODIFIED: Pass num_layers
    cob_positions = build_cob_positions(W, L, H, num_layers) # Call revised function
    X, Y = build_floor_grid(W, L)
    patches = build_patches(W, L, H)
    num_cobs = cob_positions.shape[0]
    # Corrected formula for expected COBs in a full diamond grid up to layer n = num_layers - 1
    # Sum of points per layer: 1 (L0) + 4*1 (L1) + 4*2 (L2) ... + 4*n (Ln)
    # = 1 + 4 * (1 + 2 + ... + n) = 1 + 4 * n*(n+1)/2 = 1 + 2*n*(n+1)
    n = num_layers - 1
    expected_cobs = 1 + 2 * n * (n + 1) if n >= 0 else 0 # Handle n=-1 for num_layers=0 case
    print(f"[INFO] Generated {num_cobs} COB positions (Expected for {num_layers} layers: {expected_cobs})")
    if num_cobs == 0:
         print("[WARNING] No COB positions were generated!")
    elif num_cobs != expected_cobs:
         print(f"[WARNING] Generated COB count ({num_cobs}) does not match expected count ({expected_cobs}) for a full diamond grid.")
    return (cob_positions, X, Y, patches)


# REVISED: Corrected loop for full diamond pattern
def build_cob_positions(W, L, H, num_layers):
    """Builds the full diamond-pattern COB positions based on number of layers."""
    if num_layers <= 0:
        return np.empty((0, 4), dtype=np.float64) # Return empty array if no layers

    positions = []
    # Layer 0 (center)
    positions.append((0, 0, H, 0))

    # Layers 1 up to num_layers - 1
    n = num_layers - 1 # max layer index
    for i in range(1, n + 1): # Loop through layer index 1 to n
        # Generate points for layer i by walking the perimeter
        # Start at (i, 0) and go counter-clockwise
        for dx in range(i, 0, -1): # x from i down to 1
            dy = i - dx           # y goes from 0 up to i-1
            positions.append(( dx,  dy, H, i)) # Quadrant 1
            positions.append((-dy,  dx, H, i)) # Quadrant 2
            positions.append((-dx, -dy, H, i)) # Quadrant 3
            positions.append(( dy, -dx, H, i)) # Quadrant 4
        # Add the point on the axis (0, i) - already covered by (-dy, dx) when dy=0, dx=i
        # Need the point (i, 0) explicitly if loop ends before dx=i (it does)
        positions.append((i, 0, H, i)) # Add the +X axis point for layer i
        # The others (-i,0), (0,i), (0,-i) are covered by the loop rotations

    # Remove duplicates (just in case, though logic should avoid them)
    # Convert to set of tuples for uniqueness, then back to list
    unique_positions_set = {tuple(p) for p in positions}
    positions = list(unique_positions_set)
    # Sort by layer, then maybe x, then y for consistency? Optional.
    positions.sort(key=lambda p: (p[3], p[0], p[1]))

    # --- Transform grid to room coordinates ---
    theta = math.radians(45)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centerX = W / 2.0
    centerY = L / 2.0
    n_scale = max(1, n) # Use max layer index for scaling, avoid 0 for num_layers=1
    scale_x = centerX / n_scale
    scale_y = centerY / n_scale
    mount_height = H * 0.98
    transformed = []

    for (grid_x, grid_y, _, layer_idx) in positions:
        rot_x = grid_x * cos_t - grid_y * sin_t
        rot_y = grid_x * sin_t + grid_y * cos_t
        px = centerX + rot_x * scale_x
        py = centerY + rot_y * scale_y
        px = np.clip(px, EPSILON, W - EPSILON)
        py = np.clip(py, EPSILON, L - EPSILON)
        transformed.append((px, py, mount_height, layer_idx))

    return np.array(transformed, dtype=np.float64)


def pack_luminous_flux_dynamic(params, cob_positions):
    """Assigns flux from params vector to corresponding COB based on layer index."""
    num_cobs = cob_positions.shape[0]
    num_params = len(params)
    led_intensities = np.zeros(num_cobs, dtype=np.float64)

    max_layer_index = 0
    if num_cobs > 0:
        max_layer_index = int(np.max(cob_positions[:, 3]))

    # Check if params vector is long enough for the layers present in cob_positions
    if num_params < max_layer_index + 1:
        print(f"[ERROR] pack_luminous_flux_dynamic: params vector length ({num_params}) is too short for max COB layer index ({max_layer_index}).")
        # Return zeros or raise error? Returning zeros might hide issues. Let's return zeros.
        return led_intensities # Return array of zeros

    for i in range(num_cobs):
        pos = cob_positions[i]
        # Layer index is the 4th element (index 3)
        layer_idx = int(pos[3])
        # Get intensity from the params vector using the layer index
        # params vector might be longer (padded), but we only access up to max_layer_index
        intensity = params[layer_idx]
        led_intensities[i] = intensity

    return led_intensities

# --- Floor Grid and Patch Building (Unchanged, using LRU Cache) ---
@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float):
    num_x = int(round(W / FLOOR_GRID_RES)) + 1
    num_y = int(round(L / FLOOR_GRID_RES)) + 1
    xs = np.linspace(0, W, num_x)
    ys = np.linspace(0, L, num_y)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def build_floor_grid(W, L):
    return cached_build_floor_grid(W, L)

@lru_cache(maxsize=32)
def cached_build_patches(W: float, L: float, H: float):
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []
    # Floor (Patch 0)
    patch_centers.append((W/2.0, L/2.0, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0)) # Normal pointing up
    patch_refl.append(REFL_FLOOR)
    # Ceiling
    xs_ceiling = np.linspace(0, W, CEIL_SUBDIVS_X + 1)
    ys_ceiling = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_ceiling[i] + xs_ceiling[i+1]) / 2.0
            cy = (ys_ceiling[j] + ys_ceiling[j+1]) / 2.0
            area = (xs_ceiling[i+1] - xs_ceiling[i]) * (ys_ceiling[j+1] - ys_ceiling[j])
            patch_centers.append((cx, cy, H)) # Ceiling at height H
            patch_areas.append(area)
            patch_normals.append((0.0, 0.0, -1.0)) # Normal pointing down
            patch_refl.append(REFL_CEIL)
    # Walls (Four walls: Y=0, Y=L, X=0, X=W)
    xs_wall = np.linspace(0, W, WALL_SUBDIVS_X + 1)
    zs_wall = np.linspace(0, H, WALL_SUBDIVS_Y + 1)
    # Wall Y=0
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2.0
            cz = (zs_wall[j] + zs_wall[j+1]) / 2.0
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, 0.0, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, 1.0, 0.0)) # Normal pointing into room (+Y)
            patch_refl.append(REFL_WALL)
    # Wall Y=L
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2.0
            cz = (zs_wall[j] + zs_wall[j+1]) / 2.0
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, L, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, -1.0, 0.0)) # Normal pointing into room (-Y)
            patch_refl.append(REFL_WALL)
    # Wall X=0
    ys_wall = np.linspace(0, L, WALL_SUBDIVS_X + 1) # Re-use subdiv count for consistency
    for i in range(WALL_SUBDIVS_X): # Use same count for Y subdivs on this wall
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2.0
            cz = (zs_wall[j] + zs_wall[j+1]) / 2.0
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((0.0, cy, cz))
            patch_areas.append(area)
            patch_normals.append((1.0, 0.0, 0.0)) # Normal pointing into room (+X)
            patch_refl.append(REFL_WALL)
    # Wall X=W
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2.0
            cz = (zs_wall[j] + zs_wall[j+1]) / 2.0
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((W, cy, cz))
            patch_areas.append(area)
            patch_normals.append((-1.0, 0.0, 0.0)) # Normal pointing into room (-X)
            patch_refl.append(REFL_WALL)

    print(f"[INFO] Built {len(patch_centers)} patches (1 Floor, {CEIL_SUBDIVS_X*CEIL_SUBDIVS_Y} Ceiling, {4*WALL_SUBDIVS_X*WALL_SUBDIVS_Y} Walls)")
    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

def build_patches(W, L, H):
    return cached_build_patches(W, L, H)

# ------------------------------------
# 6) The Numba-JIT Computations (Largely Unchanged, check vector accesses)
# ------------------------------------
@njit
def manual_clip(value, min_val, max_val):
    """Clips a scalar value between min_val and max_val."""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

@njit(parallel=True)
def compute_direct_floor(cob_positions, cob_lumens, X, Y):
    min_dist2 = (FLOOR_GRID_RES / 2.0) ** 2
    rows, cols = X.shape
    out = np.zeros_like(X, dtype=np.float64)
    num_cobs = cob_positions.shape[0]
    if num_cobs != len(cob_lumens): return np.zeros_like(X, dtype=np.float64)

    for r in prange(rows):
        for c in range(cols):
            fx = X[r, c]; fy = Y[r, c]; fz = 0.0
            pt_illum = 0.0
            for k in range(num_cobs):
                lx, ly, lz, _ = cob_positions[k]
                lumens_k = cob_lumens[k]
                if lumens_k <= 0: continue
                dx = fx - lx; dy = fy - ly; dz = fz - lz
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < min_dist2: d2 = min_dist2
                dist = math.sqrt(d2)
                cos_th_source = -dz / dist
                if cos_th_source <= EPSILON: continue

                # <<< FIX: Use manual_clip instead of np.clip >>>
                clipped_cos = manual_clip(cos_th_source, 0.0, 1.0)
                # Handle potential domain error for acos if clipped_cos is slightly > 1 due to float errors
                if clipped_cos > 1.0: clipped_cos = 1.0
                angle_rad = math.acos(clipped_cos)
                angle_deg = math.degrees(angle_rad)
                # angle_deg = math.degrees(math.acos(manual_clip(cos_th_source, 0.0, 1.0)))

                I_theta = luminous_intensity(angle_deg, lumens_k)
                cos_th_floor = cos_th_source # Already calculated
                if cos_th_floor <= EPSILON: continue
                E_local = (I_theta / d2) * cos_th_floor
                pt_illum += E_local
            out[r, c] = pt_illum
    return out

@njit
def compute_patch_direct(cob_positions, cob_lumens, patch_centers, patch_normals, patch_areas):
    """Computes direct illuminance on each patch center from all COBs."""
    min_dist2 = (FLOOR_GRID_RES / 2.0) ** 2
    num_patches = patch_centers.shape[0]
    num_cobs = cob_positions.shape[0]
    patch_direct_illum = np.zeros(num_patches, dtype=np.float64)
    if num_cobs != len(cob_lumens): return patch_direct_illum

    # <<< Constants needed for debug print - MUST be defined here if njit is active >>>
    # It's cleaner to calculate indices outside if removing njit for debugging,
    # but if keeping njit, they need to be calculable here or passed in.
    # Since CEIL_SUBDIVS are global python vars, njit can't see them.
    # For now, let's *comment out* the debug print within the njit function.
    # We will rely on the debug prints in the non-njit iterative_radiosity_solve later.

    for i_patch in range(num_patches):
        pc = patch_centers[i_patch]; n = patch_normals[i_patch]
        norm_n = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
        if norm_n < EPSILON: continue
        accum_illum = 0.0
        for k in range(num_cobs):
            lx, ly, lz, _ = cob_positions[k]
            lumens_k = cob_lumens[k]
            if lumens_k <= 0: continue

            dx = pc[0] - lx; dy = pc[1] - ly; dz = pc[2] - lz
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_dist2: d2 = min_dist2
            dist = math.sqrt(d2)

            cos_th_source = -dz / dist
            clipped_cos_source = manual_clip(cos_th_source, 0.0, 1.0)
            if clipped_cos_source > 1.0: clipped_cos_source = 1.0
            angle_rad_source = math.acos(clipped_cos_source)
            angle_deg_source = math.degrees(angle_rad_source)
            I_theta = luminous_intensity(angle_deg_source, lumens_k)

            dot_patch = (-dx * n[0]) + (-dy * n[1]) + (-dz * n[2])
            cos_th_patch = dot_patch / (dist * norm_n)
            if cos_th_patch <= EPSILON: continue

            E_local = (I_theta / d2) * cos_th_patch
            accum_illum += E_local

        patch_direct_illum[i_patch] = accum_illum

    # --- FIX: Comment out or remove debug print inside @njit function ---
    # It relies on global subdivision constants which Numba can't access
    # floor_idx = 0
    # ceil_idx = 1
    # # wall_idx = 1 + (CEIL_SUBDIVS_X * CEIL_SUBDIVS_Y) # Cannot use globals
    # print(f"  [DEBUG PatchDirect Inside NJIT is disabled]")
    # print(f"  Floor[{floor_idx}]: {patch_direct_illum[floor_idx]}")
    # if ceil_idx < num_patches: print(f"  Ceiling[{ceil_idx}]: {patch_direct_illum[ceil_idx]}")
    # # if wall_idx < num_patches: print(f"  Wall[{wall_idx}]: {patch_direct_illum[wall_idx]}")
    # ---

    return patch_direct_illum

@njit
def compute_form_factors(patch_centers, patch_normals, patch_areas):
    """Computes the form factor matrix F[i, j] (fraction of energy leaving j reaching i)."""
    Np = patch_centers.shape[0]
    F = np.zeros((Np, Np), dtype=np.float64)
    min_dist2_ff = 1e-4 # Minimum distance for form factor to avoid self/close issues

    for i in range(Np):
        pi = patch_centers[i]
        ni = patch_normals[i]
        norm_ni = math.sqrt(ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2])
        area_i = patch_areas[i]
        if norm_ni < EPSILON or area_i < EPSILON: continue

        for j in range(i): # Calculate F[i,j] and use reciprocity for F[j,i]
            pj = patch_centers[j]
            nj = patch_normals[j]
            norm_nj = math.sqrt(nj[0]*nj[0] + nj[1]*nj[1] + nj[2]*nj[2])
            area_j = patch_areas[j]
            if norm_nj < EPSILON or area_j < EPSILON: continue

            dx = pi[0] - pj[0]
            dy = pi[1] - pj[1]
            dz = pi[2] - pj[2]
            dist2 = dx*dx + dy*dy + dz*dz

            if dist2 < min_dist2_ff: continue # Patches too close or identical

            dist = math.sqrt(dist2)

            # Cosine of angle at j (vector j->i dotted with normal j)
            dot_j = dx*nj[0] + dy*nj[1] + dz*nj[2]
            cos_j = dot_j / (dist * norm_nj)

            # Cosine of angle at i (vector j->i dotted with normal i, note direction)
            dot_i = -(dx*ni[0] + dy*ni[1] + dz*ni[2]) # Vector j->i is - (i->j)
            cos_i = dot_i / (dist * norm_ni)

            # Visibility check (basic: are patches facing each other?)
            if cos_i > EPSILON and cos_j > EPSILON:
                # Form Factor formula (point-to-point approximation)
                ff_ji = (cos_j * cos_i) / (math.pi * dist2) * area_i # F_ji * A_j = F_ij * A_i -> F_ji = (cosj*cosi*Ai)/(pi*r^2)
                F[j, i] = max(0.0, ff_ji) # Assign F[j, i] (energy from i to j)
                # Reciprocity: A_i * F_ij = A_j * F_ji
                ff_ij = ff_ji * area_j / area_i if area_i > EPSILON else 0.0
                F[i, j] = max(0.0, ff_ij) # Assign F[i, j] (energy from j to i)
                # Clamp? Sum of F[i, :] * A[i] should approach A[j]?

    # Normalize rows? The sum of F[i, j] over all i should be <= 1 for patch j.
    # Sum F[i, j] * A[i] / A[j] should be <= 1 ? Let's skip normalization for now.
    return F


@njit
def iterative_radiosity_solve(patch_direct_illum, patch_areas, patch_refl, form_factors,
                              max_bounces, convergence_threshold):
    """Solves for total patch radiosity (leaving flux density) using iteration."""
    Np = patch_direct_illum.shape[0]
    E_incident = patch_direct_illum.copy()
    B_radiosity = E_incident * patch_refl
    B_prev = B_radiosity.copy()
    delta_B_rel = HIGH_LOSS

    # --- DEBUG PRINTS ---
    # These run outside njit, so they CAN access globals like CEIL_SUBDIVS_X/Y
    #print(f"  [DEBUG Radiosity Start] Np={Np}, Max Bounces={max_bounces}")
    ceil_idx = 1
    wall_idx = 1 + (CEIL_SUBDIVS_X * CEIL_SUBDIVS_Y) # Globals OK here
    #print(f"  [DEBUG Radiosity Start] Checking indices: Ceiling={ceil_idx}, Wall={wall_idx}")
    # Use standard Python formatting now
    #if ceil_idx < Np: print(f"  [DEBUG Radiosity Start] Direct Illum Ceiling[{ceil_idx}]: {patch_direct_illum[ceil_idx]:.4e}, Refl: {patch_refl[ceil_idx]:.2f}")
    #if wall_idx < Np: print(f"  [DEBUG Radiosity Start] Direct Illum Wall[{wall_idx}]: {patch_direct_illum[wall_idx]:.4e}, Refl: {patch_refl[wall_idx]:.2f}")
    #if ceil_idx < Np: print(f"  [DEBUG Radiosity Start] Initial Radiosity Ceiling[{ceil_idx}]: {B_radiosity[ceil_idx]:.4e}")
    #if wall_idx < Np: print(f"  [DEBUG Radiosity Start] Initial Radiosity Wall[{wall_idx}]: {B_radiosity[wall_idx]:.4e}")
    #if wall_idx < Np and ceil_idx < Np : print(f"  [DEBUG Radiosity Start] Form Factor F[{ceil_idx}, {wall_idx}] (Ceil from Wall): {form_factors[ceil_idx, wall_idx]:.4e}")
    # --- END DEBUG PRINTS ---

    for bounce in range(max_bounces):
        E_reflected = np.zeros(Np, dtype=np.float64)
        # ... (Calculation using numpy arrays) ...
        for i in range(Np):
            accum_reflected_E = 0.0
            for j in range(Np):
                 if i == j: continue
                 accum_reflected_E += B_prev[j] * form_factors[i, j]
            E_reflected[i] = accum_reflected_E

        E_incident = patch_direct_illum + E_reflected
        B_radiosity = E_incident * patch_refl

        delta_B_abs = np.sum(np.abs(B_radiosity - B_prev))
        sum_B_prev_abs = np.sum(np.abs(B_prev))
        relative_change = delta_B_abs / (sum_B_prev_abs + EPSILON)
        delta_B_rel = relative_change

        # --- DEBUG PRINTS inside loop ---
        #if (bounce == 0 or bounce == max_bounces - 1 ):
             #print(f"  [DEBUG Radiosity Bounce {bounce+1}] Rel Change: {relative_change:.4e}")
             #if ceil_idx < Np: print(f"    Ceiling[{ceil_idx}]: E_refl={E_reflected[ceil_idx]:.4e}, E_inc={E_incident[ceil_idx]:.4e}, B={B_radiosity[ceil_idx]:.4e}")
             #if wall_idx < Np: print(f"    Wall[{wall_idx}]: E_refl={E_reflected[wall_idx]:.4e}, E_inc={E_incident[wall_idx]:.4e}, B={B_radiosity[wall_idx]:.4e}")
        # --- END DEBUG PRINTS ---

        B_prev = B_radiosity.copy()
        #if relative_change < convergence_threshold:
            #print(f"  [DEBUG Radiosity Converged] Bounce {bounce+1}")
            #break
    #else:
       #print(f"  [DEBUG Radiosity Max Bounces] Bounce {bounce+1}, Final Rel Change: {relative_change:.4e}")

    #print(f"  [DEBUG Radiosity End] Max Radiosity Calculated: {np.max(B_radiosity):.4e}")
    return B_radiosity


# --- Reflection Calculation using Joblib/Numba (Unchanged Logic, check types) ---
# Function to compute reflection for a single row (for parallel processing)
@njit
def compute_row_reflection(r, X, Y, patch_centers, patch_normals, patch_areas, patch_radiosity, patch_refl, mc_samples):
    """Computes reflected illuminance on one row of the floor grid using MC."""
    cols = X.shape[1]
    row_illum = np.zeros(cols, dtype=np.float64)
    num_patches = patch_centers.shape[0]
    floor_normal = np.array([0.0, 0.0, 1.0]) # Floor normal always points up

    for c in range(cols):
        fx = X[r, c]
        fy = Y[r, c]
        fz = 0.0 # Floor point z
        accum_reflect_illum = 0.0

        for p_idx in range(num_patches):
            # Only consider reflection from patches *other* than the floor itself (patch 0)
            if p_idx == 0: continue
            # Also skip patches with zero reflectivity or radiosity
            if patch_refl[p_idx] <= EPSILON or patch_radiosity[p_idx] <= EPSILON:
                continue

            # Get patch properties
            pc = patch_centers[p_idx]
            n = patch_normals[p_idx] # Patch normal
            norm_n = np.linalg.norm(n)
            if norm_n < EPSILON: continue # Skip invalid patch

            area_p = patch_areas[p_idx]
            radiosity_p = patch_radiosity[p_idx] # Flux leaving per area (Lumen/m2)
            total_flux_leaving_p = radiosity_p * area_p # Total Lumens leaving patch p

            # --- Monte Carlo Sampling for Form Factor from Patch p to Floor Point (fx, fy) ---
            # Estimate form factor F_{floor_point, patch_p}
            # F_{i, j} = integral over Ai, Aj of (cos(th_i)*cos(th_j))/(pi*r^2) dA_i dA_j
            # For point i to area j: F_{i, j} = integral over Aj of (cos(th_i)*cos(th_j))/(pi*r^2) dA_j
            # We need flux arriving at floor point from patch p.
            # dE_floor = B_p * dF_{floor, patch} where dF is differential form factor
            # E_floor = integral over Ap of B_p * (cos(th_floor)*cos(th_p))/(pi*r^2) dA_p
            # MC estimate: E_floor = (B_p * Area_p / N_samples) * sum[ (cos_floor * cos_p) / (pi * r^2) ]

            # Define tangent vectors for sampling on the patch surface
            # Ensure normal is normalized
            n_norm = n / norm_n
            if abs(n_norm[2]) > 0.999: # Normal is mostly vertical (ceiling?)
                tangent1 = np.array([1.0, 0.0, 0.0])
            else: # Normal has significant X or Y component
                tangent1 = np.array([-n_norm[1], n_norm[0], 0.0]) # Orthogonal in XY plane
            tangent1 /= np.linalg.norm(tangent1) # Normalize tangent1
            tangent2 = np.cross(n_norm, tangent1) # tangent2 is orthogonal to both
            tangent2 /= np.linalg.norm(tangent2) # Normalize tangent2

            half_side = math.sqrt(area_p) / 2.0 # Approx side length for square patch
            form_factor_sum = 0.0

            for _ in range(mc_samples):
                # Generate random offsets on the patch plane
                offset1 = np.random.uniform(-half_side, half_side)
                offset2 = np.random.uniform(-half_side, half_side)
                # Calculate sample point coordinates on the patch
                sample_point = pc + offset1*tangent1 + offset2*tangent2

                # Vector from sample point SP on patch p to floor point FP
                dx = fx - sample_point[0]
                dy = fy - sample_point[1]
                dz = fz - sample_point[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < EPSILON: continue # Avoid self-contribution? Unlikely here.
                dist = math.sqrt(dist2)

                # Cosine angle at floor point (relative to floor normal +Z)
                # Vector SP->FP dotted with (0,0,1) is dz. Note dz is negative.
                cos_floor = -dz / dist
                if cos_floor <= EPSILON: continue # Sample point cannot see floor point

                # Cosine angle at patch sample point (relative to patch normal n)
                # Vector SP->FP dotted with n
                dot_patch = dx*n[0] + dy*n[1] + dz*n[2]
                cos_patch = dot_patch / (dist * norm_n) # Use original norm_n
                if cos_patch <= EPSILON: continue # Floor point cannot see sample point face

                # Add contribution to form factor term sum
                form_factor_term = (cos_floor * cos_patch) / (math.pi * dist2)
                form_factor_sum += form_factor_term

            # Average contribution and scale by total flux leaving patch p / N_samples
            # E_floor_from_p = (total_flux_leaving_p / mc_samples) * form_factor_sum
            # E_floor_from_p = (radiosity_p * area_p / mc_samples) * form_factor_sum
            # Alternatively, using average form factor term:
            avg_ff_term = form_factor_sum / mc_samples if mc_samples > 0 else 0.0
            # Illuminance = Radiosity * Area * AvgFormFactorTerm ??? No.
            # Illuminance E_i = sum ( B_j * F_ij )
            # E_floor_from_p = B_p * F_{floor_point, patch_p}
            # F_{floor_point, patch_p} approx = Area_p * avg( (cos_f * cos_p)/(pi*r^2) )
            E_floor_from_p = radiosity_p * area_p * avg_ff_term
            accum_reflect_illum += E_floor_from_p

        row_illum[c] = accum_reflect_illum

    return row_illum

# Main function for computing reflection using parallel rows
def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_radiosity, patch_refl, mc_samples=MC_SAMPLES):
    """Computes reflected illuminance on floor grid using parallel processing."""
    rows, cols = X.shape
    print(f"[INFO] Computing reflections on floor grid ({rows}x{cols}) using {mc_samples} MC samples per patch per point...")
    start_time = time.time()

    # Use joblib for parallel execution of compute_row_reflection
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_radiosity, patch_refl, mc_samples
    ) for r in range(rows))

    # Combine results from parallel rows
    reflect_illum_floor = np.zeros((rows, cols), dtype=np.float64)
    for r, row_vals in enumerate(results):
        if row_vals is not None and len(row_vals) == cols:
             reflect_illum_floor[r, :] = row_vals
        else:
             print(f"[WARNING] Received invalid result for reflection row {r}")

    duration = time.time() - start_time
    print(f"[INFO] Reflection calculation finished. Duration: {duration:.2f}s")
    return reflect_illum_floor

# ------------------------------------
# 7) Heatmap Plotting Function (Unchanged)
# ------------------------------------
def plot_heatmap(floor_ppfd, X, Y, cob_positions, annotation_step=5):
    """Plots PPFD heatmap with optional annotations and COB locations."""
    if X is None or Y is None or floor_ppfd is None:
        print("[Plot Warning] Insufficient data to plot heatmap.")
        return

    rows, cols = floor_ppfd.shape
    print(f"[INFO] Plotting heatmap ({rows}x{cols})...")

    fig, ax = plt.subplots(figsize=(10, 8)) # Slightly larger figure
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    # Determine color limits based on data range, avoid issues with all zeros
    vmin = np.min(floor_ppfd)
    vmax = np.max(floor_ppfd)
    if vmax <= vmin: vmax = vmin + 1.0 # Avoid zero range for colorbar

    heatmap = ax.imshow(floor_ppfd, cmap='hot', interpolation='nearest',
                        origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(heatmap, ax=ax, shrink=0.7)
    cbar.set_label('PPFD (µmol/m²/s)')

    # Add annotations (optional, can be slow for large grids)
    if annotation_step > 0:
        step = max(1, annotation_step)
        for i in range(0, rows, step):
            for j in range(0, cols, step):
                try:
                     ax.text(X[i, j], Y[i, j], f"{floor_ppfd[i, j]:.0f}",
                             ha="center", va="center", color="white", fontsize=6,
                             bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.4))
                except IndexError:
                    print(f"[Plot Warning] Annotation index error at ({i},{j})")

    # Plot COB positions if available
    if cob_positions is not None and cob_positions.shape[0] > 0:
        ax.scatter(cob_positions[:, 0], cob_positions[:, 1], marker='o',
                   color='cyan', edgecolors='black', s=40, label="COB positions", alpha=0.8)
        # Optional: Annotate COB layers
        # for k in range(cob_positions.shape[0]):
        #    ax.text(cob_positions[k,0], cob_positions[k,1]+0.1, f"L{int(cob_positions[k,3])}",
        #            color='white', fontsize=7, ha='center')

    ax.set_title(f"Floor PPFD Heatmap (Mean: {np.mean(floor_ppfd):.1f} µmol/m²/s)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', adjustable='box') # Ensure correct aspect ratio
    ax.legend()
    plt.tight_layout()
    plt.show(block=False) # Non-blocking display
    plt.pause(0.1) # Allow plot to render

# ------------------------------------
# 8) CSV Output Function (MODIFIED)
# ------------------------------------
# MODIFIED: Infer num_layers from cob_positions
def write_ppfd_to_csv(filename, floor_ppfd, X, Y, cob_positions, W, L):
    """
    Writes PPFD data to CSV, organized by distance-based rings (layers),
    using boundaries defined between COB layer radii. Infers num_layers.
    """
    print(f"[INFO] Writing layer-based PPFD data to {filename}...")
    if cob_positions is None or cob_positions.shape[0] == 0:
        print("[ERROR] Cannot write layer-based CSV: No COB positions provided.")
        return

    # --- Infer number of layers from COB positions ---
    try:
        cob_layer_indices = cob_positions[:, 3].astype(int)
        num_layers = np.max(cob_layer_indices) + 1
        print(f"[INFO] Inferred {num_layers} layers from COB positions.")
    except Exception as e:
        print(f"[ERROR] Could not determine number of layers from COB positions: {e}")
        return

    layer_data = {i: [] for i in range(num_layers)} # Initialize dict for each layer

    # --- Calculate layer radii based on actual COB positions ---
    layer_radii = np.zeros(num_layers)
    center_x, center_y = W / 2.0, L / 2.0
    for i in range(num_layers):
        layer_cob_indices = np.where(cob_layer_indices == i)[0]
        if len(layer_cob_indices) > 0:
            layer_cob_pos = cob_positions[layer_cob_indices, :2] # Get XY coords
            # Calculate distance from center for COBs in this layer
            distances = np.sqrt((layer_cob_pos[:, 0] - center_x)**2 +
                                (layer_cob_pos[:, 1] - center_y)**2)
            layer_radii[i] = np.max(distances) # Max distance defines the layer's extent
        # else: layer_radii[i] remains 0 if layer is empty (shouldn't happen with build_cob_positions)

    # --- Define sampling boundaries BETWEEN layer radii ---
    sampling_boundaries = [0.0] # Boundary 0 is at radius 0
    for i in range(num_layers - 1):
        # Midpoint between max radius of layer i and max radius of layer i+1
        midpoint = (layer_radii[i] + layer_radii[i+1]) / 2.0
        # Ensure boundary increases (handle cases where radii might be equal, e.g., layer 0)
        if midpoint <= sampling_boundaries[-1] + EPSILON:
             # If midpoint isn't larger, use the outer radius if it's distinct, else nudge slightly
             if layer_radii[i+1] > sampling_boundaries[-1] + EPSILON:
                 sampling_boundaries.append(layer_radii[i+1])
             else: # Nudge boundary slightly if radii are identical/too close
                 sampling_boundaries.append(sampling_boundaries[-1] + FLOOR_GRID_RES / 2.0)
        else:
            sampling_boundaries.append(midpoint)

    # Add a final outer boundary (e.g., room corner distance)
    max_room_dist = math.sqrt((W/2.0)**2 + (L/2.0)**2)
    # Use slightly more than the last layer's radius, but cap at room boundary
    final_boundary = layer_radii[-1] + (sampling_boundaries[-1] - sampling_boundaries[-2] if num_layers > 1 else layer_radii[-1]*0.1)
    sampling_boundaries.append(max(final_boundary, max_room_dist + FLOOR_GRID_RES)) # Ensure it covers the whole grid

    # --- Assign floor points to layers based on distance and boundaries ---
    rows, cols = floor_ppfd.shape
    points_assigned = 0
    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            dist_to_center = math.sqrt((fx - center_x)**2 + (fy - center_y)**2)

            assigned_layer = -1
            # Assign to layer 'i' if dist is between boundary i and boundary i+1
            for i in range(num_layers):
                inner_b = sampling_boundaries[i]
                outer_b = sampling_boundaries[i+1]
                # Layer 0 includes center exactly: dist >= inner_b AND dist < outer_b
                # Make comparison robust for floating point values near boundaries
                if dist_to_center >= inner_b - EPSILON and dist_to_center < outer_b - EPSILON:
                    assigned_layer = i
                    break

            # Handle points potentially outside the last boundary? Assign to last layer.
            if assigned_layer == -1 and dist_to_center >= sampling_boundaries[num_layers] - EPSILON:
                 assigned_layer = num_layers - 1

            if assigned_layer != -1:
                # Ensure layer index is valid
                if 0 <= assigned_layer < num_layers:
                     layer_data[assigned_layer].append(floor_ppfd[r, c])
                     points_assigned += 1
                else:
                     print(f"[CSV Warning] Point ({fx:.2f},{fy:.2f}) assigned invalid layer {assigned_layer}")
            # else: # Point not assigned - should not happen with expanded last boundary
            #    print(f"[CSV Warning] Point ({fx:.2f},{fy:.2f}, dist {dist_to_center:.2f}) not assigned to any layer.")


    print(f"[INFO] Assigned {points_assigned}/{rows*cols} floor points to layers for CSV.")

    # --- Write data to CSV ---
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer', 'PPFD']) # Header
            for layer_idx in range(num_layers):
                if layer_idx in layer_data and layer_data[layer_idx]:
                    for ppfd_value in layer_data[layer_idx]:
                        writer.writerow([layer_idx, f"{ppfd_value:.4f}"]) # Write layer index and PPFD value
        print(f"[INFO] Successfully wrote PPFD data to {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV file {filename}: {e}")

# ------------------------------------
# 9) Putting It All Together (MODIFIED)
# ------------------------------------
# MODIFIED: Simulate_lighting now receives pre-built geometry
def simulate_lighting(params, geo):
    """
    Simulates lighting distribution using pre-calculated geometry.
    params: Padded flux vector (length MAX_LAYERS from newml.py).
    geo: Tuple containing (cob_positions, X, Y, patches).
    """
    print("[INFO] Starting lighting simulation...")
    if not geo or len(geo) != 4:
         print("[ERROR] Invalid geometry tuple passed to simulate_lighting.")
         return None, None, None, None # Indicate error

    cob_positions, X, Y, patches = geo
    if cob_positions is None or X is None or Y is None or patches is None or len(patches)!=4:
         print("[ERROR] Invalid geometry components in simulate_lighting.")
         return None, None, None, None # Indicate error
    p_centers, p_areas, p_normals, p_refl = patches

    num_cobs = cob_positions.shape[0]
    if num_cobs == 0:
         print("[WARNING] No COBs in geometry, simulation will result in zero light.")
         # Return zeros of appropriate shape
         return np.zeros_like(X, dtype=np.float64), X, Y, cob_positions

    # 1. Assign flux from potentially padded params vector to the actual COBs
    # This uses the layer index stored in cob_positions
    lumens_per_cob = pack_luminous_flux_dynamic(params, cob_positions)
    print(f"[INFO] Assigned flux to {num_cobs} COBs. Sum Flux = {np.sum(lumens_per_cob):.1f} lumens.")

    # 2. Compute direct illuminance on the floor
    print("[INFO] Computing direct illuminance on floor...")
    direct_lux_floor = compute_direct_floor(cob_positions, lumens_per_cob, X, Y)
    print(f"[INFO] Direct floor illuminance computed. Mean = {np.mean(direct_lux_floor):.1f} lux.")

    # 3. Compute direct illuminance on all patches (walls, ceiling, floor patch 0)
    print("[INFO] Computing direct illuminance on patches...")
    patch_direct_illum = compute_patch_direct(cob_positions, lumens_per_cob, p_centers, p_normals, p_areas)
    print("[INFO] Patch direct illuminance computed.")

    # 4. Solve for radiosity (reflections)
    print("[INFO] Calculating form factors...")
    form_factors = compute_form_factors(p_centers, p_normals, p_areas)
    print(f"[INFO] Form factors computed (Shape: {form_factors.shape}).")
    print("[INFO] Solving for patch radiosity...")
    patch_radiosity = iterative_radiosity_solve(patch_direct_illum, p_areas, p_refl, form_factors,
                                                MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD)
    print(f"[INFO] Radiosity solve complete. Max Radiosity = {np.max(patch_radiosity):.1f} lumens/m2.")

    # 5. Compute reflected illuminance back onto the floor grid
    reflect_lux_floor = compute_reflection_on_floor(X, Y, p_centers, p_normals, p_areas, patch_radiosity, p_refl, MC_SAMPLES)
    print(f"[INFO] Reflected floor illuminance computed. Mean = {np.mean(reflect_lux_floor):.1f} lux.")

    # 6. Calculate total illuminance (lux) and convert to PPFD (µmol/m²/s)
    total_luminous_floor = direct_lux_floor + reflect_lux_floor
    # Convert Lux (lumens/m^2) to Irradiance (W/m^2) using luminous efficacy
    total_radiant_Wm2 = total_luminous_floor / LUMINOUS_EFFICACY
    # Convert Irradiance (W/m^2) to PPFD (µmol/m²/s) using SPD-based factor
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR

    print(f"[INFO] Simulation complete. Mean Total PPFD = {np.mean(floor_ppfd):.1f} µmol/m²/s.")
    # Return floor_ppfd grid, grid coordinates, and cob_positions used
    return floor_ppfd, X, Y, cob_positions

# ------------------------------------
# 10) Main Function for Standalone Execution (MODIFIED)
# ------------------------------------
def main():
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Lighting Simulation Script")
    parser.add_argument('--layers', type=int, default=10, help='Number of COB layers to simulate (e.g., 10)')
    parser.add_argument('--width', type=float, default=6.10, help='Room width in meters (e.g., 6.10 for 20ft)')
    parser.add_argument('--length', type=float, default=None, help='Room length in meters (defaults to width)')
    parser.add_argument('--height', type=float, default=0.9144, help='Mounting height H in meters (e.g., 0.9144 for 3ft)')
    parser.add_argument('--flux', nargs='+', type=float, default=None, help='List of flux values per layer (optional, overrides default)')
    parser.add_argument('--output-csv', type=str, default="ppfd_data.csv", help='Output CSV file name')
    parser.add_argument('--no-plot', action='store_true', help='Disable the heatmap plot')
    args = parser.parse_args()

    # --- Setup Simulation Parameters ---
    NUM_LAYERS = args.layers
    W = args.width
    L = args.length if args.length is not None else W # Default to square room
    H = args.height

    # Example flux params for the specified number of layers
    if args.flux:
        if len(args.flux) == NUM_LAYERS:
            params = np.array(args.flux, dtype=np.float64)
        else:
            print(f"[ERROR] Provided --flux list has {len(args.flux)} values, but --layers requires {NUM_LAYERS}. Using default.")
            params = np.array([10000.0] * NUM_LAYERS, dtype=np.float64) # Default: 10k lumens per layer
    else:
        # Default flux if not provided via command line
        params = np.array([10000.0] * NUM_LAYERS, dtype=np.float64)

    print("\n--- Running Standalone Simulation ---")
    print(f"Layers: {NUM_LAYERS}, Width: {W:.2f}m, Length: {L:.2f}m, Height: {H:.2f}m")
    print(f"Flux per layer: {params}")

    # --- Prepare Geometry ---
    # MODIFIED: Pass NUM_LAYERS
    try:
        geo = prepare_geometry(W, L, H, NUM_LAYERS)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to prepare geometry: {e}")
        return

    # --- Run Simulation ---
    # Simulate lighting expects a params vector potentially padded to MAX_LAYERS.
    # For standalone, we just pass the params vector we defined. If MAX_LAYERS
    # was needed, we'd pad here. However, simulate_lighting internally calls
    # pack_luminous_flux which handles mapping based on layer index.
    # Let's ensure params has at least NUM_LAYERS elements.
    if len(params) < NUM_LAYERS:
        print(f"[ERROR] Internal setup error: params length {len(params)} < NUM_LAYERS {NUM_LAYERS}")
        return

    # No need to pad here as pack_luminous handles it based on geometry layers
    floor_ppfd, X, Y, cob_positions = simulate_lighting(params, geo)

    if floor_ppfd is None:
        print("[ERROR] Simulation failed to return PPFD results.")
        return

    # --- Analyze Results ---
    mean_ppfd = np.mean(floor_ppfd[np.isfinite(floor_ppfd)])
    finite_ppfd = floor_ppfd[np.isfinite(floor_ppfd)]
    if finite_ppfd.size > 0 and mean_ppfd > EPSILON:
        mad = np.mean(np.abs(finite_ppfd - mean_ppfd))
        rmse = np.sqrt(np.mean((finite_ppfd - mean_ppfd)**2))
        dou = 100.0 * (1.0 - rmse / mean_ppfd) # Christiaens DOU
        mdou = 100.0 * (1.0 - mad / mean_ppfd) # Mean Deviation Uniformity
        cv = 100.0 * (np.std(finite_ppfd) / mean_ppfd) # Coefficient of Variation
    else:
        mad, rmse, dou, mdou, cv = 0, 0, 0, 0, 0

    print("\n--- Simulation Results ---")
    print(f"Average PPFD: {mean_ppfd:.2f} µmol/m²/s")
    print(f"MAD: {mad:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"DOU (%): {dou:.2f}") # Uniformity based on RMSE
    print(f"CV (%): {cv:.2f}")
    print(f"M-DOU (%): {mdou:.2f}") # Uniformity based on MAD

    # --- CSV Output ---
    if args.output_csv:
        write_ppfd_to_csv(args.output_csv, floor_ppfd, X, Y, cob_positions, W, L)

    # --- Heatmap (Optional) ---
    if not args.no_plot:
        plot_heatmap(floor_ppfd, X, Y, cob_positions, annotation_step=10)
        plt.show(block=True) # Keep plot open until closed manually in standalone mode

if __name__ == "__main__":
    # Define constants needed if running standalone after removing @njit
    HIGH_LOSS = 1e12
    EPSILON = 1e-9
    CEIL_SUBDIVS_X = 10 # Make sure these are accessible if needed by debug prints
    CEIL_SUBDIVS_Y = 10
    main()