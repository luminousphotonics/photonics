#!/usr/bin/env python3
"""
Full Radiosities Simulation with Persistent Gaussian Process Surrogate & Angular Data
------------------------------------------------------------------------------------
This script implements a radiosity‐based simulation to optimize luminous
flux assignments for COB LEDs in a simple rectangular room, incorporating
COB angular intensity distribution. It uses a Gaussian Process surrogate
trained persistently over runs.

The surrogate training data is stored in a file (TRAINING_DATA_FILE).

Usage:
    python ml-training_updated.py
"""

import math
import numpy as np
from scipy.optimize import minimize
from numba import njit
from functools import lru_cache
# import matplotlib # Keep if plotting might be added later
# matplotlib.use("Agg") # Keep if plotting might be added later
import os

# Additional imports for the surrogate model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.interpolate import CubicSpline
from sklearn.multioutput import MultiOutputRegressor # For regression model
from sklearn.ensemble import RandomForestRegressor # For regression model

# ------------------------------
# Configuration & Constants
# ------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

# --- UPDATED BASED ON USER EXAMPLES (Interpreting dimensions as side lengths) ---
# Mapping: Number of COB layers to the *maximum* floor area (m^2) it's suitable for.
# Derived from user examples like "6.10m x 6.10m (Area 37.21 m^2) -> 10 Layers"
COB_LAYERS_TO_MAX_AREA_M2 = {
     1: 0.37,   # Extrapolated: (0.61 * 1)^2
     2: 1.49,   # Extrapolated: (0.61 * 2)^2
     3: 3.35,   # Extrapolated: (0.61 * 3)^2
     4: 5.95,   # Extrapolated: (0.61 * 4)^2
     5: 9.30,   # Extrapolated: (0.61 * 5)^2
     6: 13.40,  # Extrapolated: (0.61 * 6)^2
     7: 18.23,  # Extrapolated: (0.61 * 7)^2
     8: 23.81,  # Extrapolated: (0.61 * 8)^2
     9: 30.14,  # Extrapolated: (0.61 * 9)^2
    10: 37.21,  # User Example: 6.10 * 6.10
    11: 45.43,  # User Example: 6.74 * 6.74
    12: 54.46,  # User Example: 7.38 * 7.38
    13: 64.32,  # User Example: 8.02 * 8.02
    14: 75.00,  # User Example: 8.66 * 8.66  (Rounded 74.9956)
    15: 86.49,  # User Example: 9.30 * 9.30
    16: 98.80,  # User Example: 9.94 * 9.94
    17: 111.94, # User Example: 10.58 * 10.58
    18: 125.89, # User Example: 11.22 * 11.22
    19: 136.42, # User Example: 11.68 * 11.68
    20: 151.78  # User Example: 12.32 * 12.32
}

# --- IMPORTANT: Update the sorted lists used by the script ---
_sorted_layers = sorted(COB_LAYERS_TO_MAX_AREA_M2.keys())
_sorted_max_areas = [COB_LAYERS_TO_MAX_AREA_M2[l] for l in _sorted_layers]

LUMINOUS_EFFICACY = 182.0         # lm/W
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv" # ADJUST PATH AS NEEDED

NUM_RADIOSITY_BOUNCES = 5
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10
FLOOR_GRID_RES = 0.08

MIN_LUMENS = 1000.0
MAX_LUMENS_MAIN = 20000.0

# Global constant: Force all candidate vectors and surrogate features to relate to this many layers.
# This defines the dimensionality of the optimization problem solved by the surrogate and the main optimizer.
FIXED_NUM_LAYERS = 10

# Number of spline samples for surrogate normalization. Let's tie it to FIXED_NUM_LAYERS for consistency.
# The normalized feature vector will have dimension M_SPLINE_SAMPLES + 1.
M_SPLINE_SAMPLES = FIXED_NUM_LAYERS

# Persistent training data file for the surrogate model
TRAINING_DATA_FILE = "simul_training_data_angular.npz" # Use new file for angular data

# Unused global variables (commented out)
# running = False
# sim_thread = None

PRE_DEFINED_FLOOR_SIZES = [
    (2, 2), (4, 4), (6, 6), (8, 8), (10, 10),
    (12, 12), (14, 14), (16, 16), (12, 16),
]
TARGET_PPFD_RANGE = range(0, 2001, 50) # 0 to 2000 inclusive

# ------------------------------------
# COB Datasheet Angular Data
# ------------------------------------
COB_ANGLE_DATA = np.array([
    [  0, 1.00], [ 10, 0.98], [ 20, 0.95], [ 30, 0.88], [ 40, 0.78],
    [ 50, 0.65], [ 60, 0.50], [ 70, 0.30], [ 80, 0.10], [ 90, 0.00],
], dtype=np.float64)
COB_angles_deg = COB_ANGLE_DATA[:, 0]
COB_shape_factor = COB_ANGLE_DATA[:, 1] # Relative intensity factor

# ------------------------------
# SPD Conversion Factor Calculation
# ------------------------------
def compute_conversion_factor(spd_file):
    try:
        # Try space delimiter first, then comma
        try:
            spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
        except ValueError:
            spd = np.loadtxt(spd_file, delimiter=',', skiprows=1)
        if spd.shape[1] != 2:
            raise ValueError(f"SPD file should have 2 columns, found {spd.shape[1]}")
    except Exception as e:
        print(f"[ERROR] Error loading SPD data from {spd_file}: {e}")
        print("[WARN] Using fallback CONVERSION_FACTOR = 0.0138 µmol/J")
        return 0.0138 # Fallback value

    wl = spd[:, 0]
    intens = spd[:, 1]
    if not np.all(np.diff(wl) > 0):
        print("[WARN] Wavelengths in SPD file are not strictly increasing. Sorting.")
        sort_idx = np.argsort(wl)
        wl = wl[sort_idx]
        intens = intens[sort_idx]

    mask_par = (wl >= 400) & (wl <= 700)
    if not np.any(mask_par):
        print("[WARN] No data found in PAR range (400-700nm). PAR fraction will be 0.")
        PAR_fraction = 0.0
        conversion_factor = 0.0
    else:
        # Use np.trapz for integration
        tot = np.trapz(intens, wl)
        tot_par = np.trapz(intens[mask_par], wl[mask_par])
        PAR_fraction = tot_par / tot if tot > 1e-9 else 0.0 # Avoid division by zero

        wl_m = wl * 1e-9
        h, c, N_A = 6.626e-34, 3.0e8, 6.022e23

        # Ensure PAR range has data for effective lambda calculation
        wl_m_par = wl_m[mask_par]
        intens_par = intens[mask_par]

        numerator = np.trapz(wl_m_par * intens_par, wl_m_par)
        denominator = np.trapz(intens_par, wl_m_par)

        if abs(denominator) < 1e-15:
             print("[WARN] Denominator for lambda_eff calculation is near zero.")
             lambda_eff = 0.0
             E_photon = 1.0 # Avoid division by zero later
        else:
            lambda_eff = numerator / denominator

        if lambda_eff <= 0:
            print(f"[WARN] Calculated effective wavelength lambda_eff is non-positive ({lambda_eff:.2e} m).")
            E_photon = 1.0 # Avoid division by zero later
            conversion_factor = 0.0
        else:
             E_photon = (h * c / lambda_eff)
             # conversion_factor = (1 / E_photon) * (1 / N_A) * PAR_fraction * 1e6 # µmol/J
             conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction # Corrected units µmol/J

    print(f"[INFO] SPD: PAR fraction={PAR_fraction:.3f}, Effective Lambda (PAR)={lambda_eff*1e9:.1f} nm, Conv Factor={conversion_factor:.5f} µmol/J")
    return conversion_factor

CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)

# ------------------------------
# Geometry Preparation Functions
# ------------------------------
def prepare_geometry(W, L, H):
    """
    Build geometry: COB positions, floor grid, patches.
    Returns tuple: (cob_positions, X, Y, (patch_centers, patch_areas, patch_normals, patch_refl))
    """
    cob_positions = build_cob_positions(W, L, H)
    X, Y = build_floor_grid(W, L)
    patches = build_patches(W, L, H)
    return (cob_positions, X, Y, patches)

def build_cob_positions(W, L, H):
    """
    Build dynamic COB positions using Diamond:61 pattern scaled by area.
    Number of layers 'n' determined by floor area via COB_LAYERS_TO_MAX_AREA_M2.
    Returns positions as (x, y, z, layer_index). COBs at 95% height.
    """
    floor_area_m2 = W * L
    target_n = 0 # Default to layer 0 (center only) if area is tiny
    if floor_area_m2 <= 0:
         print("[WARN] Floor area is zero or negative. Generating only center COB.")
         target_n = 0
    elif floor_area_m2 <= _sorted_max_areas[0]:
        target_n = _sorted_layers[0]
    else:
        found = False
        for i in range(len(_sorted_layers)):
            if floor_area_m2 <= _sorted_max_areas[i]:
                target_n = _sorted_layers[i]
                found = True
                break
        if not found:
            target_n = _sorted_layers[-1]
            print(f"[WARN] Floor area {floor_area_m2:.2f} m^2 exceeds max area in mapping ({_sorted_max_areas[-1]:.2f} m^2). Using max layers: {target_n}.")

    positions = []
    # Layer 0: Center COB
    if target_n >= 0: # Ensure at least the center COB exists if target_n is determined
        positions.append((0, 0, H, 0))
    # Layers 1 to target_n: Rings
    if target_n > 0:
        for i in range(1, target_n + 1):
            for x in range(-i, i + 1):
                y_abs = i - abs(x)
                if y_abs == 0:
                    if x != 0: # Avoid double-counting origin
                       positions.append((x, 0, H, i))
                else:
                    positions.append((x, y_abs, H, i))
                    positions.append((x, -y_abs, H, i))

    # --- Transformation and Scaling ---
    theta = math.radians(45)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centerX = W / 2
    centerY = L / 2

    # Avoid division by zero if target_n is 0
    # Scale relative to the 'radius' n of the diamond pattern
    if target_n > 0:
        scale_x = (W / 2 * 0.95 * math.sqrt(2)) / target_n
        scale_y = (L / 2 * 0.95 * math.sqrt(2)) / target_n
    else: # Only the center COB exists (n=0), place it slightly scaled just in case
        scale_x = W / 2 * 0.95
        scale_y = L / 2 * 0.95

    transformed = []
    cob_height = H * 0.95 # Place COBs slightly below ceiling
    for (x, y, _, layer) in positions: # Original H is placeholder
        # Rotate
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        # Scale and translate
        px = centerX + rx * scale_x
        py = centerY + ry * scale_y
        transformed.append((px, py, cob_height, layer))

    num_cobs_generated = len(transformed)
    # Ensure we always return at least one COB if area > 0
    if num_cobs_generated == 0 and W > 0 and L > 0:
         print("[WARN] No COBs generated despite positive area. Adding center COB.")
         transformed.append((centerX, centerY, cob_height, 0))
         num_cobs_generated = 1
         target_n = 0 # Reset target_n if it was somehow negative

    print(f"[INFO] Floor Area: {floor_area_m2:.2f} m^2 -> Target Layers (n): {target_n} -> COBs Generated: {num_cobs_generated}")

    return np.array(transformed, dtype=np.float64)


def pack_luminous_flux_dynamic(params, cob_positions):
    """
    Assigns luminous flux from the fixed-size 'params' vector (length FIXED_NUM_LAYERS)
    to the actual COBs based on their layer index.
    Uses the last value in 'params' for COBs whose layer index exceeds params length.
    """
    if cob_positions.shape[0] == 0:
        return np.array([], dtype=np.float64) # Handle case with no COBs

    led_intensities = np.zeros(cob_positions.shape[0], dtype=np.float64)
    num_params = len(params)
    if num_params == 0: # Should not happen if params has FIXED_NUM_LAYERS
        print("[WARN] Parameter vector is empty in pack_luminous_flux_dynamic.")
        return led_intensities # Return zeros

    for i, pos in enumerate(cob_positions):
        layer = int(pos[3]) # Layer index is the 4th element
        if layer < num_params:
            intensity = params[layer]
        else:
            # If COB layer index is >= num_params, use the intensity of the outermost layer defined in params
            intensity = params[num_params - 1]
        # Ensure intensity is within bounds (safety check)
        led_intensities[i] = max(MIN_LUMENS, min(MAX_LUMENS_MAIN, intensity))

    return led_intensities


@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float):
    if W <= 0 or L <= 0: return np.array([[]]), np.array([[]])
    num_x = max(2, int(round(W / FLOOR_GRID_RES)) + 1)
    num_y = max(2, int(round(L / FLOOR_GRID_RES)) + 1)
    xs = np.linspace(0, W, num_x)
    ys = np.linspace(0, L, num_y)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def build_floor_grid(W, L):
    return cached_build_floor_grid(W, L)

@lru_cache(maxsize=32)
def cached_build_patches(W: float, L: float, H: float):
    if W <= 0 or L <= 0 or H <=0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []

    # Floor patch (single patch at z=0)
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0)) # Normal points up into the room
    patch_refl.append(REFL_FLOOR)

    # Ceiling patches (at z=H)
    xs_ceiling = np.linspace(0, W, CEIL_SUBDIVS_X + 1)
    ys_ceiling = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_ceiling[i] + xs_ceiling[i+1]) / 2
            cy = (ys_ceiling[j] + ys_ceiling[j+1]) / 2
            area = (xs_ceiling[i+1] - xs_ceiling[i]) * (ys_ceiling[j+1] - ys_ceiling[j])
            patch_centers.append((cx, cy, H)) # Ceiling at z=H
            patch_areas.append(area)
            patch_normals.append((0.0, 0.0, -1.0)) # Normal points down into the room
            patch_refl.append(REFL_CEIL)

    # Walls: front (y=0), back (y=L), left (x=0), right (x=W)
    xs_wall = np.linspace(0, W, WALL_SUBDIVS_X + 1)
    zs_wall = np.linspace(0, H, WALL_SUBDIVS_Y + 1) # Walls go from z=0 to z=H

    # Front wall (y=0)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, 0.0, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, 1.0, 0.0)) # Normal points into the room (positive y)
            patch_refl.append(REFL_WALL)

    # Back wall (y=L)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, L, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, -1.0, 0.0)) # Normal points into the room (negative y)
            patch_refl.append(REFL_WALL)

    # Need y subdivisions for side walls
    ys_wall = np.linspace(0, L, WALL_SUBDIVS_X + 1) # Reuse X subdivisions count for Y

    # Left wall (x=0)
    for i in range(WALL_SUBDIVS_X): # Iterate along Y axis
        for j in range(WALL_SUBDIVS_Y): # Iterate along Z axis
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((0.0, cy, cz))
            patch_areas.append(area)
            patch_normals.append((1.0, 0.0, 0.0)) # Normal points into the room (positive x)
            patch_refl.append(REFL_WALL)

    # Right wall (x=W)
    for i in range(WALL_SUBDIVS_X): # Iterate along Y axis
        for j in range(WALL_SUBDIVS_Y): # Iterate along Z axis
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((W, cy, cz))
            patch_areas.append(area)
            patch_normals.append((-1.0, 0.0, 0.0)) # Normal points into the room (negative x)
            patch_refl.append(REFL_WALL)

    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

def build_patches(W, L, H):
    return cached_build_patches(W, L, H)


# ------------------------------
# Simulation Functions (Radiosity) - Using Angular Data
# ------------------------------

# Numba compatible interpolation for COB shape factor
@njit
def get_cob_shape_factor(angle_deg, angles_data_deg, shape_data):
    """Linearly interpolates the COB shape factor for a given emission angle in degrees."""
    # np.interp handles boundaries (uses endpoint values for angles outside range)
    return np.interp(angle_deg, angles_data_deg, shape_data)

@njit
def compute_direct_floor(light_positions, light_fluxes, X, Y, cob_angles_deg, cob_shape_factor):
    """Calculates direct irradiance W/m^2 on the floor grid using COB angular data."""
    out = np.zeros_like(X, dtype=np.float64)
    rows, cols = X.shape
    if rows == 0 or cols == 0: return out # Handle empty grid
    num_lights = light_positions.shape[0]
    if num_lights == 0: return out # Handle no lights

    for r in range(rows):
        for c in range(cols):
            fx, fy, fz = X[r, c], Y[r, c], 0.0
            accum_irr = 0.0
            for k in range(num_lights):
                lx, ly, lz = light_positions[k, 0], light_positions[k, 1], light_positions[k, 2]
                dx, dy, dz = fx - lx, fy - ly, fz - lz
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15: continue # Avoid singularity if point is exactly at light
                dist = math.sqrt(dist2)

                # Cosine of emission angle (light normal (0,0,-1) dot vector_to_point) / dist
                cos_th_led = -dz / dist
                if cos_th_led <= 1e-6: continue # Light must emit downwards

                # Calculate emission angle in degrees for interpolation
                angle_deg = math.degrees(math.acos(max(0.0, min(1.0, cos_th_led)))) # Clamp cos_th_led to [0,1]
                shape_f = get_cob_shape_factor(angle_deg, cob_angles_deg, cob_shape_factor)

                # Intensity (W/sr) in direction of the point = I0 * shape_factor
                # I0 = Flux / pi for Lambertian total flux reference
                intensity_at_angle = (light_fluxes[k] / math.pi) * shape_f

                # Cosine of incidence angle at floor (floor normal (0,0,1) dot vector_from_point_to_light) / dist
                # Vector from point to light = (-dx, -dy, -dz). Dot product = -dz.
                cos_th_floor = -dz / dist # Same as cos_th_led for horizontal floor
                if cos_th_floor <= 1e-6: continue # Should be redundant

                # Irradiance E = Intensity * cos(theta_surface) / dist^2
                accum_irr += (intensity_at_angle * cos_th_floor) / dist2

            out[r, c] = accum_irr
    return out


@njit
def compute_patch_direct(light_positions, light_fluxes, patch_centers, patch_normals, patch_areas, cob_angles_deg, cob_shape_factor):
    """Calculates direct irradiance W/m^2 on patches using COB angular data."""
    Np = patch_centers.shape[0]
    out = np.zeros(Np, dtype=np.float64)
    if Np == 0: return out # Handle no patches
    num_lights = light_positions.shape[0]
    if num_lights == 0: return out # Handle no lights

    for ip in range(Np):
        pc = patch_centers[ip]
        n_patch = patch_normals[ip]
        norm_n_patch_sq = n_patch[0]*n_patch[0] + n_patch[1]*n_patch[1] + n_patch[2]*n_patch[2]
        if norm_n_patch_sq < 1e-15: continue # Skip degenerate patches
        norm_n_patch = math.sqrt(norm_n_patch_sq)

        accum_E_i = 0.0
        for j in range(num_lights):
            lx, ly, lz = light_positions[j, :3]
            dx, dy, dz = pc[0] - lx, pc[1] - ly, pc[2] - lz
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2 < 1e-15: continue
            dist = math.sqrt(dist2)

            # Cosine of emission angle
            cos_th_led = -dz / dist
            if cos_th_led <= 1e-6: continue

            # Emission angle in degrees
            angle_deg = math.degrees(math.acos(max(0.0, min(1.0, cos_th_led)))) # Clamp cos_th_led to [0,1]
            shape_f = get_cob_shape_factor(angle_deg, cob_angles_deg, cob_shape_factor)

            # Intensity (W/sr) in direction of the patch
            intensity_at_angle = (light_fluxes[j] / math.pi) * shape_f

            # Cosine of incidence angle at patch surface
            # Dot product of patch normal and vector *from* light *to* patch / (norm_n_patch * dist)
            # Vector from light to patch = (dx, dy, dz)
            dot_patch = n_patch[0]*dx + n_patch[1]*dy + n_patch[2]*dz
            # We need angle between normal and vector FROM point TO light (-dx, -dy, -dz)
            cos_in_patch = -(dot_patch) / (dist * norm_n_patch) # Corrected sign

            if cos_in_patch <= 1e-6: continue # Light must hit the front face

            # Irradiance E = Intensity * cos(theta_surface) / dist^2
            accum_E_i += (intensity_at_angle * cos_in_patch) / dist2

        out[ip] = accum_E_i
    return out


@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct_W_m2, patch_areas, patch_refl, num_bounces):
    """
    Iterative radiosity calculation. B_i = E_i + rho_i * H_i. Returns final Radiosity B_i [W/m^2].
    Input patch_direct_W_m2 is the direct irradiance E_i.
    """
    Np = patch_direct_W_m2.shape[0]

    # --- FIX: Initialize patch_radiosity *before* the Np==0 check ---
    # If Np=0, patch_direct_W_m2 is shape (0,), so patch_radiosity will be shape (0,) float64[::1]
    # If Np>0, it's initialized correctly as shape (Np,) float64[::1]
    patch_radiosity = patch_direct_W_m2.copy() # B_i initial state (bounce 0)

    # Handle the case of no patches by returning the initialized (empty) array
    if Np == 0:
        return patch_radiosity
    # --- END FIX ---

    # Iterate for specified number of bounces
    for bounce in range(num_bounces):
        prev_patch_radiosity = patch_radiosity.copy() # B_j from previous bounce
        incident_irradiance = np.zeros(Np, dtype=np.float64) # H_i for this bounce

        # Calculate incident irradiance H_i for each patch i from all other patches j
        for i in range(Np):
            pi, ni = patch_centers[i], patch_normals[i]
            norm_ni_sq = ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2]
            if norm_ni_sq < 1e-15: continue
            norm_ni = math.sqrt(norm_ni_sq)

            accum_H_i = 0.0
            for j in range(Np):
                if i == j: continue # Patch doesn't illuminate itself directly

                B_j_prev = prev_patch_radiosity[j]
                if B_j_prev <= 1e-9: continue # No light leaving patch j

                pj, nj = patch_centers[j], patch_normals[j]
                norm_nj_sq = nj[0]*nj[0] + nj[1]*nj[1] + nj[2]*nj[2]
                if norm_nj_sq < 1e-15: continue
                norm_nj = math.sqrt(norm_nj_sq)

                # Vector from j to i
                dx, dy, dz = pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15: continue
                dist = math.sqrt(dist2)

                # Cosine at source j (angle between nj and vector j->i)
                dot_j = (nj[0]*dx + nj[1]*dy + nj[2]*dz)
                cos_j = dot_j / (dist * norm_nj)

                # Cosine at receiver i (angle between ni and vector i->j = -vector j->i)
                dot_i = -(ni[0]*dx + ni[1]*dy + ni[2]*dz) # Note the sign change for vector i->j
                cos_i = dot_i / (dist * norm_ni)

                # Visibility check
                if cos_j <= 1e-6 or cos_i <= 1e-6: continue

                # Form factor kernel K(j, i) = (cos_j * cos_i) / (pi * dist^2)
                ff_kernel = (cos_j * cos_i) / (math.pi * dist2)

                # Irradiance H_i from j = B_j_prev * K(j, i) * A_j
                accum_H_i += B_j_prev * ff_kernel * patch_areas[j] # W/m^2

            incident_irradiance[i] = accum_H_i

        # Update radiosity for the current bounce: B_i = E_i + rho_i * H_i
        # Use the original direct illumination for E_i in every bounce update
        patch_radiosity = patch_direct_W_m2 + patch_refl * incident_irradiance

    return patch_radiosity # Final radiosity B_i [W/m^2]


@njit
def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, final_patch_radiosity_W_m2, patch_refl):
    """
    Calculates indirect (reflected) irradiance [W/m^2] on the floor grid from all patches,
    using the final patch radiosity B_p.
    """
    rows, cols = X.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return out
    Np = patch_centers.shape[0]
    if Np == 0: return out

    floor_normal = np.array([0.0, 0.0, 1.0])

    for r in range(rows):
        for c in range(cols):
            fx, fy, fz = X[r, c], Y[r, c], 0.0
            floor_point = np.array([fx, fy, fz])
            total_incident_irradiance = 0.0

            for p in range(Np):
                # Use the final radiosity B_p [W/m^2]
                B_p = final_patch_radiosity_W_m2[p]
                if B_p <= 1e-9: continue

                # Skip the floor patch itself (assuming patch 0 is floor) - prevents self-illumination in this step
                # A more robust check would be based on z-coordinate near 0 and normal pointing up.
                if p == 0 and abs(patch_centers[p, 2]) < 1e-6 and patch_normals[p, 2] > 0.9:
                    continue

                pc, n_patch = patch_centers[p], patch_normals[p]
                norm_n_patch_sq = n_patch[0]*n_patch[0] + n_patch[1]*n_patch[1] + n_patch[2]*n_patch[2]
                if norm_n_patch_sq < 1e-15: continue
                norm_n_patch = math.sqrt(norm_n_patch_sq)

                # Vector from patch p to floor point
                dx, dy, dz = fx - pc[0], fy - pc[1], fz - pc[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15: continue
                dist = math.sqrt(dist2)

                # Cosine at source patch p (angle between n_patch and vector p->floor)
                dot_p = n_patch[0]*dx + n_patch[1]*dy + n_patch[2]*dz
                cos_p = dot_p / (dist * norm_n_patch)

                # Cosine at floor point (angle between floor_normal and vector floor->p = -vector p->floor)
                dot_f = -(floor_normal[0]*dx + floor_normal[1]*dy + floor_normal[2]*dz) # Note the sign change
                cos_f = dot_f / dist # norm floor_normal = 1

                # Visibility check
                if cos_p <= 1e-6 or cos_f <= 1e-6: continue

                # Form factor kernel K(p, floor_point) = (cos_p * cos_f) / (pi * dist^2)
                ff_kernel = (cos_p * cos_f) / (math.pi * dist2)

                # Irradiance contribution = B_p * K(p, floor_point) * A_p
                total_incident_irradiance += B_p * ff_kernel * patch_areas[p] # W/m^2

            out[r, c] = total_incident_irradiance
    return out


# --- Main Simulation Function ---
def simulate_lighting(params, geo):
    """
    Run the full simulation pipeline: direct + indirect illumination.
    Returns total PPFD (µmol/m²/s) on the floor grid.
    Uses COB angular data.
    """
    cob_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geo
    if X.size == 0 or p_centers.size == 0: # Handle empty geometry
        print("[WARN] Empty geometry provided to simulate_lighting.")
        # Return a sensible default, maybe based on target or zero
        return np.zeros_like(X, dtype=np.float64) if X.size > 0 else np.array([[0.0]])

    # Ensure params has the correct fixed dimension
    params_adjusted = adjust_candidate_dimension(params, fixed_dim=FIXED_NUM_LAYERS)
    led_intensities = pack_luminous_flux_dynamic(params_adjusted, cob_positions) # Lumens

    # Convert lumens to Watts
    power_arr = led_intensities / LUMINOUS_EFFICACY # Watts per LED

    # Access global angular data (ensure it's defined)
    global COB_angles_deg, COB_shape_factor
    if 'COB_angles_deg' not in globals() or 'COB_shape_factor' not in globals():
        raise NameError("COB angular data (COB_angles_deg, COB_shape_factor) not defined globally.")

    # 1. Direct irradiance on floor grid from LEDs [W/m^2]
    direct_irr_watts = compute_direct_floor(
        cob_positions, power_arr, X, Y, COB_angles_deg, COB_shape_factor
    )

    # 2. Direct irradiance on patches from LEDs (E_i) [W/m^2]
    patch_direct_watts = compute_patch_direct(
        cob_positions, power_arr, p_centers, p_normals, p_areas, COB_angles_deg, COB_shape_factor
    )

    # 3. Calculate final patch radiosity (B_i) after bounces [W/m^2]
    final_patch_radiosity_watts = iterative_radiosity_loop(
        p_centers, p_normals, patch_direct_watts, p_areas, p_refl, NUM_RADIOSITY_BOUNCES
    )

    # 4. Calculate indirect (reflected) irradiance on floor grid from patches [W/m^2]
    reflect_irr_watts = compute_reflection_on_floor(
        X, Y, p_centers, p_normals, p_areas, final_patch_radiosity_watts, p_refl
    )

    # 5. Total irradiance on floor grid [W/m^2]
    total_irr_watts = direct_irr_watts + reflect_irr_watts

    # 6. Convert total irradiance (W/m^2) to PPFD (µmol/m²/s)
    if CONVERSION_FACTOR <= 0:
        print("[WARN] CONVERSION_FACTOR is zero or negative. PPFD will be zero.")
        return np.zeros_like(total_irr_watts)
    return total_irr_watts * CONVERSION_FACTOR

# ------------------------------
# Objective Function & Optimization
# ------------------------------
def objective_function(params, geo, target_ppfd):
    """
    Objective: Minimize MAD (uniformity) + Penalty for deviation from target mean PPFD.
    Returns: objective_value, mean_ppfd, mad
    """
    floor_ppfd = simulate_lighting(params, geo)
    if floor_ppfd.size == 0 or np.all(np.isnan(floor_ppfd)):
        print("[WARN] Simulation returned empty or NaN PPFD. Returning high objective value.")
        # Return defaults indicating failure for unpacking in caller
        return 1e12, 0.0, 1e12 # objective, mean, mad

    mean_ppfd = np.mean(floor_ppfd)
    mad = 0.0 # Initialize MAD

    # Use a small tolerance for checking if mean_ppfd is near zero
    if abs(mean_ppfd) < 1e-9:
        # If mean is effectively zero, MAD is just the mean of absolute values
        mad = np.mean(np.abs(floor_ppfd))
        # Set penalty high if target is non-zero and mean is zero? Or let it be calculated?
        # Let's calculate normally: penalty = (0 - target)**2
    else:
        # Calculate MAD relative to the non-zero mean
        mad = np.mean(np.abs(floor_ppfd - mean_ppfd))

    # Penalize deviation from target PPFD using squared difference
    ppfd_penalty = (mean_ppfd - target_ppfd) ** 2

    # Combine MAD and penalty
    # Adjust the weighting factor (e.g., 2.0) as needed based on desired balance
    objective_value = mad + 2.0 * ppfd_penalty

    # Return all three values
    return objective_value, mean_ppfd, mad

def optimize_lighting(geo, target_ppfd, x0=None, progress_callback=None):
    """
    Runs the full simulation optimization using SLSQP.
    Uses FIXED_NUM_LAYERS for the optimization vector dimension.
    Includes verbose output per iteration (Objective, Mean PPFD, MAD).
    """
    # Ensure initial guess x0 has the correct fixed dimension
    if x0 is None:
        x0_adjusted = np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64) # Default guess
    else:
        x0_adjusted = adjust_candidate_dimension(x0, fixed_dim=FIXED_NUM_LAYERS)
        # Clip initial guess to bounds to ensure feasibility
        x0_adjusted = np.clip(x0_adjusted, MIN_LUMENS, MAX_LUMENS_MAIN)

    bounds = [(MIN_LUMENS, MAX_LUMENS_MAIN)] * FIXED_NUM_LAYERS
    # Estimate can be refined, depends heavily on problem complexity
    total_iterations_estimate = 500 # Adjust as needed
    iteration = 0 # Reset iteration counter for each optimization run

    def wrapped_obj(p):
        nonlocal iteration
        iteration += 1

        # --- Get metrics from objective_function ---
        # objective_function now returns: objective_value, mean_ppfd, mad
        val, mean_p, mad_p = objective_function(p, geo, target_ppfd)

        # --- VERBOSE OUTPUT ---
        # Print metrics available from objective_function directly
        log_msg = (f"[OPTIM] Iter {iteration:03d}: Objective={val:,.3f}, "
                   f"MeanPPFD={mean_p:,.1f}, MAD={mad_p:,.2f}")
        print(log_msg) # Print directly to console

        # Update progress callback if provided
        if progress_callback:
            # Calculate progress percentage
            progress_pct = min(98, (iteration / total_iterations_estimate) * 100)
            progress_callback(f"PROGRESS:{int(progress_pct)}")
            # Optionally send the detailed log message via callback too
            # progress_callback(log_msg)

        # Return only the objective value to the SciPy optimizer
        return val

    if progress_callback:
        progress_callback("[INFO] Starting SLSQP optimization...")

    # Run the minimization
    # Use disp=False to suppress SciPy's standard output if desired
    res = minimize(wrapped_obj, x0_adjusted, method='SLSQP', bounds=bounds,
                   options={'maxiter': total_iterations_estimate, 'disp': False, 'ftol': 1e-7}) # Adjust ftol/maxiter as needed

    # --- Post-optimization Messages ---
    if not res.success:
        # Provide details if convergence failed
        msg = f"[WARN] Optimization did not converge: {res.message} (Final Objective: {res.fun:.3f}, Iterations: {res.nit})"
        print(msg)
        if progress_callback: progress_callback(msg)
    else:
        # Confirm successful convergence
        msg = f"[INFO] Optimization finished successfully. Objective: {res.fun:.3f}, Iterations: {res.nit}"
        print(msg)
        if progress_callback: progress_callback(msg)

    if progress_callback:
        progress_callback("PROGRESS:99") # Indicate optimization phase complete

    # Return the best parameters found (res.x) and the geometry
    return res.x, geo

# ------------------------------
# Surrogate Model & Persistence Functions
# ------------------------------

def adjust_candidate_dimension(candidate, fixed_dim=FIXED_NUM_LAYERS, default_value=8000.0):
    """Adjusts candidate vector to fixed_dim entries."""
    candidate = np.asarray(candidate, dtype=np.float64)
    current_dim = candidate.shape[0]
    if current_dim == fixed_dim:
        return candidate
    elif current_dim < fixed_dim:
        pad = np.full((fixed_dim - current_dim,), default_value)
        return np.concatenate([candidate, pad])
    else: # current_dim > fixed_dim
        return candidate[:fixed_dim]

# --- Surrogate Normalization ---
# M_SPLINE_SAMPLES controls the dimension of the normalized 'shape' vector.
# Let's keep it tied to FIXED_NUM_LAYERS.
# The full normalized feature vector 'z' will have dimension M_SPLINE_SAMPLES + 1.

def forward_transform(raw_candidate):
    """
    Transforms a raw candidate vector (assumed length FIXED_NUM_LAYERS)
    into: Total lumens (T) and a normalized shape vector (y) sampled at M_SPLINE_SAMPLES points.
    Returns: T, u (sample points 0..1), y (normalized shape samples).
    """
    x = np.asarray(raw_candidate, dtype=np.float64)
    # Ensure input has fixed dimension before transform
    if len(x) != FIXED_NUM_LAYERS:
         print(f"[WARN] forward_transform input length {len(x)} != FIXED_NUM_LAYERS {FIXED_NUM_LAYERS}. Adjusting.")
         x = adjust_candidate_dimension(x, fixed_dim=FIXED_NUM_LAYERS)

    n = len(x) # Should be FIXED_NUM_LAYERS
    T = np.sum(x)
    if T < 1e-9: T = 1e-9 # Avoid division by zero

    if n == 0: return T, np.array([]), np.array([])
    if n == 1:
        u_orig = np.array([0.0])
        y_orig = x / T # Should be [1.0]
    else:
        u_orig = np.linspace(0.0, 1.0, n) # Original indices
        y_orig = x / T # Original normalized shape

    # Interpolate y_orig onto M_SPLINE_SAMPLES points (v) using a spline
    if M_SPLINE_SAMPLES <= 0: return T, np.array([]), np.array([])
    if M_SPLINE_SAMPLES == 1:
        v = np.array([0.5]) # Sample at midpoint? Or just average? Let's average y_orig.
        y_sampled = np.array([np.mean(y_orig)])
    elif n <= 3: # Spline needs >3 points, use linear interpolation for few layers
         v = np.linspace(0.0, 1.0, M_SPLINE_SAMPLES)
         y_sampled = np.interp(v, u_orig, y_orig)
    else:
        try:
             spline = CubicSpline(u_orig, y_orig, bc_type='natural')
             v = np.linspace(0.0, 1.0, M_SPLINE_SAMPLES)
             y_sampled = spline(v)
             # Ensure sampled values are non-negative and sum approximately to 1 (renormalize if needed)
             y_sampled = np.maximum(0, y_sampled)
             y_sum = np.sum(y_sampled)
             if y_sum > 1e-9: y_sampled /= y_sum
        except Exception as e:
             print(f"[WARN] CubicSpline failed in forward_transform: {e}. Using linear interpolation.")
             v = np.linspace(0.0, 1.0, M_SPLINE_SAMPLES)
             y_sampled = np.interp(v, u_orig, y_orig)


    return T, v, y_sampled # Return T and the fixed-dimension shape vector y_sampled


def inverse_transform(T, v, y_sampled, target_dim=FIXED_NUM_LAYERS):
    """
    Reconstructs a raw candidate vector of 'target_dim' from total lumens T
    and the normalized shape vector 'y_sampled' defined at points 'v'.
    Uses interpolation (spline or linear) to get values at target_dim points.
    """
    if target_dim <= 0: return np.array([])
    if T <= 0: return np.zeros(target_dim)

    m_sampled = len(y_sampled) # Should be M_SPLINE_SAMPLES
    if m_sampled == 0: return np.full(target_dim, T / target_dim) # Distribute total evenly
    if m_sampled == 1: return np.full(target_dim, T * y_sampled[0] / target_dim) # Scale single value? Return average.

    # Target indices where we need the shape value
    u_target = np.linspace(0.0, 1.0, target_dim)

    # Interpolate the sampled shape (y_sampled at v) onto the target indices (u_target)
    if m_sampled <= 3: # Use linear interpolation if few sample points
        y_new_norm = np.interp(u_target, v, y_sampled)
    else:
        try:
            spline = CubicSpline(v, y_sampled, bc_type='natural')
            y_new_norm = spline(u_target)
        except Exception as e:
            print(f"[WARN] CubicSpline failed in inverse_transform: {e}. Using linear interpolation.")
            y_new_norm = np.interp(u_target, v, y_sampled)

    # Ensure non-negative and renormalize the interpolated shape vector
    y_new_norm = np.maximum(0, y_new_norm)
    y_sum = np.sum(y_new_norm)
    if y_sum > 1e-9:
        y_new_norm /= y_sum # Renormalize to sum to 1

    # Scale by total lumens T
    x_new = T * y_new_norm

    # Clip to bounds as a safety measure
    x_new = np.clip(x_new, MIN_LUMENS, MAX_LUMENS_MAIN)

    return x_new

# --- Training Data Handling ---

def load_training_data():
    """Loads normalized features (X), objective values (y), and geometry features (G)."""
    if os.path.exists(TRAINING_DATA_FILE):
        try:
            with np.load(TRAINING_DATA_FILE, allow_pickle=True) as data:
                X_train = data["X_train"] # Normalized features [T, y1..M]
                y_train = data["y_train"] # Objective function values
                G_train = data["G_train"] # Geometry features [W, L, PPFD_target, H]
            print(f"[INFO] Loaded {len(X_train)} training samples from {TRAINING_DATA_FILE}.")
            # Basic validation
            if X_train.shape[0] != y_train.shape[0] or X_train.shape[0] != G_train.shape[0]:
                 print("[ERROR] Training data dimensions mismatch. Discarding data.")
                 return None, None, None
            # Check expected feature dimension based on M_SPLINE_SAMPLES
            expected_dim = M_SPLINE_SAMPLES + 1
            if X_train.shape[1] != expected_dim:
                 print(f"[WARN] Loaded X_train dimension ({X_train.shape[1]}) doesn't match expected ({expected_dim}). May cause issues.")
            return X_train, y_train, G_train
        except Exception as e:
            print(f"[ERROR] Failed to load training data from {TRAINING_DATA_FILE}: {e}")
            return None, None, None
    else:
        print(f"[INFO] No training data file found ({TRAINING_DATA_FILE}). Starting fresh.")
        return None, None, None

def save_training_data(X_train, y_train, G_train):
    """Saves normalized features, objectives, and geometry features."""
    try:
        # Check for NaN or Inf before saving
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)): print("[WARN] NaN/Inf found in X_train before saving.")
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)): print("[WARN] NaN/Inf found in y_train before saving.")
        if np.any(np.isnan(G_train)) or np.any(np.isinf(G_train)): print("[WARN] NaN/Inf found in G_train before saving.")

        np.savez(TRAINING_DATA_FILE, X_train=X_train, y_train=y_train, G_train=G_train)
        print(f"[INFO] Saved {len(X_train)} training samples to {TRAINING_DATA_FILE}.")
    except Exception as e:
        print(f"[ERROR] Failed to save training data to {TRAINING_DATA_FILE}: {e}")


def update_training_data(raw_candidate, objective_value, geom_features):
    """
    Adds a new sample (raw_candidate, objective_value) and its geometry features
    to the persistent training data. Normalizes the candidate first.
    """
    # 1. Ensure raw_candidate has the fixed dimension
    candidate_adjusted = adjust_candidate_dimension(raw_candidate, fixed_dim=FIXED_NUM_LAYERS)

    # 2. Normalize the candidate using forward transform
    T, _, y_sampled = forward_transform(candidate_adjusted) # v (indices) is not needed here
    x_norm = np.hstack(([T], y_sampled)) # Normalized feature vector [T, y1..M]

    # Check dimension
    expected_dim = M_SPLINE_SAMPLES + 1
    if len(x_norm) != expected_dim:
        print(f"[ERROR] Dimension mismatch in update_training_data: generated {len(x_norm)}, expected {expected_dim}. Skipping update.")
        return load_training_data() # Return existing data

    # Validate objective value
    if np.isnan(objective_value) or np.isinf(objective_value):
        print(f"[WARN] Invalid objective value ({objective_value}) received. Skipping update.")
        return load_training_data()

    print(f"[DEBUG] Adding training sample: Geom={geom_features}, Objective={objective_value:.3f}, T={T:.1f}")

    # 3. Load old data
    X_old, y_old, G_old = load_training_data()

    # 4. Append new sample
    if X_old is not None:
        # Ensure dimensions match before stacking
        if X_old.shape[1] != expected_dim:
            print(f"[WARN] Existing X_train dimension ({X_old.shape[1]}) differs from new ({expected_dim}). Starting new dataset.")
            X_train = np.array([x_norm], dtype=np.float64)
            y_train = np.array([objective_value], dtype=np.float64)
            G_train = np.array([geom_features], dtype=np.float64)
        else:
            # Optional: Check for duplicate geometry features to avoid redundant entries?
            # Or check for very similar x_norm vectors? For now, just append.
            X_train = np.vstack((X_old, x_norm))
            y_train = np.hstack((y_old, objective_value))
            G_train = np.vstack((G_old, geom_features))
    else:
        X_train = np.array([x_norm], dtype=np.float64)
        y_train = np.array([objective_value], dtype=np.float64)
        G_train = np.array([geom_features], dtype=np.float64)

    # 5. Save updated data
    save_training_data(X_train, y_train, G_train)

    return X_train, y_train, G_train


# --- Surrogate Training & Optimization ---

def train_surrogate(X_train, y_train):
    """Trains a Gaussian Process on the normalized training data [T, y1..M]."""
    if X_train is None or y_train is None or len(X_train) == 0:
        print("[WARN] No training data available for surrogate.")
        return None
    if X_train.shape[0] != y_train.shape[0]:
        print("[ERROR] Mismatch between X_train and y_train samples for surrogate training.")
        return None

    # Ensure correct feature dimension
    expected_dim = M_SPLINE_SAMPLES + 1
    if X_train.shape[1] != expected_dim:
        print(f"[ERROR] X_train dimension ({X_train.shape[1]}) does not match expected ({expected_dim}) for surrogate training.")
        return None

    # Define Kernel
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) \
             + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2)) # Allow some noise

    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=8, # Increase restarts
                                  normalize_y=True, # Normalize target objective values
                                  random_state=42)
    try:
        gp.fit(X_train, y_train)
        print(f"[INFO] Surrogate GP trained on {len(X_train)} samples. Final kernel: {gp.kernel_}")
        print(f"[INFO] Log-Marginal-Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")
        return gp
    except Exception as e:
        print(f"[ERROR] Gaussian Process training failed: {e}")
        # Example: Check if X_train contains NaNs or Infs
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)): print("   >> NaN/Inf detected in X_train.")
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)): print("   >> NaN/Inf detected in y_train.")
        return None


def optimize_with_surrogate(gp_model, x0_raw=None):
    """
    Optimizes the surrogate model in the normalized space [T, y1..M].
    Returns the predicted best *raw* candidate vector (length FIXED_NUM_LAYERS).
    """
    if gp_model is None:
        print("[WARN] No GP model provided to optimize_with_surrogate. Returning default guess.")
        return np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64)

    # Determine M from the GP model's training data dimension
    # (Should match M_SPLINE_SAMPLES if consistent)
    trained_feature_dim = gp_model.X_train_.shape[1]
    M = trained_feature_dim - 1
    if M != M_SPLINE_SAMPLES:
         print(f"[WARN] Mismatch: M_SPLINE_SAMPLES={M_SPLINE_SAMPLES}, M from GP data={M}. Using M={M}.")

    # Create initial guess in normalized space
    if x0_raw is None:
        x0_raw = np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64)
    x0_adjusted = adjust_candidate_dimension(x0_raw, fixed_dim=FIXED_NUM_LAYERS)
    T0, v0, y0_sampled = forward_transform(x0_adjusted)
    z0 = np.hstack(([T0], y0_sampled)) # z = [T, y1..M]

    # Ensure z0 has the correct dimension expected by the GP model
    if len(z0) != trained_feature_dim:
        print(f"[ERROR] Dimension mismatch for surrogate optimization: z0 len {len(z0)}, expected {trained_feature_dim}. Using default.")
        # Fallback: Use the mean of the training data as starting point?
        # For now, just use default T and flat shape
        T0_default = 10000.0
        y0_default = np.full(M, 1.0/M if M > 0 else 0.0)
        z0 = np.hstack(([T0_default], y0_default))
        if len(z0) != trained_feature_dim: # Still wrong? Abort.
             print("[ERROR] Cannot create valid z0 for surrogate optimization.")
             return np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64)


    def surrogate_objective(z):
        # Predict objective value using the GP
        z_reshaped = z[np.newaxis, :]
        try:
            # GP prediction is the mean of the posterior distribution
            mean_prediction = gp_model.predict(z_reshaped, return_std=False)[0]
            return mean_prediction
        except Exception as e:
            print(f"[WARN] GP prediction failed during optimization: {e}. Returning high value.")
            return 1e12 # Return large value on prediction failure

    # Bounds for normalized space: T and shape fractions y_i
    # Sum of raw lumens can be up to FIXED_NUM_LAYERS * MAX_LUMENS_MAIN
    max_total_lumens = FIXED_NUM_LAYERS * MAX_LUMENS_MAIN
    min_total_lumens = FIXED_NUM_LAYERS * MIN_LUMENS # Lower bound for T
    # Bounds: T in [min_T, max_T], shape samples y_i in ~[0, 1/M]? More relaxed [0,1] might be okay.
    # The constraint that y_i's should sum to 1 is implicitly handled by normalization in inverse_transform.
    bounds_norm = [(min_total_lumens, max_total_lumens)] + [(0.0, 1.0)] * M

    print(f"[INFO] Starting surrogate optimization from T={z0[0]:.1f}...")
    res_surr = minimize(surrogate_objective, z0, method='L-BFGS-B', # L-BFGS-B is often good for GPs
                        bounds=bounds_norm, options={'maxiter': 150, 'disp': False})

    if not res_surr.success:
        print(f"[WARN] Surrogate optimization did not converge: {res_surr.message} (Objective: {res_surr.fun:.3f})")
    else:
        print(f"[INFO] Surrogate optimization complete. Predicted objective= {res_surr.fun:.3f}")

    z_best_norm = res_surr.x
    T_best = z_best_norm[0]
    shape_best = z_best_norm[1:]

    # Reconstruct the raw candidate vector of length FIXED_NUM_LAYERS
    # Need the sampling points 'v' corresponding to shape_best (linspace 0..1, M points)
    v_best = np.linspace(0.0, 1.0, M) if M > 0 else np.array([])
    x_raw_best = inverse_transform(T_best, v_best, shape_best, target_dim=FIXED_NUM_LAYERS)

    print(f"[INFO] Best raw candidate from surrogate: T={T_best:.1f}, first val={x_raw_best[0]:.1f} (if exists)")
    return x_raw_best

# --- Regression Model for Initial Guess ---

def train_regression_model_for_normalized(X_train_norm, G_train):
    """
    Trains a model mapping Geometry Features G -> Normalized Features X_norm = [T, y1..M].
    Returns the trained model (e.g., RandomForestRegressor wrapped in MultiOutputRegressor).
    """
    if X_train_norm is None or G_train is None or len(X_train_norm) < 10: # Need sufficient data
        print("[INFO] Not enough training data (<10 samples) for geometry->normalized features regression model.")
        return None

    if X_train_norm.shape[0] != G_train.shape[0]:
        print("[ERROR] Mismatch between X_train_norm and G_train samples for regression training.")
        return None

    try:
        # Use RandomForest for potentially non-linear relationships
        # Wrap with MultiOutputRegressor to predict the multi-dimensional output X_norm
        base_estimator = RandomForestRegressor(n_estimators=50, # More estimators
                                                random_state=42,
                                                n_jobs=-1, # Use all cores
                                                max_depth=10, # Limit depth
                                                min_samples_leaf=3) # Avoid overfitting tiny leaves
        reg_model = MultiOutputRegressor(base_estimator)
        reg_model.fit(G_train, X_train_norm)
        print(f"[INFO] Regression model (G -> Normalized Features) trained on {len(G_train)} samples.")
        # Optional: Evaluate model score? reg_model.score(G_train, X_train_norm)
        return reg_model
    except Exception as e:
        print(f"[ERROR] Failed to train regression model: {e}")
        return None


def predict_initial_guess_regression(geom_features, regression_model):
    """
    Uses the trained regression model to predict a normalized feature vector 'z'
    from geometry features, then converts it back to a raw candidate vector guess.
    Returns a raw candidate vector (length FIXED_NUM_LAYERS).
    """
    if regression_model is None:
        print("[WARN] No regression model available for initial guess. Using default.")
        return np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64)

    try:
        # Predict the normalized feature vector z = [T, y1..M]
        z_pred_norm = regression_model.predict([geom_features])[0]

        # Check prediction dimension
        expected_dim = M_SPLINE_SAMPLES + 1
        if len(z_pred_norm) != expected_dim:
             print(f"[ERROR] Regression model predicted wrong dimension ({len(z_pred_norm)} vs {expected_dim}). Using default guess.")
             return np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64)

        T_pred = z_pred_norm[0]
        shape_pred = z_pred_norm[1:]
        M = len(shape_pred) # Should be M_SPLINE_SAMPLES

        # Need the corresponding sampling points 'v' for inverse transform
        v_pred = np.linspace(0.0, 1.0, M) if M > 0 else np.array([])

        # Convert back to raw candidate vector
        x_guess_raw = inverse_transform(T_pred, v_pred, shape_pred, target_dim=FIXED_NUM_LAYERS)

        print(f"[INFO] Regression-based initial guess computed: T={T_pred:.1f}")
        # print(f"   Raw guess (first few): {x_guess_raw[:min(5, FIXED_NUM_LAYERS)]}") # Debug: print first few values
        return x_guess_raw

    except Exception as e:
        print(f"[ERROR] Failed to predict initial guess using regression model: {e}")
        return np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64) # Fallback guess

# ------------------------------
# Main Simulation Orchestration
# ------------------------------

def run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height=3.0,
                      progress_callback=None):
    """
    Main function to run the ML-enhanced simulation for a single scenario.
    Includes detailed final console output.
    """
    print("\n" + "="*60)
    print(f"Starting ML Simulation: {floor_width_ft}x{floor_length_ft} ft, Target {target_ppfd} PPFD, Height {floor_height} ft")
    print("="*60)

    # Clear geometry caches for new dimensions
    cached_build_floor_grid.cache_clear()
    cached_build_patches.cache_clear()

    # Convert units (ft to m)
    ft2m = 0.3048
    W_m = floor_width_ft * ft2m
    L_m = floor_length_ft * ft2m
    H_m = floor_height * ft2m

    if W_m <= 0 or L_m <= 0 or H_m <= 0:
        print("[ERROR] Invalid room dimensions (must be positive). Aborting.")
        return None

    # 1. Prepare Geometry
    if progress_callback: progress_callback("PROGRESS:5")
    if progress_callback: progress_callback("[INFO] Preparing geometry...")
    geo = prepare_geometry(W_m, L_m, H_m)
    cob_positions, X_grid, Y_grid, patches = geo
    if X_grid.size == 0 or patches[0].size == 0:
         print("[ERROR] Geometry preparation failed (empty grid or patches). Aborting.")
         return None
    if progress_callback: progress_callback("[INFO] Geometry prepared.")

    # 2. Load Training Data
    if progress_callback: progress_callback("PROGRESS:10")
    X_train_norm, y_train_obj, G_train = load_training_data()

    # 3. Train Regression Model (Geometry -> Normalized Features)
    if progress_callback: progress_callback("PROGRESS:15")
    regression_model = train_regression_model_for_normalized(X_train_norm, G_train)

    # 4. Predict Initial Guess using Regression Model
    if progress_callback: progress_callback("PROGRESS:20")
    geom_features = [floor_width_ft, floor_length_ft, target_ppfd, floor_height]
    initial_guess_raw = predict_initial_guess_regression(geom_features, regression_model)
    if progress_callback: progress_callback("[INFO] Initial guess generated.")

    # 5. Train GP Surrogate (Normalized Features -> Objective Value)
    if progress_callback: progress_callback("PROGRESS:25")
    gp_model = train_surrogate(X_train_norm, y_train_obj)
    if progress_callback: progress_callback("[INFO] GP Surrogate training attempted.")

    # 6. Optimize with Surrogate (Optional - currently skipped, using regression guess directly)
    x0_for_full_opt = initial_guess_raw

    # 7. Run Full Optimization (SLSQP)
    if progress_callback: progress_callback("PROGRESS:40")
    print("[INFO] Starting full optimization (SLSQP)...")
    best_params_raw, _ = optimize_lighting(geo, target_ppfd, x0=x0_for_full_opt, progress_callback=progress_callback)
    print(f"[INFO] Full optimization finished.")

    # 8. Calculate Final Metrics using Optimized Parameters
    if progress_callback: progress_callback("PROGRESS:99")
    print("[INFO] Calculating final PPFD map and metrics...")
    # Initialize metrics to default/error values
    mean_ppfd, mad, rmse, dou, cv, final_obj = 0.0, 0.0, 0.0, 0.0, 0.0, 1e12
    final_ppfd_map = np.array([[]]) # Default empty map

    # Only simulate and calculate if optimization returned valid parameters
    if best_params_raw is not None:
        final_ppfd_map = simulate_lighting(best_params_raw, geo)

        if final_ppfd_map.size == 0 or np.all(np.isnan(final_ppfd_map)):
            print("[ERROR] Final simulation failed. Cannot calculate metrics.")
            # Keep default error values for metrics
        else:
            mean_ppfd = float(np.mean(final_ppfd_map))
            if abs(mean_ppfd) < 1e-9:
                print("[WARN] Mean PPFD is near zero. Metrics might be unreliable.")
                mad = float(np.mean(np.abs(final_ppfd_map)))
                rmse = float(np.sqrt(np.mean(final_ppfd_map**2)))
                std_dev = float(np.std(final_ppfd_map))
                dou = 0.0
                cv = float('inf') if std_dev > 1e-9 else 0.0 # Avoid division by zero noise
            else:
                mad = float(np.mean(np.abs(final_ppfd_map - mean_ppfd)))
                rmse = float(np.sqrt(np.mean((final_ppfd_map - mean_ppfd)**2)))
                std_dev = float(np.std(final_ppfd_map))
                # Ensure DOU calculation is safe and capped at 0
                dou = float(max(0.0, 100 * (1 - rmse / mean_ppfd)))
                cv = float(100 * (std_dev / mean_ppfd))

            # Calculate the final objective function value using the final parameters
            # Re-use the objective_function logic
            final_obj_calc, _, _ = objective_function(best_params_raw, geo, target_ppfd)
            final_obj = float(final_obj_calc)

    print(f"[RESULT] Mean PPFD: {mean_ppfd:.2f} µmol/m²/s")
    print(f"[RESULT] Uniformity - DOU: {dou:.2f}%, CV: {cv:.2f}%, MAD: {mad:.2f}, RMSE: {rmse:.2f}")
    print(f"[RESULT] Final Objective Value: {final_obj:.3f}")

    # 9. Update Training Data (only if optimization succeeded and metrics are valid)
    if progress_callback: progress_callback("PROGRESS:100")
    if best_params_raw is not None and final_ppfd_map.size > 0 and not np.all(np.isnan(final_ppfd_map)):
        print("[INFO] Updating training data...")
        # Save the result of the *full* optimization
        update_training_data(best_params_raw, final_obj, geom_features)
        print("[INFO] Training data updated.")
    else:
        print("[WARN] Skipping training data update due to optimization/simulation failure.")

    # --- FINAL CONSOLE OUTPUT ---
    print("\n" + "-"*25 + " Final Result Summary " + "-"*25)
    print(f"  Floor Size (WxL):   {floor_width_ft} x {floor_length_ft} ft")
    print(f"  Target PPFD:        {target_ppfd} µmol/m²/s")
    print(f"  MAD:                {mad:.3f} µmol/m²/s")
    print(f"  Mean PPFD (Result): {mean_ppfd:.3f} µmol/m²/s")
    print(f"  DOU (%):            {dou:.2f}%")
    print(f"  Objective Value:    {final_obj:.4f}")
    print("-" * 60)
    print("  Optimized Lumens per Layer:")
    # Check if best_params_raw is valid before iterating
    if best_params_raw is not None and len(best_params_raw) > 0:
        # Iterate up to the length of the optimized vector (should be FIXED_NUM_LAYERS)
        for i, lumens in enumerate(best_params_raw):
             print(f"    Layer {i:02d}: {lumens:,.1f} lm")
    else:
        print("    N/A (Optimization failed or no parameters found)")
    print("-" * 60)
    # --- END FINAL CONSOLE OUTPUT ---

    print("ML Simulation Run Complete.")
    print("="*60 + "\n")

    # Determine status based on whether valid parameters and results were obtained
    sim_status = "Success" if best_params_raw is not None and final_ppfd_map.size > 0 else "Failure"

    # Prepare result dictionary
    result = {
        "optimized_lumens_by_layer": best_params_raw.tolist() if best_params_raw is not None else [],
        "objective_value": final_obj if sim_status == "Success" else None, # Return None if failed
        "mean_ppfd": mean_ppfd if sim_status == "Success" else None,
        "mad": mad if sim_status == "Success" else None,
        "rmse": rmse if sim_status == "Success" else None,
        "dou": dou if sim_status == "Success" else None,
        "cv": cv if sim_status == "Success" else None,
        "floor_width_ft": floor_width_ft,
        "floor_length_ft": floor_length_ft,
        "target_ppfd": target_ppfd,
        "floor_height_ft": floor_height,
        "heatmap_grid_ppfd": final_ppfd_map.tolist() if final_ppfd_map.size > 0 else [],
        "status": sim_status
    }
    return result

def run_simulation(floor_width_ft=14.0, floor_length_ft=14.0, target_ppfd=1250.0, floor_height=3.0):
    """
    Simplified entry point for running a single simulation.
    Removed the non-existent 'side_by_side' argument.
    """
    result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height)
    if result:
        print("\n[FINAL RESULT] Simulation Output:")
        # Print a summary, full result dict is returned
        print(f"  Mean PPFD: {result['mean_ppfd']:.2f}")
        print(f"  DOU: {result['dou']:.2f}%")
        print(f"  Objective: {result['objective_value']:.3f}")
        print(f"  Optimized Lumens (first 5): {result['optimized_lumens_by_layer'][:5]}")
    else:
        print("\n[ERROR] Simulation run failed.")
    return result


# --- Batch Processing Functions ---

def continuous_run(floor_size):
    """
    Runs simulations for a given floor size across the TARGET_PPFD_RANGE.
    Results are saved to the training data file via run_ml_simulation.
    """
    floor_width_ft, floor_length_ft = floor_size # Corrected order W, L
    floor_height = 3.0  # Default height

    print(f"\n>>> Starting Batch Simulations for Floor Size: {floor_width_ft}x{floor_length_ft} ft <<<\n")

    results_summary = []
    for target_ppfd in TARGET_PPFD_RANGE:
        print(f"--- Running Target PPFD = {target_ppfd} ---")

        # Run the full ML simulation - this now handles training data update internally
        result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height)

        if result and result['status'] == 'Success':
            print(f"--- Completed Target PPFD = {target_ppfd} ---")
            print(f"  Optimized Mean PPFD: {result['mean_ppfd']:.2f}")
            print(f"  DOU: {result['dou']:.2f}%")
            print(f"  Objective: {result['objective_value']:.3f}")
            results_summary.append({
                "target": target_ppfd,
                "mean": result['mean_ppfd'],
                "dou": result['dou']
            })
        else:
            print(f"[ERROR] Simulation failed for Target PPFD = {target_ppfd}")
            results_summary.append({
                "target": target_ppfd,
                "mean": "Failed", "dou": "Failed"
            })
        print("-" * 50)

    print(f"\n>>> Finished Batch Simulations for Floor Size: {floor_width_ft}x{floor_length_ft} ft <<<")
    print("Summary:")
    for res in results_summary:
        print(f"  Target: {res['target']}, Mean PPFD: {res['mean']}, DOU: {res['dou']}")
    print("-" * 50 + "\n")

def run_all_floor_sizes():
    """Runs batch simulations for all PRE_DEFINED_FLOOR_SIZES."""
    print("*"*70)
    print(" Starting Batch Simulations for ALL Pre-defined Floor Sizes ")
    print("*"*70 + "\n")

    for floor_size in PRE_DEFINED_FLOOR_SIZES:
        continuous_run(floor_size) # Assumes floor_size is (Width, Length)

    print("\n" + "*"*70)
    print(" ALL BATCH SIMULATIONS COMPLETED! ")
    print("*"*70)

# ------------------------------
# Script Execution
# ------------------------------
if __name__ == '__main__':
    # --- Choose Action ---

    # Action 1: Run a single simulation
    def run_simulation(floor_width_ft=40.0, floor_length_ft=40.0, target_ppfd=1250.0, floor_height=3.0):
        """
        Simplified entry point for running a single simulation.
        Relies on run_ml_simulation for detailed console output.
        """
        print(f"\n--- Running Simulation: {floor_width_ft}x{floor_length_ft} ft, Target {target_ppfd} PPFD ---")
        # Call the main simulation function which now handles its own detailed output
        result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height)

        # Check the status returned by run_ml_simulation
        if not result or result.get('status') == 'Failure':
            print("\n[ERROR] Simulation run failed.")
            # Optionally return None or a specific error indicator if needed upstream
        else:
            # If needed, you can still print a very brief confirmation here
            print(f"--- Simulation Complete for {floor_width_ft}x{floor_length_ft} ft ---")

        # Return the full result dictionary (or None/error indicator on failure)
        return result
    run_simulation()

    # Action 2: Run batch simulations for all pre-defined floor sizes
    #print("Running batch simulations for all pre-defined floor sizes...")
    #run_all_floor_sizes()

    # Action 3: (Optional) Bootstrap training data if file is empty
    # _, _, G_train = load_training_data()
    # if G_train is None or len(G_train) < 5:
    #      print("\nBootstrapping training data with a few runs...")
    #      run_simulation(floor_width_ft=4.0, floor_length_ft=4.0, target_ppfd=500.0)
    #      run_simulation(floor_width_ft=8.0, floor_length_ft=8.0, target_ppfd=1000.0)
    #      run_simulation(floor_width_ft=12.0, floor_length_ft=12.0, target_ppfd=1500.0)
    #      print("Bootstrapping complete.")