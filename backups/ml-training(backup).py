#!/usr/bin/env python3
"""
Full Radiosities Simulation with Persistent Gaussian Process Surrogate
-----------------------------------------------------------------------
This script implements your radiosity‐based simulation to optimize luminous
flux assignments for COB LEDs in a simple rectangular room. It uses a
Gaussian Process surrogate that is trained and updated persistently over runs,
so that subsequent runs (with the same room dimensions and target PPFD) start
with an informed initial guess for the full optimization.

The surrogate training data is stored in a file (TRAINING_DATA_FILE) so that it
"learns" from each full simulation run.

Usage:
    python ml_simulation.py

The main entry point is the run_simulation() function.
"""

import math
import numpy as np
from scipy.optimize import minimize

from numba import njit
from functools import lru_cache
import matplotlib
matplotlib.use("Agg")
import os


# Additional imports for the surrogate model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from scipy.interpolate import CubicSpline


# ------------------------------
# Configuration & Constants
# ------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

# Mapping: Number of COB layers to the *maximum* floor area (m^2) it's suitable for.
# Derived from the user's specification (e.g., 1 layer up to 0.34m², 2 layers up to 0.98m², etc.)
# Ensure keys are sorted for easier lookup later.
COB_LAYERS_TO_MAX_AREA_M2 = {
     1: 0.34,  2: 0.98,  3: 1.62,  4: 2.26,  5: 2.9,
     6: 3.54,  7: 4.18,  8: 4.82,  9: 5.46, 10: 6.10,
    11: 6.74, 12: 7.38, 13: 8.02, 14: 8.66, 15: 9.30,
    16: 9.94, 17: 10.58, 18: 11.22, 19: 11.68, 20: 12.32
}
# Extract sorted layers and corresponding max areas for efficient lookup
_sorted_layers = sorted(COB_LAYERS_TO_MAX_AREA_M2.keys())
_sorted_max_areas = [COB_LAYERS_TO_MAX_AREA_M2[l] for l in _sorted_layers]

LUMINOUS_EFFICACY = 182.0         # lm/W
# Update SPD_FILE path as needed.
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv"

NUM_RADIOSITY_BOUNCES = 5
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

# Floor grid resolution (meters)
FLOOR_GRID_RES = 0.08

MIN_LUMENS = 1000.0
MAX_LUMENS_MAIN = 24000.0

# Global constant: Force all candidate vectors to have this many layers.
FIXED_NUM_LAYERS = 10

# Persistent training data file for the surrogate model
TRAINING_DATA_FILE = "simul_training_data.npz"

# Global flag to control continuous learning
running = False

# Global variable to hold the simulation thread
sim_thread = None

# Pre-defined floor sizes (length x width in feet)
PRE_DEFINED_FLOOR_SIZES = [
    (2, 2),
    (4, 4),
    (6, 6),
    (8, 8),
    (10, 10),
    (12, 12),
    (14, 14),
    (16, 16),
    (12, 16),   
]

# Target PPFD range (0 to 2000 in increments of 50)
TARGET_PPFD_RANGE = range(0, 2001, 50)

# ------------------------------
# SPD Conversion Factor Calculation
# ------------------------------
def compute_conversion_factor(spd_file):
    try:
        spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
    except Exception as e:
        print("Error loading SPD data:", e)
        return 0.0138
    wl = spd[:, 0]
    intens = spd[:, 1]
    mask_par = (wl >= 400) & (wl <= 700)
    tot = np.trapezoid(intens, wl)  # Deprecated; you may use np.trapezoid
    tot_par = np.trapezoid(intens[mask_par], wl[mask_par])
    PAR_fraction = tot_par / tot if tot > 0 else 1.0

    wl_m = wl * 1e-9
    h, c, N_A = 6.626e-34, 3.0e8, 6.022e23
    numerator = np.trapezoid(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
    denominator = np.trapezoid(intens[mask_par], wl_m[mask_par]) or 1e-15
    lambda_eff = numerator / denominator
    E_photon = (h * c / lambda_eff) if lambda_eff > 0 else 1.0
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction
    print(f"[INFO] SPD: PAR fraction={PAR_fraction:.3f}, conv_factor={conversion_factor:.5f} µmol/J")
    return conversion_factor

CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)

# ------------------------------
# Geometry Preparation Functions
# ------------------------------
def prepare_geometry(W, L, H):
    """
    Build geometry used in the simulation:
      - COB positions (with dynamic staggered layout)
      - Floor grid (X, Y)
      - Patches for walls, floor, ceiling (centers, areas, normals, reflectances)
    Returns a tuple: (cob_positions, X, Y, (patch_centers, patch_areas, patch_normals, patch_refl))
    """
    cob_positions = build_cob_positions(W, L, H)
    X, Y = build_floor_grid(W, L)  # Uses caching for efficiency.
    patches = build_patches(W, L, H)
    return (cob_positions, X, Y, patches)

def build_cob_positions(W, L, H):
    """
    Build dynamic COB positions using a Diamond:61 pattern.
    The number of layers ('n' in the pattern) is determined by the floor area (W * L)
    to maintain a consistent COB density based on COB_LAYERS_TO_MAX_AREA_M2 mapping.
    Each position is (x, y, H, layer) where layer indicates the ring number (0 to n).
    """
    floor_area_m2 = W * L

    # Determine the target number of layers 'n' based on floor area
    target_n = 0 # Corresponds to the radius of the diamond pattern
    if floor_area_m2 <= _sorted_max_areas[0]:
        target_n = _sorted_layers[0] # Use layer 1 for the smallest areas
    else:
        for i in range(len(_sorted_layers)):
            if floor_area_m2 <= _sorted_max_areas[i]:
                target_n = _sorted_layers[i]
                break
        else:
            # If area is larger than the max defined area, use the max defined layers
            target_n = _sorted_layers[-1]
            print(f"[WARN] Floor area {floor_area_m2:.2f} m^2 exceeds max area in mapping ({_sorted_max_areas[-1]:.2f} m^2). Using max layers: {target_n}.")

    # --- Original Diamond:61 Pattern Generation ---
    # 'target_n' now controls the number of rings/layers generated
    positions = []
    # Layer 0: Center COB
    positions.append((0, 0, H, 0))
    # Layers 1 to target_n: Rings
    if target_n > 0: # Check required as target_n could theoretically be 0 if mapping changes
        for i in range(1, target_n + 1): # Loop up to the determined number of layers
            for x in range(-i, i + 1):
                y_abs = i - abs(x)
                if y_abs == 0:
                    # On x-axis (excluding origin if i>0)
                    if x != 0: # Avoid double-counting origin
                       positions.append((x, 0, H, i))
                else:
                    # Off-axis points
                    positions.append((x, y_abs, H, i))
                    positions.append((x, -y_abs, H, i))

    # --- Transformation and Scaling ---
    # Scale the pattern to fit the room dimensions.
    # The scaling now depends on 'target_n' (which is derived from area)
    # to properly distribute the generated COBs within the W x L space.
    theta = math.radians(45)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centerX = W / 2
    centerY = L / 2

    # Scaling factor based on the number of layers 'target_n' determined by area.
    # Avoid division by zero if target_n is somehow 0.
    # Use 0.95 factor to keep COBs slightly away from walls.
    # The denominator 'target_n' ensures that as more layers are added (for larger areas),
    # the relative spacing is maintained correctly within the larger room.
    scale_x = (W / 2 * 0.95 * math.sqrt(2)) / target_n if target_n > 0 else (W / 2 * 0.95)
    scale_y = (L / 2 * 0.95 * math.sqrt(2)) / target_n if target_n > 0 else (L / 2 * 0.95)

    transformed = []
    for (x, y, h, layer) in positions:
        # Rotate
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        # Scale and translate
        px = centerX + rx * scale_x
        py = centerY + ry * scale_y
        # Add position with layer info (use 95% height for COBs)
        transformed.append((px, py, H * 0.95, layer))

    num_cobs_generated = len(transformed)
    print(f"[INFO] Floor Area: {floor_area_m2:.2f} m^2 -> Target Layers (n): {target_n} -> COBs Generated: {num_cobs_generated}")

    return np.array(transformed, dtype=np.float64)

def pack_luminous_flux_dynamic(params, cob_positions):
    """
    For each COB position, select the corresponding luminous flux from params.
    The index in params corresponds to the layer number.
    If a COB’s layer index exceeds the length of params, use the last candidate value.
    """
    led_intensities = []
    for pos in cob_positions:
        layer = int(pos[3])
        if layer < len(params):
            intensity = params[layer]
        else:
            intensity = params[-1]  # Use the last parameter if layer index is too high
        led_intensities.append(intensity)
    return np.array(led_intensities, dtype=np.float64)


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
    
    # Floor patch
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0))
    patch_refl.append(REFL_FLOOR)
    
    # Ceiling patches
    xs_ceiling = np.linspace(0, W, CEIL_SUBDIVS_X + 1)
    ys_ceiling = np.linspace(0, L, CEIL_SUBDIVS_Y + 1)
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (xs_ceiling[i] + xs_ceiling[i+1]) / 2
            cy = (ys_ceiling[j] + ys_ceiling[j+1]) / 2
            area = (xs_ceiling[i+1] - xs_ceiling[i]) * (ys_ceiling[j+1] - ys_ceiling[j])
            patch_centers.append((cx, cy, H + 0.01))
            patch_areas.append(area)
            patch_normals.append((0.0, 0.0, -1.0))
            patch_refl.append(REFL_CEIL)
    
    # Walls: front (y=0), back (y=L), left (x=0), right (x=W)
    xs_wall = np.linspace(0, W, WALL_SUBDIVS_X + 1)
    zs_wall = np.linspace(0, H, WALL_SUBDIVS_Y + 1)
    
    # Front wall
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, 0.0, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, -1.0, 0.0))
            patch_refl.append(REFL_WALL)
    
    # Back wall
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, L, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, 1.0, 0.0))
            patch_refl.append(REFL_WALL)
    
    ys_wall = np.linspace(0, L, WALL_SUBDIVS_X + 1)
    # Left wall
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((0.0, cy, cz))
            patch_areas.append(area)
            patch_normals.append((-1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)
    
    # Right wall
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((W, cy, cz))
            patch_areas.append(area)
            patch_normals.append((1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)
    
    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

def build_patches(W, L, H):
    return cached_build_patches(W, L, H)

# ------------------------------
# Simulation Functions (Radiosity) - CORRECTED
# ------------------------------
@njit
def compute_direct_floor(light_positions, light_fluxes, X, Y):
    # This function calculates direct illumination from COBs to the floor grid.
    # It seems logically correct based on the inverse square law and cosine falloff.
    # No change needed here based on the request about reflectivity.
    out = np.zeros_like(X, dtype=np.float64)
    rows, cols = X.shape
    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            fz = 0.0 # Floor point z coordinate
            val = 0.0
            for k in range(light_positions.shape[0]):
                lx = light_positions[k, 0]
                ly = light_positions[k, 1]
                lz = light_positions[k, 2] # Light height
                dx = fx - lx
                dy = fy - ly
                dz = fz - lz # Vector from light to floor point
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)
                # Cosine angle at the floor (normal is 0,0,1)
                # Dot product of (0,0,1) and (-dx, -dy, -dz) is -dz
                # cos_th_floor = -dz / dist
                # Cosine angle at the LED (Lambertian emission, normal assumed 0,0,-1)
                # Dot product of (0,0,-1) and (dx, dy, dz) is -dz
                cos_th_led = -dz / dist # = abs(dz)/dist since lz > fz=0

                if cos_th_led < 1e-6: # Ensure light is emitted downwards
                    continue

                # Irradiance = Intensity * cos(theta_source) / distance^2
                # Intensity I = Flux / pi (for Lambertian source)
                # E = (Flux / pi) * cos_th_led / dist^2
                # Flux here is power_arr (W), so E is W/m^2
                val += (light_fluxes[k] / math.pi) * (cos_th_led / dist2)
            out[r, c] = val
    return out


@njit
def compute_patch_direct(light_positions, light_fluxes, patch_centers, patch_normals, patch_areas):
    # This function calculates the direct irradiance (E_i) on each patch from the LEDs.
    # It seems logically correct.
    # No change needed here based on the request about reflectivity.
    Np = patch_centers.shape[0]
    out = np.zeros(Np, dtype=np.float64) # Direct Irradiance E_i [W/m^2 or lm/m^2]
    for ip in range(Np):
        pc = patch_centers[ip]
        n_patch = patch_normals[ip]
        norm_n_patch = math.sqrt(n_patch[0]*n_patch[0] + n_patch[1]*n_patch[1] + n_patch[2]*n_patch[2])
        if norm_n_patch < 1e-9: norm_n_patch=1.0

        accum_E_i = 0.0
        for j in range(light_positions.shape[0]):
            lx, ly, lz = light_positions[j, :3]
            # Vector from light source to patch center
            dx = pc[0] - lx
            dy = pc[1] - ly
            dz = pc[2] - lz
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2 < 1e-15:
                continue
            dist = math.sqrt(dist2)

            # Cosine angle at LED source (normal 0,0,-1)
            # Dot product of (0,0,-1) and (dx, dy, dz) is -dz
            cos_th_led = -dz / dist # Assumes LED points straight down
            if cos_th_led <= 1e-6: # Light must go towards patch
                 continue

            # Direct irradiance hitting a surface perpendicular to the light ray
            E_normal = (light_fluxes[j] / math.pi) * (cos_th_led / dist2)

            # Cosine angle at the patch surface
            # Dot product of patch normal and vector *from* light *to* patch (-dx, -dy, -dz)
            dot_patch = -(n_patch[0]*dx + n_patch[1]*dy + n_patch[2]*dz)
            cos_in_patch = dot_patch / (dist * norm_n_patch)
            if cos_in_patch <= 1e-6: # Light must hit the front face of the patch
                continue

            # Irradiance on patch = E_normal * cos_in_patch
            accum_E_i += E_normal * cos_in_patch

        out[ip] = accum_E_i
    return out


@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct, patch_areas, patch_refl, num_bounces):
    """
    Corrected iterative radiosity calculation.
    B_i = E_i + rho_i * H_i
    H_i = sum_j (B_j_prev * K(j, i) * A_j)
    """
    Np = patch_direct.shape[0]
    # Initialize radiosity B_i with direct illumination E_i (W/m^2 or lm/m^2)
    patch_radiosity = patch_direct.copy() # B_i for current bounce

    # Iterate for specified number of bounces
    for bounce in range(num_bounces):
        # Store radiosity from the *previous* bounce to calculate incident irradiance
        prev_patch_radiosity = patch_radiosity.copy()
        # Array to accumulate incident irradiance (H_i) for each patch i in this bounce
        incident_irradiance = np.zeros(Np, dtype=np.float64)

        # Calculate incident irradiance H_i for each patch i
        for i in range(Np):
            accum_H_i = 0.0
            pi = patch_centers[i]
            ni = patch_normals[i]
            norm_ni = math.sqrt(ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2])
            if norm_ni < 1e-9: norm_ni = 1.0 # Avoid division by zero

            # Sum contribution to H_i from all other patches j
            for j in range(Np):
                if i == j: # A patch does not illuminate itself directly
                    continue

                # Use radiosity B_j from the *previous* bounce
                B_j_prev = prev_patch_radiosity[j]
                if B_j_prev <= 1e-9: # No light leaving patch j
                    continue

                pj = patch_centers[j]
                nj = patch_normals[j]
                norm_nj = math.sqrt(nj[0]*nj[0] + nj[1]*nj[1] + nj[2]*nj[2])
                if norm_nj < 1e-9: norm_nj = 1.0 # Avoid division by zero

                # Vector from patch j center to patch i center
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dz = pi[2] - pj[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)

                # Cosine term at source patch j (angle between normal nj and vector pj->pi)
                dot_j = (nj[0]*dx + nj[1]*dy + nj[2]*dz)
                cos_j = dot_j / (dist * norm_nj)

                # Cosine term at receiving patch i (angle between normal ni and vector pi->pj)
                # Vector pi->pj is (-dx, -dy, -dz)
                dot_i = -(ni[0]*dx + ni[1]*dy + ni[2]*dz)
                cos_i = dot_i / (dist * norm_ni)

                # Check visibility and orientation: both cosines must be positive
                # (Light must leave j towards i, and arrive at i from j)
                if cos_j <= 1e-6 or cos_i <= 1e-6:
                    continue

                # Form factor kernel K(j, i) = (cos_j * cos_i) / (math.pi * dist2)
                ff_kernel = (cos_j * cos_i) / (math.pi * dist2) # Units: 1/m^2

                # Irradiance contribution H_i_from_j = B_j_prev * K(j, i) * A_j
                H_i_from_j = B_j_prev * ff_kernel * patch_areas[j] # (W/m^2)*(1/m^2)*m^2 = W/m^2
                accum_H_i += H_i_from_j

            incident_irradiance[i] = accum_H_i

        # Update radiosity for the current bounce: B_i = E_i + rho_i * H_i
        # E_i is patch_direct[i], rho_i is patch_refl[i], H_i is incident_irradiance[i]
        patch_radiosity = patch_direct + patch_refl * incident_irradiance

        # Optional: Check for convergence if needed, e.g., by comparing patch_radiosity with prev_patch_radiosity

    return patch_radiosity # Return the final radiosity after all bounces


@njit
def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, final_patch_radiosity, patch_refl):
    """
    Corrected calculation of irradiance on the floor grid from all patches.
    Uses the final radiosity B_p of each patch p.
    Irradiance at floor point = sum_p ( B_p * K(p, floor_point) * A_p )
    """
    rows, cols = X.shape
    out = np.zeros((rows, cols), dtype=np.float64) # Reflected Irradiance [W/m^2 or lm/m^2]
    floor_normal = np.array([0.0, 0.0, 1.0]) # Normal vector of the floor points

    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            fz = 0.0 # Floor is at z=0
            floor_point = np.array([fx, fy, fz])

            total_incident_irradiance = 0.0 # Accumulate irradiance at this floor point

            # Sum contribution from all patches p
            for p in range(patch_centers.shape[0]):
                # Use the final radiosity B_p calculated by iterative_radiosity_loop
                B_p = final_patch_radiosity[p]
                if B_p <= 1e-9: # No light leaving this patch
                    continue

                # Ensure patch p is not the floor itself if floor is patch 0 (optional check)
                # if p == 0 and abs(patch_centers[p][2]) < 1e-6: continue

                pc = patch_centers[p]
                n_patch = patch_normals[p]
                norm_n_patch = math.sqrt(n_patch[0]*n_patch[0] + n_patch[1]*n_patch[1] + n_patch[2]*n_patch[2])
                if norm_n_patch < 1e-9: norm_n_patch = 1.0

                # Vector from patch center p to floor point
                dx = floor_point[0] - pc[0]
                dy = floor_point[1] - pc[1]
                dz = floor_point[2] - pc[2] # Should generally be negative (floor below patches)

                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)

                # Cosine at source patch p (angle between n_patch and vector pc->floor_point)
                dot_p = n_patch[0]*dx + n_patch[1]*dy + n_patch[2]*dz
                cos_p = dot_p / (dist * norm_n_patch)

                # Cosine at floor point (angle between floor_normal and vector floor_point->pc)
                # Vector floor_point->pc is (-dx, -dy, -dz)
                dot_f = floor_normal[0]*(-dx) + floor_normal[1]*(-dy) + floor_normal[2]*(-dz)
                cos_f = dot_f / dist # norm of floor_normal is 1

                # Check visibility and orientation: both cosines must be positive
                if cos_p <= 1e-6 or cos_f <= 1e-6:
                    continue

                # Form factor kernel K(p, floor_point) = (cos_p * cos_f) / (math.pi * dist2)
                ff_kernel = (cos_p * cos_f) / (math.pi * dist2) # Units: 1/m^2

                # Irradiance contribution at floor point from patch p = B_p * K(p, floor_point) * A_p
                irradiance_from_p = B_p * ff_kernel * patch_areas[p] # (W/m^2)*(1/m^2)*m^2 = W/m^2

                total_incident_irradiance += irradiance_from_p

            out[r, c] = total_incident_irradiance
    return out

# Ensure simulate_lighting calls the corrected functions with the right arguments
def simulate_lighting(params, geo):
    """
    Run the full simulation. Uses corrected radiosity functions.
    """
    cob_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geo
    led_intensities = pack_luminous_flux_dynamic(params, cob_positions) # Lumens

    # Convert lumens to Watts for radiosity calculation
    power_arr = led_intensities / LUMINOUS_EFFICACY # Watts per LED

    # 1. Direct irradiance on floor grid from LEDs [W/m^2]
    direct_irr_watts = compute_direct_floor(cob_positions, power_arr, X, Y)

    # 2. Direct irradiance on patches from LEDs (E_i) [W/m^2]
    patch_direct_watts = compute_patch_direct(cob_positions, power_arr, p_centers, p_normals, p_areas)

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
    # CONVERSION_FACTOR is µmol/J = µmol/(W*s)
    # PPFD = Total Irradiance [W/m^2] * CONVERSION_FACTOR [µmol/J]
    # Units: (W/m^2) * (µmol / (W*s)) = µmol / (m^2 * s) --- This is correct.
    return total_irr_watts * CONVERSION_FACTOR

# IMPORTANT: Replace the original functions `iterative_radiosity_loop` and
# `compute_reflection_on_floor` in your script with these corrected versions.
# Also ensure `simulate_lighting` uses these corrected functions as shown above.
# The functions `compute_direct_floor` and `compute_patch_direct` have been included
# for completeness and context, although they were likely correct already.

def objective_function(params, geo, target_ppfd):
    """
    Objective function for optimization.
    Computes the mean absolute deviation (MAD) of PPFD around the mean,
    plus a penalty on the difference between the mean PPFD and the target.
    """
    floor_ppfd = simulate_lighting(params, geo)
    mean_ppfd = np.mean(floor_ppfd)
    mad = np.mean(np.abs(floor_ppfd - mean_ppfd))
    ppfd_penalty = (mean_ppfd - target_ppfd) ** 2
    return mad + 2.0 * ppfd_penalty

# ------------------------------
# Full Simulation Optimization Function
# ------------------------------
def optimize_lighting(geo, target_ppfd, x0=None, progress_callback=None):
    cob_positions = geo[0]
    # Use the same minimum features as in surrogate optimization
    # Use the fixed candidate dimension for the surrogate
    fixed_dim = FIXED_NUM_LAYERS  
    if x0 is None:
        x0 = np.full(fixed_dim, 8000.0, dtype=np.float64)
    else:
        x0 = adjust_candidate_dimension(x0, fixed_dim=fixed_dim)

    if x0 is None:
        x0 = np.array([8000.0] * fixed_dim)
    if len(x0) < fixed_dim:
        pad = np.array([8000.0] * (fixed_dim - len(x0)))
        x0 = np.concatenate([x0, pad])
    elif len(x0) > fixed_dim:
        x0 = x0[:fixed_dim]
    bounds = [(MIN_LUMENS, MAX_LUMENS_MAIN)] * fixed_dim
    total_iterations_estimate = 500
    iteration = 0

    def wrapped_obj(p):
        nonlocal iteration
        iteration += 1
        val = objective_function(p, geo, target_ppfd)
        floor_ppfd = simulate_lighting(p, geo)
        mp = np.mean(floor_ppfd)
        msg = f"[DEBUG] param={p}, mean_ppfd={mp:.1f}, obj={val:.3f}"
        if progress_callback:
            progress_pct = min(98, (iteration / total_iterations_estimate) * 100)
            progress_callback(f"PROGRESS:{progress_pct}")
            progress_callback(msg)
        return val

    if progress_callback:
        progress_callback("[INFO] Starting SLSQP optimization...")
    res = minimize(wrapped_obj, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': total_iterations_estimate, 'disp': True})
    if not res.success and progress_callback:
        progress_callback(f"[WARN] Optimization did not converge: {res.message}")
    if progress_callback:
        progress_callback("PROGRESS:99")
    return res.x, geo

# ------------------------------
# Surrogate Model & Persistence Functions
# ------------------------------

########################
# Helper: Adjust candidate vector dimension
########################

def adjust_candidate_dimension(candidate, fixed_dim=FIXED_NUM_LAYERS, default_value=8000.0):
    """
    Adjust candidate vector to fixed_dim entries:
      - Pad with default_value if too short.
      - Truncate if too long.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    current_dim = candidate.shape[0]
    if current_dim < fixed_dim:
        pad = np.full((fixed_dim - current_dim,), default_value)
        return np.concatenate([candidate, pad])
    elif current_dim > fixed_dim:
        return candidate[:fixed_dim]
    return candidate

########################
# 1) forward_transform / inverse_transform
########################

def forward_transform(x):
    """
    Given a candidate vector x, compute:
      - T: total lumens (sum(x))
      - u: normalized indices (linspace from 0 to 1) with length = len(x)
      - y: fraction of total (x/T)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    T = np.sum(x)
    if T < 1e-12:
        T = 1e-12
    if n == 1:
        u = np.array([0.0])
        y = x / T
    else:
        u = np.linspace(0.0, 1.0, n)
        y = x / T
    return T, u, y

def inverse_transform(T, u, y, m):
    """
    Given T, u, and y from forward_transform, re-sample y(u) at m points
    using a cubic spline, and return the raw candidate vector.
    """
    if m == 1:
        return np.array([T], dtype=np.float64)
    spline = CubicSpline(u, y, bc_type='natural')
    v = np.linspace(0.0, 1.0, m)
    y_new = spline(v)
    x_new = T * y_new
    return x_new

# --- Update load_training_data to also load geometry info ---
def load_training_data():
    if os.path.exists(TRAINING_DATA_FILE):
        with np.load(TRAINING_DATA_FILE, allow_pickle=True) as data:
            X_train = data["X_train"]
            y_train = data["y_train"]
            G_train = data["G_train"] if "G_train" in data else None
        print(f"[INFO] Loaded {len(X_train)} training samples (feature dimension = {X_train.shape[1]}).")
        return X_train, y_train, G_train
    else:
        print("[INFO] No persistent training data found. Starting fresh.")
        return None, None, None

# --- Update save_training_data to also save geometry info ---
def save_training_data(X_train, y_train, G_train):
    """
    Saves training data to disk.
    """
    np.savez(TRAINING_DATA_FILE, X_train=X_train, y_train=y_train, G_train=G_train)
    print(f"[INFO] Saved {len(X_train)} training samples.")

########################
# 2) update_training_data
########################

def update_training_data(geo, target_ppfd, new_sample, geom_features):
    M = 10  # number of spline samples; final feature dimension = M+1
    # Force candidate to FIXED_NUM_LAYERS
    raw_candidate = adjust_candidate_dimension(new_sample[0], fixed_dim=FIXED_NUM_LAYERS)
    print(f"[DEBUG] Received candidate of shape {np.asarray(new_sample[0]).shape}, adjusted to {raw_candidate.shape}")
    obj_value = new_sample[1]

    # Normalize candidate vector
    T, u, y = forward_transform(raw_candidate)
    spline = CubicSpline(u, y, bc_type='natural')
    v = np.linspace(0.0, 1.0, M)
    y_fixed = spline(v)  # fixed shape vector of length M
    x_norm = np.hstack((T, y_fixed))  # final normalized feature vector (length M+1)

    # Load old training data
    X_old, y_old, G_old = load_training_data()
    if X_old is not None:
        if X_old.shape[1] != x_norm.shape[0]:
            print("[WARN] Dimension mismatch in training data. Discarding old data.")
            X_train = np.array([x_norm], dtype=np.float64)
            y_train = np.array([obj_value], dtype=np.float64)
            G_train = np.array([geom_features], dtype=np.float64)
        else:
            X_train = np.vstack((X_old, x_norm))
            y_train = np.hstack((y_old, [obj_value]))
            G_train = np.vstack((G_old, geom_features))
    else:
        X_train = np.array([x_norm], dtype=np.float64)
        y_train = np.array([obj_value], dtype=np.float64)
        G_train = np.array([geom_features], dtype=np.float64)
    
    save_training_data(X_train, y_train, G_train)
    print(f"[INFO] Saved training sample #{len(X_train)} with normalized dimension {x_norm.shape[0]}")
    return X_train, y_train, G_train


def generate_new_sample(geo, target_ppfd, sample_params):
    """Evaluate the full objective at sample_params."""
    obj_value = objective_function(sample_params, geo, target_ppfd)
    return sample_params, obj_value

# --- New helper function to adjust candidate vector dimensions ---
def adjust_sample_dim(sample, target_dim, default=8000.0):
    current_dim = sample.shape[0]
    if current_dim < target_dim:
        pad = np.full((target_dim - current_dim,), default)
        return np.concatenate([sample, pad])
    elif current_dim > target_dim:
        return sample[:target_dim]
    return sample

########################
# 3) train_surrogate
########################

def train_surrogate(X_train, y_train):
    """
    Trains a Gaussian Process on the normalized training data.
    X_train is assumed to be of shape (N, M+1) where M+1 is the fixed feature dimension.
    """
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gp.fit(X_train, y_train)
    print("[INFO] Surrogate trained. Final kernel:", gp.kernel_)
    return gp

def surrogate_objective(params, gp_model):
    params = np.atleast_2d(params)
    y_pred, sigma = gp_model.predict(params, return_std=True)
    return y_pred[0]

########################
# 4) optimize_with_surrogate
########################

def optimize_with_surrogate(geo, target_ppfd, gp_model, x0=None):
    """
    Optimizes in the normalized space and returns a raw candidate vector.
    Uses the GP's training feature dimension to set the number of spline samples (M).
    The returned candidate is adjusted to FIXED_NUM_LAYERS.
    """
    # Determine M from GP training data: feature dimension = M+1.
    M = gp_model.X_train_.shape[1] - 1

    if x0 is None:
        x0 = np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64)
    x0 = adjust_candidate_dimension(x0, fixed_dim=FIXED_NUM_LAYERS)
    
    T0, u0, y0 = forward_transform(x0)
    spline0 = CubicSpline(u0, y0, bc_type='natural')
    v0 = np.linspace(0.0, 1.0, M)
    y0_fixed = spline0(v0)
    z0 = np.hstack((T0, y0_fixed))  # normalized initial guess

    def objective_in_normalized_space(z):
        z_reshaped = z[np.newaxis, :]
        return gp_model.predict(z_reshaped, return_std=False)[0]

    # Bounds: T in [1000,30000], shape samples in [0,1].
    bounds = [(1000.0, 30000.0)] + [(0.0, 1.0)] * M

    from scipy.optimize import minimize
    res = minimize(objective_in_normalized_space, z0, method='SLSQP', bounds=bounds, options={'maxiter':200})
    z_best = res.x
    T_best = z_best[0]
    shape_best = z_best[1:]
    
    # Reconstruct the raw candidate for simulation using inverse_transform.
    x_new = inverse_transform(T_best, np.linspace(0, 1, M), shape_best, FIXED_NUM_LAYERS)
    print(f"[INFO] Surrogate optimization complete. Best objective= {res.fun:.3f}, success={res.success}")
    return x_new

def filter_training_data_by_layers(X_train, y_train, expected_feature_dim):
    # X_train is assumed to be an array of shape (num_samples, feature_dim)
    filtered_X = []
    filtered_y = []
    for i, x in enumerate(X_train):
        if len(x) == expected_feature_dim:
            filtered_X.append(x)
            filtered_y.append(y_train[i])
    if filtered_X:
        return np.vstack(filtered_X), np.array(filtered_y)
    else:
        return None, None

# --- Revised compute_initial_guess using only training data with matching number of layers ---
def compute_initial_guess(geom_features, n_layers, tol=(1.0, 1.0, 50.0, 0.5)):
    """
    geom_features: [width_ft, length_ft, target_ppfd, height_ft] for current run.
    tol: tolerances for width, length, target_ppfd, height.
    """
    X_train, y_train, G_train = load_training_data()
    if G_train is None:
        return np.array([8000.0] * n_layers)
    
    similar = []
    # Only consider training samples whose candidate vectors have the correct dimension.
    for i, g in enumerate(G_train):
        if len(X_train[i]) != n_layers:
            continue
        # Check if geometry is within tolerance.
        if (abs(g[0] - geom_features[0]) <= tol[0] and
            abs(g[1] - geom_features[1]) <= tol[1] and
            abs(g[2] - geom_features[2]) <= tol[2] and
            abs(g[3] - geom_features[3]) <= tol[3]):
            similar.append(X_train[i])
    if similar:
        similar = np.array(similar)
        # No need to adjust dimensions here because we only included samples with len == n_layers.
        candidate = np.mean(similar, axis=0)
        print(f"[INFO] Informed initial guess from {len(similar)} similar samples: {candidate}")
        return candidate
    else:
        return np.array([8000.0] * n_layers)

########################
# 5) train_regression_model_for_normalized
########################

def train_regression_model_for_normalized(M=10):
    """
    Trains a regression model mapping geometry features to the normalized feature vector.
    The normalized vector is of length M+1 = [T, y_fixed_1, ..., y_fixed_M].
    Returns the trained model, or None if insufficient data.
    """
    X_train, y_train, G_train = load_training_data()
    if X_train is None or len(X_train) < 5:
        print("[INFO] Not enough training data for regression model.")
        return None
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor
    reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, random_state=42))
    reg.fit(G_train, X_train)
    print(f"[INFO] Regression model trained on {len(X_train)} samples.")
    return reg

########################
# 6) predict_initial_guess_regression
########################

def predict_initial_guess_regression(geom_features, n_layers):
    """
    Uses the regression model to predict a normalized feature vector from geometry,
    then converts it to a raw candidate vector adjusted to FIXED_NUM_LAYERS.
    """
    M = 10
    model = train_regression_model_for_normalized(M=M)
    if model is None:
        print("[WARN] No regression model available; using fallback guess.")
        return np.full(FIXED_NUM_LAYERS, 8000.0, dtype=np.float64)
    z_pred = model.predict([geom_features])[0]
    T_pred = z_pred[0]
    shape_pred = z_pred[1:]
    x_guess = inverse_transform(T_pred, np.linspace(0, 1, M), shape_pred, FIXED_NUM_LAYERS)
    print(f"[INFO] Regression-based guess computed: T={T_pred:.1f} for {FIXED_NUM_LAYERS} layers.")
    return x_guess

########################
# Warm-up: Initial Sampling Strategy
########################

def initial_sampling(num_samples=5):
    """
    Run an initial set of simulations with varying candidate vectors to populate training data.
    This function generates num_samples candidate vectors (each of length FIXED_NUM_LAYERS),
    runs the full simulation on each, and calls update_training_data.
    Replace the simulation call with your actual simulation function.
    """
    for i in range(num_samples):
        # Generate a candidate vector: for example, vary lumens around 8000 with random noise.
        candidate = np.full(FIXED_NUM_LAYERS, 8000.0) + np.random.uniform(-1000, 1000, size=FIXED_NUM_LAYERS)
        # Run the full simulation on the candidate (simulate_lighting returns a PPFD field)
        # For demonstration, we use a dummy objective value. Replace with your actual simulation call.
        obj_value = np.mean(candidate)  # dummy objective; in practice, compute objective_function(candidate, geo, target_ppfd)
        # geometry features example: [floor_width_ft, floor_length_ft, target_ppfd, floor_height_ft]
        geom_features = [14.0, 14.0, 1250.0, 3.0]
        # Update training data with this sample.
        update_training_data(None, 1250.0, (candidate, obj_value), geom_features)
    print("[INFO] Initial sampling complete.")

# --- Modify run_iterative_optimization to use the regression-based initial guess ---
def run_iterative_optimization(floor_width_ft=14.0, floor_length_ft=14.0,
                               target_ppfd=1250.0, floor_height=3.0,
                               num_new_samples=10, progress_callback=None, x0=None):
    ft2m = 3.28084
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = floor_height / ft2m
    geo = prepare_geometry(W_m, L_m, H_m)
    
    # Create geometry feature vector: [width, length, target_ppfd, height]
    geom_features = [floor_width_ft, floor_length_ft, target_ppfd, floor_height]
    
    # Use the fixed candidate dimension for the surrogate.
    fixed_dim = FIXED_NUM_LAYERS  
    if x0 is None:
        x0 = np.full(fixed_dim, 8000.0, dtype=np.float64)
    else:
        x0 = adjust_candidate_dimension(x0, fixed_dim=fixed_dim)
    
    # Define expected normalized feature dimension: T + M samples (M = fixed_dim, so length = fixed_dim+1)
    expected_feature_dim = fixed_dim + 1

    # Load training data and filter by expected_feature_dim:
    X_train_old, y_train_old, G_train_old = load_training_data()
    if X_train_old is not None and y_train_old is not None:
        X_train_filtered, y_train_filtered = filter_training_data_by_layers(X_train_old, y_train_old, expected_feature_dim)
        if X_train_filtered is not None:
            gp_model = train_surrogate(X_train_filtered, y_train_filtered)
        else:
            default_params = np.full(fixed_dim, 8000.0, dtype=np.float64)
            sample = generate_new_sample(geo, target_ppfd, default_params)
            X_train_filtered = sample[0][np.newaxis, :]
            y_train_filtered = np.array([sample[1]])
            save_training_data(X_train_filtered, y_train_filtered, np.array([geom_features]))
            gp_model = train_surrogate(X_train_filtered, y_train_filtered)
    else:
        default_params = np.full(fixed_dim, 8000.0, dtype=np.float64)
        sample = generate_new_sample(geo, target_ppfd, default_params)
        X_train_filtered = sample[0][np.newaxis, :]
        y_train_filtered = np.array([sample[1]])
        save_training_data(X_train_filtered, y_train_filtered, np.array([geom_features]))
        gp_model = train_surrogate(X_train_filtered, y_train_filtered)
    
    # Use the regression model to predict an informed initial guess:
    informed_candidate = predict_initial_guess_regression(geom_features, fixed_dim)
    
    # Use surrogate optimization starting from the informed candidate.
    candidate = optimize_with_surrogate(geo, target_ppfd, gp_model, x0=informed_candidate)
    print(f"Surrogate candidate: {candidate}")
    if progress_callback:
        progress_callback(f"[INFO] Using surrogate candidate as initial guess: {candidate}")
    
    best_params_full, _ = optimize_lighting(geo, target_ppfd, x0=candidate, progress_callback=progress_callback)
    
    # Evaluate the full objective at the optimized candidate:
    new_sample = generate_new_sample(geo, target_ppfd, best_params_full)
    X_train, y_train, G_train = update_training_data(geo, target_ppfd, new_sample, geom_features)
    
    # Retrain surrogate on filtered updated data:
    X_train_filtered, y_train_filtered = filter_training_data_by_layers(X_train, y_train, expected_feature_dim)
    if X_train_filtered is not None:
        gp_model = train_surrogate(X_train_filtered, y_train_filtered)
    
    candidate_check = optimize_with_surrogate(geo, target_ppfd, gp_model)
    print("Final surrogate-based candidate (post-update):", candidate_check)
    final_ppfd = simulate_lighting(candidate_check, geo)
    print("Mean PPFD from full simulation (with surrogate-opt candidate):", np.mean(final_ppfd))
    
    return best_params_full, gp_model

# ------------------------------
# Main Entry Point
# ------------------------------
def run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height=3.0):
    cached_build_floor_grid.cache_clear()
    cached_build_patches.cache_clear()

    ft2m = 3.28084
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = floor_height / ft2m

    # Run the simulation
    best_params, _ = run_iterative_optimization(floor_width_ft, floor_length_ft, target_ppfd, floor_height)
    final_ppfd = simulate_lighting(best_params, prepare_geometry(W_m, L_m, H_m))
    mean_ppfd = np.mean(final_ppfd)
    mad = np.mean(np.abs(final_ppfd - mean_ppfd))
    rmse = np.sqrt(np.mean((final_ppfd - mean_ppfd)**2))
    dou = 100 * (1 - rmse / mean_ppfd)
    cv = 100 * (np.std(final_ppfd) / mean_ppfd)

    result = {
        "optimized_lumens_by_layer": best_params.tolist(),
        "mad": float(mad),
        "optimized_ppfd": float(mean_ppfd),
        "rmse": float(rmse),
        "dou": float(dou),
        "cv": float(cv),
        "floor_width": floor_width_ft,
        "floor_length": floor_length_ft,
        "target_ppfd": target_ppfd,
        "floor_height": floor_height,
        "heatmapGrid": final_ppfd.tolist()
    }

    return result

def run_simulation(floor_width_ft=14.0, floor_length_ft=14.0, target_ppfd=1250.0, floor_height=3.0):
    result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height, side_by_side=True)
    print("\n[RESULT] Simulation Output:")
    print(result)
    return result  # Return the result so that the API response is defined


def continuous_run(floor_size):
    """
    Continuously run simulations for a given floor size across the Target PPFD range.
    Results are saved to the training data file.
    """
    floor_length_ft, floor_width_ft = floor_size
    floor_height = 3.0  # Default height (can be adjusted if needed)

    print(f"\nStarting simulations for floor size: {floor_length_ft}x{floor_width_ft} ft")

    for target_ppfd in TARGET_PPFD_RANGE:
        print(f"Running simulation for Target PPFD = {target_ppfd}...")

        # Run the simulation
        result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height)

        # Extract results
        optimized_lumens = result["optimized_lumens_by_layer"]
        final_ppfd = result["optimized_ppfd"]
        dou = result["dou"]
        rmse = result["rmse"]
        cv = result["cv"]
        mad = result["mad"]

        # Update training data
        update_training_data(
            geo=None,  # Pass None since we're not using geometry here
            target_ppfd=target_ppfd,
            new_sample=(optimized_lumens, final_ppfd),
            geom_features=[floor_width_ft, floor_length_ft, target_ppfd, floor_height]
        )

        print(f"Completed simulation for PPFD = {target_ppfd}")
        print(f"Optimized Lumens: {optimized_lumens}")
        print(f"Final PPFD: {final_ppfd:.2f}")
        print(f"DOU: {dou:.2f}%, RMSE: {rmse:.2f}, CV: {cv:.2f}%, MAD: {mad:.2f}")
        print("-" * 50)

    print(f"Finished simulations for floor size: {floor_length_ft}x{floor_width_ft} ft\n")

def run_all_floor_sizes():
    """
    Run simulations for all pre-defined floor sizes across the Target PPFD range.
    """
    print("Starting simulations for all pre-defined floor sizes...")

    for floor_size in PRE_DEFINED_FLOOR_SIZES:
        continuous_run(floor_size)

    print("All simulations completed!")

# Example usage:
if __name__ == '__main__':
    # Run simulations for all pre-defined floor sizes
    run_all_floor_sizes()
