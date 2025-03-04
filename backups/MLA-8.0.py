#!/usr/bin/env python3
"""
Lighting Optimization Script
Replicates the Original Lighting Simulation (Diamond Layout, 6 Layers + Corner COBs),
Uses Patch-Based Radiosity & a 2cm Floor Grid.

We optimize 10 total parameters:
- 6 layer intensities (in lumens) for the main COB layers
- 4 separate corner intensities (in lumens)

Room geometry, reflectances, and multi-bounce radiosity match the original script.
LED strips are ignored (flux = 0).
"""

import math
import numpy as np
from scipy.optimize import minimize
import sys
from numba import njit

# ------------------------------
# Diamond pattern: 6 sets of coordinates (plus an extra "layer 0" that we skip)
layers_coords = [
    [(0, 0)],  # This is the "center layer"
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
# The final array length is 61 total COB positions.

corner_idx = [57, 58, 59, 60]  # Indices of the 4 corner COBs in that final array

# Reflectances from the original script
REFL_WALL = 0.08
REFL_CEIL = 0.1
REFL_FLOOR = 0.0

# SPD -> PPFD conversion
LUMINOUS_EFFICACY = 182.0         # lm/W
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv"

# Bounces and geometry subdiv
NUM_RADIOSITY_BOUNCES = 2
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

# Floor resolution
FLOOR_GRID_RES = 0.02  # 2 cm

# Bounds for each of the 10 parameters (6 layers + 4 corners)
MIN_LUMENS = 2000.0
MAX_LUMENS = 30000.0

# ------------------------------
# Compute SPD -> Conversion Factor
# ------------------------------
def compute_conversion_factor(spd_file):
    try:
        spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
    except Exception as e:
        print("Error loading SPD data:", e)
        # fallback if SPD file is missing
        return 0.0138
    wl = spd[:, 0]
    intens = spd[:, 1]
    mask_par = (wl >= 400) & (wl <= 700)
    tot = np.trapz(intens, wl)
    tot_par = np.trapz(intens[mask_par], wl[mask_par])
    PAR_fraction = tot_par / tot if tot > 0 else 1.0

    wl_m = wl * 1e-9
    h, c, N_A = 6.626e-34, 3.0e8, 6.022e23
    numerator = np.trapz(wl_m[mask_par] * intens[mask_par], wl_m[mask_par])
    denominator = np.trapz(intens[mask_par], wl_m[mask_par]) or 1e-15
    lambda_eff = numerator / denominator
    E_photon = (h * c / lambda_eff) if lambda_eff > 0 else 1.0
    conversion_factor = (1.0 / E_photon) * (1e6 / N_A) * PAR_fraction
    print(f"[INFO] SPD: PAR fraction={PAR_fraction:.3f}, conv_factor={conversion_factor:.5f} µmol/J")
    return conversion_factor

CONVERSION_FACTOR = compute_conversion_factor(SPD_FILE)

# ------------------------------
# Build Diamond of 61 COB Positions
# ------------------------------
def build_cob_positions(W, L, H):
    """
    Replicates the diamond pattern from the original script:
    Returns (positions_array_of_shape_(61,3))
    """
    light_positions = []
    center = (W/2, L/2)
    # add center
    light_positions.append((center[0], center[1], H))

    for i, layer in enumerate(layers_coords):
        if i == 0:
            # already added center
            continue
        for (dx, dy) in layer:
            # rotate 45 deg
            theta = math.radians(45)
            rx = dx * math.cos(theta) - dy * math.sin(theta)
            ry = dx * math.sin(theta) + dy * math.cos(theta)
            px = center[0] + rx * (W / 7.2)
            py = center[1] + ry * (W / 7.2)
            light_positions.append((px, py, H))

    return np.array(light_positions, dtype=np.float64)

# ------------------------------
# 10-Parameter Packing
# ------------------------------
def pack_luminous_flux(params):
    """
    params => [layer1, layer2, layer3, layer4, layer5, layer6, corner1, corner2, corner3, corner4]
    We fill the 61-COB array accordingly, overwriting corner_idx with the last 4 values.
    """
    # LAYER_COUNTS = [1,4,8,12,16,20]
    LAYER_COUNTS = [1, 4, 8, 12, 16, 20]
    layers_vals = params[0:6]     # 6 layer intensities
    corner_vals = params[6:10]    # 4 corner COBs

    led_intensities = []
    for val, count in zip(layers_vals, LAYER_COUNTS):
        led_intensities.extend([val]*count)

    # Overwrite corner indices
    for i, c_idx in enumerate(corner_idx):
        led_intensities[c_idx] = corner_vals[i]

    return np.array(led_intensities, dtype=np.float64)

# ------------------------------
# Floor Grid
# ------------------------------
def build_floor_grid(W, L):
    xs = np.arange(0, W, FLOOR_GRID_RES)
    ys = np.arange(0, L, FLOOR_GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

# ------------------------------
# Direct Irradiance on Floor
# ------------------------------
@njit
def compute_direct_floor(light_positions, light_fluxes, X, Y):
    out = np.zeros_like(X, dtype=np.float64)
    rows, cols = X.shape
    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            val = 0.0
            for k in range(light_positions.shape[0]):
                lx = light_positions[k, 0]
                ly = light_positions[k, 1]
                lz = light_positions[k, 2]
                dx = fx - lx
                dy = fy - ly
                dz = -lz
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)
                cos_th = -dz / dist
                if cos_th < 0:
                    cos_th = 0.0
                val += (light_fluxes[k] / math.pi) * (cos_th / dist2)
            out[r, c] = val
    return out

# ------------------------------
# Build Surfaces
# ------------------------------
def build_patches(W, L, H):
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []

    # Floor patch
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W*L)
    patch_normals.append((0.0, 0.0, 1.0))
    patch_refl.append(REFL_FLOOR)

    # Ceiling
    dx_c = W/CEIL_SUBDIVS_X
    dy_c = L/CEIL_SUBDIVS_Y
    for i in range(CEIL_SUBDIVS_X):
        for j in range(CEIL_SUBDIVS_Y):
            cx = (i+0.5)*dx_c
            cy = (j+0.5)*dy_c
            patch_centers.append((cx, cy, H))
            patch_areas.append(dx_c*dy_c)
            patch_normals.append((0.0, 0.0, -1.0))
            patch_refl.append(REFL_CEIL)

    # Walls
    dx = W/WALL_SUBDIVS_X
    dz = H/WALL_SUBDIVS_Y
    # y=0
    for ix in range(WALL_SUBDIVS_X):
        for iz in range(WALL_SUBDIVS_Y):
            px = (ix+0.5)*dx
            py = 0.0
            pz = (iz+0.5)*dz
            patch_centers.append((px, py, pz))
            patch_areas.append(dx*dz)
            patch_normals.append((0.0, -1.0, 0.0))
            patch_refl.append(REFL_WALL)
    # y=L
    for ix in range(WALL_SUBDIVS_X):
        for iz in range(WALL_SUBDIVS_Y):
            px = (ix+0.5)*dx
            py = L
            pz = (iz+0.5)*dz
            patch_centers.append((px, py, pz))
            patch_areas.append(dx*dz)
            patch_normals.append((0.0, 1.0, 0.0))
            patch_refl.append(REFL_WALL)

    dy = L/WALL_SUBDIVS_X
    # x=0
    for iy in range(WALL_SUBDIVS_X):
        for iz in range(WALL_SUBDIVS_Y):
            px = 0.0
            py = (iy+0.5)*dy
            pz = (iz+0.5)*dz
            patch_centers.append((px, py, pz))
            patch_areas.append(dy*dz)
            patch_normals.append((-1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)
    # x=W
    for iy in range(WALL_SUBDIVS_X):
        for iz in range(WALL_SUBDIVS_Y):
            px = W
            py = (iy+0.5)*dy
            pz = (iz+0.5)*dz
            patch_centers.append((px, py, pz))
            patch_areas.append(dy*dz)
            patch_normals.append((1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)

    return (np.array(patch_centers, dtype=np.float64),
            np.array(patch_areas, dtype=np.float64),
            np.array(patch_normals, dtype=np.float64),
            np.array(patch_refl, dtype=np.float64))

# ------------------------------
# Direct Irradiance on Patches (with Patch-Normal Cos)
# ------------------------------
def compute_patch_direct(light_positions, light_fluxes,
                         patch_centers, patch_normals, patch_areas):
    Np = patch_centers.shape[0]
    out = np.zeros(Np, dtype=np.float64)
    for ip in range(Np):
        pc = patch_centers[ip]
        n = patch_normals[ip]
        norm_n = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
        accum = 0.0
        for j in range(light_positions.shape[0]):
            lx, ly, lz = light_positions[j]
            dx = pc[0] - lx
            dy = pc[1] - ly
            dz = pc[2] - lz
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2 < 1e-15:
                continue
            dist = math.sqrt(dist2)
            cos_th_led = -dz/dist
            if cos_th_led < 0:
                continue
            E_led = (light_fluxes[j]/math.pi)*(cos_th_led/dist2)
            dot_patch = -(dx*n[0] + dy*n[1] + dz*n[2])
            cos_in_patch = dot_patch/(dist*norm_n)
            if cos_in_patch < 0:
                cos_in_patch = 0
            accum += E_led*cos_in_patch
        out[ip] = accum
    return out

# ------------------------------
# Radiosity
# ------------------------------
@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct,
                             patch_areas, patch_refl, num_bounces):
    Np = patch_direct.shape[0]
    patch_rad = patch_direct.copy()
    for _ in range(num_bounces):
        new_flux = np.zeros(Np, dtype=np.float64)
        for j in range(Np):
            if patch_refl[j] <= 0:
                continue
            outF = patch_rad[j]*patch_areas[j]*patch_refl[j]
            pj = patch_centers[j]
            nj = patch_normals[j]
            norm_nj = math.sqrt(nj[0]*nj[0] + nj[1]*nj[1] + nj[2]*nj[2])
            for i in range(Np):
                if i == j:
                    continue
                pi = patch_centers[i]
                ni = patch_normals[i]
                norm_ni = math.sqrt(ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2])
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dz = pi[2] - pj[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)
                dot_j = (nj[0]*dx + nj[1]*dy + nj[2]*dz)
                cos_j = dot_j/(dist*norm_nj)
                if cos_j < 0:
                    continue
                dot_i = - (ni[0]*dx + ni[1]*dy + ni[2]*dz)
                cos_i = dot_i/(dist*norm_ni)
                if cos_i < 0:
                    continue
                ff = (cos_j*cos_i)/(math.pi*dist2)
                new_flux[i] += outF*ff

        patch_rad = patch_direct + new_flux/patch_areas
    return patch_rad

# ------------------------------
# Reflected Irradiance on Floor
# ------------------------------
@njit
def compute_reflection_on_floor(X, Y,
                                patch_centers, patch_normals, patch_areas, patch_rad, patch_refl):
    rows, cols = X.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            val = 0.0
            for p in range(patch_centers.shape[0]):
                if patch_refl[p] <= 0:
                    continue
                outF = patch_rad[p]*patch_areas[p]*patch_refl[p]
                pc = patch_centers[p]
                dx = fx - pc[0]
                dy = fy - pc[1]
                dz = -pc[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)
                n = patch_normals[p]
                norm_n = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
                dot_p = dx*n[0] + dy*n[1] + dz*n[2]
                cos_p = dot_p/(dist*norm_n)
                if cos_p < 0:
                    cos_p = 0
                cos_f = -dz/dist
                if cos_f < 0:
                    cos_f = 0
                ff = (cos_p*cos_f)/(math.pi*dist2)
                val += outF*ff
            out[r, c] = val
    return out

# ------------------------------
# Full Simulation
# ------------------------------
def simulate_lighting(params, W, L, H):
    # 1) build geometry
    cob_positions = build_cob_positions(W, L, H)
    lumens_arr = pack_luminous_flux(params)
    power_arr = lumens_arr / LUMINOUS_EFFICACY  # W

    # 2) floor grid
    X, Y = build_floor_grid(W, L)
    direct_irr = compute_direct_floor(cob_positions, power_arr, X, Y)

    # 3) patches & direct
    patch_centers, patch_areas, patch_normals, patch_refl = build_patches(W, L, H)
    patch_direct = compute_patch_direct(cob_positions, power_arr,
                                        patch_centers, patch_normals, patch_areas)
    # 4) multi-bounce
    patch_rad = iterative_radiosity_loop(patch_centers, patch_normals,
                                         patch_direct, patch_areas,
                                         patch_refl, NUM_RADIOSITY_BOUNCES)
    # 5) reflection on floor
    reflect_irr = compute_reflection_on_floor(X, Y,
                                              patch_centers, patch_normals,
                                              patch_areas, patch_rad, patch_refl)

    # total floor irradiance
    floor_irr = direct_irr + reflect_irr
    # convert to PPFD
    floor_ppfd = floor_irr * CONVERSION_FACTOR
    return floor_ppfd

# ------------------------------
# Objective
# ------------------------------
def objective_function(params, W, L, H, target_ppfd):
    floor_ppfd = simulate_lighting(params, W, L, H)
    mean_ppfd = np.mean(floor_ppfd)
    mad = np.mean(np.abs(floor_ppfd - mean_ppfd))
    ppfd_penalty = (mean_ppfd - target_ppfd)**2
    obj = mad + 2.0*ppfd_penalty
    return obj

# ------------------------------
# Optimization
# ------------------------------
def optimize_lighting(W, L, H, target_ppfd):
    """
    We have 10 parameters: 6 layer intensities + 4 corner intensities
    Each param is in [MIN_LUMENS, MAX_LUMENS].
    verbose => always True (hard-coded).
    """
    # Example initial guess
    x0 = np.array([8000, 8000, 8000, 8000, 8000, 8000,
                   18000, 18000, 18000, 18000], dtype=np.float64)
    bounds = [(MIN_LUMENS, MAX_LUMENS)]*10

    def wrapped_obj(p):
        val = objective_function(p, W, L, H, target_ppfd)
        # Print debug info for each iteration
        floor_ppfd = simulate_lighting(p, W, L, H)
        mp = np.mean(floor_ppfd)
        print(f"[DEBUG] param={p}, mean_ppfd={mp:.1f}, obj={val:.3f}")
        return val

    print("[INFO] Starting SLSQP optimization...")
    res = minimize(
        wrapped_obj, x0, method='SLSQP', bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )
    if not res.success:
        print(f"[WARN] Optimization did not converge: {res.message}", file=sys.stderr)
    return res.x

# ------------------------------
# Main
# ------------------------------
def run_simulation(floor_width_ft=12.0, floor_length_ft=12.0, target_ppfd=1250.0):
    # Convert from feet to meters
    ft2m = 3.28084
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = 3.0 / ft2m  # LED height = 3 ft => ~0.9144 m

    print(f"[INFO] Room: {floor_width_ft}×{floor_length_ft} ft => {W_m:.3f}×{L_m:.3f} m, LED height={H_m:.3f} m")
    print(f"[INFO] Target PPFD={target_ppfd:.1f} µmol/m²/s")

    # Optimize
    best_params = optimize_lighting(W_m, L_m, H_m, target_ppfd)
    final_ppfd = simulate_lighting(best_params, W_m, L_m, H_m)
    mean_ppfd = np.mean(final_ppfd)
    mad = np.mean(np.abs(final_ppfd - mean_ppfd))

    print("\n[RESULT] 6 layer intensities + 4 corner intensities (lumens):")
    for i, val in enumerate(best_params):
        print(f"   Param {i+1}: {val:.1f}")
    print(f"[RESULT] Mean PPFD={mean_ppfd:.1f}, MAD={mad:.2f}")

def main():
    # Always runs with hard-coded defaults
    run_simulation()

if __name__ == "__main__":
    main()
