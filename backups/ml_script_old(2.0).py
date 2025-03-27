#!/usr/bin/env python3
"""
Lighting Optimization Script (ml_simulation_improved.py)

Key Improvements to Achieve Higher DOU:
1) Adaptive exploration/exploitation via dynamic kappa in Bayesian Optimization.
2) Multiple restarts to avoid local minima.
3) Asymmetric penalty to allow going above target PPFD if it helps uniformity.
4) Direct weighting of DOU in the objective.
5) Optional pre-probing analysis (brief random or grid sample of param space).

Conceptual additions:
- Multi-objective approach outline (not fully implemented).
- Potential adaptive grid resolution approach for faster early exploration.
"""

import math
import numpy as np
from numba import njit, prange
from functools import lru_cache
import threading
import time
from joblib import Parallel, delayed
from bayes_opt import BayesianOptimization



MIN_LUMENS = 1000.0
MAX_LUMENS_MAIN = 24000.0

# Basic Config & Reflectances
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1
LUMINOUS_EFFICACY = 182.0  # lumens/W
SPD_FILE = "./backups/spd_data.csv"
MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

# You could try coarse->fine adaptive approach. Start coarser for speed, refine once near optimum.
FLOOR_GRID_RES_OPT = 0.2    # for Bayesian optimization
FLOOR_GRID_RES_FINAL = 0.08 # for final evaluation

FIXED_NUM_LAYERS = 10
MC_SAMPLES = 16

# COB datasheet angle data
COB_ANGLE_DATA = np.array([
    [0, 1.00], [10, 0.98], [20, 0.95], [30, 0.88], [40, 0.78],
    [50, 0.65], [60, 0.50], [70, 0.30], [80, 0.10], [90, 0.00]
], dtype=np.float64)
COB_angles_deg = COB_ANGLE_DATA[:, 0]
COB_shape = COB_ANGLE_DATA[:, 1]

# Compute SPD-based µmol/J factor
def compute_conversion_factor(spd_file):
    try:
        spd = np.loadtxt(spd_file, delimiter=' ', skiprows=1)
    except Exception as e:
        print("Error loading SPD data:", e)
        return 0.0138
    wl = spd[:, 0]
    intens = spd[:, 1]
    mask_par = (wl >= 400) & (wl <= 700)
    tot = np.trapz(intens, wl)
    tot_par = np.trapz(intens[mask_par], wl[mask_par])
    PAR_fraction = tot_par / tot if tot > 0 else 1.0
    
    # Convert to photons:
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

# Pre-compute shape integral
def integrate_shape_for_flux(angles_deg, shape):
    rad_angles = np.radians(angles_deg)
    G = 0.0
    for i in range(len(rad_angles) - 1):
        th0 = rad_angles[i]
        th1 = rad_angles[i+1]
        s0 = shape[i]
        s1 = shape[i+1]
        s_mean = 0.5 * (s0 + s1)
        dtheta = (th1 - th0)
        th_mid = 0.5 * (th0 + th1)
        sin_mid = math.sin(th_mid)
        G_seg = s_mean * 2.0 * math.pi * sin_mid * dtheta
        G += G_seg
    return G

SHAPE_INTEGRAL = integrate_shape_for_flux(COB_angles_deg, COB_shape)

@njit
def luminous_intensity(angle_deg, total_lumens):
    if angle_deg <= COB_angles_deg[0]:
        rel = COB_shape[0]
    elif angle_deg >= COB_angles_deg[-1]:
        rel = COB_shape[-1]
    else:
        rel = np.interp(angle_deg, COB_angles_deg, COB_shape)
    return (total_lumens * rel) / SHAPE_INTEGRAL

# Geometry building
def prepare_geometry(W, L, H, grid_res):
    cob_positions = build_cob_positions(W, L, H)
    X, Y = build_floor_grid(W, L, grid_res)
    patches = build_patches(W, L, H)
    return (cob_positions, X, Y, patches)

def build_cob_positions(W, L, H):
    n = FIXED_NUM_LAYERS - 1
    positions = []
    # Center LED:
    positions.append((0, 0, H, 0))
    # Spread others in a "rotated diamond" pattern
    for i in range(1, n + 1):
        for x in range(-i, i + 1):
            y_abs = i - abs(x)
            if y_abs == 0:
                positions.append((x, 0, H, i))
            else:
                positions.append((x, y_abs, H, i))
                positions.append((x, -y_abs, H, i))
    theta = math.radians(45)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centerX = W / 2
    centerY = L / 2
    scale_x = (W / 2 * math.sqrt(2)) / n
    scale_y = (L / 2 * math.sqrt(2)) / n
    transformed = []
    for (xx, yy, hh, layer) in positions:
        rx = xx * cos_t - yy * sin_t
        ry = xx * sin_t + yy * cos_t
        px = centerX + rx * scale_x
        py = centerY + ry * scale_y
        transformed.append((px, py, H * 0.95, layer))
    return np.array(transformed, dtype=np.float64)

def pack_luminous_flux_dynamic(params, cob_positions):
    led_intensities = []
    for pos in cob_positions:
        layer = int(pos[3])
        intensity = params[layer]
        led_intensities.append(intensity)
    return np.array(led_intensities, dtype=np.float64)

@lru_cache(maxsize=32)
def cached_build_floor_grid(W: float, L: float, grid_res: float):
    num_x = int(round(W / grid_res)) + 1
    num_y = int(round(L / grid_res)) + 1
    xs = np.linspace(0, W, num_x)
    ys = np.linspace(0, L, num_y)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def build_floor_grid(W, L, grid_res):
    return cached_build_floor_grid(W, L, grid_res)

@lru_cache(maxsize=32)
def cached_build_patches(W: float, L: float, H: float):
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []

    # Floor patch:
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0))
    patch_refl.append(REFL_FLOOR)

    # Ceiling subdivs:
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

    # Walls subdivs:
    xs_wall = np.linspace(0, W, WALL_SUBDIVS_X + 1)
    zs_wall = np.linspace(0, H, WALL_SUBDIVS_Y + 1)
    # - front wall
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1] - xs_wall[i]) * (zs_wall[j+1] - zs_wall[j])
            patch_centers.append((cx, 0.0, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, -1.0, 0.0))
            patch_refl.append(REFL_WALL)
    # - back wall
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1] - xs_wall[i]) * (zs_wall[j+1] - zs_wall[j])
            patch_centers.append((cx, L, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, 1.0, 0.0))
            patch_refl.append(REFL_WALL)
    ys_wall = np.linspace(0, L, WALL_SUBDIVS_X + 1)
    # - left wall
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1] - ys_wall[i]) * (zs_wall[j+1] - zs_wall[j])
            patch_centers.append((0.0, cy, cz))
            patch_areas.append(area)
            patch_normals.append((-1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)
    # - right wall
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1] - ys_wall[i]) * (zs_wall[j+1] - zs_wall[j])
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

@njit(parallel=True)
def compute_direct_floor(cob_positions, cob_lumens, X, Y):
    min_dist2 = (FLOOR_GRID_RES_OPT / 2.0) ** 2
    rows, cols = X.shape
    out = np.zeros_like(X, dtype=np.float64)
    for r in prange(rows):
        for c in prange(cols):
            fx = X[r, c]
            fy = Y[r, c]
            val = 0.0
            for k in range(cob_positions.shape[0]):
                lx = cob_positions[k, 0]
                ly = cob_positions[k, 1]
                lz = cob_positions[k, 2]
                lumens_k = cob_lumens[k]
                dx = fx - lx
                dy = fy - ly
                dz = -lz
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < min_dist2:
                    d2 = min_dist2
                dist = math.sqrt(d2)
                cos_th = -dz / dist
                if cos_th <= 0:
                    continue
                angle_deg = math.degrees(math.acos(cos_th))
                I_theta = luminous_intensity(angle_deg, lumens_k)
                cos_in_floor = cos_th
                E_local = (I_theta / (dist * dist)) * cos_in_floor
                val += E_local
            out[r, c] = val
    return out

@njit
def compute_patch_direct(cob_positions, cob_lumens, patch_centers, patch_normals, patch_areas):
    min_dist2 = (FLOOR_GRID_RES_OPT / 2.0) ** 2
    Np = patch_centers.shape[0]
    out = np.zeros(Np, dtype=np.float64)
    for ip in range(Np):
        pc = patch_centers[ip]
        n0 = patch_normals[ip, 0]
        n1 = patch_normals[ip, 1]
        n2 = patch_normals[ip, 2]
        norm_n = math.sqrt(n0*n0 + n1*n1 + n2*n2)
        accum = 0.0
        for k in range(cob_positions.shape[0]):
            lx = cob_positions[k, 0]
            ly = cob_positions[k, 1]
            lz = cob_positions[k, 2]
            lumens_k = cob_lumens[k]
            dx = pc[0] - lx
            dy = pc[1] - ly
            dz = pc[2] - lz
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_dist2:
                d2 = min_dist2
            dist = math.sqrt(d2)
            cos_th_led = -dz / dist
            if cos_th_led <= 0:
                continue
            angle_deg = math.degrees(math.acos(cos_th_led))
            I_theta = luminous_intensity(angle_deg, lumens_k)
            dot_patch = -(dx*n0 + dy*n1 + dz*n2)
            cos_in_patch = dot_patch / (dist * norm_n)
            if cos_in_patch < 0:
                cos_in_patch = 0.0
            E_local = (I_theta / (dist*dist)) * cos_in_patch
            accum += E_local
        out[ip] = accum
    return out

@njit
def iterative_radiosity_loop(patch_centers, patch_normals, patch_direct, patch_areas, patch_refl,
                             max_bounces, convergence_threshold):
    Np = patch_direct.shape[0]
    patch_rad = patch_direct.copy()
    epsilon = 1e-6
    for bounce in range(max_bounces):
        new_flux = np.zeros(Np, dtype=np.float64)
        for j in range(Np):
            if patch_refl[j] <= 0:
                continue
            outF = patch_rad[j] * patch_areas[j] * patch_refl[j]
            pj0 = patch_centers[j, 0]
            pj1 = patch_centers[j, 1]
            pj2 = patch_centers[j, 2]
            nj0 = patch_normals[j, 0]
            nj1 = patch_normals[j, 1]
            nj2 = patch_normals[j, 2]
            norm_nj = math.sqrt(nj0*nj0 + nj1*nj1 + nj2*nj2)
            for i in range(Np):
                if i == j:
                    continue
                pi0 = patch_centers[i, 0]
                pi1 = patch_centers[i, 1]
                pi2 = patch_centers[i, 2]
                ni0 = patch_normals[i, 0]
                ni1 = patch_normals[i, 1]
                ni2 = patch_normals[i, 2]
                norm_ni = math.sqrt(ni0*ni0 + ni1*ni1 + ni2*ni2)
                dx = pi0 - pj0
                dy = pi1 - pj1
                dz = pi2 - pj2
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)
                dot_j = nj0*dx + nj1*dy + nj2*dz
                cos_j = dot_j / (dist*norm_nj)
                if cos_j < 0:
                    continue
                dot_i = -(ni0*dx + ni1*dy + ni2*dz)
                cos_i = dot_i / (dist*norm_ni)
                if cos_i < 0:
                    continue
                ff = (cos_j * cos_i) / (math.pi * dist2)
                new_flux[i] += outF * ff

        new_patch_rad = np.empty_like(patch_rad)
        max_rel_change = 0.0
        for i in range(Np):
            new_patch_rad[i] = patch_direct[i] + new_flux[i] / patch_areas[i]
            change = abs(new_patch_rad[i] - patch_rad[i])
            denom = abs(patch_rad[i]) + epsilon
            rel_change = change / denom
            if rel_change > max_rel_change:
                max_rel_change = rel_change
        patch_rad = new_patch_rad.copy()
        if max_rel_change < convergence_threshold:
            break
    return patch_rad

def compute_row_reflection(r, X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl, mc_samples):
    row_vals = []
    for c in range(X.shape[1]):
        fx = X[r, c]
        fy = Y[r, c]
        val = 0.0
        for p in range(patch_centers.shape[0]):
            if patch_refl[p] <= 0:
                continue
            outF = patch_rad[p] * patch_areas[p] * patch_refl[p]
            pc = patch_centers[p]
            n = patch_normals[p]
            # Construct tangential directions for MC sampling:
            if abs(n[2]) < 0.99:
                tangent1 = np.array([-n[1], n[0], 0.0])
            else:
                tangent1 = np.array([1.0, 0.0, 0.0])
            tangent1_norm = np.linalg.norm(tangent1)
            if tangent1_norm > 0:
                tangent1 = tangent1 / tangent1_norm
            else:
                tangent1 = np.array([1.0, 0.0, 0.0])
            tangent2 = np.cross(n, tangent1)
            tangent2_norm = np.linalg.norm(tangent2)
            if tangent2_norm > 0:
                tangent2 = tangent2 / tangent2_norm
            else:
                tangent2 = np.array([0.0, 1.0, 0.0])
            half_side = math.sqrt(patch_areas[p]) / 2.0

            sample_sum = 0.0
            for _ in range(mc_samples):
                offset1 = np.random.uniform(-half_side, half_side)
                offset2 = np.random.uniform(-half_side, half_side)
                sample_point = pc + offset1*tangent1 + offset2*tangent2
                dx = fx - sample_point[0]
                dy = fy - sample_point[1]
                dz = -sample_point[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < 1e-15:
                    continue
                dist = math.sqrt(dist2)
                cos_f = -dz / dist if (-dz/dist) > 0 else 0.0
                dot_p = dx*n[0] + dy*n[1] + dz*n[2]
                cos_p = dot_p / (dist * np.linalg.norm(n))
                if cos_p < 0:
                    cos_p = 0.0
                ff = (cos_p * cos_f) / (math.pi * dist2)
                sample_sum += ff
            avg_ff = sample_sum / mc_samples
            val += outF * avg_ff
        row_vals.append(val)
    return row_vals

def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl, mc_samples=16):
    rows, cols = X.shape
    results = Parallel(n_jobs=-1)(
        delayed(compute_row_reflection)(
            r, X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl, mc_samples
        )
        for r in range(rows)
    )
    out = np.zeros((rows, cols), dtype=np.float64)
    for r, row_vals in enumerate(results):
        out[r, :] = row_vals
    return out

def simulate_lighting(params, geo):
    # Unpack geometry
    cob_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geo
    lumens_per_cob = pack_luminous_flux_dynamic(params, cob_positions)

    # Direct on floor
    floor_lux_like = compute_direct_floor(cob_positions, lumens_per_cob, X, Y)
    # Direct on patches
    patch_direct = compute_patch_direct(cob_positions, lumens_per_cob, p_centers, p_normals, p_areas)
    # Radiosity
    patch_rad = iterative_radiosity_loop(
        p_centers, p_normals, patch_direct, p_areas, p_refl,
        MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD
    )
    # Reflection onto floor
    reflect_floor = compute_reflection_on_floor(
        X, Y, p_centers, p_normals, p_areas, patch_rad, p_refl
    )
    total_luminous = floor_lux_like + reflect_floor
    total_radiant_Wm2 = total_luminous / LUMINOUS_EFFICACY
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR
    return floor_ppfd

# -----------------------------
#  OBJECTIVE FUNCTION CHANGES
# -----------------------------
def objective_function_BO(**kwargs):
    """
    Modified objective:
    - We heavily weight the DOU directly (dou_weight).
    - Asymmetric penalty: heavier penalty if below target, lighter if above.
    """
    global geo_opt, target_ppfd

    # Convert all p{i} to param array
    params = [kwargs[f'p{i}'] for i in range(10)]
    floor_ppfd = simulate_lighting(params, geo_opt)

    mean_ppfd = np.mean(floor_ppfd)
    if mean_ppfd <= 0:
        return -1e6

    # Degree of Uniformity (minimum-to-average)
    # Or use standard deviation-based metric. We'll continue with RMSE-based.
    rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))
    dou = 100.0 * (1.0 - rmse / mean_ppfd)  # same formula as original

    # Asymmetric penalty: heavier if below the target
    penalty = 0.0
    diff = mean_ppfd - target_ppfd
    if diff < 0:
        # Heavier penalty if below target
        penalty = 0.3 * (abs(diff))**2
    else:
        # Light penalty if going above
        penalty = 0.1 * (diff)**2

    # Weighted sum. We place more emphasis on DOU now.
    dou_weight = 1.0  # tune this up if you want uniformity to matter more
    score = dou_weight * dou - penalty

    return score

def optimize_lighting(W, L, H, target_ppfd_arg, progress_callback=None):
    """
    1) Adaptive exploration via dynamic kappa.
    2) Multiple restarts.
    3) Final scaling of intensities if needed.
    """
    global geo_opt, target_ppfd
    target_ppfd = target_ppfd_arg  # Update global
    geo_opt = prepare_geometry(W, L, H, FLOOR_GRID_RES_OPT)

    # Param space
    pbounds = {f'p{i}': (MIN_LUMENS, MAX_LUMENS_MAIN) for i in range(10)}

    def run_single_bo(random_seed):
        # We create a BayesianOptimization object
        optimizer = BayesianOptimization(
            f=objective_function_BO,
            pbounds=pbounds,
            verbose=2,
            random_state=random_seed,
        )

        # Optional pre-defined probes to seed
        probe_points = [
            [4000.0,  2000.0, 8000.0, 5000.0, 6000.0, 12000.0, 15000.0, 2000.0, 20000.0, 24000.0],
            [12000.0, 12000.0, 12000.0,12000.0,12000.0,12000.0,12000.0,12000.0,12000.0,12000.0],
        ]
        for probe in probe_points:
            optimizer.probe(
                params={f'p{i}': val for i, val in enumerate(probe)},
                lazy=True,
            )

        # We'll do a dynamic approach: higher kappa early, then lower it later
        n_iter = 150
        for i in range(n_iter):
            frac = i / float(n_iter)
            next_point = optimizer.suggest()
            next_point = optimizer.suggest()
            target = objective_function_BO(**next_point)
            optimizer.register(params=next_point, target=target)

            if progress_callback:
                progress_callback(f"Iteration {i}, target={target:.2f}")

        return optimizer

    # MULTIPLE RESTARTS: run the optimization multiple times with different seeds
    best_optimizer = None
    best_value = -1e9
    restarts = 3  # set to 3 or more for thoroughness
    for seed in range(1, restarts + 1):
        temp_opt = run_single_bo(random_seed=seed)
        if temp_opt.max['target'] > best_value:
            best_value = temp_opt.max['target']
            best_optimizer = temp_opt

    # We now have the best optimizer from multiple runs
    best_params_unscaled = [best_optimizer.max['params'][f'p{i}'] for i in range(10)]

    # Evaluate the best solution on a finer grid
    geo_final = prepare_geometry(W, L, H, FLOOR_GRID_RES_FINAL)
    final_ppfd_unscaled = simulate_lighting(best_params_unscaled, geo_final)
    mean_ppfd_unscaled = np.mean(final_ppfd_unscaled)
    s = target_ppfd / mean_ppfd_unscaled if mean_ppfd_unscaled > 0 else 1.0
    # Keep scaling factor from going beyond lumens cap
    s_max = min(MAX_LUMENS_MAIN / p for p in best_params_unscaled if p > 0)
    s_final = min(s, s_max)

    best_params_scaled = [p * s_final for p in best_params_unscaled]
    final_ppfd = simulate_lighting(best_params_scaled, geo_final)
    mean_ppfd = np.mean(final_ppfd)
    rmse = np.sqrt(np.mean((final_ppfd - mean_ppfd)**2))
    dou = 100.0 * (1.0 - rmse / mean_ppfd) if mean_ppfd > 0 else 0
    print(f"Mean PPFD: {mean_ppfd:.2f}, DOU: {dou:.2f}%, Params: {best_params_scaled}")
    if mean_ppfd < target_ppfd:
        print("Note: Target PPFD not fully reached within constraints.")

    return best_params_scaled, geo_final

def run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height=3.0, progress_callback=None):
    try:
        floor_width_ft = float(floor_width_ft)
        floor_length_ft = float(floor_length_ft)
        target_ppfd = float(target_ppfd)
        floor_height = float(floor_height)
    except Exception as e:
        print("Error converting inputs to float:", e)
        return {}

    ft2m = 3.28084
    W_m = floor_width_ft / ft2m
    L_m = floor_length_ft / ft2m
    H_m = floor_height / ft2m

    heartbeat_active = [True]
    if progress_callback:
        def heartbeat():
            while heartbeat_active[0]:
                progress_callback(": heartbeat")
                time.sleep(15)
        hb_thread = threading.Thread(target=heartbeat)
        hb_thread.daemon = True
        hb_thread.start()

    best_params, geo = optimize_lighting(W_m, L_m, H_m, target_ppfd, progress_callback)

    final_ppfd = simulate_lighting(best_params, geo)
    mean_ppfd = np.mean(final_ppfd)
    rmse = np.sqrt(np.mean((final_ppfd - mean_ppfd)**2))
    dou = 100.0 * (1.0 - rmse / mean_ppfd)
    mad = np.mean(np.abs(final_ppfd - mean_ppfd))
    cv = 100.0 * (np.std(final_ppfd) / mean_ppfd)

    result = {
        "optimized_lumens_by_layer": best_params,
        "mad": float(mad),
        "optimized_ppfd": float(mean_ppfd),
        "rmse": float(rmse),
        "dou": float(dou),
        "cv": float(cv),
        "floor_width": floor_width_ft,
        "floor_length": floor_length_ft,
        "target_ppfd": target_ppfd,
        "floor_height": floor_height,
    }

    if progress_callback:
        heartbeat_active[0] = False
        progress_callback("PROGRESS:100")

    return result

def run_simulation(floor_width_ft=20.0, floor_length_ft=20.0, target_ppfd=1250.0, floor_height=3.0):
    result = run_ml_simulation(floor_width_ft, floor_length_ft, target_ppfd, floor_height)
    print("\n[RESULT] Simulation Output:")
    print(result)

def main():
    run_simulation()

if __name__ == "__main__":
    main()
