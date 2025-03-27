#!/usr/bin/env python3
import csv
import math
import numpy as np
from numba import njit, prange
from functools import lru_cache
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse  # Import argparse

# ------------------------------------
# 1) Basic Config & Reflectances
# ------------------------------------
REFL_WALL = 0.8
REFL_CEIL = 0.8
REFL_FLOOR = 0.1

LUMINOUS_EFFICACY = 182.0  # lumens/W
SPD_FILE = "/Users/austinrouse/photonics/backups/spd_data.csv"

MAX_RADIOSITY_BOUNCES = 10
RADIOSITY_CONVERGENCE_THRESHOLD = 1e-3

WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10

FLOOR_GRID_RES = 0.08  # m
FIXED_NUM_LAYERS = 20
MC_SAMPLES = 16

# ------------------------------------
# 2) COB Datasheet Angular Data
# ------------------------------------
COB_ANGLE_DATA = np.array([
    [  0, 1.00],
    [ 10, 0.98],
    [ 20, 0.95],
    [ 30, 0.88],
    [ 40, 0.78],
    [ 50, 0.65],
    [ 60, 0.50],
    [ 70, 0.30],
    [ 80, 0.10],
    [ 90, 0.00],
], dtype=np.float64)

COB_angles_deg = COB_ANGLE_DATA[:, 0]
COB_shape = COB_ANGLE_DATA[:, 1]

# ------------------------------------
# 3) Compute SPD-based µmol/J Factor
# ------------------------------------
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

# ------------------------------------
# 4) Pre-Compute Normalization Factor
# ------------------------------------
def integrate_shape_for_flux(angles_deg, shape):
    rad_angles = np.radians(angles_deg)
    G = 0.0
    for i in range(len(rad_angles) - 1):
        th0 = rad_angles[i]
        th1 = rad_angles[i+1]
        s0 = shape[i]
        s1 = shape[i+1]
        s_mean = 0.5*(s0 + s1)
        dtheta = (th1 - th0)
        th_mid = 0.5*(th0 + th1)
        sin_mid = math.sin(th_mid)
        G_seg = s_mean * 2.0*math.pi * sin_mid * dtheta
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

# ------------------------------------
# 5) Geometry Building
# ------------------------------------
def prepare_geometry(W, L, H):
    cob_positions = build_cob_positions(W, L, H)
    X, Y = build_floor_grid(W, L)
    patches = build_patches(W, L, H)
    return (cob_positions, X, Y, patches)

def build_cob_positions(W, L, H):
    n = FIXED_NUM_LAYERS - 1
    positions = []
    positions.append((0, 0, H, 0))
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
    patch_centers.append((W/2, L/2, 0.0))
    patch_areas.append(W * L)
    patch_normals.append((0.0, 0.0, 1.0))
    patch_refl.append(REFL_FLOOR)
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
    xs_wall = np.linspace(0, W, WALL_SUBDIVS_X + 1)
    zs_wall = np.linspace(0, H, WALL_SUBDIVS_Y + 1)
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cx = (xs_wall[i] + xs_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (xs_wall[i+1]-xs_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((cx, 0.0, cz))
            patch_areas.append(area)
            patch_normals.append((0.0, -1.0, 0.0))
            patch_refl.append(REFL_WALL)
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
    for i in range(WALL_SUBDIVS_X):
        for j in range(WALL_SUBDIVS_Y):
            cy = (ys_wall[i] + ys_wall[i+1]) / 2
            cz = (zs_wall[j] + zs_wall[j+1]) / 2
            area = (ys_wall[i+1]-ys_wall[i]) * (zs_wall[j+1]-zs_wall[j])
            patch_centers.append((0.0, cy, cz))
            patch_areas.append(area)
            patch_normals.append((-1.0, 0.0, 0.0))
            patch_refl.append(REFL_WALL)
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

# ------------------------------------
# 6) The Numba-JIT Computations
# ------------------------------------
@njit(parallel=True)
def compute_direct_floor(cob_positions, cob_lumens, X, Y):
    min_dist2 = (FLOOR_GRID_RES / 2.0) ** 2
    rows, cols = X.shape
    out = np.zeros_like(X, dtype=np.float64)
    for r in prange(rows):
        for c in range(cols):
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
                cos_th = -dz/dist
                if cos_th <= 0:
                    continue
                angle_deg = math.degrees(math.acos(cos_th))
                I_theta = luminous_intensity(angle_deg, lumens_k)
                cos_in_floor = cos_th
                E_local = (I_theta / (dist*dist)) * cos_in_floor
                val += E_local
            out[r, c] = val
    return out

@njit
def compute_patch_direct(cob_positions, cob_lumens, patch_centers, patch_normals, patch_areas):
    min_dist2 = (FLOOR_GRID_RES / 2.0) ** 2
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
            cos_th_led = -dz/dist
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
                cos_j = dot_j/(dist*norm_nj)
                if cos_j < 0:
                    continue
                dot_i = -(ni0*dx + ni1*dy + ni2*dz)
                cos_i = dot_i/(dist*norm_ni)
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

#Using joblib as in the original script
def compute_reflection_on_floor(X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl,
                                mc_samples=MC_SAMPLES):
    rows, cols = X.shape
    results = Parallel(n_jobs=-1)(delayed(compute_row_reflection)(
        r, X, Y, patch_centers, patch_normals, patch_areas, patch_rad, patch_refl, mc_samples
    ) for r in range(rows))
    out = np.zeros((rows, cols), dtype=np.float64)
    for r, row_vals in enumerate(results):
        out[r, :] = row_vals
    return out

@njit
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
                cos_f = -dz/dist if (-dz/dist) > 0 else 0.0
                dot_p = dx*n[0] + dy*n[1] + dz*n[2]
                cos_p = dot_p/(dist*np.linalg.norm(n))
                if cos_p < 0:
                    cos_p = 0.0
                ff = (cos_p * cos_f) / (math.pi * dist2)
                sample_sum += ff
            avg_ff = sample_sum / mc_samples
            val += outF * avg_ff
        row_vals.append(val)
    return row_vals

# ------------------------------------
# 7) Heatmap Plotting Function (Optional, but useful for visualization)
# ------------------------------------
def plot_heatmap(floor_ppfd, X, Y, cob_positions, annotation_step=2):
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    heatmap = ax.imshow(floor_ppfd, cmap='hot', interpolation='nearest',
                        origin='lower', extent=extent)
    fig.colorbar(heatmap, ax=ax)

    rows, cols = floor_ppfd.shape
    for i in range(0, rows, annotation_step):
        for j in range(0, cols, annotation_step):
            ax.text(X[i, j], Y[i, j], f"{floor_ppfd[i, j]:.1f}",
                    ha="center", va="center", color="white", fontsize=6)
    ax.scatter(cob_positions[:, 0], cob_positions[:, 1], marker='o',
               color='blue', edgecolors='black', s=50, label="COB positions")

    ax.set_title("Floor PPFD Heatmap with COB Positions")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    plt.show(block=False)  # Add block=False
    plt.pause(0.1)

# ------------------------------------
# 8) CSV Output Function (NEW - with Sampling)
# ------------------------------------
def write_ppfd_to_csv(filename, floor_ppfd, X, Y, cob_positions, W, L):
    """
    Writes PPFD data to CSV, organized by distance-based rings (layers).
    """
    layer_data = {}
    for i in range(FIXED_NUM_LAYERS):
        layer_data[i] = []

    # Calculate layer radii based on COB positions
    cob_layers = cob_positions[:, 3]
    layer_radii = []
    center_x, center_y = W / 2, L / 2

    for i in range(FIXED_NUM_LAYERS):
      layer_cob_indices = np.where(cob_layers == i)[0]
      if len(layer_cob_indices) > 0:  # Handle cases with empty layers
          layer_cob_positions = cob_positions[layer_cob_indices, :2]
          distances = np.sqrt((layer_cob_positions[:, 0] - center_x)**2 +
                              (layer_cob_positions[:, 1] - center_y)**2)
          # Use the maximum distance as the outer radius for this layer
          layer_radii.append(np.max(distances))
      else:
          # If no COBs in layer, use a placeholder (e.g., previous layer's value)
          layer_radii.append(layer_radii[-1] if i > 0 else 0)


    rows, cols = floor_ppfd.shape
    for r in range(rows):
        for c in range(cols):
            fx = X[r, c]
            fy = Y[r, c]
            dist_to_center = np.sqrt((fx - center_x)**2 + (fy - center_y)**2)

            # Assign to layer based on distance rings
            assigned_layer = -1
            for i in range(FIXED_NUM_LAYERS):
                outer_radius = layer_radii[i]
                # For the inner radius, use the previous layer's outer radius,
                # or 0 for the innermost layer.
                inner_radius = layer_radii[i-1] if i > 0 else 0

                if inner_radius <= dist_to_center <= outer_radius:
                    assigned_layer = i
                    break

            if assigned_layer != -1:
                layer_data[assigned_layer].append(floor_ppfd[r, c])


    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Layer', 'PPFD'])
        for layer in sorted(layer_data.keys()):
            for ppfd_value in layer_data[layer]:
                writer.writerow([layer, ppfd_value])

# ------------------------------------
# 9) Putting It All Together
# ------------------------------------
def simulate_lighting(params, geo):
    cob_positions, X, Y, (p_centers, p_areas, p_normals, p_refl) = geo
    lumens_per_cob = pack_luminous_flux_dynamic(params, cob_positions)
    floor_lux_like = compute_direct_floor(cob_positions, lumens_per_cob, X, Y)
    patch_direct = compute_patch_direct(cob_positions, lumens_per_cob, p_centers, p_normals, p_areas)
    patch_rad = iterative_radiosity_loop(p_centers, p_normals, patch_direct, p_areas, p_refl,
                                         MAX_RADIOSITY_BOUNCES, RADIOSITY_CONVERGENCE_THRESHOLD)
    reflect_floor = compute_reflection_on_floor(X, Y, p_centers, p_normals, p_areas, patch_rad, p_refl)
    total_luminous = floor_lux_like + reflect_floor
    total_radiant_Wm2 = total_luminous / LUMINOUS_EFFICACY
    floor_ppfd = total_radiant_Wm2 * CONVERSION_FACTOR
    return floor_ppfd, X, Y, cob_positions  # Return cob_positions

def main():
    W = 12.192  # 30 ft in meters
    L = 12.192  # 30 ft in meters
    H = 0.9144  # 3 ft in meters # CORRECTED height
    
    params = np.array([
    3436.5441,     # Center COB
    4515.2702,     # Layer 1
    5269.6295,     # Layer 2
    5810.3411,     # Layer 3
    6248.1242,     # Layer 4
    6693.6982,     # Layer 5
    7257.7821,     # Layer 6
    8051.0953,     # Layer 7
    9169.2352,     # Layer 8
    10421.5236,     # Layer 9
    11361.5126,     # Layer 10
    11533.7583,     # Layer 11
    10507.2603,     # Layer 12
    8539.0178,     # Layer 13
    6652.1502,     # Layer 14
    5909.9016,     # Layer 15
    7352.6524,     # Layer 16
    11263.6230,     # Layer 17
    17012.4995,     # Layer 18
    25000.0000,     # Layer 19
    ], dtype=np.float64)

    geo = prepare_geometry(W, L, H)
    floor_ppfd, X, Y, cob_positions = simulate_lighting(params, geo)

    mean_ppfd = np.mean(floor_ppfd)
    mad = np.mean(np.abs(floor_ppfd - mean_ppfd))
    rmse = np.sqrt(np.mean((floor_ppfd - mean_ppfd)**2))
    dou = 100 * (1 - rmse / mean_ppfd)
    cv = 100 * (np.std(floor_ppfd) / mean_ppfd)

    print(f"Average PPFD: {mean_ppfd:.2f} µmol/m²/s")
    print(f"MAD: {mad:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"DOU (%): {dou:.2f}")
    print(f"CV (%): {cv:.2f}")

    # --- CSV Output (with Sampling) ---
    write_ppfd_to_csv("ppfd_data.csv", floor_ppfd, X, Y, cob_positions, W, L) # Pass W and L
    print("PPFD data written to ppfd_data.csv")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Lighting Simulation Script")
    parser.add_argument('--no-plot', action='store_true', help='Disable the heatmap plot')
    args = parser.parse_args()

    # --- Heatmap (Optional) ---
    if not args.no_plot:
        plot_heatmap(floor_ppfd, X, Y, cob_positions, annotation_step=10)

if __name__ == "__main__":
    main()