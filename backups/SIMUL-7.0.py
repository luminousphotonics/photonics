import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import ttk
import json as json
import os

# ------------------------------
# User-Configurable Parameters
# ------------------------------
LAYER_COUNTS = [1, 4, 8, 12, 16, 20]
SETTINGS_FILE = "settings.json"

# For subdividing walls/ceiling:
WALL_SUBDIVS_X = 10
WALL_SUBDIVS_Y = 5
CEIL_SUBDIVS_X = 10
CEIL_SUBDIVS_Y = 10
NUM_RADIOSITY_BOUNCES = 2  # More bounces => more reflection accuracy

FLOOR_GRID_RES = 0.02  # meters
REFL_WALL = 0.85
REFL_CEIL = 0.1
REFL_FLOOR = 0.0  # Often negligible

def run_simulation():
    # 1. Fetch user inputs
    try:
        layer_intensities = [float(e.get().strip()) for e in layer_intensity_entries]
    except:
        print("Error reading layer intensities.")
        return

    led_intensities = []
    for intensity, count in zip(layer_intensities, LAYER_COUNTS):
        led_intensities.extend([intensity]*count)

    try:
        corner_vals = [float(e.get().strip()) for e in corner_intensity_entries]
    except:
        print("Error reading corner intensities.")
        return

    # Overwrite corner COBs in layer 6
    corner_idx = [41+16, 41+17, 41+18, 41+19]
    for i, idx_val in enumerate(corner_idx):
        if idx_val < len(led_intensities):
            led_intensities[idx_val] = corner_vals[i]

    try:
        led_strip_fluxes = [float(e.get().strip()) for e in led_strip_entries]
    except:
        print("Error reading LED strip fluxes.")
        return

    try:
        layer7_flux = float(layer7_intensity_entry.get().strip())
    except:
        print("Error reading Layer 7 LED strip intensity.")
        return

    # 2. Load SPD => for PAR conversion
    luminous_efficacy = 182.0  # lm/W
    try:
        spd = np.loadtxt(r"/Users/austinrouse/photonics/backups/spd_data.csv", delimiter=' ', skiprows=1)
    except:
        print("Error loading SPD data.")
        return
    wl = spd[:, 0]
    intens = spd[:, 1]
    mask_par = (wl >= 400) & (wl <= 700)
    tot = np.trapz(intens, wl)
    tot_par = np.trapz(intens[mask_par], wl[mask_par])
    PAR_fraction = tot_par / tot if tot > 0 else 1.0

    wl_m = wl * 1e-9
    h, c, N_A = 6.626e-34, 3.0e8, 6.022e23
    numerator = np.trapz(wl_m[mask_par]*intens[mask_par], wl_m[mask_par])
    denominator = np.trapz(intens[mask_par], wl_m[mask_par]) if np.trapz(intens[mask_par], wl_m[mask_par])>0 else 1
    lambda_eff = numerator / denominator
    E_photon = h*c / lambda_eff if lambda_eff>0 else 1
    conversion_factor = (1./E_photon)*(1e6/N_A)*PAR_fraction

    print(f"PAR fraction = {PAR_fraction:.3f}")
    print(f"Conversion factor = {conversion_factor:.3f} µmol/J")

    # 3. Room geometry
    W_m, L_m, H_m = 12.0, 12.0, 3.0
    ft2m = 3.28084
    W = W_m/ft2m
    L = L_m/ft2m
    H = H_m/ft2m

    # 4. COB LED arrangement
    light_positions = []
    light_fluxes = []

    # Original spacing
    original_spacing = W / 7.2
    # We'll shrink layers 1..6 so that they're inside:
    shrink_factor = 0.8  # tweak to push them further in or out
    layer6_spacing = original_spacing * shrink_factor

    # Diamond of 61 COB positions in 6 layers
    layers_coords = [
        [(0, 0)],
        [(-1, 0), (1, 0), (0, -1), (0, 1)],
        [(-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)],
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
    center = (W/2, L/2)

    # Add center (Layer 0)
    light_positions.append((center[0], center[1], H))
    light_fluxes.append(0.0)

    # For layers 1..6, use the smaller spacing
    for i, layer in enumerate(layers_coords):
        if i == 0:
            continue
        for dx, dy in layer:
            theta = math.radians(45)
            rx = dx * layer6_spacing * math.cos(theta) - dy * layer6_spacing * math.sin(theta)
            ry = dx * layer6_spacing * math.sin(theta) + dy * layer6_spacing * math.cos(theta)
            px = center[0] + rx
            py = center[1] + ry
            pz = H
            light_positions.append((px, py, pz))
            light_fluxes.append(0.0)

    # Assign intensities
    if len(led_intensities) != len(light_positions):
        print("Mismatch in LED intensities.")
        return
    for i, val in enumerate(led_intensities):
        light_fluxes[i] = val

    # 5. Existing LED strips (unchanged)
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
    points_per_seg = 5
    strip_plot_data = []

    for sn, ids in led_strips.items():
        seg_pts = []
        for i in range(len(ids)-1):
            s_idx = ids[i] - 1
            e_idx = ids[i+1] - 1
            spos = light_positions[s_idx]
            epos = light_positions[e_idx]
            for t in np.linspace(0, 1, points_per_seg, endpoint=False):
                px = spos[0] + t*(epos[0] - spos[0])
                py = spos[1] + t*(epos[1] - spos[1])
                pz = spos[2] + t*(epos[2] - spos[2])
                seg_pts.append((px, py, pz))
        seg_pts.append(light_positions[ids[-1] - 1])
        flux_tot = led_strip_fluxes[sn-1]
        npt = len(seg_pts)
        fpp = flux_tot / npt if npt else 0
        for pt in seg_pts:
            light_positions.append(pt)
            light_fluxes.append(fpp)
        strip_plot_data.append(seg_pts)

    # 5.5. Layer 7 LED strips
    # We want to place them where the old bounding box for layer 6 would have been,
    # i.e. as if we used original_spacing instead of the smaller spacing.
    def compute_original_layer6_bbox():
        coords = []
        for dx, dy in layers_coords[5]:  # layer 6 coords
            theta = math.radians(45)
            rx = dx * original_spacing * math.cos(theta) - dy * original_spacing * math.sin(theta)
            ry = dx * original_spacing * math.sin(theta) + dy * original_spacing * math.cos(theta)
            coords.append((center[0] + rx, center[1] + ry))
        arr = np.array(coords)
        min_x, max_x = arr[:,0].min(), arr[:,0].max()
        min_y, max_y = arr[:,1].min(), arr[:,1].max()
        return (min_x, max_x, min_y, max_y)

    min_x, max_x, min_y, max_y = compute_original_layer6_bbox()

    # We'll define two rings:
    # - Outer ring = exactly the old bounding box
    # - Inner ring = offset inwards slightly
    offset = 0.2  # how close the inner ring is to the outer ring
    outer_rect = [min_x, max_x, min_y, max_y]
    inner_rect = [min_x + offset, max_x - offset, min_y + offset, max_y - offset]

    def build_perimeter_points(rect, n_subdiv=20):
        x_min, x_max, y_min, y_max = rect
        segs = []
        # top
        top = []
        for t in np.linspace(0, 1, n_subdiv, endpoint=False):
            x = x_min + t*(x_max - x_min)
            y = y_max
            top.append((x, y, H))
        top.append((x_max, y_max, H))
        segs.append(top)
        # right
        right = []
        for t in np.linspace(0, 1, n_subdiv, endpoint=False):
            x = x_max
            y = y_max - t*(y_max - y_min)
            right.append((x, y, H))
        right.append((x_max, y_min, H))
        segs.append(right)
        # bottom
        bot = []
        for t in np.linspace(0, 1, n_subdiv, endpoint=False):
            x = x_max - t*(x_max - x_min)
            y = y_min
            bot.append((x, y, H))
        bot.append((x_min, y_min, H))
        segs.append(bot)
        # left
        left = []
        for t in np.linspace(0, 1, n_subdiv, endpoint=False):
            x = x_min
            y = y_min + t*(y_max - y_min)
            left.append((x, y, H))
        left.append((x_min, y_max, H))
        segs.append(left)
        return segs

    layer7_outer = build_perimeter_points(outer_rect, n_subdiv=20)
    layer7_inner = build_perimeter_points(inner_rect, n_subdiv=20)

    # We'll distribute half the flux to each ring
    flux_out = layer7_flux * 0.5
    flux_in  = layer7_flux * 0.5

    layer7_plot_data = []
    # Outer ring
    total_out_pts = sum(len(s) for s in layer7_outer)
    fpp_out = flux_out / total_out_pts if total_out_pts else 0
    for seg in layer7_outer:
        layer7_plot_data.append(seg)
        for pt in seg:
            light_positions.append(pt)
            light_fluxes.append(fpp_out)

    # Inner ring
    total_in_pts = sum(len(s) for s in layer7_inner)
    fpp_in = flux_in / total_in_pts if total_in_pts else 0
    for seg in layer7_inner:
        layer7_plot_data.append(seg)
        for pt in seg:
            light_positions.append(pt)
            light_fluxes.append(fpp_in)

    # 6. Floor grid
    xs = np.arange(0, W, FLOOR_GRID_RES)
    ys = np.arange(0, L, FLOOR_GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    direct_irr = np.zeros_like(X)

    # 7. Physically consistent direct: Lambertian downward
    for (pos, lum) in zip(light_positions, light_fluxes):
        P = lum / luminous_efficacy
        x0, y0, z0 = pos
        dx = X - x0
        dy = Y - y0
        dz = -z0
        dist2 = dx*dx + dy*dy + dz*dz
        dist = np.sqrt(dist2)
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_th = -dz / dist
        cos_th[cos_th < 0] = 0
        with np.errstate(divide='ignore'):
            E = (P / math.pi) * (cos_th / dist2)
        direct_irr += np.nan_to_num(E)

    # 8. Patch-based surfaces for multi-bounce
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []

    # Floor
    patch_centers.append((W/2, L/2, 0))
    patch_areas.append(W*L)
    patch_normals.append((0,0,1))
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
            patch_normals.append((0, 0, -1))
            patch_refl.append(REFL_CEIL)

    # Walls
    def subdiv_wall_y0():
        dx = W/WALL_SUBDIVS_X
        dz = H/WALL_SUBDIVS_Y
        for ix in range(WALL_SUBDIVS_X):
            for iz in range(WALL_SUBDIVS_Y):
                px = (ix+0.5)*dx
                py = 0
                pz = (iz+0.5)*dz
                patch_centers.append((px, py, pz))
                patch_areas.append(dx*dz)
                patch_normals.append((0, -1, 0))
                patch_refl.append(REFL_WALL)
    subdiv_wall_y0()

    def subdiv_wall_yL():
        dx = W/WALL_SUBDIVS_X
        dz = H/WALL_SUBDIVS_Y
        for ix in range(WALL_SUBDIVS_X):
            for iz in range(WALL_SUBDIVS_Y):
                px = (ix+0.5)*dx
                py = L
                pz = (iz+0.5)*dz
                patch_centers.append((px, py, pz))
                patch_areas.append(dx*dz)
                patch_normals.append((0, 1, 0))
                patch_refl.append(REFL_WALL)
    subdiv_wall_yL()

    def subdiv_wall_x0():
        dy = L/WALL_SUBDIVS_X
        dz = H/WALL_SUBDIVS_Y
        for iy in range(WALL_SUBDIVS_X):
            for iz in range(WALL_SUBDIVS_Y):
                px = 0
                py = (iy+0.5)*dy
                pz = (iz+0.5)*dz
                patch_centers.append((px, py, pz))
                patch_areas.append(dy*dz)
                patch_normals.append((-1, 0, 0))
                patch_refl.append(REFL_WALL)
    subdiv_wall_x0()

    def subdiv_wall_xW():
        dy = L/WALL_SUBDIVS_X
        dz = H/WALL_SUBDIVS_Y
        for iy in range(WALL_SUBDIVS_X):
            for iz in range(WALL_SUBDIVS_Y):
                px = W
                py = (iy+0.5)*dy
                pz = (iz+0.5)*dz
                patch_centers.append((px, py, pz))
                patch_areas.append(dy*dz)
                patch_normals.append((1, 0, 0))
                patch_refl.append(REFL_WALL)
    subdiv_wall_xW()

    patch_centers = np.array(patch_centers)
    patch_areas = np.array(patch_areas)
    patch_normals = np.array(patch_normals)
    patch_refl = np.array(patch_refl)
    Np = len(patch_centers)

    patch_direct = np.zeros(Np)
    for ip in range(Np):
        pc = patch_centers[ip]
        n = patch_normals[ip]
        accum = 0.0
        for (lp, lum) in zip(light_positions, light_fluxes):
            P = lum / luminous_efficacy
            dx = pc[0] - lp[0]
            dy = pc[1] - lp[1]
            dz = pc[2] - lp[2]
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2 < 1e-12:
                continue
            dist = math.sqrt(dist2)
            dd = np.array([dx, dy, dz])
            cos_th_led = -dz/dist
            if cos_th_led < 0:
                continue
            E_led = (P/math.pi)*(cos_th_led/dist2)
            cos_in_patch = np.dot(-dd, n)/(dist*np.linalg.norm(n))
            if cos_in_patch < 0:
                cos_in_patch = 0
            accum += E_led*cos_in_patch
        patch_direct[ip] = accum

    patch_flux = patch_direct*patch_areas
    patch_rad = np.copy(patch_direct)

    for b in range(NUM_RADIOSITY_BOUNCES):
        new_flux = np.zeros(Np)
        for j in range(Np):
            if patch_refl[j] <= 0:
                continue
            outF = patch_rad[j]*patch_areas[j]*patch_refl[j]
            pc_j = patch_centers[j]
            n_j = patch_normals[j]
            for i2 in range(Np):
                if i2 == j:
                    continue
                pc_i = patch_centers[i2]
                n_i = patch_normals[i2]
                dd = pc_i - pc_j
                dist2 = np.dot(dd, dd)
                if dist2 < 1e-12:
                    continue
                dist = math.sqrt(dist2)
                cos_j = np.dot(n_j, dd)/(dist*np.linalg.norm(n_j))
                cos_i = np.dot(-n_i, dd)/(dist*np.linalg.norm(n_i))
                if cos_j < 0 or cos_i < 0:
                    continue
                ff = (cos_j*cos_i)/(math.pi*dist2)
                new_flux[i2] += outF*ff
        patch_rad = patch_direct + new_flux/patch_areas

    reflect_irr = np.zeros_like(X)
    floor_pts = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=1)
    floor_n = np.array([0,0,1], dtype=float)

    for p in range(Np):
        outF = patch_rad[p]*patch_areas[p]*patch_refl[p]
        if outF < 1e-15:
            continue
        pc = patch_centers[p]
        n = patch_normals[p]
        dv = floor_pts - pc
        dist2 = np.einsum('ij,ij->i', dv, dv)
        dist = np.sqrt(dist2)
        cos_p = np.einsum('ij,j->i', dv, n)/(dist*np.linalg.norm(n)+1e-15)
        cos_f = np.einsum('ij,j->i', -dv, floor_n)/(dist+1e-15)
        cos_p[cos_p<0] = 0
        cos_f[cos_f<0] = 0
        ff = (cos_p*cos_f)/(math.pi*dist2+1e-15)
        cell_flux = outF*ff
        cell_area = FLOOR_GRID_RES*FLOOR_GRID_RES
        cell_irr = cell_flux/cell_area
        reflect_irr.ravel()[:] += np.nan_to_num(cell_irr)

    floor_irr = direct_irr + reflect_irr
    floor_ppfd = floor_irr * conversion_factor

    avg_ppfd = np.mean(floor_ppfd)
    rmse = np.sqrt(np.mean((floor_ppfd - avg_ppfd)**2))
    du = 100*(1-(rmse/avg_ppfd))
    print(f"Average PPFD = {avg_ppfd:.1f} µmol/m²/s")
    print(f"RMSE         = {rmse:.1f} µmol/m²/s")
    print(f"Degree of Uniformity = {du:.1f} %")

    plt.figure(figsize=(6,5))
    plt.imshow(floor_ppfd, origin='lower', extent=[0,W,0,L], cmap='viridis')
    plt.colorbar(label='PPFD (µmol/m²/s)')
    plt.xlabel('Width (m)')
    plt.ylabel('Length (m)')
    plt.title(f"Lambertian Down + {NUM_RADIOSITY_BOUNCES}-Bounce Radiosity\nAvg={avg_ppfd:.1f}, DU={du:.1f}%")

    # Measurement labels
    num_measure = 10
    x_measure = np.linspace(0, W, num_measure)
    y_measure = np.linspace(0, L, num_measure)
    for xm in x_measure:
        for ym in y_measure:
            ix = min(int(xm / FLOOR_GRID_RES), floor_ppfd.shape[1]-1)
            iy = min(int(ym / FLOOR_GRID_RES), floor_ppfd.shape[0]-1)
            val = floor_ppfd[iy, ix]
            plt.text(xm, ym, f'{val:.1f}', color='white', ha='center', va='center',
                     fontsize=6, bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # Plot COB LEDs (Layer 6 is shrunk inside)
    num_cob_leds = 61
    for pos in light_positions[:num_cob_leds]:
        plt.plot(pos[0], pos[1], 'rx', markersize=8)

    # Plot existing LED strips (pink)
    for seg_pts in strip_plot_data:
        sx = [p[0] for p in seg_pts]
        sy = [p[1] for p in seg_pts]
        plt.plot(sx, sy, 'm--', linewidth=1)
        for p in seg_pts:
            plt.plot(p[0], p[1], 'mo', markersize=3)

    # Plot the new Layer 7 perimeter strips (green)
    # Outer ring is exactly old bounding box, inner ring is offset inwards
    # so we have two rows for Layer 7
    plt.plot([], [], ' ', label="Layer 7 perimeter")  # for legend
    layer7_plot_data = layer7_outer + layer7_inner
    for seg in layer7_plot_data:
        gx = [p[0] for p in seg]
        gy = [p[1] for p in seg]
        plt.plot(gx, gy, 'g--', linewidth=1)
        for p in seg:
            plt.plot(p[0], p[1], 'go', markersize=3)

    plt.legend(loc='upper right', fontsize=8)
    plt.show()

# ------------------------------
# Basic Tkinter UI
# ------------------------------
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_settings():
    s = {
        "layer_intensities": [e.get() for e in layer_intensity_entries],
        "corner_intensities": [e.get() for e in corner_intensity_entries],
        "led_strip_fluxes": [e.get() for e in led_strip_entries],
        "layer7_intensity": layer7_intensity_entry.get()
    }
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f)
    except:
        pass

def on_closing():
    save_settings()
    root.destroy()

root = tk.Tk()
root.title("Physically Consistent PPFD Radiosity")
root.protocol("WM_DELETE_WINDOW", on_closing)

layer_frame = ttk.LabelFrame(root, text="Layer Settings")
layer_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

layer_int_frame = ttk.LabelFrame(layer_frame, text="COB Intensities (lm)")
layer_int_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
layer_intensity_entries = []
defaults = [6000, 8000, 12000, 8000, 10000, 18000]
for i, df in enumerate(defaults):
    lb = ttk.Label(layer_int_frame, text=f"Layer {i+1}:")
    lb.grid(row=i, column=0, padx=2, pady=2, sticky="e")
    e = ttk.Entry(layer_int_frame, width=7)
    e.insert(0, str(df))
    e.grid(row=i, column=1, padx=2, pady=2, sticky="w")
    layer_intensity_entries.append(e)

corner_frame = ttk.LabelFrame(root, text="Corner COB Intensities (lm)")
corner_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
corner_intensity_entries = []
corner_labels = ["Left:", "Right:", "Bottom:", "Top:"]
for i in range(4):
    lb = ttk.Label(corner_frame, text=corner_labels[i])
    lb.grid(row=i, column=0, padx=2, pady=2, sticky="e")
    e = ttk.Entry(corner_frame, width=7)
    e.insert(0, "18000")
    e.grid(row=i, column=1, padx=2, pady=2, sticky="w")
    corner_intensity_entries.append(e)

led_strip_frame = ttk.LabelFrame(root, text="LED Strip Luminous Flux (lm)")
led_strip_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
led_strip_entries = []
n_strips = 20
cols = 2
for i in range(n_strips):
    r = i // cols
    c = i % cols
    lb = ttk.Label(led_strip_frame, text=f"Strip #{i+1}:")
    lb.grid(row=r, column=2*c, padx=2, pady=2, sticky="e")
    e = ttk.Entry(led_strip_frame, width=10)
    e.insert(0, "3000")
    e.grid(row=r, column=2*c+1, padx=2, pady=2, sticky="w")
    led_strip_entries.append(e)

layer7_frame = ttk.LabelFrame(root, text="Layer 7 LED Strip Intensity (lm)")
layer7_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
layer7_intensity_entry = ttk.Entry(layer7_frame, width=10)
layer7_intensity_entry.insert(0, "6000")
layer7_intensity_entry.grid(row=0, column=0, padx=2, pady=2)

loaded = load_settings()
if loaded:
    if "layer_intensities" in loaded:
        for e, v in zip(layer_intensity_entries, loaded["layer_intensities"]):
            e.delete(0, tk.END)
            e.insert(0, v)
    if "corner_intensities" in loaded:
        for e, v in zip(corner_intensity_entries, loaded["corner_intensities"]):
            e.delete(0, tk.END)
            e.insert(0, v)
    if "led_strip_fluxes" in loaded:
        for e, v in zip(led_strip_entries, loaded["led_strip_fluxes"]):
            e.delete(0, tk.END)
            e.insert(0, v)
    if "layer7_intensity" in loaded:
        layer7_intensity_entry.delete(0, tk.END)
        layer7_intensity_entry.insert(0, loaded["layer7_intensity"])

run_button = ttk.Button(root, text="Run Simulation", command=run_simulation)
run_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
