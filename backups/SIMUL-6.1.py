import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import ttk
import json as json  # Use json alias
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
REFL_WALL = 0.9
REFL_CEIL = 0.1
REFL_FLOOR = 0.0  # Often negligible

# ------------------------------
# Core Simulation
# ------------------------------
def run_simulation():
    global floor_ppfd
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
    corner_idx = [41+16, 41+17, 41+18, 41+19]
    for i, idx_val in enumerate(corner_idx):
        if idx_val < len(led_intensities):
            led_intensities[idx_val] = corner_vals[i]

    try:
        led_strip_fluxes = [float(e.get().strip()) for e in led_strip_entries]
    except:
        print("Error reading LED strip fluxes.")
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
    lambda_eff = numerator/denominator
    E_photon = h*c/lambda_eff if lambda_eff>0 else 1
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

    # Diamond of 61 COB positions in 6 layers
    layers_coords = [
        [(0, 0)],
        [(-1, 0), (1, 0), (0, -1), (0, 1)],
        [(-1, -1), (1, -1), (-1, 1), (1, 1), ( -2, 0), (2, 0), (0, -2), (0, 2)],
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
    spacing = W/7.2  # same scale in x,y

    # Add center
    light_positions.append((center[0], center[1], H))
    light_fluxes.append(0.0)

    for i, layer in enumerate(layers_coords):
        if i==0:
            continue
        for dx, dy in layer:
            # rotate 45 deg
            theta = math.radians(45)
            rx = dx*spacing*math.cos(theta) - dy*spacing*math.sin(theta)
            ry = dx*spacing*math.sin(theta) + dy*spacing*math.cos(theta)
            px = center[0]+rx
            py = center[1]+ry
            pz = H
            light_positions.append((px, py, pz))
            light_fluxes.append(0.0)

    if len(led_intensities)!=len(light_positions):
        print("Mismatch in LED intensities.")
        return
    for i,val in enumerate(led_intensities):
        light_fluxes[i]=val

    # 5. LED strips
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
    for sn,ids in led_strips.items():
        seg_pts=[]
        for i in range(len(ids)-1):
            s_idx=ids[i]-1
            e_idx=ids[i+1]-1
            spos=light_positions[s_idx]
            epos=light_positions[e_idx]
            for t in np.linspace(0,1,points_per_seg,endpoint=False):
                px=spos[0]+t*(epos[0]-spos[0])
                py=spos[1]+t*(epos[1]-spos[1])
                pz=spos[2]+t*(epos[2]-spos[2])
                seg_pts.append((px,py,pz))
        seg_pts.append(light_positions[ids[-1]-1])
        flux_tot = led_strip_fluxes[sn-1]
        npt = len(seg_pts)
        fpp = flux_tot/npt if npt else 0
        for pt in seg_pts:
            light_positions.append(pt)
            light_fluxes.append(fpp)

    # 6. Floor grid
    xs = np.arange(0, W, FLOOR_GRID_RES)
    ys = np.arange(0, L, FLOOR_GRID_RES)
    X, Y = np.meshgrid(xs, ys)
    direct_irr = np.zeros_like(X)

    # 7. Physically consistent direct: Lambertian downward
    for (pos, lum) in zip(light_positions, light_fluxes):
        P = lum / luminous_efficacy  # W
        x0, y0, z0 = pos
        # Vector from LED to floor cell
        dx = X - x0
        dy = Y - y0
        dz = -z0  # floor is at z=0, LED at z=z0 => direction is downward
        dist2 = dx*dx + dy*dy + dz*dz
        dist = np.sqrt(dist2)

        with np.errstate(divide='ignore', invalid='ignore'):
            cos_th = -dz / dist
        cos_th[cos_th<0] = 0
        # E = (P/π)*(cos_th/dist^2)
        with np.errstate(divide='ignore'):
            E = (P/math.pi)* (cos_th/(dist2))
        direct_irr += np.nan_to_num(E)

    # 8. Patch-based surfaces for multi-bounce
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_refl = []

    # Floor patch (z=0) - reflectance
    patch_centers.append((W/2, L/2, 0))
    patch_areas.append(W*L)
    patch_normals.append((0, 0, 1))
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

    # Walls y=0, y=L, x=0, x=W
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

    # Direct on patches
    patch_direct = np.zeros(Np)
    for ip in range(Np):
        pc = patch_centers[ip]
        n = patch_normals[ip]
        # sum from all LED sources, lambertian downward
        accum = 0.0
        for (lp, lum) in zip(light_positions, light_fluxes):
            P = lum/luminous_efficacy
            dx = pc[0]-lp[0]
            dy = pc[1]-lp[1]
            dz = pc[2]-lp[2]
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2<1e-12:
                continue
            dist = math.sqrt(dist2)

            dd = np.array([dx, dy, dz])
            dist = np.linalg.norm(dd)
            cos_th_led = -dz/dist
            if cos_th_led < 0:
                # patch is "above" LED or sideways
                continue
            E_led = (P/math.pi)*(cos_th_led/(dist2))

            # Now cos_in on the patch side:
            cos_in_patch = np.dot(-dd, n)/(dist*np.linalg.norm(n))
            if cos_in_patch<0:
                cos_in_patch=0
            accum+=E_led*cos_in_patch
        patch_direct[ip] = accum

    patch_flux = patch_direct * patch_areas
    patch_rad = np.copy(patch_direct)

    # iterative radiosity
    for b in range(NUM_RADIOSITY_BOUNCES):
        new_flux = np.zeros(Np)
        for j in range(Np):
            if patch_refl[j]<=0:
                continue
            outF = patch_rad[j]*patch_areas[j]*patch_refl[j]
            pc_j = patch_centers[j]
            n_j = patch_normals[j]
            for i2 in range(Np):
                if i2==j:
                    continue
                pc_i = patch_centers[i2]
                n_i = patch_normals[i2]
                dd = pc_i - pc_j
                dist2 = np.dot(dd, dd)
                if dist2<1e-12:
                    continue
                dist = math.sqrt(dist2)
                cos_j = np.dot(n_j, dd)/(dist*np.linalg.norm(n_j))
                cos_i = np.dot(-n_i, dd)/(dist*np.linalg.norm(n_i))
                if cos_j<0 or cos_i<0:
                    continue
                # standard form factor approx:
                ff = (cos_j*cos_i)/(math.pi*dist2)
                new_flux[i2]+=outF*ff

        patch_rad = patch_direct + new_flux/patch_areas

    # final reflection on the floor
    reflect_irr = np.zeros_like(X)
    floor_pts = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=1)
    floor_n = np.array([0,0,1], dtype=float)

    for p in range(Np):
        outF = patch_rad[p]*patch_areas[p]*patch_refl[p]
        if outF<1e-15:
            continue
        pc = patch_centers[p]
        n = patch_normals[p]
        dv = floor_pts - pc
        dist2 = np.einsum('ij,ij->i', dv, dv)
        dist = np.sqrt(dist2)
        cos_p = np.einsum('ij,j->i', dv, n)/(dist*np.linalg.norm(n)+1e-15)
        cos_f = np.einsum('ij,j->i', -dv, floor_n)/(dist+1e-15)
        cos_p[cos_p<0]=0
        cos_f[cos_f<0]=0
        ff = (cos_p*cos_f)/(math.pi*dist2+1e-15)
        cell_flux = outF*ff
        cell_area = FLOOR_GRID_RES*FLOOR_GRID_RES
        cell_irr = cell_flux/cell_area
        reflect_irr.ravel()[:] += np.nan_to_num(cell_irr)

    floor_irr = direct_irr + reflect_irr
    floor_ppfd = floor_irr*conversion_factor

    # Summaries
    avg_ppfd = np.mean(floor_ppfd)
    rmse = np.sqrt(np.mean((floor_ppfd-avg_ppfd)**2))
    du = 100*(1-(rmse/avg_ppfd))
    print(f"Average PPFD = {avg_ppfd:.1f} µmol/m²/s")
    print(f"RMSE         = {rmse:.1f} µmol/m²/s")
    print(f"Degree of Uniformity = {du:.1f} %")

    # Plot
    plt.figure(figsize=(6,5))
    plt.imshow(floor_ppfd, origin='lower', extent=[0,W,0,L], cmap='viridis')
    plt.colorbar(label='PPFD (µmol/m²/s)')
    plt.xlabel('Width (m)')
    plt.ylabel('Length (m)')
    plt.title(f"Lambertian Down + {NUM_RADIOSITY_BOUNCES}-Bounce Radiosity\nAvg={avg_ppfd:.1f}, DOU={du:.1f}%")

    # Add PPFD labels at measurement points
    num_measure = 10  # Example: 10x10 grid of measurement points
    x_measure = np.linspace(0, W, num_measure)
    y_measure = np.linspace(0, L, num_measure)

    for xm in x_measure:
        for ym in y_measure:
            # Find the nearest grid cell index
            ix = int(xm / FLOOR_GRID_RES)
            iy = int(ym / FLOOR_GRID_RES)

            # Ensure indices are within bounds
            ix = min(ix, floor_ppfd.shape[1] - 1)
            iy = min(iy, floor_ppfd.shape[0] - 1)

            ppfd_val = floor_ppfd[iy, ix]  # Note: iy, ix for matrix indexing
            plt.text(xm, ym, f'{ppfd_val:.1f}', color='white', ha='center', va='center',
                     fontsize=6, bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # Add markers for COB LEDs
    num_cob_leds = 61  # Total number of COB LEDs
    for pos in light_positions[:num_cob_leds]:
        plt.plot(pos[0], pos[1], 'rx', markersize=8)  # Red 'x' for COB LEDs

    # Add markers and lines for LED strips
    for strip_num, led_ids in led_strips.items():
        strip_points = []
        for i in range(len(led_ids) - 1):
            start_idx = led_ids[i] - 1
            end_idx = led_ids[i + 1] - 1
            start_pos = light_positions[start_idx]
            end_pos = light_positions[end_idx]
            for t in np.linspace(0, 1, points_per_seg, endpoint=False):
                px = start_pos[0] + t * (end_pos[0] - start_pos[0])
                py = start_pos[1] + t * (end_pos[1] - start_pos[1])
                pz = start_pos[2] + t * (end_pos[2] - start_pos[2])
                strip_points.append((px, py, pz))
        strip_points.append(light_positions[led_ids[-1] - 1])

        # Plot the strip as a dashed line
        strip_x = [p[0] for p in strip_points]
        strip_y = [p[1] for p in strip_points]
        plt.plot(strip_x, strip_y, 'm--', linewidth=1)  # Magenta dashed line

        # Plot markers at each point along the strip
        for pos in strip_points:
            plt.plot(pos[0], pos[1], 'mo', markersize=3)  # Magenta 'o' for strip points
            
    plt.show()


# ------------------------------
# Basic Tkinter UI
# ------------------------------
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE,"r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_settings():
    s = {
        "layer_intensities":[e.get() for e in layer_intensity_entries],
        "corner_intensities":[e.get() for e in corner_intensity_entries],
        "led_strip_fluxes":[e.get() for e in led_strip_entries]
    }
    try:
        with open(SETTINGS_FILE,"w") as f:
            json.dump(s,f)
    except:
        pass

def on_closing():
    save_settings()
    root.destroy()

root = tk.Tk()
root.title("Physically Consistent PPFD Radiosity")
root.protocol("WM_DELETE_WINDOW", on_closing)

layer_frame = ttk.LabelFrame(root, text="Layer Settings")
layer_frame.grid(row=0,column=0,padx=10,pady=10,sticky="ew")

layer_int_frame = ttk.LabelFrame(layer_frame, text="COB Intensities (lm)")
layer_int_frame.grid(row=0,column=0,padx=5,pady=5,sticky="ew")
layer_intensity_entries = []
defaults = [6000,8000,12000,8000,10000,18000]
for i,df in enumerate(defaults):
    lb = ttk.Label(layer_int_frame, text=f"Layer {i+1}:")
    lb.grid(row=i,column=0,padx=2,pady=2,sticky="e")
    e = ttk.Entry(layer_int_frame, width=7)
    e.insert(0,str(df))
    e.grid(row=i,column=1,padx=2,pady=2,sticky="w")
    layer_intensity_entries.append(e)

corner_frame = ttk.LabelFrame(root, text="Corner COB Intensities (lm)")
corner_frame.grid(row=1,column=0,padx=10,pady=10,sticky="ew")
corner_intensity_entries = []
corner_labels = ["Left:", "Right:", "Bottom:", "Top:"]
for i in range(4):
    lb = ttk.Label(corner_frame, text=corner_labels[i])
    lb.grid(row=i,column=0,padx=2,pady=2,sticky="e")
    e = ttk.Entry(corner_frame, width=7)
    e.insert(0,"18000")
    e.grid(row=i,column=1,padx=2,pady=2,sticky="w")
    corner_intensity_entries.append(e)

led_strip_frame = ttk.LabelFrame(root, text="LED Strip Luminous Flux (lm)")
led_strip_frame.grid(row=0,column=1,padx=10,pady=10,sticky="nsew")
led_strip_entries = []
n_strips=20
cols=2
for i in range(n_strips):
    r=i//cols
    c=i%cols
    lb = ttk.Label(led_strip_frame, text=f"Strip #{i+1}:")
    lb.grid(row=r,column=2*c,padx=2,pady=2,sticky="e")
    e = ttk.Entry(led_strip_frame, width=10)
    e.insert(0,"3000")
    e.grid(row=r,column=2*c+1,padx=2,pady=2,sticky="w")
    led_strip_entries.append(e)

loaded = load_settings()
if loaded:
    if "layer_intensities" in loaded:
        for e,v in zip(layer_intensity_entries, loaded["layer_intensities"]):
            e.delete(0,tk.END)
            e.insert(0,v)
    if "corner_intensities" in loaded:
        for e,v in zip(corner_intensity_entries, loaded["corner_intensities"]):
            e.delete(0,tk.END)
            e.insert(0,v)
    if "led_strip_fluxes" in loaded:
        for e,v in zip(led_strip_entries, loaded["led_strip_fluxes"]):
            e.delete(0,tk.END)
            e.insert(0,v)

run_button = ttk.Button(root, text="Run Simulation", command=run_simulation)
run_button.grid(row=2,column=0,columnspan=2,padx=10,pady=10)

root.mainloop()