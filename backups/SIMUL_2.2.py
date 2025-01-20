import math
from math import pi, cos, sin
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d
import tkinter as tk
from tkinter import Tk, Label, Entry, Button, BooleanVar, Checkbutton, StringVar, OptionMenu, filedialog
import json
import tkinter.simpledialog
import csv
from scipy.interpolate import interp1d

SETTINGS_FILE = 'settings.json'

# Initialize flags for GUI elements
show_light_sources = True
show_measurement_points = True

# Global variables for SPD/IES integration and others
ies_data = None
spd_wavelengths = None
spd_power = None
normalized_spd = None
radiance_distribution = None
center_source = None
v_values = None  # Global photopic response values

def simple_ies_parse(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: IES file '{filename}' not found.")
        raise

    tilt_none_index = None
    for i, line in enumerate(lines):
        if line.strip() == "TILT=NONE":
            tilt_none_index = i
            break

    if tilt_none_index is None:
        print("Warning: 'TILT=NONE' not found. Proceeding from start of file.")
        tilt_none_index = 0

    data_start_index = tilt_none_index + 1

    total_lumens = 1000.0
    if data_start_index < len(lines):
        try:
            header_numbers = lines[data_start_index].split()
            total_lumens = float(header_numbers[1])
        except (IndexError, ValueError):
            print("Warning: Header parsing failed. Using default lumens = 1000.")
        data_start_index += 1
    else:
        print("Warning: No header line found after 'TILT=NONE'. Using default lumens = 1000.")

    candela_values = []
    for line in lines[data_start_index:]:
        try:
            candela_values.extend([float(x) for x in line.split()])
        except ValueError:
            continue

    if not candela_values:
        print("Warning: No candela values parsed from the IES file.")

    return {"lumens": total_lumens, "candela_values": candela_values}

def load_ies_data(ies_filename):
    global ies_data
    try:
        ies_data = simple_ies_parse(ies_filename)
    except FileNotFoundError:
        ies_data = None
    return ies_data

def load_spd_data(spd_filename):
    global spd_wavelengths, spd_power
    with open(spd_filename, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    wavelengths = []
    powers = []
    for line in lines:
        parts = line.replace(',', ' ').split()
        if len(parts) >= 2:
            wavelengths.append(float(parts[0]))
            powers.append(float(parts[1]))
    spd_wavelengths = np.array(wavelengths)
    spd_power = np.array(powers)
    return spd_wavelengths, spd_power

def normalize_spd_to_ies():
    global normalized_spd, spd_wavelengths, spd_power, ies_data, v_values
    v_lambda = np.loadtxt('/Users/austinrouse/photonics/backups/luminosity_function.txt')  # [wavelength, V(λ)]
    v_interp = interp1d(v_lambda[:, 0], v_lambda[:, 1], bounds_error=False, fill_value=0)
    v_values = v_interp(spd_wavelengths)

    k_m = 683  # lm/W
    luminous_flux = np.sum(spd_power * v_values * k_m * np.gradient(spd_wavelengths))
    total_lumens = ies_data['lumens']
    scaling_factor = total_lumens / luminous_flux
    normalized_spd = spd_power * scaling_factor

def combine_spd_ies():
    global radiance_distribution, normalized_spd, ies_data
    intensity_distribution = np.array(ies_data['candela_values'])
    radiance_distribution = intensity_distribution[..., None] * normalized_spd

def calculate_lux_at_point(measurement_point, par_mask):
    lux_total = 0.0
    # Sum over angular distribution and spectrum
    for angle_index in range(radiance_distribution.shape[0]):
        radiance = radiance_distribution[angle_index, par_mask]
        # Multiply by photopic response and 683 lm/W conversion factor
        lux_value = np.sum(radiance * v_values[par_mask] * 683)
        lux_total += lux_value
    return lux_total / radiance_distribution.shape[0]

def calculate_distance(light_position, measurement_point):
    light_x, light_y = light_position
    point_x, point_y, point_z = measurement_point
    distance = math.sqrt((light_x - point_x) ** 2 + (light_y - point_y) ** 2 + (point_z ** 2))
    return distance

def load_settings():
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        return {}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

def load_intensities():
    filename = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')])
    if filename:
        with open(filename, 'r') as f:
            intensities = [float(line.strip().split(',')[0]) for line in f]
        for i, entry in enumerate(light_intensity_entries):
            entry.delete(0, 'end')
            if i < len(intensities):
                entry.insert(0, str(intensities[i]))
            else:
                entry.insert(0, '1.0')

def save_intensities():
    filename = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')])
    if filename:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            intensities = [entry.get() for entry in light_intensity_entries]
            for intensity in intensities:
                writer.writerow([intensity])

def reorder_dps(dps, measurement_points, center_point):
    relative_coords = [(x - center_point[0], y - center_point[1]) for x, y, _ in measurement_points]
    dps_coords = list(zip(dps, relative_coords))
    dps_coords.sort(key=lambda x: max(abs(x[1][0]), abs(x[1][1])))
    sorted_dps = [dps for dps, coord in dps_coords]
    return sorted_dps

def prepare_heatmap_data():
    global show_light_sources, show_measurement_points, intensity
    global floor_height, floor_width, light_sources, num_points
    global light_intensities, layer_intensities, min_int, max_int, measurement_points
    global average_intensity, total_intensity, intensity_variance, total_lumens
    global center_source

    measurement_points = []

    floor_width = float(floor_width_entry.get())
    floor_height = float(floor_height_entry.get())
    light_array_width = float(light_array_width_entry.get())
    light_array_height = float(light_array_height_entry.get())
    layer_intensities = [float(entry.get()) for entry in light_intensity_entries]
    min_int = float(min_int_entry.get())
    max_int = float(max_int_entry.get())
    perimeter_reflectivity = perimeter_reflectivity_var.get()

    if perimeter_reflectivity > 1.0:
        perimeter_reflectivity /= 100.0
    elif perimeter_reflectivity < 0.0:
        perimeter_reflectivity = 0.0

    settings = {
        'floor_width': floor_width,
        'floor_height': floor_height,
        'light_array_width': light_array_width,
        'light_array_height': light_array_height,
        'layer_intensities': layer_intensities,
        'min_int': min_int,
        'max_int': max_int,
        'perimeter_reflectivity': perimeter_reflectivity,
    }
    save_settings(settings)

    num_points = 225
    light_array_width_ft = light_array_width / 12
    light_array_height_ft = light_array_height / 12

    center_x = floor_width / 2
    center_y = floor_height / 2
    center_source = (center_x, center_y)

    pattern_option = selected_pattern.get()
    if pattern_option == "5x5 Grid":
        spacing = light_array_width_ft / 7
        square_sources = []
        for i in np.arange(-2, 3, 1):
            for j in np.arange(-2, 3, 1):
                x_offset = spacing * i
                y_offset = spacing * j
                light_pos = (center_x + x_offset, center_y + y_offset)
                if light_pos != center_source:
                    square_sources.append(light_pos)
        light_sources = square_sources

    if pattern_option == "8x8 Grid":
        spacing = light_array_width_ft / 7
        square_sources = []
        for i in np.arange(-3.5, 4, 1):
            for j in np.arange(-3.5, 4, 1):
                x_offset = spacing * i
                y_offset = spacing * j
                light_pos = (center_x + x_offset, center_y + y_offset)
                if light_pos != center_source:
                    square_sources.append(light_pos)
        light_sources = square_sources

    if pattern_option == "Diamond: 13":
        layers = [
            [(0, 0)],
            [(-1, 0), (1, 0), (0, -1), (0, 1)],
            [(-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
        ]
    elif pattern_option == "Diamond: 25":
        layers = [
            [(0, 0)],
            [(-1, 0), (1, 0), (0, -1), (0, 1)],
            [(-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)],
            [(-2, -1), (2, -1), (-2, 1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2),
             (-3, 0), (3, 0), (0, -3), (0, 3)]
        ]
    elif pattern_option == "Diamond: 41":
        layers = [
            [(0, 0)],
            [(-1, 0), (1, 0), (0, -1), (0, 1)],
            [(-1, -1), (1, -1), (-1, 1), (1, 1),
             (-2, 0), (2, 0), (0, -2), (0, 2)],
            [(-2, -1), (2, -1), (-2, 1), (2, 1),
             (-1, -2), (1, -2), (-1, 2), (1, 2),
             (-3, 0), (3, 0), (0, -3), (0, 3)],
            [(-2, -2), (2, -2), (-2, 2), (2, 2),
             (-3, -1), (3, -1), (-3, 1), (3, 1),
             (-1, -3), (1, -3), (-1, 3), (1, 3),
             (-4, 0), (4, 0), (0, -4), (0, 4)]
        ]
    elif pattern_option == "Diamond: 61":
        layers = [
            [(0, 0)],
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
    elif pattern_option == "Diamond: 221":
        layers = [
            [(0, 0)],
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
             (-5, 0), (5, 0), (0, -5), (0, 5)],
            [(-3, -3), (3, -3), (-3, 3), (3, 3),
             (-4, -2), (4, -2), (-4, 2), (4, 2),
             (-2, -4), (2, -4), (-2, 4), (2, 4),
             (-5, -1), (5, -1), (-5, 1), (5, 1),
             (-1, -5), (1, -5), (-1, 5), (1, 5),
             (-6, 0), (6, 0), (0, -6), (0, 6)],
            [(-4, -3), (4, -3), (-4, 3), (4, 3),
             (-3, -4), (3, -4), (-3, 4), (3, 4),
             (-5, -2), (5, -2), (-5, 2), (5, 2),
             (-2, -5), (2, -5), (-2, 5), (2, 5),
             (-6, -1), (6, -1), (-6, 1), (6, 1),
             (-1, -6), (1, -6), (-1, 6), (1, 6),
             (-7, 0), (7, 0), (0, -7), (0, 7)],
            [(-4, -4), (4, -4), (-4, 4), (4, 4),
             (-5, -3), (5, -3), (-5, 3), (5, 3),
             (-3, -5), (3, -5), (-3, 5), (3, 5),
             (-6, -2), (6, -2), (-6, 2), (6, 2),
             (-2, -6), (2, -6), (-2, 6), (2, 6),
             (-7, -1), (7, -1), (-7, 1), (7, 1),
             (-1, -7), (1, -7), (-1, 7), (1, 7),
             (-8, 0), (8, 0), (0, -8), (0, 8)],
            [(-5, -4), (5, -4), (-5, 4), (5, 4),
             (-4, -5), (4, -5), (-4, 5), (4, 5),
             (-6, -3), (6, -3), (-6, 3), (6, 3),
             (-3, -6), (3, -6), (-3, 6), (3, 6),
             (-7, -2), (7, -2), (-7, 2), (7, 2),
             (-2, -7), (2, -7), (-2, 7), (2, 7),
             (-8, -1), (8, -1), (-8, 1), (8, 1),
             (-1, -8), (1, -8), (-1, 8), (1, 8),
             (-9, 0), (9, 0), (0, -9), (0, 9)],
            [(-5, -5), (5, -5), (-5, 5), (5, 5),  # Added (-5, 5)
             (-6, -4), (6, -4), (-6, 4), (6, 4),
             (-4, -6), (4, -6), (-4, 6), (4, 6),
             (-7, -3), (7, -3), (-7, 3), (7, 3),
             (-3, -7), (3, -7), (-3, 7), (3, 7),
             (-8, -2), (8, -2), (-8, 2), (8, 2),
             (-2, -8), (2, -8), (-2, 8), (2, 8),
             (-9, -1), (9, -1), (-9, 1), (9, 1),
             (-1, -9), (1, -9), (-1, 9), (1, 9),
             (-10, 0), (10, 0), (0, -10), (0, 10)]
        ]

    if pattern_option in ["Diamond: 13", "Diamond: 25", "Diamond: 41", "Diamond: 61", "Diamond: 221"]:
        spacing_x = light_array_width_ft / 7.2
        spacing_y = light_array_height_ft / 7.2
        light_sources = [center_source]  # Initialize with the center light
        light_intensities = [layer_intensities[0]] if len(layer_intensities) > 0 else [1.0]  # Center light intensity

        for layer_index, layer in enumerate(layers):
            if layer_index == 0:
                continue

            layer_intensity = layer_intensities[layer_index] if layer_index < len(layer_intensities) else 1.0

            for dot in layer:
                x_offset = spacing_x * dot[0]
                y_offset = spacing_y * dot[1]
                theta = math.radians(45)
                rotated_x_offset = x_offset * math.cos(theta) - y_offset * math.sin(theta)
                rotated_y_offset = x_offset * math.sin(theta) + y_offset * math.cos(theta)
                light_pos = (center_x + rotated_x_offset, center_y + rotated_y_offset)
                light_sources.append(light_pos)
                light_intensities.append(layer_intensity)

    def get_measurement_points():
        num_measurement_points = 225
        measurement_points_per_dimension = int(math.sqrt(num_measurement_points))
        step_size_x = floor_width / (measurement_points_per_dimension + 1)
        step_size_y = floor_height / (measurement_points_per_dimension + 1)
        points = []
        for i in range(1, measurement_points_per_dimension + 1):
            for j in range(1, measurement_points_per_dimension + 1):
                points.append((i * step_size_x, j * step_size_y, 0.0))
        return points

    measurement_points = get_measurement_points()

    load_ies_data('/Users/austinrouse/photonics/backups/ray_file.ies')
    load_spd_data('/Users/austinrouse/photonics/backups/spd_data.csv')
    normalize_spd_to_ies()
    combine_spd_ies()

    par_mask = (spd_wavelengths >= 400) & (spd_wavelengths <= 700)

    intensity = np.zeros(num_points)
    for idx, measurement_point in enumerate(measurement_points):
        intensity[idx] = calculate_lux_at_point(measurement_point, par_mask)

    total_lumens = sum(light_intensities)
    average_intensity = np.mean(intensity)
    total_intensity = np.sum(intensity)
    intensity_variance = np.mean(np.abs(intensity - average_intensity))

    dps = np.abs(intensity - average_intensity)
    center_point = (floor_width / 2, floor_height / 2)
    sorted_dps = reorder_dps(dps, measurement_points, center_point)

    return sorted_dps, center_source

def on_generate_heatmap_click():
    sorted_dps, center_source = prepare_heatmap_data()
    generate_heatmap(center_source)

def plot_dps(sorted_intensity):
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_intensity, color='dodgerblue', marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.title('Deviation from Average Illuminance Across Measurement Points')
    plt.xlabel('Measurement Point Index')
    plt.ylabel('Deviation (Lux)')
    plt.grid(True)
    plt.ylim([0, max_int])
    plt.show()

def generate_line_graph():
    sorted_intensity, _ = prepare_heatmap_data()
    plot_dps(sorted_intensity)

def find_nearest_light_source(x, y, light_sources):
    distances = [np.sqrt((x - lx)**2 + (y - ly)**2) for lx, ly in light_sources]
    return np.argmin(distances)

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    ix, iy = event.xdata, event.ydata
    idx = find_nearest_light_source(ix, iy, light_sources)
    print(f'Clicked on source {idx}, intensity: {light_intensities[idx]}')
    generate_heatmap(center_source)

def generate_heatmap(center_source):
    sorted_dps, center_source = prepare_heatmap_data()
    global floor_width, floor_height, intensity, measurement_points, num_points, min_int, max_int, intensity_variance, total_lumens, average_intensity

    num_points_per_dim = int(math.sqrt(num_points))
    intensity_matrix = intensity.reshape((num_points_per_dim, num_points_per_dim))
    intensity_matrix = np.flip(intensity_matrix, axis=0)

    plt.figure(figsize=(8, 8))
    plt.xlim(0, floor_width)
    plt.ylim(0, floor_height)

    if show_light_sources.get():
        plt.plot(center_source[0], center_source[1], 'mo', markersize=6)
        for idx, light_source in enumerate(light_sources):
            if light_source == center_source:
                continue
            plt.plot(light_source[0], light_source[1], 'mo', markersize=6)
            plt.text(light_source[0], light_source[1], str(idx + 1), color='red', fontsize=12, ha='right')

    selected_option = selected_cmap.get()
    cmap_dict = {
        "Cool": "cool",
        "Jet": "jet",
        "Viridis": "viridis",
        "Plasma": "plasma",
        "Inferno": "inferno",
        "Turbo": "turbo",
        "Autumn": "autumn",
        "Spring": "spring",
        "Gray": "gray",
    }
    cmap_value = cmap_dict.get(selected_option, "jet")

    plt.imshow(intensity_matrix, cmap=cmap_value, interpolation='bicubic', vmin=min_int, vmax=max_int, extent=[0, floor_width, 0, floor_height])
    plt.colorbar(label='Illuminance (Lux)')
    plt.title('Light Intensity Heat Map (Lux)')

    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    for row in range(intensity_matrix.shape[0]):
        for col in range(intensity_matrix.shape[1]):
            point_intensity = intensity_matrix[row, col]
            rounded_intensity = int(round(point_intensity))
            text_x = col * (floor_width / intensity_matrix.shape[1]) + (floor_width / (2 * intensity_matrix.shape[1]))
            text_y = (intensity_matrix.shape[0] - row - 1) * (floor_height / intensity_matrix.shape[0]) + (floor_height / (2 * intensity_matrix.shape[0]))
            plt.text(text_x, text_y, str(rounded_intensity), ha='center', va='center', color='black', fontsize=8, fontweight='bold')

    if show_measurement_points.get():
        measurement_xs, measurement_ys, _ = zip(*measurement_points)
        plt.plot(measurement_xs, measurement_ys, 'r+', markersize=7, alpha=1.0)

    text = f'Total Lumens: {total_lumens}\nAverage Lux: {average_intensity:.2f}\nMean Absolute Deviation: {intensity_variance:.2f}'
    text_lines = text.split('\n')
    ax = plt.gca()
    for i, line in enumerate(text_lines):
        if ':' in line:
            title, value = line.split(':', 1)
            ax.text(0.01, -0.12 - i * 0.05, f"{title}:", transform=ax.transAxes, fontsize=10, fontweight='bold', color='blue')
            ax.text(0.5, -0.12 - i * 0.05, value.strip(), transform=ax.transAxes, fontsize=10)

    plt.show()

def generate_surface_graph(flatten_factor=1.0, scale_factor=1.0):
    sorted_dps, center_source = prepare_heatmap_data()
    global floor_width, floor_height, intensity, num_points

    min_int_val = float(min_int_entry.get())
    max_int_val = float(max_int_entry.get())

    x = []
    y = []
    for i in range(15):
        for j in range(15):
            x.append(-7.5 + j * 15. / 14)
            y.append(7.5 - i * 15. / 14)
    x = np.array(x) * scale_factor
    y = np.array(y) * scale_factor

    z = np.array(intensity)
    assert (len(z) == 225), f"Wrong length for input z. Should be 225, but it is {len(z)}!"

    x_pad = np.array(
        [-18., -9., 0., 9., 18., -18., -18., -18., -18., -9., 0., 9., 18., 18., 18., 18., 
         -15., -7.5, 0., 7.5, 15., -15., -15., -15., -15., -7.5, 0., 7.5, 15., 15., 
         15., 15.]) * scale_factor
    y_pad = np.array(
        [-18., -18., -18., -18., -18., -9., 0., 9., 18., 18., 18., 18., 18., 
         -9., 0., 9., -15., -15., -15., -15., -15., -7.5, 0., 7.5, 15., 15., 
         15., 15., 15., -7.5, 0., 7.5]) * scale_factor
    z_pad = np.zeros(x_pad.shape)

    x_ = np.concatenate((x, x_pad))
    y_ = np.concatenate((y, y_pad))
    z_ = np.concatenate((z, z_pad))

    if flatten_factor != 1.0:
        z_ = np.power(z_, flatten_factor)

    triang = mtri.Triangulation(x_, y_)
    x_grid, y_grid = np.mgrid[x_.min():x_.max():.02, y_.min():y_.max():.02]
    interp_lin = mtri.LinearTriInterpolator(triang, z_)
    zi_lin = interp_lin(x_grid, y_grid)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([x_grid.min(), x_grid.max()])
    ax.set_ylim([y_grid.min(), y_grid.max()])
    ax.set_zlim([z_.min(), z_.max()])

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    selected_option = selected_cmap.get()
    cmap_dict = {
        "Cool": "cool",
        "Jet": "jet",
        "Viridis": "viridis",
        "Plasma": "plasma",
        "Inferno": "inferno",
        "Turbo": "turbo",
        "Autumn": "autumn",
        "Spring": "spring",
        "Gray": "gray",
    }
    cmap_value = cmap_dict.get(selected_option, "jet")

    pc = ax.plot_surface(x_grid, y_grid, zi_lin, cmap=cmap_value, vmin=z_.min(), vmax=z_.max())
    fig.colorbar(pc, fraction=0.032, pad=0.04)
    ax.plot_wireframe(x_grid, y_grid, zi_lin, rstride=50, cstride=50, color='k', linewidth=0.4)

    plt.title('Light Intensity Surface Graph (Lux)')
    plt.show()

# GUI Setup
root = Tk()
root.title("Illuminance Simulation Software")

settings = load_settings()

floor_width_label = Label(root, text="Floor Width (feet):")
floor_width_label.grid(row=0, column=0)
floor_width_entry = Entry(root)
floor_width_entry.grid(row=0, column=1)
floor_width_entry.insert(0, settings.get('floor_width', '10'))

floor_height_label = Label(root, text="Floor Height (feet):")
floor_height_label.grid(row=1, column=0)
floor_height_entry = Entry(root)
floor_height_entry.grid(row=1, column=1)
floor_height_entry.insert(0, settings.get('floor_height', '10'))

min_int_label = Label(root, text="Min. Intensity (Lux):")
min_int_label.grid(row=1, column=2)
min_int_entry = Entry(root)
min_int_entry.grid(row=1, column=3)
min_int_entry.insert(0, settings.get('min_int', '10'))

max_int_label = Label(root, text="Max. Intensity (Lux):")
max_int_label.grid(row=2, column=2)
max_int_entry = Entry(root)
max_int_entry.grid(row=2, column=3)
max_int_entry.insert(0, settings.get('max_int', '10'))

light_array_width_label = Label(root, text="Lighting Array Width (inches):")
light_array_width_label.grid(row=2, column=0)
light_array_width_entry = Entry(root)
light_array_width_entry.grid(row=2, column=1)
light_array_width_entry.insert(0, settings.get('light_array_width', '60'))

light_array_height_label = Label(root, text="Lighting Array Height (inches):")
light_array_height_label.grid(row=3, column=0)
light_array_height_entry = Entry(root)
light_array_height_entry.grid(row=3, column=1)
light_array_height_entry.insert(0, settings.get('light_array_height', '60'))

perimeter_reflectivity_var = tk.DoubleVar(value=settings.get('perimeter_reflectivity', 0.3))
perimeter_reflectivity_label = Label(root, text="Perimeter Reflectivity (0-1):")
perimeter_reflectivity_label.grid(row=3, column=2)
perimeter_reflectivity_entry = Entry(root, textvariable=perimeter_reflectivity_var)
perimeter_reflectivity_entry.grid(row=3, column=3)

light_intensity_labels = []
light_intensity_entries = []
num_layers = 11
num_columns = 2

for i in range(num_layers):
    label_text = f"Layer {i} Intensity (lm):" if i > 0 else "Center Intensity (lm):"
    label = Label(root, text=label_text)
    row = i // num_columns + 4
    column = i % num_columns * 2
    label.grid(row=row, column=column)
    entry = Entry(root)
    entry.grid(row=row, column=column + 1)
    if 'layer_intensities' in settings and i < len(settings['layer_intensities']):
        entry.insert(0, str(settings['layer_intensities'][i]))
    else:
        entry.insert(0, '1000')
    light_intensity_labels.append(label)
    light_intensity_entries.append(entry)

show_light_sources = BooleanVar()
show_light_sources.set(True)
show_measurement_points = BooleanVar()
show_measurement_points.set(True)

pattern_options = ["Diamond: 13", "Diamond: 25", "Diamond: 41", "Diamond: 61", "Diamond: 221", "5x5 Grid", "8x8 Grid"]
selected_pattern = StringVar(root)
selected_pattern.set(pattern_options[0])

cmap_options = ["Cool", "Jet", "Inferno", "Viridis", "Plasma", "Turbo", "Autumn", "Spring", "Gray"]
selected_cmap = StringVar(root)
selected_cmap.set(cmap_options[1])

pattern_label = Label(root, text="Select Pattern:")
pattern_label.grid(row=1, column=5)
pattern_dropdown = OptionMenu(root, selected_pattern, *pattern_options)
pattern_dropdown.grid(row=2, column=5)

cmap_label = Label(root, text="Select Color Map:")
cmap_label.grid(row=1, column=4)
cmap_dropdown = OptionMenu(root, selected_cmap, *cmap_options)
cmap_dropdown.grid(row=2, column=4)

light_sources_checkbutton = Checkbutton(root, text="Show Light Sources", variable=show_light_sources)
light_sources_checkbutton.grid(row=25, column=0, columnspan=1)

measurement_points_checkbutton = Checkbutton(root, text="Show Measurement Points", variable=show_measurement_points)
measurement_points_checkbutton.grid(row=26, column=0, columnspan=1)

generate_button = Button(root, text="Generate Heatmap (Lux)", command=on_generate_heatmap_click)
generate_button.grid(row=27, column=0, columnspan=1)

surface_button = Button(root, text="Generate 3D Surface Plot (Lux)", command=generate_surface_graph)
surface_button.grid(row=27, column=1, columnspan=1)

line_graph_button = Button(root, text="Generate 2D Line Graph (Deviation)", command=generate_line_graph)
line_graph_button.grid(row=27, column=2, columnspan=1)

load_button = Button(root, text="Load Intensities", command=load_intensities)
load_button.grid(row=27, column=3, columnspan=1)

save_button = Button(root, text="Save Intensities", command=save_intensities)
save_button.grid(row=27, column=4, columnspan=1)

root.mainloop()
