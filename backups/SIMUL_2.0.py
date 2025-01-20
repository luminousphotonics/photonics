import math
from math import pi, cos, sin
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d
import tkinter as tk
from tkinter import Tk, ttk, Label, Entry, Button, BooleanVar, Checkbutton, StringVar, OptionMenu
import json
import tkinter.simpledialog
from tkinter import filedialog
import csv
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

# Initialize light_sources as an empty list
light_sources = []

SETTINGS_FILE = 'settings.json'
#LUMENS_TO_PPFD_CONVERSION = 65  # Updated Conversion Factor

luminous_efficacy = 182  # lm/W (editable if needed for different light sources)

show_light_sources = True
show_measurement_points = True

#plt.rcParams['figure.dpi'] = 125  # or any other value

def load_spd_data(filename="/Users/austinrouse/photonics/backups/spd_data.csv"):  # Using absolute path
    """Loads SPD data from a CSV file.

    Args:
        filename (str): The path to the CSV file containing SPD data.

    Returns:
        tuple: A tuple containing two NumPy arrays: wavelengths (nm) and
               corresponding relative intensities.
    """

    wavelengths = []
    intensities = []

    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)  # Use csv.reader with space delimiter
            next(reader)  # Skip the header row "wavelength,intensity"
            for row in reader:
                wavelengths.append(float(row[0]))
                intensities.append(float(row[1]))
    except FileNotFoundError:
        print(f"Error: SPD data file not found at {filename}")
        return None, None

    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)

    # Sort data by wavelength
    sort_indices = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_indices]
    intensities = intensities[sort_indices]

    return wavelengths, intensities

# Load SPD data
spd_wavelengths, spd_intensities = load_spd_data()

# Check if SPD data was loaded successfully
if spd_wavelengths is None or spd_intensities is None:
    # Handle the error appropriately, e.g., display an error message and exit
    print("Error loading SPD data. Exiting.")
    exit()

def interpolate_spd(wavelengths, intensities, new_wavelengths):
    """Interpolates SPD data to a new set of wavelengths.

    Args:
        wavelengths (np.ndarray): Original wavelengths from SPD data.
        intensities (np.ndarray): Original relative intensities.
        new_wavelengths (np.ndarray): New wavelengths for interpolation.

    Returns:
        np.ndarray: Interpolated relative intensities at new_wavelengths.
    """

    f_interp = interp1d(wavelengths, intensities, kind='linear', bounds_error=False, fill_value=(0, 0))
    interpolated_intensities = f_interp(new_wavelengths)
    return interpolated_intensities

# Generate target wavelengths for interpolation (e.g., 400 to 795.7 nm inclusive)
target_wavelengths = np.linspace(400, 717.94, 396)

# Interpolate SPD data to target wavelengths
interpolated_intensities = interpolate_spd(spd_wavelengths, spd_intensities, target_wavelengths)

def calculate_photon_flux(wavelengths, intensities):
    """Calculates the photon flux (Q) for the PAR range (400-795.7 nm).

    Args:
        wavelengths (np.ndarray): Wavelengths in nanometers.
        intensities (np.ndarray): Relative intensities (proportional to P(λ)).

    Returns:
        float: Photon flux (Q) in photons/s.
    """

    h = 6.626e-34  # Planck's constant (J.s)
    c = 3.0e8      # Speed of light (m/s)

    wavelengths_m = wavelengths * 1e-9  # Convert nm to meters

    # Calculate energy per photon for each wavelength
    photon_energies = (h * c) / wavelengths_m

    # Calculate integrand: (P(λ) * λ) / (h * c)
    integrand = intensities * wavelengths_m / (h * c)

    # Approximate the integral using the trapezoidal rule
    photon_flux = trapezoid(integrand, x=wavelengths_m)

    return photon_flux

def lumens_to_radiometric_power(lumens, luminous_efficacy):
    """Converts lumens to radiometric power (Watts).

    Args:
        lumens (float): Lumens.
        luminous_efficacy (float): Luminous efficacy (lm/W).

    Returns:
        float: Radiometric power (Watts).
    """

    radiometric_power = lumens / luminous_efficacy
    return radiometric_power

AVOGADRO = 6.022e23  # photons per mole

def calculate_ppfd(photon_flux, area):
    """
    Calculates PPFD from photon flux and area.

    Args:
        photon_flux (float): Photon flux (photons/s).
        area (float): Area (m^2).

    Returns:
        float: PPFD (µmol/m^2/s).
    """
    # Convert photons/s to µmol/s by dividing by Avogadro's number in micromoles:
    # 1 µmol = 6.022e17 photons
    photon_flux_micromol = photon_flux / (AVOGADRO / 1e6)
    ppfd = photon_flux_micromol / area
    return ppfd


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
    # Open a file dialog for the user to select a CSV file
    filename = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')])

    # If the user didn't cancel the dialog
    if filename:
        with open(filename, 'r') as f:
            # Split each line into fields and take the first field
            intensities = [float(line.strip().split(',')[0]) for line in f]

        # Insert the intensities into the entry fields
        for i, entry in enumerate(light_intensity_entries):
            entry.delete(0, 'end')
            if i < len(intensities):
                entry.insert(0, str(intensities[i]))
            else:
                entry.insert(0, '1.0')

def save_intensities():
    # Open a file dialog for the user to select a location and filename to save the CSV file
    filename = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')])

    # If the user didn't cancel the dialog
    if filename:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Get the intensities from the entry fields
            intensities = [entry.get() for entry in light_intensity_entries]
            # Write the intensities to the CSV file
            for intensity in intensities:
                writer.writerow([intensity])

def reorder_dps(dps, measurement_points, center_point):
    # Calculate coordinates relative to the central point
    relative_coords = [(x - center_point[0], y - center_point[1]) for x, y, _ in measurement_points]

    # Create a list of tuples, where each tuple is (dps, relative coordinate)
    dps_coords = list(zip(dps, relative_coords))

    # Sort by the maximum absolute coordinate
    dps_coords.sort(key=lambda x: max(abs(x[1][0]), abs(x[1][1])))

    # Extract and return the sorted dps values
    sorted_dps = [dps for dps, coord in dps_coords]
    return sorted_dps

# Add the Lambertian emission function from Script 2 (outside prepare_heatmap_data)
def lambertian_emission(intensity, distance, z):
    if distance == 0:
        return intensity
    return (intensity * z) / ((distance ** 2 + z ** 2) ** 1.5)

def calculate_wall_reflection(measurement_point, floor_width, floor_height, perimeter_reflectivity, direct_ppfd, decay_constant=1.0):
    x_m, y_m, z_m = measurement_point

    reflected_ppfd_exp = 0.0
    reflected_ppfd_inv = 0.0

    # Calculate distances to each wall
    distances = {
        'left': x_m,
        'right': floor_width - x_m,
        'bottom': y_m,
        'top': floor_height - y_m
    }

    # Calculate reflected light from each wall using both models
    for wall, distance in distances.items():
        # Exponential Decay Model
        reflected_light_exp = direct_ppfd * perimeter_reflectivity * math.exp(-distance / decay_constant)
        reflected_ppfd_exp += reflected_light_exp

        # Inverse Square Law Model
        reflected_light_inv = direct_ppfd * perimeter_reflectivity / (distance + 1)**2
        reflected_ppfd_inv += reflected_light_inv

        # Debug (Optional): Print reflected light from each wall
        # print(f"Point ({x_m:.2f}, {y_m:.2f}), Wall: {wall}, Exp. Refl: {reflected_light_exp:.4f}, Inv. Refl: {reflected_light_inv:.4f}")

    return reflected_ppfd_exp, reflected_ppfd_inv

def calculate_reflected_light(measurement_points, floor_width, floor_height, perimeter_reflectivity, intensity, num_iterations=3, decay_constant=1.0):
    reflected_intensity_exp = np.zeros(len(measurement_points))
    reflected_intensity_inv = np.zeros(len(measurement_points))

    for iteration in range(num_iterations):
        print(f"Reflection Iteration {iteration + 1}")
        temp_intensity_exp = np.zeros(len(measurement_points))
        temp_intensity_inv = np.zeros(len(measurement_points))

        for i, point in enumerate(measurement_points):
            direct_or_updated_ppfd = intensity[i] if iteration == 0 else intensity[i] + reflected_intensity_exp[i] # Use exponential reflection for subsequent iterations
            reflection_at_point_exp, reflection_at_point_inv = calculate_wall_reflection(
                point, floor_width, floor_height, perimeter_reflectivity, direct_or_updated_ppfd, decay_constant
            )
            temp_intensity_exp[i] += reflection_at_point_exp
            temp_intensity_inv[i] += reflection_at_point_inv
        
        reflected_intensity_exp += temp_intensity_exp
        reflected_intensity_inv += temp_intensity_inv

        print(f"Total Reflected Intensity (Exp) after iteration {iteration + 1}: {np.sum(temp_intensity_exp):.4f}")
        print(f"Total Reflected Intensity (Inv) after iteration {iteration + 1}: {np.sum(temp_intensity_inv):.4f}")

    print(f"Total Reflected Intensity (Exp): {np.sum(reflected_intensity_exp):.4f}")
    print(f"Total Reflected Intensity (Inv): {np.sum(reflected_intensity_inv):.4f}")
    return reflected_intensity_exp, reflected_intensity_inv

# Moved this function up so it can be called in on_generate_heatmap_click()
def update_light_intensity_entries():
    global light_intensity_entries, light_intensity_labels, light_sources

    pattern_option = selected_pattern.get()

    try:
        floor_width = float(floor_width_entry.get())
        floor_height = float(floor_height_entry.get())
        light_array_width_ft = float(light_array_width_entry.get()) / 12
        light_array_height_ft = float(light_array_height_entry.get()) / 12
    except ValueError:
        print("Error: Invalid floor or lighting array dimensions.")
        return

    center_x = floor_width / 2
    center_y = floor_height / 2
    center_source = (center_x, center_y)  # Define center_source

    light_sources = generate_light_sources(pattern_option, center_x, center_y, light_array_width_ft, light_array_height_ft)

    # Remove existing entry widgets
    for entry in light_intensity_entries:
        entry.destroy()
    for label in light_intensity_labels:
        label.destroy()
    light_intensity_entries = []
    light_intensity_labels = []

    # Create new entry widgets based on the number of light sources
    num_light_sources = len(light_sources)
    num_columns = 2  # Number of columns for layout

    for i in range(num_light_sources):
        label_text = f"Layer {i} Lumens:" if i > 0 else "Center Lumens:"
        label = Label(root, text=label_text)
        light_intensity_labels.append(label)

        # Calculate the row and column index for grid layout
        row = i // num_columns + 4
        column = i % num_columns * 2

        label.grid(row=row, column=column, padx=5, pady=2)
        entry = Entry(root)
        entry.insert(0, '1.0')  # Default value
        entry.grid(row=row, column=column + 1, padx=5, pady=2)
        light_intensity_entries.append(entry)
        
def generate_light_sources(pattern_option, center_x, center_y, light_array_width_ft, light_array_height_ft):
    """
    Generates light source positions based on the selected pattern.

    Args:
        pattern_option (str): The selected lighting pattern.
        center_x (float): X-coordinate of the center point.
        center_y (float): Y-coordinate of the center point.
        light_array_width_ft (float): Width of the lighting array in feet.
        light_array_height_ft (float): Height of the lighting array in feet.

    Returns:
        list: A list of tuples representing light source positions (x, y).
    """
    generated_light_sources = []

    if pattern_option == "5x5 Grid":
        spacing = light_array_width_ft / 7
        for i in range(-2, 3):
            for j in range(-2, 3):
                x_offset = spacing * i
                y_offset = spacing * j
                light_pos = (center_x + x_offset, center_y + y_offset)
                if light_pos != (center_x, center_y):
                    generated_light_sources.append(light_pos)

    elif pattern_option == "8x8 Grid":
        spacing = light_array_width_ft / 7
        for i in range(-3, 4):
            for j in range(-3, 4):
                x_offset = spacing * i
                y_offset = spacing * j
                light_pos = (center_x + x_offset, center_y + y_offset)
                if light_pos != (center_x, center_y):
                    generated_light_sources.append(light_pos)

    elif pattern_option.startswith("Diamond"):
        # Define layers based on the pattern
        layers = []
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
        light_intensities = [layer_intensities[0]] if len(layer_intensities) > 0 else [1.0] # Center light intensity

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

    # Add more patterns as necessary...

    return generated_light_sources

def prepare_heatmap_data():
    global show_light_sources, show_measurement_points, intensity
    global floor_height, floor_width, light_sources, num_points
    global light_intensities, layer_intensities, min_int, max_int, measurement_points
    global average_intensity, total_intensity, ppfd, intensity_variance
    global total_lumens, center_source

    measurement_points = []

    # Get the user-defined settings from the input fields
    try:
        floor_width = float(floor_width_entry.get())
        floor_height = float(floor_height_entry.get())
        light_array_width = float(light_array_width_entry.get())
        light_array_height = float(light_array_height_entry.get())
        min_int = float(min_int_entry.get())
        max_int = float(max_int_entry.get())
        perimeter_reflectivity = perimeter_reflectivity_var.get()
    except ValueError:
        print("Error: Invalid input in one of the fields.")
        return [], ()

    # Load and scale SPD data
    spd_wavelengths, spd_intensities = load_spd_data()
    if spd_wavelengths is None or spd_intensities is None:
        print("Error loading SPD data. Exiting.")
        exit()

    # Compute global scaling for SPD data
    total_relative = trapezoid(spd_intensities, x=spd_wavelengths)
    baseline_total_power = 1.0  # W
    spd_scaling_factor = baseline_total_power / total_relative
    absolute_intensities = spd_intensities * spd_scaling_factor

    target_wavelengths = np.linspace(400, 795.7, 396)
    interpolated_intensities = interpolate_spd(spd_wavelengths, absolute_intensities, target_wavelengths)

    baseline_photon_flux = calculate_photon_flux(target_wavelengths, interpolated_intensities)
    print(f"Baseline photon flux (for {baseline_total_power} W): {baseline_photon_flux}")

    # Calculate the unscaled photon flux with scaled SPD data
    #unscaled_photon_flux = calculate_photon_flux(target_wavelengths, interpolated_intensities)
    #print(f"Unscaled photon flux (for {assumed_total_power} W): {unscaled_photon_flux}")


    # Normalize perimeter_reflectivity to [0, 1]
    if perimeter_reflectivity > 1.0:
        print(f"Perimeter Reflectivity input {perimeter_reflectivity} > 1. Normalizing to {perimeter_reflectivity / 100.0}")
        perimeter_reflectivity /= 100.0
    elif perimeter_reflectivity < 0.0:
        print(f"Perimeter Reflectivity input {perimeter_reflectivity} < 0. Setting to 0.")
        perimeter_reflectivity = 0.0

    # Store the settings for next time
    settings = {
        'floor_width': floor_width,
        'floor_height': floor_height,
        'light_array_width': light_array_width,
        'light_array_height': light_array_height,
        'min_int': min_int,
        'max_int': max_int,
        'perimeter_reflectivity': perimeter_reflectivity,
    }
    save_settings(settings)

    # Constants
    num_points = 225

    # Convert lighting array dimensions to feet
    light_array_width_ft = light_array_width / 12
    light_array_height_ft = light_array_height / 12

    # Calculate the position of the center light source
    center_x = floor_width / 2
    center_y = floor_height / 2

    center_source = (center_x, center_y)

    # Dropdown menu for pattern arrangement
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

        for layer_index, layer in enumerate(layers):
            if layer_index == 0:
                continue  # Skip the center for diamond patterns

            for dot in layer:
                x_offset = spacing_x * dot[0]
                y_offset = spacing_y * dot[1]
                theta = math.radians(45)
                rotated_x_offset = x_offset * math.cos(theta) - y_offset * math.sin(theta)
                rotated_y_offset = x_offset * math.sin(theta) + y_offset * math.cos(theta)
                light_pos = (center_x + rotated_x_offset, center_y + rotated_y_offset)
                light_sources.append(light_pos)

    # -------------------------------------------------------------------------
    # Generate measurement points at z=0 (on the floor).
    # We'll place lights at z=2.5 ft
    # -------------------------------------------------------------------------
    def get_measurement_points(floor_width, floor_height):
        num_measurement_points = 225
        measurement_points_per_dimension = int(math.sqrt(num_measurement_points))
        step_size_x = floor_width / (measurement_points_per_dimension + 1)
        step_size_y = floor_height / (measurement_points_per_dimension + 1)

        points = []
        for x in range(1, measurement_points_per_dimension + 1):
            for y in range(1, measurement_points_per_dimension + 1):
                point_x = x * step_size_x
                point_y = y * step_size_y
                # Measurement on the floor -> z=0
                point_z = 0.0
                points.append((point_x, point_y, point_z))

        return points


    measurement_points = get_measurement_points(floor_width, floor_height)
    intensity = np.zeros(num_points)

    # Convert floor dimensions to meters
    floor_width_m = floor_width * 0.3048
    floor_height_m = floor_height * 0.3048

    # We'll consider the light(s) at z=2.5 ft (convert to meters)
    light_z = 2.5 * 0.3048 # light source height

    # Create a dictionary to map light sources to layers
    light_to_layer_map = {}
    if pattern_option in ["Diamond: 13", "Diamond: 25", "Diamond: 41", "Diamond: 61", "Diamond: 221"]:
        layer_index = 0
        for layer in layers:
            if layer_index == 0:
                # Center light
                light_to_layer_map[light_sources[0]] = layer_index
            else:
                for pos in layer:
                    # Find the corresponding light source
                    # Calculate expected position
                    spacing_x = light_array_width_ft / 7.2
                    spacing_y = light_array_height_ft / 7.2
                    x_offset = spacing_x * pos[0]
                    y_offset = spacing_y * pos[1]
                    theta = math.radians(45)
                    rotated_x_offset = x_offset * math.cos(theta) - y_offset * math.sin(theta)
                    rotated_y_offset = x_offset * math.sin(theta) + y_offset * math.cos(theta)
                    expected_pos = (floor_width / 2 + rotated_x_offset, floor_height / 2 + rotated_y_offset)
                    try:
                        idx = light_sources.index(expected_pos)
                        light_to_layer_map[light_sources[idx]] = layer_index
                    except ValueError:
                        print(f"Warning: Light source at {expected_pos} not found in light_sources.")
            layer_index += 1
    else:
        # For grid patterns, each light source is its own layer
        for i, pos in enumerate(light_sources):
            light_to_layer_map[pos] = i

    # Generate measurement points at z=0 (on the floor)
    measurement_points = get_measurement_points(floor_width, floor_height)

    intensity = np.zeros(num_points)

    # Find the index of the point closest to the center
    center_point = (floor_width / 2, floor_height / 2)
    center_point_index = min(range(len(measurement_points)), key=lambda i: math.dist(measurement_points[i][:2], center_point))

    # ... [previous code in prepare_heatmap_data] ...

    # Calculate PPFD for each measurement point
    for point_index, measurement_point in enumerate(measurement_points):
        total_ppfd = 0.0
        x_m, y_m, z_m = measurement_point

        for layer_index in range(len(light_intensity_entries)):
            try:
                lumens = float(light_intensity_entries[layer_index].get())
            except ValueError:
                lumens = 1.0

            # Convert lumens to radiometric power for this layer
            layer_radiometric_power = lumens_to_radiometric_power(lumens, luminous_efficacy)
            layer_scaling_factor = layer_radiometric_power  # assuming baseline_total_power = 1 W
            layer_photon_flux = baseline_photon_flux * layer_scaling_factor

            layer_light_sources = [pos for pos, layer in light_to_layer_map.items() if layer == layer_index]

            for lx, ly in layer_light_sources:
                area = floor_width_m * floor_height_m
                ppfd_at_point = calculate_ppfd(layer_photon_flux, area)
                horizontal_distance = calculate_distance((lx, ly), measurement_point)
                intensity_at_point = lambertian_emission(ppfd_at_point, horizontal_distance, light_z)
                total_ppfd += intensity_at_point

        intensity[point_index] = total_ppfd

    # --- Adaptive Calculation of Multiple Bounce Reflections ---
    tolerance = 0.01  # Threshold for negligible contribution
    max_iterations = 20  # Safety limit to prevent infinite loops
    iteration = 0
    prev_total_reflection_exp = float('inf')
    prev_total_reflection_inv = float('inf')
    accumulated_reflection_exp = np.zeros_like(intensity)
    accumulated_reflection_inv = np.zeros_like(intensity)

    while iteration < max_iterations:
        # Compute reflection for this bounce (using both models)
        reflected_intensity_exp, reflected_intensity_inv = calculate_reflected_light(
            measurement_points,
            floor_width,
            floor_height,
            perimeter_reflectivity,
            intensity + accumulated_reflection_exp,  # Use exponential reflection for subsequent iterations
            num_iterations=1,  # Single bounce iteration per loop
            decay_constant=1.5  # You might want to make this a user input later
        )

        # Accumulate the new reflection (for both models)
        accumulated_reflection_exp += reflected_intensity_exp
        accumulated_reflection_inv += reflected_intensity_inv

        # Check total energy of this bounce (for both models)
        current_total_reflection_exp = np.sum(reflected_intensity_exp)
        current_total_reflection_inv = np.sum(reflected_intensity_inv)
        print(f"Iteration {iteration+1}: Total new reflected energy (Exp) = {current_total_reflection_exp:.4f}")
        print(f"Iteration {iteration+1}: Total new reflected energy (Inv) = {current_total_reflection_inv:.4f}")

        # Break if change is below threshold (check both models)
        if (abs(current_total_reflection_exp - prev_total_reflection_exp) < tolerance) and \
           (abs(current_total_reflection_inv - prev_total_reflection_inv) < tolerance):
            print("Reflections have converged below tolerance.")
            break

        prev_total_reflection_exp = current_total_reflection_exp
        prev_total_reflection_inv = current_total_reflection_inv
        iteration += 1

    # Add accumulated reflections to original intensity (choose one model or combine them)
    # For now, let's use the exponential model:
    #intensity += accumulated_reflection_exp
    intensity += accumulated_reflection_inv # Uncomment if you want to use inverse square instead
    # intensity += (accumulated_reflection_exp + accumulated_reflection_inv) / 2 # Uncomment if you want to average the two models
    # Debug statements
    print(f"Perimeter Reflectivity: {perimeter_reflectivity}")
    #print(f"Total Reflected Intensity: {np.sum(reflected_intensity)}")


    # Calculate the total lumens (sum from entry widgets)
    total_lumens = sum(float(entry.get()) for entry in light_intensity_entries if entry.get().replace('.', '', 1).isdigit())

    # Remove existing total lumens label if any
    for widget in root.grid_slaves(row=27, column=5):
        widget.destroy()
    total_lumens_label = Label(root, text=f"Total Lumens: {total_lumens:.2f}")
    total_lumens_label.grid(row=27, column=5, columnspan=5)

    average_intensity = np.mean(intensity)
    total_intensity = np.sum(intensity)

    # PPFD is average µmol·m⁻²·s⁻¹ across all measurement points
    ppfd = total_intensity / len(measurement_points)
    print(f"Average PPFD: {average_intensity}")
    print(f"Total PPFD: {total_intensity}")
    print(f"PPFD: {ppfd}")

    intensity_variance = np.mean(np.abs(intensity - average_intensity))

    # Remove existing variance labels if any
    for widget in root.grid_slaves(row=25, column=5):
        widget.destroy()
    for widget in root.grid_slaves(row=26, column=5):
        widget.destroy()

    variance_label = Label(root, text=f"Mean Absolute Deviation: {intensity_variance:.2f}")
    variance_label.grid(row=25, column=5, columnspan=5)

    variance_label = Label(root, text=f"Total Intensity: {total_intensity:.2f}")
    variance_label.grid(row=26, column=5, columnspan=5)   

    dps = np.abs(intensity - average_intensity)
    sorted_dps = reorder_dps(dps, measurement_points, center_point)

    return sorted_dps, center_source

def on_generate_heatmap_click():
    sorted_dps, center_source = prepare_heatmap_data()  # Only expecting two values
    generate_heatmap(center_source)

def plot_dps(sorted_intensity):
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_intensity, color='dodgerblue', marker='o', linestyle='-', linewidth=1, markersize=3,)
    plt.title('DPS Across Measurement Points')
    plt.xlabel('Measurement Point Index')
    plt.ylabel('DPS Value')
    plt.grid(True)
    plt.ylim([0, 420])  # Set the limits of y-axis
    plt.show()

def generate_line_graph():
    sorted_intensity, _ = prepare_heatmap_data()  # Unpack and ignore center_source
    plot_dps(sorted_intensity)

def edit_intensity(idx):
    global layer_intensities  # Access the global list

    # Create a dialog and get the user's input
    new_intensity = tkinter.simpledialog.askfloat("Edit Light Intensity", "Enter new intensity:")
    
    if new_intensity is not None:  # None if the user cancelled the dialog
        # Update the light source's intensity
        layer_intensities[idx] = new_intensity
        # Also update the corresponding Entry in the GUI
        light_intensity_entries[idx].delete(0, tk.END)  # Remove the old intensity
        light_intensity_entries[idx].insert(0, str(new_intensity))  # Insert the new intensity
        # Call generate_heatmap again to update the plot
        generate_heatmap(center_source)

def find_nearest_light_source(x, y, light_sources):
    distances = [np.sqrt((x - lx)**2 + (y - ly)**2) for lx, ly in light_sources]
    return np.argmin(distances)

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return  # Click was outside the axes
    ix, iy = event.xdata, event.ydata
    idx = find_nearest_light_source(ix, iy, light_sources)
    print(f'Clicked on source {idx}, intensity: {light_intensities[idx]}')
    # edit_intensity(idx) # commented out this line since we dont need it anymore due to switching from light source to layer
    generate_heatmap(center_source)

def generate_heatmap(center_source):
    # Call prepare_heatmap_data and unpack both sorted_dps and center_source
    sorted_dps, center_source = prepare_heatmap_data()
    
    # Access necessary globals
    global floor_width, floor_height, intensity, measurement_points, num_points, min_int, max_int, intensity_variance, ppfd, total_lumens
    
    # Reshape intensity for heatmap
    num_points_per_dim = int(math.sqrt(num_points))
    intensity_matrix = intensity.reshape((num_points_per_dim, num_points_per_dim))
    intensity_matrix = np.flip(intensity_matrix, axis=0)

    # Plot the floor space
    plt.figure(figsize=(8, 8))
    plt.xlim(0, floor_width)
    plt.ylim(0, floor_height)

    if show_light_sources.get():
        plt.plot(center_source[0], center_source[1], 'mo', markersize=6)  # Center light
        for idx, light_source in enumerate(light_sources):
            if light_source == center_source:
                continue  # Skip the center light
            plt.plot(light_source[0], light_source[1], 'mo', markersize=6)  # Magenta circle for light sources
            plt.text(light_source[0], light_source[1], str(idx + 1), color='red', fontsize=12, ha='right')

    # Dropdown menu for cmap colors
    selected_option = selected_cmap.get()

    # Dictionary mapping options to cmap values
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

    # Use the selected option to get the cmap value
    cmap_value = cmap_dict.get(selected_option, "jet")  # Default to "jet" if the option is not recognized

    # Code to generate heat map
    plt.imshow(intensity_matrix, cmap=cmap_value, interpolation='bicubic', vmin=min_int, vmax=max_int, extent=[0, floor_width, 0, floor_height])
    plt.colorbar(label='Intensity')
    plt.title('Light Intensity Heat Map')

    # Add the click event handler
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    # Add intensity labels to each data point with adjusted size and positioning
    for row in range(intensity_matrix.shape[0]):
        for col in range(intensity_matrix.shape[1]):
            point_intensity = intensity_matrix[row, col]
            rounded_intensity = int(round(point_intensity))  # Round and convert to int  
            text_x = col * (floor_width / intensity_matrix.shape[1]) + (floor_width / (2 * intensity_matrix.shape[1]))
            text_y = (intensity_matrix.shape[0] - row - 1) * (floor_height / intensity_matrix.shape[0]) + (floor_height / (2 * intensity_matrix.shape[0]))
            plt.text(text_x, text_y, str(rounded_intensity), ha='center', va='center', color='black',
                     fontsize=8, fontweight='bold')

    if show_measurement_points.get():
        # Plot the measurement points
        measurement_xs, measurement_ys, _ = zip(*measurement_points)
        plt.plot(measurement_xs, measurement_ys, 'r+', markersize=7, alpha=1.0)

    # Display summary texts
    text = f'Total Lumens: {total_lumens}\nPPFD: {ppfd:.2f}\nMean Absolute Deviation: {intensity_variance:.2f}'
    text_lines = text.split('\n')

    # Get the current axis
    ax = plt.gca()

    for i, line in enumerate(text_lines):
        if ':' in line:
            title, value = line.split(':', 1)
            ax.text(0.01, -0.12 - i * 0.05, f"{title}:", transform=ax.transAxes, fontsize=10, fontweight='bold', color='blue')
            ax.text(0.5, -0.12 - i * 0.05, value.strip(), transform=ax.transAxes, fontsize=10)

    plt.show()

def generate_surface_graph(flatten_factor=1.0, scale_factor=1.0, x_limit=20, y_limit=20):
    sorted_dps, center_source = prepare_heatmap_data()
    
    # Access necessary globals
    global floor_width, floor_height, intensity, num_points
    
    # Get the user-defined settings from the input fields
    min_int = float(min_int_entry.get())
    max_int = float(max_int_entry.get())

    x = []
    y = []

    for i in range(15):
        for j in range(15):
            x.append(-7.5 + j * 15. / 14)
            y.append(7.5 - i * 15. / 14)

    x = np.array(x) * scale_factor
    y = np.array(y) * scale_factor

    # Data for surface plot
    z = np.array(intensity)

    # Ensure z has correct length
    assert (len(z) == 225), f"Wrong length for input z. Should be 225, but it is {len(z)}!"

    # Padding arrays (ensure they are fully defined)
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

    # Apply a power transformation to z_ to flatten the intensity (conditionally)
    if flatten_factor != 1.0:
        z_ = np.power(z_, flatten_factor)

    triang = mtri.Triangulation(x_, y_)
    x_grid, y_grid = np.mgrid[x_.min():x_.max():.02, y_.min():y_.max():.02]
    interp_lin = mtri.LinearTriInterpolator(triang, z_)
    zi_lin = interp_lin(x_grid, y_grid)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Set the x and y limits based on the data
    ax.set_xlim([x_grid.min(), x_grid.max()])
    ax.set_ylim([y_grid.min(), y_grid.max()])
    ax.set_zlim([z_.min(), z_.max()])  # Set z-axis limits based on z_ data

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Dropdown menu for cmap colors
    selected_option = selected_cmap.get()

    # Dictionary mapping options to cmap values
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

    # Use the selected option to get the cmap value
    cmap_value = cmap_dict.get(selected_option, "jet")  # Default to "jet" if the option is not recognized

    pc = ax.plot_surface(x_grid, y_grid, zi_lin, cmap=cmap_value, vmin=z_.min(), vmax=z_.max())

    fig.colorbar(pc, fraction=0.032, pad=0.04)

    ax.plot_wireframe(x_grid, y_grid, zi_lin, rstride=50, cstride=50, color='k', linewidth=0.4)

    plt.title('Light Intensity Surface Graph')

    plt.show()
 
# Create the GUI
root = Tk()
root.title("PPFD Simulation Software")

# Load previous settings if available
settings = load_settings()

# Floor space dimensions
floor_width_label = Label(root, text="Floor Width (feet):")
floor_width_label.grid(row=0, column=0)
floor_width_entry = Entry(root)
floor_width_entry.grid(row=0, column=1)

if 'floor_width' in settings:
    floor_width_entry.insert(0, str(settings['floor_width']))
else:
    floor_width_entry.insert(0, '12') # Changed default to 12

floor_height_label = Label(root, text="Floor Height (feet):")
floor_height_label.grid(row=1, column=0)
floor_height_entry = Entry(root)
floor_height_entry.grid(row=1, column=1)

if 'floor_height' in settings:
    floor_height_entry.insert(0, str(settings['floor_height']))
else:
    floor_height_entry.insert(0, '12') # Changed default to 12

# Min / Max intensity scale
min_int_label = Label(root, text="Min. Intensity (scale):")
min_int_label.grid(row=1, column=2)
min_int_entry = Entry(root)
min_int_entry.grid(row=1, column=3)

if 'min_int' in settings:
    min_int_entry.insert(0, str(settings['min_int']))
else:
    min_int_entry.insert(0, '0') # Changed default to 0

max_int_label = Label(root, text="Max. Intensity (scale):")
max_int_label.grid(row=2, column=2)
max_int_entry = Entry(root)
max_int_entry.grid(row=2, column=3)

if 'max_int' in settings:
    max_int_entry.insert(0, str(settings['max_int']))
else:
    max_int_entry.insert(0, '500') # Changed default to 500

# Lighting array dimensions
light_array_width_label = Label(root, text="Lighting Array Width (inches):")
light_array_width_label.grid(row=2, column=0)
light_array_width_entry = Entry(root)
light_array_width_entry.grid(row=2, column=1)

if 'light_array_width' in settings:
    light_array_width_entry.insert(0, str(settings['light_array_width']))
else:
    light_array_width_entry.insert(0, '60')

light_array_height_label = Label(root, text="Lighting Array Height (inches):")
light_array_height_label.grid(row=3, column=0)
light_array_height_entry = Entry(root)
light_array_height_entry.grid(row=3, column=1)

if 'light_array_height' in settings:
    light_array_height_entry.insert(0, str(settings['light_array_height']))
else:
    light_array_height_entry.insert(0, '60')

# Perimeter Reflectivity input field
# Create a Tkinter variable to hold the perimeter reflectivity value
perimeter_reflectivity_var = tk.DoubleVar(value=0.3)  # Set default value to 0.3

# Perimeter Reflectivity input field
perimeter_reflectivity_label = Label(root, text="Perimeter Reflectivity (0-1):")
perimeter_reflectivity_label.grid(row=3, column=2)
perimeter_reflectivity_entry = Entry(root, textvariable=perimeter_reflectivity_var)  # Link Entry to the variable
perimeter_reflectivity_entry.grid(row=3, column=3)

# Light intensity input fields
light_intensity_labels = []
light_intensity_entries = []
num_layers = 11  # Maximum number of layers (including center)
num_columns = 2  # Number of columns to divide the intensity fields into

# Create the intensity input fields in the GUI
for i in range(num_layers):
    label_text = f"Layer {i} Lumens:" if i > 0 else "Center Lumens:"
    label = Label(root, text=label_text)

    # Calculate the row and column index
    row = i // num_columns + 4
    column = i % num_columns * 2

    label.grid(row=row, column=column)
    entry = Entry(root)
    entry.grid(row=row, column=column + 1)

    # Load saved intensities if available
    if 'layer_intensities' in settings and i < len(settings['layer_intensities']):
        entry.insert(0, str(settings['layer_intensities'][i]))
    else:
        entry.insert(0, '1.0')  # Default intensity

    light_intensity_labels.append(label)
    light_intensity_entries.append(entry)

# Toggle checkboxes for light sources and measurement points
show_light_sources = BooleanVar()
show_light_sources.set(True)
show_measurement_points = BooleanVar()
show_measurement_points.set(True)

# Define the available options for pattern choice dropdown menu
pattern_options = ["Diamond: 13", "Diamond: 25", "Diamond: 41", "Diamond: 61", "Diamond: 221", "5x5 Grid", "8x8 Grid"]

# Create a variable to store the selected pattern
selected_pattern = StringVar(root)
selected_pattern.set(pattern_options[0]) # Set the default option

# Define the available options for the colormap dropdown menu
cmap_options = ["Cool", "Jet", "Inferno",
                "Viridis", "Plasma", "Turbo", 
                "Autumn", "Spring", "Gray"]

# Create a variable to store the selected colormap option
selected_cmap = StringVar(root)
selected_cmap.set(cmap_options[1])  # Set the default option

# Create a label for the pattern dropdown menu
pattern_label = Label(root, text="Select Pattern:")
pattern_label.grid(row=1, column=5)

# Create the pattern dropdown menu
pattern_dropdown = OptionMenu(root, selected_pattern, *pattern_options, command=lambda _: update_light_intensity_entries())
pattern_dropdown.grid(row=2, column=5)

# Create a label for the colormap dropdown menu
cmap_label = Label(root, text="Select Color Map:")
cmap_label.grid(row=1, column=4)

# Create the colormap dropdown menu
cmap_dropdown = OptionMenu(root, selected_cmap, *cmap_options)
cmap_dropdown.grid(row=2, column=4)

light_sources_checkbutton = Checkbutton(root, text="Show Light Sources", variable=show_light_sources)
light_sources_checkbutton.grid(row=25, column=0, columnspan=1)

measurement_points_checkbutton = Checkbutton(root, text="Show Measurement Points", variable=show_measurement_points)
measurement_points_checkbutton.grid(row=26, column=0, columnspan=1)

# Generate heatmap button
generate_button = Button(root, text="Generate Heatmap", command=on_generate_heatmap_click)
generate_button.grid(row=27, column=0, columnspan=1)

surface_button = Button(root, text="Generate 3D Surface Plot", command=generate_surface_graph)
surface_button.grid(row=27, column=1, columnspan=1)

# Generate 2D Line Graph button
line_graph_button = Button(root, text="Generate 2D Line Graph", command=generate_line_graph)
line_graph_button.grid(row=27, column=2, columnspan=1)

# Add a button for loading intensities from a file
load_button = Button(root, text="Load Intensities", command=load_intensities)
load_button.grid(row=27, column=3, columnspan=1)

# Add a button for saving intensities to a file
save_button = Button(root, text="Save Intensities", command=save_intensities)
save_button.grid(row=27, column=4, columnspan=1)  # adjust the row and column as needed

root.mainloop()