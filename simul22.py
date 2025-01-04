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
from scipy.optimize import curve_fit

SETTINGS_FILE = 'settings.json'
#LUMENS_TO_PPFD_CONVERSION = 23  # Updated Conversion Factor

# Constants and Parameters
HEIGHT_FROM_FLOOR = 2.5
BEAM_ANGLE_DEG = 120
REFLECTIVITY = 0.0
NUM_MEASUREMENT_POINTS = 225
DEFAULT_TARGET_PPFD = 1500
PPFD_THRESHOLD_FACTOR = 0.5
NUM_LAYERS = 6
DEFAULT_FLOOR_WIDTH = 12.0  # Add this line, set to your desired default value
DEFAULT_FLOOR_HEIGHT = 12.0 # Add this line, set to your desired default value
DEFAULT_LIGHT_ARRAY_WIDTH = 12.0 # Add this line, set to your desired default value
DEFAULT_LIGHT_ARRAY_HEIGHT = 12.0 # Add this line, set to your desired default value

show_light_sources = True
show_measurement_points = True

#plt.rcParams['figure.dpi'] = 125  # or any other value

# Data for Curve Fitting
angles_deg = np.array([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])
relative_intensities = np.array([0, 10, 20, 35, 55, 80, 100, 80, 55, 35, 20, 10, 0]) / 100.0

# 4th-Order Polynomial Function
def polynomial_func(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Curve Fitting
params, covariance = curve_fit(polynomial_func, angles_deg, relative_intensities)
print("Fitted parameters (a, b, c, d, e):", params)

# Fitted Intensity Function
def fitted_intensity_func(angle_deg):
    a, b, c, d, e = params
    return polynomial_func(angle_deg, a, b, c, d, e)

# Lambertian Emission Model with Fitted Function
def lambertian_emission(ppfd, distance, z, beam_angle_rad):
    theta = math.acos(max(min(z / distance, 1), -1))
    theta_deg = abs(math.degrees(theta))
    relative_intensity = fitted_intensity_func(theta_deg)
    relative_intensity = max(0, min(relative_intensity, 1))
    ppfd_at_point = ppfd * relative_intensity
    ppfd_at_point /= distance ** 2
    return ppfd_at_point

# --- Calculate PPFD from Lumens ---
def calculate_ppfd_from_lumens(lumens, effective_area):
    """
    Calculates PPFD from lumens, given the effective area.
    """
    # Simple conversion for now (can be made more accurate later)
    ppfd = lumens / effective_area * 21.62513 # Using 15 as the conversion factor: 1 μmol/s ≈ 15 lumens
    return ppfd

def estimate_effective_area_per_emitter(emitter_x, emitter_y, measurement_points, ppfd_threshold_factor):
    """
    Estimates the effective area illuminated by a single emitter.
    """
    # Calculate PPFD distribution from this emitter
    ppfd_values = []
    for point in measurement_points:
        distance = math.sqrt((emitter_x - point[0])**2 + (emitter_y - point[1])**2 + HEIGHT_FROM_FLOOR**2)
        beam_angle_radians = math.radians(BEAM_ANGLE_DEG)
        # Use a placeholder PPFD value (e.g., 1) for the emitter in lambertian_emission
        ppfd = lambertian_emission(1, distance, HEIGHT_FROM_FLOOR, beam_angle_radians)  # Assume 1 as placeholder intensity
        ppfd_values.append(ppfd)

    # Find max PPFD (directly below the emitter)
    max_ppfd = max(ppfd_values)

    # Calculate PPFD threshold
    ppfd_threshold = max_ppfd * ppfd_threshold_factor

    # Estimate effective area based on the threshold
    effective_area = sum(1 for ppfd in ppfd_values if ppfd >= ppfd_threshold) * (DEFAULT_FLOOR_WIDTH * DEFAULT_FLOOR_HEIGHT) / NUM_MEASUREMENT_POINTS

    return effective_area

# --- Calculate PPFD at a Point with Reflection ---
def calculate_ppfd_at_point(point, light_sources, light_intensities_by_layer, pattern_option):
    total_ppfd = 0
    for layer_index, layer_sources in enumerate(light_sources):
        if pattern_option == "5x5 Grid" or pattern_option == "8x8 Grid":
            layer_ppfd = light_intensities_by_layer[0]
        else:
            layer_ppfd = light_intensities_by_layer[layer_index]
        for source in layer_sources:
            distance = math.sqrt((source[0] - point[0])**2 + (source[1] - point[1])**2 + HEIGHT_FROM_FLOOR**2)
            beam_angle_radians = math.radians(BEAM_ANGLE_DEG)
            direct_ppfd = lambertian_emission(layer_ppfd, distance, HEIGHT_FROM_FLOOR, beam_angle_radians)
            reflected_ppfd = direct_ppfd * REFLECTIVITY
            total_ppfd += direct_ppfd + reflected_ppfd

    return total_ppfd

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

def prepare_heatmap_data():
    global show_light_sources, show_measurement_points, intensity 
    global floor_height, floor_width, light_sources, num_points
    global light_intensities, min_int, max_int, measurement_points
    global average_intensity, total_intensity, ppfd, intensity_variance
    global total_lumens

    measurement_points = []

    # Get the user-defined settings from the input fields
    floor_width = float(floor_width_entry.get())
    floor_height = float(floor_height_entry.get())
    light_array_width = float(light_array_width_entry.get())
    light_array_height = float(light_array_height_entry.get())
    light_intensities = [float(entry.get()) for entry in light_intensity_entries]
    min_int = float(min_int_entry.get())
    max_int = float(max_int_entry.get())

    # Store the settings for next time
    settings = {
        'floor_width': floor_width,
        'floor_height': floor_height,
        'light_array_width': light_array_width,
        'light_array_height': light_array_height,
        'light_intensities': light_intensities,
        'min_int': min_int,
        'max_int': max_int,
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

    # --- Light Source Arrangement Generation (Based on Pattern) ---
    if pattern_option == "5x5 Grid":
        # Calculate the spacing between lights
        spacing = light_array_width_ft / 5

        # Prepare list for light sources
        light_sources = []

        # Create a 5x5 square pattern
        for i in np.arange(-2, 3, 1):
            for j in np.arange(-2, 3, 1):
                x_offset = spacing * i
                y_offset = spacing * j
                light_pos = (center_x + x_offset, center_y + y_offset)
                light_sources.append(light_pos)

    elif pattern_option == "8x8 Grid":
        # Calculate the spacing between lights
        spacing = light_array_width_ft / 8

        # Prepare list for light sources
        light_sources = []

        # Create an 8x8 square pattern
        for i in np.arange(-3.5, 4.5, 1):
            for j in np.arange(-3.5, 4.5, 1):
                x_offset = spacing * i
                y_offset = spacing * j
                light_pos = (center_x + x_offset, center_y + y_offset)
                light_sources.append(light_pos)

    elif pattern_option in ["Diamond: 13", "Diamond: 25", "Diamond: 41", "Diamond: 61"]:
        # Define layers for Diamond patterns
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
                [(-2, -1), (2, -1), (-2, 1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2), (-3, 0), (3, 0), (0, -3), (0, 3)]
            ]
        elif pattern_option == "Diamond: 41": 
            layers = [
                [(0, 0)],
                [(-1, 0), (1, 0), (0, -1), (0, 1)],
                [(-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)],
                [(-2, -1), (2, -1), (-2, 1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2), (-3, 0), (3, 0), (0, -3), (0, 3)],
                [(-2, -2), (2, -2), (-2, 2), (2, 2), (-3, -1), (3, -1), (-3, 1), (3, 1), (-1, -3), (1, -3), (-1, 3), (1, 3), (-4, 0), (4, 0), (0, -4), (0, 4)]
            ]
        elif pattern_option == "Diamond: 61":
            layers = [
                [(0, 0)],
                [(-1, 0), (1, 0), (0, -1), (0, 1)],
                [(-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)],
                [(-2, -1), (2, -1), (-2, 1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2), (-3, 0), (3, 0), (0, -3), (0, 3)],
                [(-2, -2), (2, -2), (-2, 2), (2, 2), (-3, -1), (3, -1), (-3, 1), (3, 1), (-1, -3), (1, -3), (-1, 3), (1, 3), (-4, 0), (4, 0), (0, -4), (0, 4)],
                [(-3, -2), (3, -2), (-3, 2), (3, 2), (-2, -3), (2, -3), (-2, 3), (2, 3), (-4, -1), (4, -1), (-4, 1), (4, 1), (-1, -4), (1, -4), (-1, 4), (1, 4), (-5, 0), (5, 0), (0, -5), (0, 5)]
            ]

        # Calculate the spacing between lights
        spacing_x = light_array_width_ft / 7.2
        spacing_y = light_array_height_ft / 7.2

        # Prepare list for light sources
        light_sources = []  # Initialize light_sources here

        # Add the center light source to the list
        light_sources.append(center_source)

        # Generate light sources for Diamond patterns
        for layer in layers:
            for dot in layer:
                if dot != (0, 0):
                    x_offset = spacing_x * dot[0]
                    y_offset = spacing_y * dot[1]
                    theta = math.radians(45)
                    rotated_x_offset = x_offset * math.cos(theta) - y_offset * math.sin(theta)
                    rotated_y_offset = x_offset * math.sin(theta) + y_offset * math.cos(theta)
                    light_pos = (center_x + rotated_x_offset, center_y + rotated_y_offset)
                    light_sources.append(light_pos)

    else:
        # Default to an empty list if pattern option is not recognized
        light_sources = []

    # --- Measurement Points Generation ---
    def get_measurement_points():
        measurement_points = []
        measurement_points_per_dimension = int(math.sqrt(num_points))
        light_array_height_from_floor = HEIGHT_FROM_FLOOR

        step_size_x = floor_width / (measurement_points_per_dimension + 1)
        step_size_y = floor_height / (measurement_points_per_dimension + 1)

        for x in range(1, measurement_points_per_dimension + 1):
            for y in range(1, measurement_points_per_dimension + 1):
                point_x = x * step_size_x
                point_y = y * step_size_y
                point_z = light_array_height_from_floor
                measurement_points.append((point_x, point_y, point_z))

        return measurement_points

    measurement_points = get_measurement_points()

    # --- PPFD Calculation ---
    intensity = np.zeros(num_points)

    # Determine the number of lights per layer based on pattern option
    if pattern_option in ["5x5 Grid", "8x8 Grid"]:
        num_lights_per_layer = len(light_sources)  # All lights in one layer
    elif pattern_option in ["Diamond: 13", "Diamond: 25", "Diamond: 41", "Diamond: 61"]:
        num_lights_per_layer = len(layers[0]) if layers else 0  # Assuming layers is defined based on pattern_option
    else:
        num_lights_per_layer = 0  # Default to 0 if pattern is not recognized
        print(f"Warning: Pattern option '{pattern_option}' not recognized. No lights assigned to layers.")

    # Calculate effective area per emitter for each layer
    effective_areas_per_emitter = []
    for light_source in light_sources:
        # Ensure light_source is treated as a tuple (coordinates) for both grid and diamond patterns
        if isinstance(light_source, tuple):
            emitter_x, emitter_y = light_source
        else:
            # Handle cases where light_source is not a tuple (e.g., for grid patterns)
            emitter_x, emitter_y = light_source, 0.0  # Adjust default y-value as needed

        effective_area = estimate_effective_area_per_emitter(emitter_x, emitter_y, measurement_points, PPFD_THRESHOLD_FACTOR)
        effective_areas_per_emitter.append(effective_area)

    # Create a list to store PPFD values for each layer
    light_ppfds_by_layer = []

    # Convert lumens to PPFD for each layer
    if pattern_option == "5x5 Grid" or pattern_option == "8x8 Grid":
        # For grid patterns, use the first intensity value for all lights
        ppfd = calculate_ppfd_from_lumens(light_intensities[0], effective_areas_per_emitter[0])
        light_ppfds_by_layer = [ppfd] * len(light_sources)  # Repeat the same PPFD for all lights
    else:
        # For diamond patterns, use the intensity value for each layer
        for i, layer in enumerate(layers):
            # Check if there are more layers than intensity values provided
            if i < len(light_intensities):
                layer_lumens = light_intensities[i]
                # Calculate PPFD for the layer based on the layer's total lumens and effective area
                ppfd = calculate_ppfd_from_lumens(layer_lumens, effective_areas_per_emitter[i] * len(layer))
                light_ppfds_by_layer.extend([ppfd] * len(layer))  # Repeat PPFD for each light in the layer
            else:
                # Handle the case where there are more layers than intensity values
                print(f"Warning: No intensity provided for layer {i}. Using a default PPFD value.")
                default_ppfd = 0.0
                light_ppfds_by_layer.extend([default_ppfd] * len(layer))

    # --- Intensity Calculation at Each Measurement Point ---
    for point_index, measurement_point in enumerate(measurement_points):
        total_ppfd = 0
        for light_index, light_position in enumerate(light_sources):
            distance = calculate_distance(light_position, measurement_point)

            # Get the z-coordinate (height) of the light source
            light_z = HEIGHT_FROM_FLOOR  # Assuming this is constant for all lights

            # Use the fitted Lambertian emission model
            if light_index < len(light_ppfds_by_layer):
                # Calculate the PPFD at the measurement point using the lambertian_emission function
                beam_angle_radians = math.radians(BEAM_ANGLE_DEG)
                ppfd_at_point = lambertian_emission(light_ppfds_by_layer[light_index], distance, light_z, beam_angle_radians)

                # Add to the total PPFD
                total_ppfd += ppfd_at_point

        intensity[point_index] = total_ppfd

    # Calculate the total lumens
    total_lumens = sum(light_intensities)

    # Display the total lumens on the GUI
    total_lumens_label = Label(root, text=f"Total Lumens: {total_lumens:.2f}")
    total_lumens_label.grid(row=27, column=5, columnspan=5)

# Calculate the average intensity
    average_intensity = np.mean(intensity)

    # Calculate the total intensity
    total_intensity = np.sum(intensity)

    # Calculate the ppfd
    ppfd = total_intensity / len(measurement_points)
    print(ppfd)

    # Calculate the Averaged Intensity Variance
    intensity_variance = np.mean(np.abs(intensity - average_intensity))

    # Display the Averaged Intensity Variance on the GUI
    variance_label = Label(root, text=f"Mean Absolute Deviation: {intensity_variance:.2f}")
    variance_label.grid(row=25, column=5, columnspan=5)

    # Display the total intensity on the GUI
    variance_label = Label(root, text=f"Total Intensity: {total_intensity:.2f}")
    variance_label.grid(row=26, column=5, columnspan=5)

    # Calculate DPS
    dps = np.abs(intensity - average_intensity)

    center_point = (floor_width / 2, floor_height / 2)  # Assuming your center point is the center of the floor

    sorted_dps = reorder_dps(dps, measurement_points, center_point)

    # Generate 2D intensity profile
    #generate_2d_intensity_profile(center_x, center_y, measurement_points, intensity)

    return sorted_dps

def plot_dps(sorted_intensity):
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_intensity, color='dodgerblue', marker='o', linestyle='-', linewidth=1, markersize=3,)
    plt.title('DPS Across Measurement Points')
    plt.xlabel('Measurement Point Index')
    plt.ylabel('DPS Value')
    plt.grid(True)
    plt.ylim([0, 420])  # Set the limits of x-axis
    plt.show()

def generate_line_graph():
    sorted_intensity = prepare_heatmap_data()
    plot_dps(sorted_intensity)

def edit_intensity(idx):
    global light_intensities  # Access the global list

    # Create a dialog and get the user's input
    new_intensity = tkinter.simpledialog.askfloat("Edit Light Intensity", "Enter new intensity:")
    
    if new_intensity is not None:  # None if the user cancelled the dialog
        # Update the light source's intensity
        light_intensities[idx] = new_intensity
        # Also update the corresponding Entry in the GUI
        light_intensity_entries[idx].delete(0, tk.END)  # Remove the old intensity
        light_intensity_entries[idx].insert(0, str(new_intensity))  # Insert the new intensity
        # Call generate_heatmap again to update the plot
        generate_heatmap()

def find_nearest_light_source(x, y, light_sources):
    distances = [np.sqrt((x - lx)**2 + (y - ly)**2) for lx, ly in light_sources]
    return np.argmin(distances)

def onclick(event):
    ix, iy = event.xdata, event.ydata
    idx = find_nearest_light_source(ix, iy, light_sources)
    print(f'Clicked on source {idx}, intensity: {light_intensities[idx]}')
    edit_intensity(idx)

def generate_heatmap():
    # First, call prepare_heatmap_data to get the updated variables
    prepare_heatmap_data()
    num_points_per_dim = int(np.sqrt(num_points))
    intensity_matrix = intensity.reshape((num_points_per_dim, num_points_per_dim))
    intensity_matrix = np.flip(intensity_matrix, axis=0)

    # Plot the floor space
    plt.figure(figsize=(8, 8))
    plt.xlim(0, floor_width)
    plt.ylim(0, floor_height)

    if show_light_sources.get():
        for idx, light_source in enumerate(light_sources):
            if isinstance(light_source, tuple):  # Check if the light_source is a tuple
                plt.plot(light_source[0], light_source[1], 'mo', markersize=6)  # Plot a single point
                plt.text(light_source[0], light_source[1], str(idx), color='red', fontsize=12, ha='right')
            else:
                for source in light_source:
                    plt.plot(source[0], source[1], 'mo', markersize=6)  # Plot a single point for each source in the layer
                    plt.text(source[0], source[1], str(idx), color='red', fontsize=12, ha='right')

    # dropdown menu for cmap colors
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
    cmap_value = cmap_dict.get(selected_option, "jet")  # Default to "cool" if the option is not recognized

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
    # Subplot adjust
    #plt.subplots_adjust(bottom=0.25)

    if show_measurement_points.get():
        # Plot the measurement points
        measurement_xs, measurement_ys, _ = zip(*measurement_points)
        plt.plot(measurement_xs, measurement_ys, 'r+', markersize=7, alpha=1.0)
        
    text = f'Total Lumens: {total_lumens}\nPPFD: {ppfd}\nMean Absolute Deviation: {intensity_variance}'
    text_lines = text.split('\n')

    # Get the current axis
    ax = plt.gca()

    for i, line in enumerate(text_lines):
        title, value = line.split(':')
        ax.text(0.01, -0.12 - i * 0.05, title + ':', transform=ax.transAxes, fontsize=10, fontweight='bold', color='blue')
        ax.text(0.5, -0.12 - i * 0.05, value, transform=ax.transAxes, fontsize=10)

    plt.show()

def generate_surface_graph(flatten_factor=1.0, scale_factor=1.0,x_limit=20, y_limit=20):
    prepare_heatmap_data()
    # Get the user-defined settings from the input fields
    min_int = float(min_int_entry.get())
    max_int = float(max_int_entry.get())

    x = []
    y = []

    for i in range(15) :
        for j in range(15) :
            x.append(-7.5 + j*15./14)
            y.append(7.5 - i*15./14)
        
    x = np.array(x) * scale_factor
    y = np.array(y) * scale_factor

    # put your data here:
    z = np.array(intensity)

    # The line below simply checks that the input (z) has the correct number of elements.
    assert(len(z) == 225) , "Wrong length for input z. Should be 225, but it is " + str(len(z)) + " !"

    x_pad = np.array([-18.,-9.,0.,9.,18.,-18.,-18.,-18.,-18.,-9.,0.,9.,18.,18.,18.,18.,-15.,-7.5,0.,7.5,15.,-15.,-15.,-15.,-15.,-7.5,0.,7.5,15.,15.,15.,15.]) * scale_factor
    y_pad = np.array([-18.,-18.,-18.,-18.,-18.,-9.,0.,9.,18.,18.,18.,18.,18.,-9.,0.,9.,-15.,-15.,-15.,-15.,-15.,-7.5,0.,7.5,15.,15.,15.,15.,15.,-7.5,0.,7.5]) * scale_factor
    z_pad = np.zeros(x_pad.shape)

    x_ = np.concatenate((x,x_pad))
    y_ = np.concatenate((y,y_pad))
    z_ = np.concatenate((z,z_pad))

    # Add a constant to z_ to raise the surface plot
    z_ = z_ + 0  # Adjust this value as needed

    # Apply a power transformation to z_ to flatten the intensity
    z_ = np.power(z_, flatten_factor)

    triang = mtri.Triangulation(x_, y_)
    x_grid, y_grid = np.mgrid[x_.min():x_.max():.02, y_.min():y_.max():.02]
    interp_lin = mtri.LinearTriInterpolator(triang, z_)
    zi_lin = interp_lin(x_grid, y_grid)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    # Set the x and y limits
    ax.set_xlim([-x_limit, x_limit])
    ax.set_ylim([-y_limit, y_limit])

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    #ax.set_zlim3d(0, 1650)
    ax.set_zlim3d(min_int, max_int)

    #dropdown menu for cmap colors
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
    cmap_value = cmap_dict.get(selected_option, "jet")  # Default to "cool" if the option is not recognized

    #pc = ax.plot_surface(x_grid, y_grid, zi_lin, cmap=cmap_value, vmin=0, vmax=1650)
    pc = ax.plot_surface(x_grid, y_grid, zi_lin, cmap=cmap_value, vmin=min_int, vmax=max_int)

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
    floor_width_entry.insert(0, '10')

floor_height_label = Label(root, text="Floor Height (feet):")
floor_height_label.grid(row=1, column=0)
floor_height_entry = Entry(root)
floor_height_entry.grid(row=1, column=1)

if 'floor_height' in settings:
    floor_height_entry.insert(0, str(settings['floor_height']))
else:
    floor_height_entry.insert(0, '10')

# Min / Max intensity scale
min_int_label = Label(root, text="Min. Intensity (scale):")
min_int_label.grid(row=1, column=2)
min_int_entry = Entry(root)
min_int_entry.grid(row=1, column=3)

if 'min_int' in settings:
    min_int_entry.insert(0, str(settings['min_int']))
else:
    min_int_entry.insert(0, '10')

max_int_label = Label(root, text="Max. Intensity (scale):")
max_int_label.grid(row=2, column=2)
max_int_entry = Entry(root)
max_int_entry.grid(row=2, column=3)

if 'max_int' in settings:
    max_int_entry.insert(0, str(settings['max_int']))
else:
    max_int_entry.insert(0, '10')

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

# Light intensity input fields
light_intensity_labels = []
light_intensity_entries = []
num_columns = 3  # Number of columns to divide the intensity fields into

# Create the intensity input fields in the GUI
for i in range(61):
    if i in (0,1,2,3,4):
        label = Label(root, text=f"L1 {i} Intensity")
    elif i in range(5, 13):
        label = Label(root, text=f"L2 {i} Intensity")
    elif i in range(13, 25):
        label = Label(root, text=f"L3 {i} Intensity")
    elif i in range(25, 41):
        label = Label(root, text=f"L4 {i} Intensity")
    elif i in range(41, 61):
        label = Label(root, text=f"L5 {i} Intensity")
    else:
        label = Label(root, text=f"other {i} Intensity")

    # Calculate the row and column index based on the desired number of columns
    row = i // num_columns + 4
    column = i % num_columns * 2

    label.grid(row=row, column=column)
    entry = Entry(root)
    entry.grid(row=row, column=column + 1)
    if 'light_intensities' in settings and i < len(settings['light_intensities']):
        entry.insert(0, str(settings['light_intensities'][i]))
    else:
        entry.insert(0, '1.0')
    light_intensity_labels.append(label)
    light_intensity_entries.append(entry)

# Toggle checkboxes for light sources and measurement points
show_light_sources = BooleanVar()
show_light_sources.set(True)
show_measurement_points = BooleanVar()
show_measurement_points.set(True)

# Define the avaialble options for pattern choice dropdown menu
pattern_options = ["Diamond: 13", "Diamond: 25", "Diamond: 41", "Diamond: 61", "5x5 Grid", "8x8 Grid"]

# Create a variable to store the selected pattern
selected_pattern = StringVar(root)
selected_pattern.set(pattern_options[0]) # Set the default option

# Define the available options for the dropdown menu
cmap_options = ["Cool", "Jet", "Inferno",
                "Viridis", "Plasma", "Turbo", 
                "Autumn", "Spring", "Gray"]

# Create a variable to store the selected option
selected_cmap = StringVar(root)
selected_cmap.set(cmap_options[1])  # Set the default option

# Create a label for the pattern dropdown menu
cmap_label = Label(root, text="Select Pattern:")
cmap_label.grid(row=1, column=5)

# Create the pattern dropdown menu
pattern_dropdown = OptionMenu(root, selected_pattern, *pattern_options)
pattern_dropdown.grid(row=2, column=5)

# Create a label for the dropdown menu
cmap_label = Label(root, text="Select Color Map:")
cmap_label.grid(row=1, column=4)

# Create the dropdown menu
cmap_dropdown = OptionMenu(root, selected_cmap, *cmap_options)
cmap_dropdown.grid(row=2, column=4)

light_sources_checkbutton = Checkbutton(root, text="Show Light Sources", variable=show_light_sources)
light_sources_checkbutton.grid(row=25, column=0, columnspan=1)

measurement_points_checkbutton = Checkbutton(root, text="Show Measurement Points", variable=show_measurement_points)
measurement_points_checkbutton.grid(row=26, column=0, columnspan=1)

# Generate heatmap button
generate_button = Button(root, text="Generate Heatmap", command=generate_heatmap)
generate_button.grid(row=27, column=0, columnspan=1)

surface_button = Button(root, text="Generate 3D Surface Plot", command=generate_surface_graph)
surface_button.grid(row=27, column=1, columnspan=1)

# Button initialization
surface_button = Button(root, text="Generate 2D Line Graph", command=generate_line_graph)
surface_button.grid(row=27, column=2, columnspan=1)

# Add a button for loading intensities from a file
load_button = Button(root, text="Load Intensities", command=load_intensities)
load_button.grid(row=27, column=3, columnspan=1)

# Add a button for saving intensities to a file
save_button = Button(root, text="Save Intensities", command=save_intensities)
save_button.grid(row=27, column=4, columnspan=1)  # adjust the row and column as needed

root.mainloop()