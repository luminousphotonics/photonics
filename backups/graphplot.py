import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import tkinter as tk
from tkinter import filedialog

def load_measurement_data():
    """
    Opens a file dialog to select a CSV file.
    Expects the CSV to contain 98 measurement values (one per row or first column).
    Returns a NumPy array of measurements.
    """
    root = tk.Tk()
    root.withdraw()  # hide main window
    filename = filedialog.askopenfilename(
        title="Select CSV file with 98 measurement points",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not filename:
        print("No file selected. Exiting.")
        sys.exit(1)
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # ignore blank rows
                try:
                    data.append(float(row[0]))
                except ValueError:
                    continue
    if len(data) != 98:
        print(f"Warning: Expected 98 measurements but got {len(data)}")
    return np.array(data)

def generate_line_graph():
    """
    Loads measurement data from CSV, builds a 14x7 grid over a 20'x10' space,
    calculates the deviation from the average intensity, and plots a 2D line graph.
    The points are sorted by distance from the grid center.
    """
    data = load_measurement_data()  # 98 values expected
    cols, rows = 14, 7
    # Build grid coordinates assuming the grid spans 20 feet in x and 10 feet in y.
    # Here we assume the CSV ordering is row-major starting at the top row (y=10) down to y=0.
    x_coords = np.linspace(0, 20, cols)
    # For y, we define the top row as y=10 and bottom as y=0.
    y_coords = np.linspace(10, 0, rows)
    coords = []
    for i in range(rows):
        for j in range(cols):
            coords.append((x_coords[j], y_coords[i]))
    coords = np.array(coords)
    center = np.array([10, 5])  # center of 20' x 10' space

    avg_intensity = np.mean(data)
    deviations = np.abs(data - avg_intensity)
    # Sort points by distance from the center.
    distances = np.linalg.norm(coords - center, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_devs = deviations[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_devs, marker='o', linestyle='-', color='dodgerblue')
    plt.title("Deviation from Average Illuminance")
    plt.xlabel("Measurement Point (sorted by distance from center)")
    plt.ylabel("Deviation (Lux)")
    plt.grid(True)
    plt.show()

def generate_surface_graph(flatten_factor=1.0, scale_factor=1.0):
    """
    Loads measurement data from CSV, builds a 14x7 grid over a 20'x10' space,
    and creates a 3D surface plot of the intensity values.
    The 'flatten_factor' parameter allows nonlinear scaling of the intensity,
    and 'scale_factor' scales the spatial coordinates.
    """
    data = load_measurement_data()  # 98 values expected
    cols, rows = 14, 7
    x_coords = np.linspace(0, 20, cols)
    y_coords = np.linspace(0, 10, rows)
    X, Y = np.meshgrid(x_coords, y_coords)  # shape (rows, cols)
    # Reshape the measurement data into the grid (assumes row-major order from CSV)
    Z = data.reshape((rows, cols))

    if flatten_factor != 1.0:
        Z = np.power(Z, flatten_factor)

    # Flatten the grid for triangulation
    x_flat = (X.flatten() * scale_factor)
    y_flat = (Y.flatten() * scale_factor)
    z_flat = Z.flatten()

    triang = mtri.Triangulation(x_flat, y_flat)
    # Build a fine grid for interpolation
    xi = np.linspace(x_flat.min(), x_flat.max(), 200)
    yi = np.linspace(y_flat.min(), y_flat.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    interp_lin = mtri.LinearTriInterpolator(triang, z_flat)
    Zi = interp_lin(Xi, Yi)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', edgecolor='none')
    ax.set_title("Light Intensity Surface Graph (Lux)")
    ax.set_xlabel("X (feet)")
    ax.set_ylabel("Y (feet)")
    ax.set_zlabel("Illuminance (Lux)")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()

if __name__ == "__main__":
    # Run one or both functions based on command-line argument.
    # Usage:
    #   python new_plot_script.py line     -> show 2D line graph
    #   python new_plot_script.py surface  -> show 3D surface graph
    #   python new_plot_script.py          -> default to surface graph
    if len(sys.argv) > 1:
        if sys.argv[1] == "line":
            generate_line_graph()
        elif sys.argv[1] == "surface":
            generate_surface_graph()
        else:
            print("Unknown argument. Use 'line' or 'surface'.")
    else:
        generate_surface_graph()
