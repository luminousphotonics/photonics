import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Input Parameters ---
floor_width = 12  # meters - NOW 12 feet
floor_length = 12  # meters - NOW 12 feet
light_height = 2.5  # meters
cob_wattage = 50  # watts
cob_efficiency = 182  # lumens per watt
measurement_height = 0.5  # meters

# --- COB Coordinates (12' x 12' Diamond Pattern) ---
cob_layers = [
    [(0, 0)],  # Central lighting element (Layer 0)
    [(-1.5, 0), (1.5, 0), (0, -1.5), (0, 1.5)],  # Layer 1
    [(-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5),
     (-3, 0), (3, 0), (0, -3), (0, 3)],  # Layer 2
    [(-3, -1.5), (3, -1.5), (-3, 1.5), (3, 1.5),
     (-1.5, -3), (1.5, -3), (-1.5, 3), (1.5, 3),
     (-4.5, 0), (4.5, 0), (0, -4.5), (0, 4.5)],  # Layer 3
    [(-3, -3), (3, -3), (-3, 3), (3, 3),
     (-4.5, -1.5), (4.5, -1.5), (-4.5, 1.5), (4.5, 1.5),
     (-1.5, -4.5), (1.5, -4.5), (-1.5, 4.5), (1.5, 4.5),
     (-6, 0), (6, 0), (0, -6), (0, 6)],  # Layer 4
    [(-4.5, -3), (4.5, -3), (-4.5, 3), (4.5, 3),
     (-3, -4.5), (3, -4.5), (-3, 4.5), (3, 4.5),
     (-6, -1.5), (6, -1.5), (-6, 1.5), (6, 1.5),
     (-1.5, -6), (1.5, -6), (-1.5, 6), (1.5, 6),
     (-7.5, 0), (7.5, 0), (0, -7.5), (0, 7.5)],  # Layer 5
    [(-4.5, -4.5), (4.5, -4.5), (-4.5, 4.5), (4.5, 4.5),
     (-6, -3), (6, -3), (-6, 3), (6, 3),
     (-3, -6), (3, -6), (-3, 6), (3, 6),
     (-7.5, -1.5), (7.5, -1.5), (-7.5, 1.5), (7.5, 1.5),
     (-1.5, -7.5), (1.5, -7.5), (-1.5, 7.5), (1.5, 7.5),
     (-9, 0), (9, 0), (0, -9), (0, 9)],  # Layer 6
    [(-6, -4.5), (6, -4.5), (-6, 4.5), (6, 4.5),
     (-4.5, -6), (4.5, -6), (-4.5, 6), (4.5, 6),
     (-7.5, -3), (7.5, -3), (-7.5, 3), (7.5, 3),
     (-3, -7.5), (3, -7.5), (-3, 7.5), (3, 7.5),
     (-9, -1.5), (9, -1.5), (-9, 1.5), (9, 1.5),
     (-1.5, -9), (1.5, -9), (-1.5, 9), (1.5, 9),
     (-10.5, 0), (10.5, 0), (0, -10.5), (0, 10.5)],  # Layer 7
    [(-6, -6), (6, -6), (-6, 6), (6, 6),
     (-7.5, -4.5), (7.5, -4.5), (-7.5, 4.5), (7.5, 4.5),
     (-4.5, -7.5), (4.5, -7.5), (-4.5, 7.5), (4.5, 7.5),
     (-9, -3), (9, -3), (-9, 3), (9, 3),
     (-3, -9), (3, -9), (-3, 9), (3, 9),
     (-10.5, -1.5), (10.5, -1.5), (-10.5, 1.5), (10.5, 1.5),
     (-1.5, -10.5), (1.5, -10.5), (-1.5, 10.5), (1.5, 10.5),
     (-12, 0), (12, 0), (0, -12), (0, 12)]  # Layer 8
]

# --- Load SPD and Radiation Pattern Data from CSV ---
# Read the SPD data, skipping the first row and handling extra spaces
spd_data = pd.read_csv(
    '/Users/austinrouse/photonics/main/spd_data.csv',
    sep='\s+',
    engine='python',
    skiprows=1,  # Skip the first row (header)
    names=['wavelength', 'intensity']
)

# Read the radiation pattern data, skipping the first row and handling extra spaces
radiation_pattern_data = pd.read_csv(
    '/Users/austinrouse/photonics/main/radiation_pattern_data.csv',
    sep='\s+',
    engine='python',
    skiprows=1,
    names=['angle', 'relative_intensity']
)

# --- Helper Functions ---

def calculate_distance(x1, y1, z1, x2, y2, z2):
    """Calculates the distance between two points."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def calculate_angle(x1, y1, z1, x2, y2, z2):
    """Calculates the angle between the z-axis and the vector from (x1, y1, z1) to (x2, y2, z2)."""
    vector = np.array([x2 - x1, y2 - y1, z2 - z1])
    unit_vector = vector / np.linalg.norm(vector)
    return np.arccos(unit_vector[2])  # Angle with the z-axis

def lumens_to_candelas(lumens, angle, radiation_pattern_df):
    """Converts lumens to candelas using the radiation pattern."""
    # Ensure the DataFrame is sorted by angle
    radiation_pattern_df = radiation_pattern_df.sort_values(by='angle')

    # Interpolate the relative intensity for the given angle
    relative_intensity = np.interp(angle, radiation_pattern_df['angle'], radiation_pattern_df['relative_intensity'])
    return lumens * relative_intensity

def spd_to_ppfd(spd_df, wavelengths, intensities):
    """Converts spectral power distribution (SPD) to PPFD.

    Args:
        spd_df (pd.DataFrame): DataFrame containing SPD data with columns 'wavelength' and 'intensity'.
        wavelengths (np.array): Array of wavelengths.
        intensities (np.array): Array of intensities corresponding to the wavelengths.

    Returns:
        float: PPFD value in µmol/m²/s.
    """
    # Constants
    h = 6.626e-34  # Planck's constant (J.s)
    c = 2.998e8    # Speed of light (m/s)
    Na = 6.022e23   # Avogadro's number

    # Ensure wavelengths are within the PAR range (400-700 nm)
    mask = (wavelengths >= 400) & (wavelengths <= 700)
    wavelengths = wavelengths[mask]
    intensities = intensities[mask]

    # Interpolate SPD data to match the given wavelengths
    interpolated_intensities = np.interp(wavelengths, spd_df['wavelength'], spd_df['intensity'])

    # Calculate photon energy for each wavelength (E = hc/λ)
    photon_energies = (h * c) / (wavelengths * 1e-9)

    # Calculate photon flux density for each wavelength
    photon_flux_densities = intensities * interpolated_intensities / photon_energies

    # Integrate over the PAR range to get PPFD
    ppfd = np.trapz(photon_flux_densities, wavelengths) / Na * 1e6  # Convert to µmol/m²/s

    return ppfd

def rotate_coordinates(x, y, angle_deg):
    """Rotates coordinates (x, y) by a given angle in degrees."""
    angle_rad = np.deg2rad(angle_deg)
    x_rotated = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rotated = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rotated, y_rotated

# --- Main Calculation ---

# Create measurement grid
x_coords = np.linspace(-floor_width / 2, floor_width / 2, 100)
y_coords = np.linspace(-floor_length / 2, floor_length / 2, 100)
X, Y = np.meshgrid(x_coords, y_coords)
ppfd_grid = np.zeros_like(X)

# Flatten the COB layers for easier iteration
all_cob_coords = [coord for layer in cob_layers for coord in layer]

# Rotate COB coordinates by 45 degrees
rotated_cob_coords = [rotate_coordinates(x, y, 45) for x, y in all_cob_coords]

# --- Loop through each measurement point ---
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]
        y = Y[i, j]
        z = measurement_height

        total_ppfd = 0

        # --- Loop through each COB LED ---
        for cob_x, cob_y in rotated_cob_coords:
            # Adjust COB wattage based on layer
            layer_index = next((i for i, layer in enumerate(cob_layers) if (cob_x, cob_y) in layer), -1)
            wattage = 50  # Default for layer 1
            if layer_index == 2:
                wattage = 40
            elif layer_index == 3:
                wattage = 30
            elif layer_index == 4:
                wattage = 20
            elif layer_index == 5:
                wattage = 10

            distance = calculate_distance(cob_x, cob_y, light_height, x, y, z)
            angle = calculate_angle(cob_x, cob_y, light_height, x, y, z)

            # Calculate luminous flux (lumens)
            luminous_flux = wattage * cob_efficiency

            # Convert lumens to candelas using radiation pattern
            candelas = lumens_to_candelas(luminous_flux, angle, radiation_pattern_data)

            # Scale intensity by the cosine of the angle (Lambertian)
            intensity = candelas * np.cos(angle)

            # Convert to spectral radiant flux (W/nm) using SPD
            spectral_radiant_flux = intensity * spd_data['intensity'].values # Scale SPD intensities

            # Calculate PPFD
            ppfd = spd_to_ppfd(spd_data, spd_data['wavelength'].values, spectral_radiant_flux)

            # Add contribution to the total PPFD
            total_ppfd += ppfd

        ppfd_grid[i, j] = total_ppfd

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 8))

# Increase the number of contour levels for a smoother heatmap
heatmap = ax.contourf(X, Y, ppfd_grid, 50, cmap='viridis')

# Add colorbar
cbar = fig.colorbar(heatmap)
cbar.set_label('PPFD (µmol/m²/s)')

# Add COB locations
for cob_x, cob_y in rotated_cob_coords:
    ax.plot(cob_x, cob_y, 'ro', markersize=2)  # 'ro' for red circles

# Set labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('PPFD Distribution')

# Set aspect ratio to equal for a square plot
ax.set_aspect('equal')

plt.show()