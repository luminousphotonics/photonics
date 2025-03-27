#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import matplotlib.pyplot as plt
import warnings

# ---------------------------
# Data: Use the latest known_configs including N=18
known_configs = {
    10: np.array([
        4036.684, 2018.342, 8074.0665, 5045.855, 6055.0856,
        12110.052, 15137.565, 2018.342, 20183.42, 24220.104
    ], dtype=np.float64),
    11: np.array([
        3343.1405, 4298.3235, 4298.9822, 2865.549, 11462.2685,
        17193.294, 7641.464, 2387.9575, 9074.2385, 22406.9105,
        22924.392
    ], dtype=np.float64),
    12: np.array([
        3393.159, 3393.159, 3393.8276, 2908.422, 9694.8,
        17450.532, 11633.688, 2423.685, 9210.003, 9694.74,
        19389.48, 23267.376
    ], dtype=np.float64),
    13: np.array([
        3431.5085, 3431.5085, 6373.4516, 2941.293, 6863.0557,
        12745.603, 12745.603, 13235.8185, 9314.0945, 3921.724,
        3921.724, 23530.344, 23530.344
    ], dtype=np.float64),
    14: np.array([
        3399.7962, 3399.7962, 5344.9613, 5344.2873, 6799.6487,
        7771.0271, 11656.5406, 15053.6553, 15053.6553, 3885.5135,
        3885.5135, 9713.7838, 19427.5677, 23313.0812
    ], dtype=np.float64),
    15: np.array([
        1932.1547, 2415.1934, 4344.4082, 8212.8061, 7728.6775,
        7728.6188, 6762.5415, 12559.0156, 16423.3299, 9660.7733,
        3864.3093, 5796.4640, 7728.6188, 23185.8563, 23185.8563
    ], dtype=np.float64),
    16: np.array([ # Successful N=16 Manual Data
        3801.6215, 3162.5110, 4132.8450, 7000.3495, 9000.9811,
        9000.4963, 7500.8053, 7000.4393, 13000.3296, 16500.3501,
        11000.8911, 4571.2174, 5127.4023, 9924.7413, 16685.5291,
        25000.0000
    ], dtype=np.float64),
    17: np.array([ # Successful N=17 Refined Data (V20)
        4332.1143, 3657.6542, 4017.4842, 5156.7357, 6708.1373,
        8319.1898, 9815.3845, 11114.2396, 12022.3055, 12091.5716,
        10714.1765, 8237.7581, 6442.0864, 6687.0190, 10137.5776,
        16513.6125, 25000.0000
    ], dtype=np.float64),
    18: np.array([ # Successful N=18 Manually Tuned Data
        4372.8801, 3801.7895, 5000.3816, 6500.3919, 8000.5562,
        7691.6100, 8000.2891, 8000.7902, 10000.7478, 13000.3536,
        12700.1186, 9909.0286, 7743.6182, 6354.5384, 6986.5800,
        10529.9883, 16699.2440, 25000.0000
    ], dtype=np.float64),
}

# Spline Parameters (match the predictive tool)
SPLINE_DEGREE = 3
SMOOTHING_FACTOR_MODE = "multiplier"
SMOOTHING_MULTIPLIER = 1.5

# Visualization Parameters
N_VALUES_TO_PLOT = [10, 12, 14, 16, 18, 20] # Layers to show slices for
X_NORM_POINTS = 100 # Number of points for plotting each slice smoothly
OUTPUT_FILENAME = "spline_flux_profiles.png"
# ---------------------------

def prepare_spline_data(known_configs):
    """Prepares data points (N, x_norm, flux)"""
    n_list = []
    x_norm_list = []
    flux_list = []
    min_n, max_n = float('inf'), float('-inf')

    for n, fluxes in known_configs.items():
        if n < 2: continue
        num_layers_in_config = len(fluxes)
        if num_layers_in_config != n:
             warnings.warn(f"Mismatch for N={n}. Using actual count {num_layers_in_config}.", UserWarning)
             n_actual = num_layers_in_config
             if n_actual < 2: continue
        else:
             n_actual = n

        min_n = min(min_n, n_actual)
        max_n = max(max_n, n_actual)
        norm_positions = np.linspace(0, 1, n_actual)
        for i, flux in enumerate(fluxes):
            n_list.append(n_actual)
            x_norm_list.append(norm_positions[i])
            flux_list.append(flux)

    if not n_list:
        raise ValueError("No valid configuration data found.")

    num_data_points = len(flux_list)
    print(f"Spline fitting using data from N={min_n} to N={max_n}. Total points: {num_data_points}")
    return np.array(n_list), np.array(x_norm_list), np.array(flux_list), num_data_points

def fit_spline(n_coords, x_norm_coords, flux_values, num_data_points, k=SPLINE_DEGREE, mode=SMOOTHING_FACTOR_MODE, multiplier=SMOOTHING_MULTIPLIER):
    """Fits the SmoothBivariateSpline"""
    s_value = None
    if mode == "num_points":
        s_value = float(num_data_points)
        print(f"Using explicit spline smoothing factor s = {s_value:.1f}")
    elif mode == "multiplier":
        s_value = float(num_data_points) * multiplier
        print(f"Using explicit spline smoothing factor s = {s_value:.1f} ({multiplier} * num_points)")
    elif isinstance(mode, (int, float)):
        s_value = float(mode)
        print(f"Using explicit spline smoothing factor s = {s_value:.1f}")
    else:
        print("Using default spline smoothing factor s (estimated by FITPACK)")

    unique_n = len(np.unique(n_coords))
    unique_x = len(np.unique(x_norm_coords))
    if unique_n <= k or unique_x <= k:
         k_eff = min(unique_n - 1, unique_x - 1, k)
         k_eff = max(1, k_eff)
         warnings.warn(f"Insufficient unique coordinates for spline degree {k}. Reducing degree to {k_eff}.", UserWarning)
         k = k_eff

    try:
        with warnings.catch_warnings():
             # Suppress common FITPACK warnings when s is provided
             warnings.filterwarnings("ignore", message="The required storage space", category=UserWarning)
             warnings.filterwarnings("ignore", message="The number of knots required", category=UserWarning)
             warnings.filterwarnings("ignore", message="Warning: s=", category=UserWarning)
             warnings.filterwarnings("ignore", message="Warning: ier=.*", category=UserWarning)
             spline = SmoothBivariateSpline(n_coords, x_norm_coords, flux_values, kx=k, ky=k, s=s_value)
             print("Spline fitting complete.")
             return spline
    except Exception as e:
         print(f"\nError fitting spline: {e}")
         raise

def plot_spline_slices(spline, n_values, x_points, filename):
    """Plots Flux vs x_norm for different N values"""
    plt.figure(figsize=(10, 6))
    x_norm_grid = np.linspace(0, 1, x_points)

    for n in n_values:
        n_array = np.full_like(x_norm_grid, n)
        # Evaluate the spline for this slice
        flux_predicted = spline(n_array, x_norm_grid, grid=False)
        # Ensure non-negative for plotting
        flux_predicted = np.maximum(flux_predicted, 0)
        plt.plot(x_norm_grid, flux_predicted, label=f'N = {n}')

    plt.xlabel("Normalized Layer Position (0=Center, 1=Outer)")
    plt.ylabel("Predicted Luminous Flux (Unscaled Spline Shape)")
    plt.title("Spline-Predicted Flux Profile Shape vs. Number of Layers (N)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0

    # Add annotation about the data source
    min_n_data = min(known_configs.keys())
    max_n_data = max(known_configs.keys())
    plt.text(0.98, 0.02, f"Spline based on data N={min_n_data}-{max_n_data}",
             horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gca().transAxes, fontsize=8, color='gray')


    try:
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()


def main():
    print("--- Visualizing the Learned Spline Shape ---")
    try:
        n_coords, x_norm_coords, flux_values, num_points = prepare_spline_data(known_configs)
        spline_model = fit_spline(n_coords, x_norm_coords, flux_values, num_points)
        plot_spline_slices(spline_model, N_VALUES_TO_PLOT, X_NORM_POINTS, OUTPUT_FILENAME)

    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()