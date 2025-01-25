import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Constants and Parameters
TARGET_PPFD = 600  # μmol/m²/s
MAX_DISTANCE = 2.0  # meters
DISTANCE_POINTS = 100
LIGHT_OUTPUT_UNIT = 1  # Arbitrary units for light output

# Distance array
distances = np.linspace(0.1, MAX_DISTANCE, DISTANCE_POINTS)  # Avoid distance=0

# Non-uniform Lighting (Inverse Square Law)
ppfd_non_uniform = LIGHT_OUTPUT_UNIT / (distances ** 2)

# To achieve target PPFD, calculate required light output
required_light_non_uniform = TARGET_PPFD * (distances ** 2)

# Uniform Lighting
ppfd_uniform = np.full_like(distances, TARGET_PPFD)

# Required light output for uniform lighting
# Assuming PPFD is directly proportional to light output and uniform across canopy
required_light_uniform = TARGET_PPFD * np.ones_like(distances)

# Energy Consumption (Assuming energy ∝ light output)
energy_non_uniform = required_light_non_uniform
energy_uniform = required_light_uniform

# Plot PPFD vs Distance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(distances, ppfd_non_uniform, label='Non-Uniform Lighting')
plt.plot(distances, ppfd_uniform, label='Uniform Lighting', linestyle='--')
plt.axhline(TARGET_PPFD, color='grey', linestyle=':', label='Target PPFD')
plt.xlabel('Distance from Light (m)')
plt.ylabel('PPFD (μmol/m²/s)')
plt.title('PPFD vs. Distance')
plt.legend()
plt.grid(True)

# Plot Energy vs Distance
plt.subplot(1, 2, 2)
plt.plot(distances, energy_non_uniform, label='Non-Uniform Lighting')
plt.plot(distances, energy_uniform, label='Uniform Lighting', linestyle='--')
plt.xlabel('Distance from Light (m)')
plt.ylabel('Required Light Output (Arbitrary Units)')
plt.title('Energy Required vs. Distance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Heatmap of Light Distribution for Both Lighting Types
# Define canopy area
canopy_size = 1.0  # meters
resolution = 100
x = np.linspace(-canopy_size, canopy_size, resolution)
y = np.linspace(-canopy_size, canopy_size, resolution)
X, Y = np.meshgrid(x, y)
distance = np.sqrt(X**2 + Y**2)

# Non-uniform PPFD distribution
ppfd_non_uniform_map = LIGHT_OUTPUT_UNIT / (distance ** 2 + 0.1)  # Add 0.1 to avoid division by zero

# Uniform PPFD distribution
ppfd_uniform_map = np.full_like(ppfd_non_uniform_map, TARGET_PPFD)

# Plot Heatmaps
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Non-Uniform Lighting Heatmap
c1 = axs[0].imshow(ppfd_non_uniform_map, extent=(-canopy_size, canopy_size, -canopy_size, canopy_size),
                   origin='lower', cmap='viridis', norm=colors.LogNorm(vmin=ppfd_non_uniform_map.min(), vmax=ppfd_non_uniform_map.max()))
axs[0].set_title('Non-Uniform Lighting PPFD Distribution')
axs[0].set_xlabel('X (m)')
axs[0].set_ylabel('Y (m)')
fig.colorbar(c1, ax=axs[0], label='PPFD (μmol/m²/s)')

# Uniform Lighting Heatmap
c2 = axs[1].imshow(ppfd_uniform_map, extent=(-canopy_size, canopy_size, -canopy_size, canopy_size),
                   origin='lower', cmap='viridis')
axs[1].set_title('Uniform Lighting PPFD Distribution')
axs[1].set_xlabel('X (m)')
axs[1].set_ylabel('Y (m)')
fig.colorbar(c2, ax=axs[1], label='PPFD (μmol/m²/s)')

plt.tight_layout()
plt.show()

# Summary Plot: Energy Savings by Uniform Lighting
plt.figure(figsize=(8,6))
energy_savings = energy_non_uniform - energy_uniform
plt.plot(distances, energy_savings, label='Energy Savings (Non-Uniform - Uniform)')
plt.fill_between(distances, energy_savings, where=(energy_savings>0), color='green', alpha=0.3)
plt.xlabel('Distance from Light (m)')
plt.ylabel('Energy Savings (Arbitrary Units)')
plt.title('Energy Savings by Using Uniform Lighting')
plt.legend()
plt.grid(True)
plt.show()

# Additional Annotations and Insights
print(f"At a distance of 0.5m:")
print(f" - Required light output (Non-Uniform): {required_light_non_uniform[np.argmin(np.abs(distances-0.5))]:.2f} units")
print(f" - Required light output (Uniform): {required_light_uniform[np.argmin(np.abs(distances-0.5))]:.2f} units")
print(f" - Energy Savings: {energy_non_uniform[np.argmin(np.abs(distances-0.5))] - energy_uniform[np.argmin(np.abs(distances-0.5))]:.2f} units")

print(f"\nAt a distance of 1.0m:")
print(f" - Required light output (Non-Uniform): {required_light_non_uniform[np.argmin(np.abs(distances-1.0))]:.2f} units")
print(f" - Required light output (Uniform): {required_light_uniform[np.argmin(np.abs(distances-1.0))]:.2f} units")
print(f" - Energy Savings: {energy_non_uniform[np.argmin(np.abs(distances-1.0))] - energy_uniform[np.argmin(np.abs(distances-1.0))]:.2f} units")

print(f"\nAt a distance of 1.5m:")
print(f" - Required light output (Non-Uniform): {required_light_non_uniform[np.argmin(np.abs(distances-1.5))]:.2f} units")
print(f" - Required light output (Uniform): {required_light_uniform[np.argmin(np.abs(distances-1.5))]:.2f} units")
print(f" - Energy Savings: {energy_non_uniform[np.argmin(np.abs(distances-1.5))] - energy_uniform[np.argmin(np.abs(distances-1.5))]:.2f} units")
