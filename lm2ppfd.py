import numpy as np
import matplotlib.pyplot as plt

# --- Data from Figure 13 (3000K curve, approximate) ---
wavelengths = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780])
relative_intensities_3000K = np.array([1, 3, 6, 10, 20, 53, 44, 30, 27, 30, 34, 39, 43, 47, 51, 55, 60, 65, 70, 76, 84, 90, 95, 98, 100, 98, 93, 85, 75, 60, 45, 34, 25, 19, 14, 10, 7, 5, 2])
relative_intensities_3000K = relative_intensities_3000K / max(relative_intensities_3000K)  # Normalize to maximum intensity

# --- Photopic Luminous Efficiency Function (Approximation) ---
photopic_x = [380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 507, 510, 520, 530, 540, 550, 555, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780]
photopic_y = [0.0001, 0.0004, 0.0012, 0.0040, 0.0116, 0.023, 0.038, 0.060, 0.091, 0.139, 0.208, 0.323, 0.503, 0.631, 0.710, 0.862, 0.954, 0.995, 1.000, 0.995, 0.952, 0.870, 0.757, 0.631, 0.503, 0.381, 0.265, 0.175, 0.107, 0.061, 0.032, 0.017, 0.0082, 0.0041, 0.0021, 0.0010, 0.0005, 0.0002, 0.0001, 0, 0, 0, 0]

# --- Estimate PPFD from Relative Intensities---
ppfd_total = 0
PAR_x = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700]
for i in range(len(PAR_x)):
    wavelength = PAR_x[i]
    if wavelength in wavelengths:
        # Find the index of the wavelength in the 3000K data
        idx = np.where(wavelengths == wavelength)[0][0]
        relative_intensity = relative_intensities_3000K[idx]

        # Convert relative intensity to photon flux (simplified conversion)
        # Assuming a linear relationship between relative intensity and PPFD within the PAR range
        photon_flux = relative_intensity
        ppfd_total += photon_flux


# --- Estimate Lumens ---
lumens_total = 0
for i in range(len(photopic_x)):
    wavelength = photopic_x[i]
    if wavelength in wavelengths:
        # Find the index of the wavelength in the 3000K data
        idx = np.where(wavelengths == wavelength)[0][0]
        relative_intensity = relative_intensities_3000K[idx]

        # Weight by the photopic luminous efficiency function
        weighted_intensity = relative_intensity * photopic_y[i]
        lumens_total += weighted_intensity

lumens_total *= 13500  # Scale by the maximum luminous output

# --- Calculate Conversion Factor ---
conversion_factor = ppfd_total / lumens_total

print(f"Estimated PPFD to Lumens conversion factor (3000K): {conversion_factor:.4f}")

# --- Plotting for visualization (optional) ---
plt.figure(figsize=(10, 5))

# Plot SPD
plt.subplot(1, 2, 1)
plt.plot(wavelengths, relative_intensities_3000K)
plt.title("Spectral Power Distribution (3000K)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative Intensity")

# Plot Photopic curve
plt.subplot(1, 2, 2)
plt.plot(photopic_x, photopic_y)
plt.title("Photopic Luminous Efficiency Function")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Luminous Efficiency")
plt.tight_layout()
plt.show()