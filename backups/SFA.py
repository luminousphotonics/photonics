import numpy as np
from scipy.fft import fft2, fftshift
from scipy.signal import detrend
import matplotlib.pyplot as plt

def spatial_frequency_analysis(ppfd_data):
    """
    Performs spatial frequency analysis.  Returns log magnitude spectrum.
    """
    detrended_data = detrend(detrend(ppfd_data, axis=0), axis=1)
    rows, cols = detrended_data.shape
    size = max(rows, cols)
    padded_data = np.zeros((size, size))
    padded_data[:rows, :cols] = detrended_data
    window = np.hanning(size)
    window2D = np.sqrt(np.outer(window, window))
    windowed_data = padded_data * window2D
    fft_result = fft2(windowed_data)
    magnitude_spectrum = np.abs(fftshift(fft_result))
    log_magnitude_spectrum = np.log10(magnitude_spectrum + 1)
    return log_magnitude_spectrum

def radial_profile(data, center=None):
    """Computes the radial profile (average value at each radius) of a 2D array."""
    y, x = np.indices((data.shape))
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


# Novel system LUX data:
novel_data = np.array([
    [44907, 44919, 43785, 43496, 43381, 43461, 43513, 43477, 43871, 43478, 43578, 43839, 44863, 44728],
    [45493, 46364, 45247, 45277, 45275, 45337, 45352, 45336, 45337, 45364, 45345, 45207, 46334, 45561],
    [44664, 45061, 44505, 45160, 45455, 45626, 45582, 45647, 45594, 45526, 45169, 44331, 45215, 44534],
    [44391, 45038, 44984, 46301, 46760, 46894, 46889, 46833, 46881, 46730, 46245, 44980, 45156, 44496],
    [44837, 45386, 44706, 45389, 45654, 45849, 45795, 45733, 45737, 45802, 45344, 44684, 45492, 44850],
    [46284, 46708, 45702, 45620, 45691, 45704, 45657, 45647, 45875, 45818, 45702, 45610, 46805, 45856],
    [45179, 45618, 44639, 44302, 44265, 44234, 44275, 44227, 44082, 44329, 44232, 44578, 45482, 45122]
])

# Conventional system LUX data:
conventional_data = np.array([
   [28370, 37917, 40369, 42096, 43461, 42320, 43442, 43543, 42327, 43438, 41950, 40406, 37649, 28144],
   [34507, 45757, 48715, 50487, 52041, 50695, 52383, 52280, 50688, 52182, 50677, 48639, 45420, 33750],
   [36608, 48605, 51598, 53655, 55492, 54209, 55616, 55636, 54103, 55455, 53823, 52311, 48250, 35928],
   [37083, 49179, 52439, 54520, 56233, 54657, 56440, 56333, 54366, 56277, 54362, 52320, 48735, 36471],
   [36385, 48410, 51624, 53336, 55076, 54086, 55243, 55058, 53576, 55153, 53419, 51418, 47848, 35632],
   [33763, 44818, 47608, 49411, 51211, 49885, 51218, 51221, 49691, 50972, 49350, 47616, 44571, 32981],
   [27138, 36552, 39074, 40386, 41773, 40707, 41861, 41864, 40792, 40767, 40128, 38712, 36181, 26895]
])

# Perform the Analysis
novel_spectrum = spatial_frequency_analysis(novel_data)
conventional_spectrum = spatial_frequency_analysis(conventional_data)

# --- Plotting ---

# 1. Different Colormaps, Same Scale
vmin = min(novel_spectrum.min(), conventional_spectrum.min())
vmax = max(novel_spectrum.max(), conventional_spectrum.max())

fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows, 3 columns

cmaps = ['viridis', 'jet', 'turbo']  # Different colormaps
titles = ['Viridis', 'Jet', 'Turbo']

for i, cmap in enumerate(cmaps):
    im1 = axes[0, i].imshow(novel_spectrum, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, i].set_title(f'Novel ({titles[i]})')
    fig.colorbar(im1, ax=axes[0, i])

    im2 = axes[1, i].imshow(conventional_spectrum, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, i].set_title(f'Conventional ({titles[i]})')
    fig.colorbar(im2, ax=axes[1, i])
plt.suptitle("Spatial Frequency Analysis with Consistent Color Scale", fontsize=16)
plt.tight_layout()
plt.show()

# 2. Difference Plot
difference_spectrum = novel_spectrum - conventional_spectrum

plt.figure(figsize=(8, 6))
plt.imshow(difference_spectrum, cmap='RdBu') # Red-Blue colormap
plt.title('Difference Spectrum (Novel - Conventional)')
plt.colorbar()
plt.tight_layout()
plt.show()

# 3. 1D Radial Profile
novel_radial = radial_profile(novel_spectrum)
conv_radial = radial_profile(conventional_spectrum)

plt.figure(figsize=(8,6))
plt.plot(novel_radial, label='Novel')
plt.plot(conv_radial, label='Conventional')
plt.xlabel('Spatial Frequency (pixels$^{-1}$)')
plt.ylabel('Average Magnitude (Log Scale)')
plt.title('Radial Profile of Magnitude Spectra')
plt.legend()
plt.grid(True)
plt.show()
print("Novel radial profile:", novel_radial)
print("Conventional radial profile:", conv_radial)