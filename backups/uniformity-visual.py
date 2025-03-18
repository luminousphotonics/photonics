import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

def visualize_beam_overlap_colored_physics(pattern_option, light_array_width_ft, light_array_height_ft,
                                            beam_spread_ft, light_distance_ft):
    """
    Creates a 2D scientific visualization of beam overlap with physically motivated color mapping.
    
    For each LED in the "Diamond: 61" pattern, we assign a relative luminous flux (layer intensity).
    Then, for each LED's floor point (directly beneath the LED), we compute the cumulative PPFD
    (Photosynthetic Photon Flux Density) from all LEDs, using:
    
        PPFD = sum ( intensity_j / ( (d_j)^2 ) )
        
    where d_j is the 3D distance from LED j to the floor point, and light_distance_ft is constant.
    
    The jet colormap is applied to these cumulative PPFD values so that regions with higher overlap
    (center of the illuminated plane) get colors like red, and regions with lower overlap (edges) get
    cooler colors.
    """
    # Beam angle for reference (not used for color mapping here)
    beam_angle_rad = 2 * math.atan(beam_spread_ft / light_distance_ft)
    beam_angle_deg = math.degrees(beam_angle_rad)
    print(f"Beam Angle: {beam_angle_deg:.2f}°")
    
    # Setup COB LED positions and relative luminous flux (layer intensities) using "Diamond: 61"
    center_x = light_array_width_ft / 2
    center_y = light_array_height_ft / 2
    spacing_x = light_array_width_ft / 7.2
    spacing_y = light_array_height_ft / 7.2
    
    if pattern_option == "Diamond: 61":
        layers = [
            [(0, 0)],
            [(-1, 0), (1, 0), (0, -1), (0, 1)],
            [(-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)],
            [(-2, -1), (2, -1), (-2, 1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2),
             (-3, 0), (3, 0), (0, -3), (0, 3)],
            [(-2, -2), (2, -2), (-2, 2), (2, 2), (-3, -1), (3, -1), (-3, 1), (3, 1),
             (-1, -3), (1, -3), (-1, 3), (1, 3), (-4, 0), (4, 0), (0, -4), (0, 4)],
            [(-3, -2), (3, -2), (-3, 2), (3, 2), (-2, -3), (2, -3), (-2, 3), (2, 3),
             (-4, -1), (4, -1), (-4, 1), (4, 1), (-1, -4), (1, -4), (-1, 4), (1, 4),
             (-5, 0), (5, 0), (0, -5), (0, 5)]
        ]
        # Relative luminous flux (lower for inner layers, higher for outer layers)
        layer_intensities = [0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        light_sources = []
        intensity_map = {}
        for layer_index, layer in enumerate(layers):
            for dot in layer:
                # Rotate offset by 45° for the centered square pattern.
                x_offset = spacing_x * dot[0]
                y_offset = spacing_y * dot[1]
                theta_rot = math.radians(45)
                rotated_x = x_offset * math.cos(theta_rot) - y_offset * math.sin(theta_rot)
                rotated_y = x_offset * math.sin(theta_rot) + y_offset * math.cos(theta_rot)
                pos = (center_x + rotated_x, center_y + rotated_y)
                light_sources.append(pos)
                intensity_map[pos] = layer_intensities[layer_index]
    else:
        print("Invalid pattern option.")
        return

    # Compute cumulative PPFD at the floor point beneath each LED.
    # For each LED i at position (x_i, y_i), sum contributions from all LEDs j:
    #   contribution = intensity_j / ( (sqrt((x_i - x_j)^2 + (y_i - y_j)^2 + light_distance_ft^2))^2 )
    cumulative_ppfd = {}
    for pos_i in light_sources:
        total = 0
        for pos_j in light_sources:
            dx = pos_i[0] - pos_j[0]
            dy = pos_i[1] - pos_j[1]
            d = math.sqrt(dx**2 + dy**2 + light_distance_ft**2)
            # Using inverse square law
            total += intensity_map[pos_j] / (d**2)
        cumulative_ppfd[pos_i] = total

    # Normalize cumulative PPFD values for colormap mapping.
    ppfd_values = list(cumulative_ppfd.values())
    norm = Normalize(vmin=min(ppfd_values), vmax=max(ppfd_values))
    cmap = cm.get_cmap('jet')
    
    # Create the 2D plot.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, light_array_width_ft)
    ax.set_ylim(0, light_array_height_ft)
    ax.set_aspect('equal')
    ax.set_title(f"{pattern_option} COB LED - Geometric Beam Overlap\nBeam Angle: {beam_angle_deg:.1f}°", fontsize=14)
    ax.set_xlabel("Width (ft)")
    ax.set_ylabel("Height (ft)")
    
    # Draw each beam as a circle outline.
    # The outline color is determined by the normalized cumulative PPFD at that LED’s floor point.
    for pos in light_sources:
        color = cmap(norm(cumulative_ppfd[pos]))
        beam_circle = patches.Circle(pos, beam_spread_ft, facecolor='none', 
                                       edgecolor=color, linestyle='dashed', linewidth=2)
        ax.add_patch(beam_circle)
    
    # Mark the COB LED positions as small black dots.
    for pos in light_sources:
        ax.plot(pos[0], pos[1], marker='o', markersize=4, color='black')
    
    plt.show()

# Example usage
pattern = "Diamond: 61"
width = 10          # Illuminated area width (ft)
height = 10         # Illuminated area height (ft)
beam_spread = 3     # Beam spread at 50% peak intensity (ft)
light_distance = 3  # Distance from LED to illuminated surface (ft)

visualize_beam_overlap_colored_physics(pattern, width, height, beam_spread, light_distance)
