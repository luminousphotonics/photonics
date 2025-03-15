import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np  # Import numpy

def visualize_cob_overlap_geometric(pattern_option, light_array_width_ft, light_array_height_ft, cob_beam_angle_degrees=120):
    """
    Generates a geometric visualization of COB LED arrangement with beam overlap,
    emphasizing the interaction between neighboring COBs.

    Args:
        pattern_option: The pattern string (e.g., "Diamond: 61").
        light_array_width_ft: Width of the LED array in feet.
        light_array_height_ft: Height of the LED array in feet.
        cob_beam_angle_degrees: Beam angle of each COB LED in degrees.
    """

    center_x = light_array_width_ft / 2
    center_y = light_array_height_ft / 2
    center_source = (center_x, center_y)

    if pattern_option == "Diamond: 61":
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
    else:
        print("Invalid pattern option.")
        return

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # --- Calculate beam radius and intersection points ---
    beam_radius = max(spacing_x, spacing_y) * 1.2 * (cob_beam_angle_degrees / 120.0)

    for i, (x1, y1) in enumerate(light_sources):
        cob_circle = patches.Circle((x1, y1), radius=0.05, color='black', zorder=3)
        ax.add_patch(cob_circle)

        # Draw the beam circle with a *thicker* line and *no fill*
        beam_circle = patches.Circle((x1, y1), radius=beam_radius, color='blue', alpha=1.0, zorder=1, linewidth=2, fill=False) # No fill
        ax.add_patch(beam_circle)

        # --- Find and draw intersections ---
        for j, (x2, y2) in enumerate(light_sources):
            if i == j:  # Don't compare a circle to itself
                continue

            # Calculate distance between circle centers
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Check for intersection
            if distance < 2 * beam_radius and distance > 1e-6: # Check if distance is less than sum of radii, avoiding division by zero

                # Calculate intersection points (using numpy for vector operations)
                a = (beam_radius**2 - beam_radius**2 + distance**2) / (2 * distance)
                h = math.sqrt(beam_radius**2 - a**2)
                x_mid = x1 + a * (x2 - x1) / distance
                y_mid = y1 + a * (y2 - y1) / distance

                ix1 = x_mid + h * (y2 - y1) / distance
                iy1 = y_mid - h * (x2 - x1) / distance
                ix2 = x_mid - h * (y2 - y1) / distance
                iy2 = y_mid + h * (x2 - x1) / distance

                # Draw small red circles at the intersection points
                intersection1 = patches.Circle((ix1, iy1), radius=0.08, color='red', zorder=2)
                intersection2 = patches.Circle((ix2, iy2), radius=0.08, color='red', zorder=2)
                ax.add_patch(intersection1)
                ax.add_patch(intersection2)

    ax.set_xlim(0, light_array_width_ft)
    ax.set_ylim(0, light_array_height_ft)
    ax.set_xlabel("Width (ft)")
    ax.set_ylabel("Height (ft)")
    ax.set_title(f"{pattern_option} COB LED Arrangement - Geometric Overlap")
    plt.grid(True)
    plt.show()

# Example Usage
pattern = "Diamond: 61"
width = 10
height = 10
beam_angle = 120

visualize_cob_overlap_geometric(pattern, width, height, beam_angle)