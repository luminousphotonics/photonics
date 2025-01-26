import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def fixed_frame_arrangement(room_width, room_length, fixture_size=3.5):
    """Simulate fixed-frame lighting fixture placement challenges"""
    plt.figure(figsize=(15, 6))
    
    # Fixed Frame Arrangement Subplot
    plt.subplot(1, 2, 1)
    plt.title(f"Fixed Frame: {room_width}'x{room_length}'")
    plt.xlim(0, room_width)
    plt.ylim(0, room_length)
    plt.xlabel("Room Width (ft)")
    plt.ylabel("Room Length (ft)")
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Calculate and draw fixtures
    fixtures_x = int(np.floor(room_width / fixture_size))
    fixtures_y = int(np.floor(room_length / fixture_size))
    
    ax = plt.gca()
    for x in range(fixtures_x):
        for y in range(fixtures_y):
            rect_x = x * fixture_size
            rect_y = y * fixture_size
            ax.add_patch(patches.Rectangle((rect_x, rect_y), fixture_size, fixture_size, 
                                         fill=False, edgecolor='red', linewidth=2))
    
    # Centered Square Sequence Subplot
    plt.subplot(1, 2, 2)
    plt.title(f"Centered Square Sequence: {room_width}'x{room_length}'")
    plt.xlim(0, room_width)
    plt.ylim(0, room_length)
    plt.xlabel("Room Width (ft)")
    plt.ylabel("Room Length (ft)")
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Draw centered square sequence points
    center_x, center_y = room_width/2, room_length/2
    layers = [1, 4, 8, 12, 16, 20, 24]
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    
    for layer_idx, point_count in enumerate(layers):
        radius = layer_idx + 1
        coords = [
            (0, radius), (0, -radius), 
            (radius, 0), (-radius, 0),
            (radius, radius), (radius, -radius),
            (-radius, radius), (-radius, -radius)
        ]
        
        for x, y in coords:
            plt.scatter(center_x + x, center_y + y, 
                        color=colors[layer_idx], s=100)
    
    plt.tight_layout()
    plt.show()

# Test with different room sizes
room_sizes = [(12, 12), (16, 16), (20, 20)]
for width, length in room_sizes:
    fixed_frame_arrangement(width, length)