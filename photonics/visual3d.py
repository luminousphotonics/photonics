import pyvista as pv
import numpy as np
from typing import List, Dict, Any
from main.dataclass_models import Fixture  # Import Fixture class
import os  # Import the os module
from django.conf import settings  # Import Django settings

# Function to calculate the number of layers based on floor dimensions
def calculate_layers(width: float, length: float) -> int:
    layer_dimensions = [
        {"width": 0, "length": 0},
        {"width": 3, "length": 3},
        {"width": 5, "length": 5},
        {"width": 8, "length": 8},
        {"width": 11, "length": 11},
        {"width": 14, "length": 14},
    ]

    fitted_layers = 0
    for i in range(1, len(layer_dimensions)):
        if width >= layer_dimensions[i]["width"] and length >= layer_dimensions[i]["length"]:
            fitted_layers = i
        else:
            break
    return fitted_layers

# Function to determine the number of fixtures in a layer
def determine_fixtures_in_layer(layer: int, selected_layers: int) -> int:
    if layer >= selected_layers:
        return 0  # Only render up to selected_layers

    fixture_count_per_layer = {
        0: 1,
        1: 2,
        2: 4,
        3: 4,
        4: 6,
    }

    return fixture_count_per_layer.get(layer, 0)


# Function to determine fixture placement based on layer and fixture index
def determine_fixture_placement(layer: int, fixture_index: int, center_x: float, center_y: float, base_layer_spacing: float) -> Fixture:
    # Define the base directory for your models using settings.BASE_DIR
    base_dir = os.path.join(settings.BASE_DIR, 'main', 'static', 'models')

    # Define refined layer-specific spacing multipliers for closer control
    spacing_multipliers = {
        0: 0,       # Center unit
        1: 0,       # Layer 2
        2: 0.75,    # Layer 3
        3: 0.37,    # Layer 4
        4: 0.37,    # Layer 5
    }

    # Calculate spacing for the current layer
    layer_spacing = base_layer_spacing * spacing_multipliers.get(layer, 1.0)

    if layer == 0:
        # Layer 1: Center Unit
        return Fixture(
            file_path=os.path.join(base_dir, "modeltest2.obj"),  # Absolute path
            type="X-Unit",
            layer=layer,
            position={"x": center_x + layer_spacing, "y": center_y - 0.5 * layer_spacing, "z": 1.5},
            rotation={"x": 90, "y": 0, "z": 0},
        )
    elif layer == 1:
        # Layer 2: 2 Fixtures (L-Shaped)
        if fixture_index == 0:  # Bottom-left
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest5.obj"),  # Absolute path
                type="L-Shaped",
                layer=layer,
                position={"x": center_x + layer_spacing - 0.35, "y": center_y * layer_spacing * 0 - 10, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 90},
            )
        elif fixture_index == 1:  # Top-right
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest5.obj"),  # Absolute path
                type="L-Shaped",
                layer=layer,
                position={"x": center_x - layer_spacing - 0.35, "y": center_y * layer_spacing * 0 + 25, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 270},
            )
    elif layer == 2:
        # Layer 3: 4 Fixtures (Linear Units)
        if fixture_index == 0:  # Top
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest4.obj"),  # Absolute path
                type="Linear",
                layer=layer,
                position={"x": center_x * layer_spacing * - 0.02, "y": center_y + layer_spacing * 1, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 90},
            )
        elif fixture_index == 1:  # Bottom
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest4.obj"),  # Absolute path
                type="Linear",
                layer=layer,
                position={"x": center_x * layer_spacing / 16, "y": center_y - 0.95 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 90},
            )
        elif fixture_index == 2:  # Left
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest14.obj"),  # Absolute path
                type="Linear",
                layer=layer,
                position={"x": center_x - 0.9 * layer_spacing, "y": center_y / layer_spacing - 10.5, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 90},
            )
        elif fixture_index == 3:  # Right
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest14.obj"),  # Absolute path
                type="Linear",
                layer=layer,
                position={"x": center_x + 0.9 * layer_spacing, "y": center_y * layer_spacing * 0 + 28, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 90},
            )
    elif layer == 3:
        # Layer 4: 4 Fixtures (Reverse L-Shaped)
        if fixture_index == 0:  # Bottom-right
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest11.obj"),  # Absolute path
                type="ReverseL-Shaped",
                layer=layer,
                position={"x": center_x * layer_spacing - 113, "y": center_y - 1.3 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 180},
            )
        elif fixture_index == 1:  # Top-right
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest11.obj"),  # Absolute path
                type="ReverseL-Shaped",
                layer=layer,
                position={"x": center_x + 1.3 * layer_spacing, "y": center_y + 2.1 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 270},
            )
        elif fixture_index == 2:  # Top-left
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest12.obj"),  # Absolute path
                type="ReverseL-Shaped",
                layer=layer,
                position={"x": center_x - 1.8 * layer_spacing, "y": center_y + 1.6 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 270},
            )
        elif fixture_index == 3:  # Bottom-left
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest13.obj"),  # Absolute path
                type="ReverseL-Shaped",
                layer=layer,
                position={"x": center_x - 1.3 * layer_spacing, "y": center_y - 1.9 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 270},
            )
    elif layer == 4:
        # Layer 5: 6 Fixtures (Various Types)
        if fixture_index == 0:  # Top-left
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest12.obj"),  # Absolute path
                type="ReverseL-Shaped",
                layer=layer,
                position={"x": center_x - 2.5 * layer_spacing, "y": center_y + 2.3 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 270},
            )
        elif fixture_index == 1:  # Top-left
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest4.obj"),  # Absolute path
                type="Linear",
                layer=layer,
                position={"x": center_x + 0.8 * layer_spacing, "y": center_y + 3.2 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 90},
            )
        elif fixture_index == 2:  # Top-right (Linear)
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest4.obj"),  # Absolute path
                type="Linear",
                layer=layer,
                position={"x": center_x + 3.1 * layer_spacing, "y": center_y + 2.3 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 180},
            )
        elif fixture_index == 3:  # Bottom-right (linear)
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest14.obj"),  # Absolute path
                type="Linear",
                layer=layer,
                position={"x": center_x + 3.1 * layer_spacing, "y": center_y - 1.7 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 90},
            )
        elif fixture_index == 4:  # bottom
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest14.obj"),  # Absolute path
                type="Linear",
                layer=layer,
                position={"x": center_x + 0.2 * layer_spacing, "y": center_y - 3.0 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 0},
            )
        elif fixture_index == 5:  # Bottom-left
            return Fixture(
                file_path=os.path.join(base_dir, "modeltest5.obj"),  # Absolute path
                type="L-Shaped",
                layer=layer,
                position={"x": center_x - 2.5 * layer_spacing, "y": center_y - 1.9 * layer_spacing, "z": 1.5},
                rotation={"x": 90, "y": 0, "z": 0},
            )
    else:
        # Handle additional layers if necessary
        return None

# Function to generate all fixture definitions based on selected layers
def generate_all_fixtures(floor_width: float, floor_length: float, center_x: float = 5.0, center_y: float = 5.0, layer_spacing: float = 3.0) -> List[Fixture]:
    selected_layers = calculate_layers(floor_width, floor_length)
    fixture_definitions = []

    for layer in range(selected_layers):
        num_fixtures = determine_fixtures_in_layer(layer, selected_layers)
        for fixture_index in range(num_fixtures):
            fixture = determine_fixture_placement(layer, fixture_index, center_x, center_y, layer_spacing)
            if fixture:
                fixture_definitions.append(fixture)
    
    return fixture_definitions


# Function to load a single OBJ model
def load_3d_model(file_path: str) -> pv.PolyData:
    try:
        model = pv.read(file_path)
        print(f"Successfully loaded model: {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model '{file_path}': {e}")
        return None

# Function to create a plant canopy matching the heatmap size and centered
def create_centered_plant_canopy(center_x: float, center_y: float, width: float, length: float, canopy_height: float) -> pv.PolyData:
    canopy = pv.Plane(
        center=(center_x, center_y, canopy_height),
        direction=(0, 0, 1),
        i_size=width,
        j_size=length
    )
    return canopy

# Function to populate the canopy with plant models, spacing, correct orientation, and scaling
def populate_canopy_with_realistic_plants(center_x: float, center_y: float, width: float, length: float, canopy_height: float, grid_size: tuple, plant_model_path: str, plant_spacing: float = 1.0, plant_scale: float = 1.0) -> List[pv.PolyData]:
    plant_models = []

    # Calculate the total width and height of the grid
    grid_width = (grid_size[1] - 1) * plant_spacing
    grid_height = (grid_size[0] - 1) * plant_spacing

    # Calculate the starting position for x and y to center the grid
    x_start = center_x - (grid_width / 2)
    y_start = center_y - (grid_height / 2)

    # Generate plant positions based on grid size and spacing
    for i in range(grid_size[0]):  # Iterate over rows (y-direction)
        for j in range(grid_size[1]):  # Iterate over columns (x-direction)
            # Calculate the x and y coordinates for each plant
            x = x_start + j * plant_spacing
            y = y_start + i * plant_spacing

            # Load a realistic plant model
            plant = load_3d_model(plant_model_path)
            if plant:
                # Scale the plant model
                plant.scale(plant_scale, inplace=True)

                # Rotate the plant to stand upright
                plant.rotate_x(90, inplace=True)

                # Position each plant at the calculated grid point
                plant.translate([x, y, canopy_height], inplace=True)
                plant_models.append(plant)

    return plant_models

# Updated function to load and arrange models
def load_and_arrange_models(fixture_definitions: List[Fixture], lighting_height: float) -> List[pv.PolyData]:
    models = []
    for idx, fixture in enumerate(fixture_definitions):
        model = load_3d_model(fixture.file_path)
        if model:
            print(f"\nProcessing fixture {idx}: {fixture.file_path} - {fixture.type}")
            
            # Rotate the model to correct orientation
            model.rotate_x(fixture.rotation["x"], inplace=True)
            model.rotate_y(fixture.rotation["y"], inplace=True)
            model.rotate_z(fixture.rotation["z"], inplace=True)
            print(f"Model {idx} bounds after rotation: {model.bounds}")
            
            # Center the model
            model_center = np.array(model.center)
            model.translate(-model_center, inplace=True)
            print(f"Model {idx} center after centering: {model.center}")
            
            # Define offsets based on fixture position
            x_offset = fixture.position["x"]
            y_offset = fixture.position["y"]
            z_offset = fixture.position["z"] + lighting_height  # Add lighting height
            print(f"Model {idx} offsets: x={x_offset}, y={y_offset}, z={z_offset}")
            
            # Translate the model to its designated position
            model.translate([x_offset, y_offset, z_offset], inplace=True)
            print(f"Model {idx} center after translation: {model.center}")
            
            models.append(model)
    return models

# Function to generate heatmap data
def generate_heatmap(grid_size: tuple = (50, 50), intensity_func: Any = None):
    x = np.linspace(0, 10, grid_size[0])  # Span 10 units
    y = np.linspace(0, 10, grid_size[1])
    xx, yy = np.meshgrid(x, y)

    if intensity_func:
        intensity = intensity_func(xx, yy)
    else:
        # Gaussian peak at center
        intensity = np.exp(-((xx - 5)**2 + (yy - 5)**2) / 10)

    # Normalize intensity to [0, 1]
    intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    return xx, yy, intensity

# Function to create heatmap plane with dynamic sizing
def create_heatmap_plane(grid_size: tuple = (50, 50), heatmap: np.ndarray = None, bounds: List[float] = None) -> pv.PolyData:
    # Define physical size based on grid_size
    # For example, each grid cell is 0.4 units in size
    cell_size = 5.0  # Adjust as needed for heatmap size
    plane_size_x = grid_size[0] * cell_size
    plane_size_y = grid_size[1] * cell_size
    
    # Create a plane with dynamic physical size and resolution
    plane = pv.Plane(
        i_size=plane_size_x, 
        j_size=plane_size_y, 
        i_resolution=grid_size[0]-1, 
        j_resolution=grid_size[1]-1
    )

    # Position the plane at z=0
    if bounds:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        plane.scale([x_max - x_min, y_max - y_min, 1.0])
        plane.translate([(x_max + x_min) / 2, (y_max + y_min) / 2, z_min])
    else:
        # Default bounds if none provided
        plane.translate([0, 0, 0])

    if heatmap is not None:
        plane['heatmap'] = heatmap.ravel()

    print(f"Heatmap plane bounds: {plane.bounds}")
    print(f"Heatmap plane center: {plane.center}")

    return plane

def create_wireframe(bounds: List[float]) -> pv.PolyData:
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    cube = pv.Box(bounds=bounds)
    wireframe = cube.extract_all_edges()  # Corrected method
    return wireframe

def set_camera_position(plotter: pv.Plotter, bounds: List[float], center_x: float, center_y: float):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    camera_distance = max(x_max - x_min, y_max - y_min) * 1.5
    camera_position = [
        (center_x + camera_distance, center_y + camera_distance, camera_distance),
        (center_x, center_y, 0),
        (0, 0, 1)
    ]
    plotter.camera_position = camera_position
    print(f"Camera position set to: {plotter.camera_position}")


def validate_fixtures(fixture_definitions: List[Fixture], heatmap_bounds: List[float]) -> bool:
    x_min, x_max, y_min, y_max, z_min, z_max = heatmap_bounds
    all_valid = True
    for fixture in fixture_definitions:
        x = fixture.position["x"]
        y = fixture.position["y"]
        if not (x_min <= x <= x_max) or not (y_min <= y <= y_max):
            print(f"Warning: Fixture {fixture.type} at ({x}, {y}) is outside heatmap bounds.")
            all_valid = False
    return all_valid


# Updated visualization function with a perspective box workaround
def visualize_heatmap_with_realistic_plants(
    heatmap_plane: pv.PolyData,
    models: List[pv.PolyData],
    heatmap_bounds: List[float],
    center_x: float,
    center_y: float,
    canopy_height: float,
    plant_models: List[pv.PolyData],
    colormap: str = "viridis"
):
    plotter = pv.Plotter()

    # Add the heatmap plane
    plotter.add_mesh(heatmap_plane, scalars="heatmap", cmap=colormap, opacity=0.8, show_edges=False)

    # Add all lighting models
    for model in models:
        plotter.add_mesh(model, color="white", opacity=0.9, show_edges=True)

    # Add the plant canopy - centered to match heatmap
    plant_canopy = create_centered_plant_canopy(center_x, center_y, heatmap_bounds[1]-heatmap_bounds[0], heatmap_bounds[3]-heatmap_bounds[2], canopy_height)
    plotter.add_mesh(plant_canopy, color="green", opacity=0.2)

    # Add plant actors to the plotter but keep a reference
    plant_actors = []
    for plant in plant_models:
        actor = plotter.add_mesh(plant, color="green", opacity=1.0)
        plant_actors.append(actor)
        actor.SetVisibility(False)  # Initially hide the plants

    # Checkbox callback function
    def toggle_plants(state):
        for actor in plant_actors:
            actor.SetVisibility(state)
        plotter.render()

    # Add the checkbox
    plotter.add_checkbox_button_widget(toggle_plants, value=False, position=(10, 150), size=30, border_size=3)

    # Camera views
    def set_camera_view(view_name):
        if view_name == "Top":
            plotter.view_yz(True)
            plotter.camera_position = [(center_x, center_y, 1), (center_x, center_y, 0), (0, 1, 0)]
            plotter.camera.roll_angle = 0  # Reset roll angle for top view
        elif view_name == "Side":
            plotter.view_xz(True)
            plotter.camera_position = [(center_x, heatmap_bounds[3], 0), (center_x, center_y, 0), (0, 0, 1)]
            plotter.camera.roll_angle = 0
        elif view_name == "Front":
            plotter.view_xy(True)
            plotter.camera_position = [(heatmap_bounds[1], center_y, 0), (center_x, center_y, 0), (0, 0, 1)]
            plotter.camera.roll_angle = 0
        elif view_name == "Diagonal":
            plotter.camera_position = plotter.get_default_camera_position()

        plotter.reset_camera()
        plotter.render()

    # Add buttons for each view using add_box_widget
    button_positions = [(10, 100), (10, 70), (10, 40), (10, 10)]
    button_labels = ["Top", "Side", "Front", "Diagonal"]
    button_sizes = [30, 30, 30, 30]  # Corrected: single integer value for width
    view_names = ["Top", "Side", "Front", "Diagonal"]

    for i, (pos, label, size, view_name) in enumerate(zip(button_positions, button_labels, button_sizes, view_names)):
        def create_callback(view_name):
            def callback(state):
                set_camera_view(view_name)
            return callback

        plotter.add_checkbox_button_widget(
            callback=create_callback(view_name),
            value=False,  # Initial state: unchecked
            position=pos,
            size=size,  # Corrected: single integer value
            border_size=3,
            color_on='lightgray',  # Color when checked
            color_off='white',  # Color when unchecked
            background_color='gray'
        )

    # Add scalar bar for heatmap
    #plotter.add_scalar_bar(title="Heatmap Intensity", n_labels=5, shadow=True, fmt="%.2f")

    # Add axes and bounds
    plotter.add_axes()
    