import os
import shutil

def move_dwg_files(destination_folder):
    # Get the user's home directory
    home_dir = os.path.expanduser("~")
    
    # Create destination folder if it doesn't exist
    destination_path = os.path.join(home_dir, destination_folder)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Created folder: {destination_path}")
    
    # Walk through all files and directories in the home directory
    for root, dirs, files in os.walk(home_dir):
        for file in files:
            if file.lower().endswith(".tsx"):  # Case-insensitive check for .dwg files
                file_path = os.path.join(root, file)
                try:
                    # Move the file to the destination folder
                    shutil.move(file_path, destination_path)
                    print(f"Moved: {file_path} -> {destination_path}")
                except Exception as e:
                    print(f"Error moving file {file_path}: {e}")

if __name__ == "__main__":
    move_dwg_files("Typescript Files - 1.05.2025")
