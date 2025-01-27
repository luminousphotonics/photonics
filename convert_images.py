import os
from PIL import Image

def convert_to_webp(image_path, output_folder, quality=80):
    """
    Converts an image to WebP format and saves it to the specified output folder.
    
    :param image_path: Path to the original image.
    :param output_folder: Folder to save the WebP image.
    :param quality: Quality of the WebP image (1-100).
    :return: Path to the converted WebP image.
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate the output file path
        file_name = os.path.splitext(os.path.basename(image_path))[0] + ".webp"
        output_path = os.path.join(output_folder, file_name)
        
        # Convert and save the image in WebP format
        img.save(output_path, "WEBP", quality=quality)
        
        print(f"Converted {image_path} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error converting {image_path} to WebP: {e}")
        return None

def convert_images_in_folder(input_folder, output_folder):
    """
    Converts all images in the input folder to WebP format and saves them to the output folder.
    
    :param input_folder: Folder containing the original images.
    :param output_folder: Folder to save the WebP images.
    """
    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg')

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(supported_formats):
            image_path = os.path.join(input_folder, file_name)
            convert_to_webp(image_path, output_folder)

if __name__ == "__main__":
    # Define your input and output folders
    input_folder = "/Users/austinrouse/photonics/static/main/images"
    output_folder = "/Users/austinrouse/photonics/static/main/images/webp"

    # Convert images
    convert_images_in_folder(input_folder, output_folder)