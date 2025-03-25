from PIL import Image
import os
import time

# Set the paths for the image folders
left_folder = "results/EDVR_L_UDC-VIT_patch_1/visualization/REDS4"
right_folder = "results/EDVR_L_UDC-VIT_patch_2/visualization/REDS4"
output_folder = "results/combined_images"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of right image subfolders
right_folders = sorted(os.listdir(right_folder))

# Perform operations for each right image subfolder
for folder_name in right_folders:
    start = time.time()
    print(folder_name)
    # Set the paths for the image subfolders inside the right and left folders
    right_img_folder = os.path.join(right_folder, folder_name)
    left_img_folder = os.path.join(left_folder, folder_name)
   
    # Create a new subfolder in the output folder with the name of each right image subfolder
    output_subfolder = os.path.join(output_folder, folder_name)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)
    
    # List of image files
    left_images = sorted([f for f in os.listdir(left_img_folder) if f.endswith('.png')])
    right_images = sorted([f for f in os.listdir(right_img_folder) if f.endswith('.png')])

    # Assuming the image file names in both folders match
    for left_img_name, right_img_name in zip(left_images, right_images):
        # Open the images
        left_img_path = os.path.join(left_img_folder, left_img_name)
        right_img_path = os.path.join(right_img_folder, right_img_name)

        left_img = Image.open(left_img_path)
        right_img = Image.open(right_img_path)
        
        # Crop the images (crop from the left for the left image, and from the right for the right image)
        left_cropped = left_img.crop((0, 0, 950, 1060))
        right_cropped = right_img.crop((right_img.width - 950, 0, right_img.width, 1060))
        
        # Create a new combined image
        new_img = Image.new('RGB', (1900, 1060))
        new_img.paste(left_cropped, (0, 0))
        new_img.paste(right_cropped, (950, 0))
        
        # Save the new image
        output_path = os.path.join(output_subfolder, left_img_name[:4]+".png")
        new_img.save(output_path)
        
    end = time.time()
    print(end - start)

print("Images combined and saved successfully.")
