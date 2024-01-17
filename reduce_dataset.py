import os
import random
import shutil

# Path to the root directory of the DFDC dataset
dfdc_root = 'dfdc/dfdc_train_part_46'

# List all folders (real and fake) in the root directory
subfolders = [f.path for f in os.scandir(dfdc_root) if f.is_dir()]

# Set the desired size of the reduced dataset
desired_size = 1000

# Create a directory for the reduced dataset
reduced_dir = 'C:/Users/vansh/Desktop/DEEP/dfdc_new'
os.makedirs(reduced_dir, exist_ok=True)

# Iterate through each subfolder (real and fake)
for subfolder in subfolders:
    # List all video files in the current subfolder
    video_files = [f.name for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(('.mp4', '.avi', '.mov'))]

    # Ensure that the desired size is not greater than the available videos
    desired_size_per_class = min(desired_size // len(subfolders), len(video_files))

    # Randomly select files for the reduced dataset
    selected_videos = random.sample(video_files, desired_size_per_class)

    # Create a subdirectory in the reduced dataset for the current class
    class_name = os.path.basename(subfolder)
    class_reduced_dir = os.path.join(reduced_dir, class_name)
    os.makedirs(class_reduced_dir, exist_ok=True)

    # Copy selected files to the reduced dataset directory
    for video_name in selected_videos:
        source_path = os.path.join(subfolder, video_name)
        destination_path = os.path.join(class_reduced_dir, video_name)
        shutil.copyfile(source_path, destination_path)

print("Reduced DFDC Dataset has been created in:", reduced_dir)
