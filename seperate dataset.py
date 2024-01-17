import pandas as pd
import os
import shutil

# Load metadata CSV file
metadata_path = 'C:/Users/vansh/Desktop/DEEP/dfdc/dfdc_train_part_46/metadata.csv'
df = pd.read_csv(metadata_path)

# Iterate through rows and move videos
for index, row in df.iterrows():
    filename = row['filename']
    label = row['label']

    # Construct source path and destination directory
    source_path = os.path.join('C:/Users/vansh/Desktop/DEEP/dfdc/dfdc_train_part_46', filename)
    destination_dir = os.path.join('C:/Users/vansh/Desktop/DEEP/dfdc/dfdc_train_part_46', label)

    # Check if the source file exists before moving
    if os.path.exists(source_path):
        # Create destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Move the file
        shutil.move(source_path, destination_dir)
    else:
        print(f"File not found: {source_path}")
