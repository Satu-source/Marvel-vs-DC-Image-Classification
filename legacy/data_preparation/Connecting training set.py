# ARCHIVAL SOURCE (2024). Absolute local file-system paths replaced with a placeholder.
# Set REPLACE_WITH_PROJECT_ROOT to a local project directory before attempting to run.
import os
import shutil

def organize_images(source_dir, target_base_dir, class_name):
    # Create class subdirectory in target directory if it doesn't exist
    target_dir = os.path.join(target_base_dir, class_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Move images to the class subdirectory
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isfile(item_path):
            shutil.copy(item_path, os.path.join(target_dir, item))  # Copy images to avoid data loss

# Define source directories
source_directories = {
    "Marvel": {
        "train": r"REPLACE_WITH_PROJECT_ROOT\Marvel\train_marvel",
        "valid": r"REPLACE_WITH_PROJECT_ROOT\Marvel\valid_marvel",
        "test": r"REPLACE_WITH_PROJECT_ROOT\Marvel\test_marvel"
    },
    "DC": {
        "train": r"REPLACE_WITH_PROJECT_ROOT\DC\train_dc",
        "valid": r"REPLACE_WITH_PROJECT_ROOT\DC\valid_dc",
        "test": r"REPLACE_WITH_PROJECT_ROOT\DC\test_dc"
    }
}

target_base_dir = r"REPLACE_WITH_PROJECT_ROOT\Combined"

# Organize images for each class and split
for class_name, splits in source_directories.items():
    for split, source_dir in splits.items():
        target_dir = os.path.join(target_base_dir, split)
        organize_images(source_dir, target_dir, class_name)

print("Images organized into class subdirectories.")