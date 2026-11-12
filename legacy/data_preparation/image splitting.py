# ARCHIVAL SOURCE (2024). Absolute local file-system paths replaced with a placeholder.
# Set REPLACE_WITH_PROJECT_ROOT to a local project directory before attempting to run.
#import needed libraries
import os
import shutil
import random

# Paths to your original Marvel and DC images on the Desktop
marvel_path = r"REPLACE_WITH_PROJECT_ROOT\Marvel\Doctor_strange_in_the_multiverse_of_madness_jpg_format"
dc_path = r"REPLACE_WITH_PROJECT_ROOT\DC\Aquaman_(2018)_jpg_format"

# Paths to the destination directories on the Desktop
dest_valid_marvel = r"REPLACE_WITH_PROJECT_ROOT\Marvel\valid_marvel"
dest_train_marvel = r"REPLACE_WITH_PROJECT_ROOT\Marvel\train_marvel"
dest_test_marvel = r"REPLACE_WITH_PROJECT_ROOT\Marvel\test_marvel"

dest_valid_dc = r"REPLACE_WITH_PROJECT_ROOT\DC\valid_dc"
dest_train_dc = r"REPLACE_WITH_PROJECT_ROOT\DC\train_dc"
dest_test_dc = r"REPLACE_WITH_PROJECT_ROOT\DC\test_dc"

# Function to split data
def split_data(src_dir, train_dir, valid_dir, test_dir, split_ratio):
    images = os.listdir(src_dir)
    random.shuffle(images)
    
    split1 = int(split_ratio[0] * len(images))
    split2 = int((split_ratio[0] + split_ratio[1]) * len(images))
    
    train_images = images[:split1]
    valid_images = images[split1:split2]
    test_images = images[split2:]
    
    for img in train_images:
        shutil.copy(os.path.join(src_dir, img), train_dir)
    
    for img in valid_images:
        shutil.copy(os.path.join(src_dir, img), valid_dir)
    
    for img in test_images:
        shutil.copy(os.path.join(src_dir, img), test_dir)

# Split Marvel images
split_data(marvel_path, dest_train_marvel, dest_valid_marvel, dest_test_marvel, [0.7, 0.15, 0.15])

# Split DC images
split_data(dc_path, dest_train_dc, dest_valid_dc, dest_test_dc, [0.7, 0.15, 0.15])