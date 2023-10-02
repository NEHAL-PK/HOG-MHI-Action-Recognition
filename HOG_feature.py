import os
import shutil

def extract_class_from_directory(directory_name):
    """Extract class label from a directory name."""
    return directory_name.split('_')[0]

base_directory = r"C:\Users\nehal\Downloads\Compressed\drive-download-20230928T233402Z-001"

# We create a dictionary to store the mapping of directories and their destination folders
move_dict = {}

for root, dirs, files in os.walk(base_directory, topdown=False):  # topdown=False processes the subdirectories first
    for dir_name in dirs:
        class_label = extract_class_from_directory(dir_name)
        
        # Prepare the destination path for the directory
        dest_path = os.path.join(base_directory, class_label)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        # Full path of current directory
        full_dir_path = os.path.join(root, dir_name)
        
        # Append the directory path and its destination to the move dictionary
        move_dict[full_dir_path] = dest_path

# Now, move the directories to their respective class folders
for src, dest in move_dict.items():
    final_dest = os.path.join(dest, os.path.basename(src))
    shutil.move(src, final_dest)