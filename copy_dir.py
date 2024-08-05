import os
import shutil


def copy_directory(src, dest):
    # Check if the source directory exists
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory '{src}' does not exist.")

    # Copy the directory tree
    shutil.copytree(src, dest, dirs_exist_ok=True)
    print(f"Copied '{src}' to '{dest}'")

# Example usage
source_directory =  '/Users/mcevo/PycharmProjects/fpl_data/data'
destination_directory = './data'
copy_directory(source_directory, destination_directory)