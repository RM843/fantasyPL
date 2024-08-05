import os

def list_folders(path):
    try:
        # Get all entries in the specified path
        entries = os.listdir(path)
        # Filter entries to include only directories
        directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
        return directories
    except FileNotFoundError:
        print(f"Path '{path}' not found.")
        return []
    except PermissionError:
        print(f"Permission denied for path '{path}'.")
        return []

# Example usage
directory_path = '/path/to/your/directory'
folders = list_folders(directory_path)
print(f"Folders in '{directory_path}': {folders}")