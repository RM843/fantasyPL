import os
from itertools import combinations

import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        duration = end_time - start_time  # Calculate the duration
        print(f"Function '{func.__name__}' took {duration:.4f} seconds to execute")
        return result  # Return the result of the function
    return wrapper

def combination_generator(iterable, r):
    for combo in combinations(iterable, r):
        yield combo

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