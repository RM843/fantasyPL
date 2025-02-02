import os
import pprint
from itertools import combinations

import time
from functools import wraps


def df_to_nested_dict(df, level1, level2):
    result = {}
    for _, row in df.iterrows():
        key1 = row[level1]
        key2 = row[level2]
        value = row.drop([level1, level2]).to_dict()
        key1 = key1+"_"+key2
        if key1 not in result:
            result[key1] = []


        result[key1].append(value)

    return result

def trim_tuples(lst):
    trimmed = []
    found_non_none = False

    for item in lst:
        if not found_non_none and item[1] is None:
            continue
        found_non_none = True
        trimmed.append(item)

    return trimmed
def replace_nones_with_previous(lst):
    last_value = None
    for i, val in enumerate(lst):
        if val is None:
            lst[i] = last_value
        else:
            last_value = val
    return lst
def print_list_of_dicts(list_of_dicts):
    """Print a list of dictionaries in a nicely formatted way."""
    pp = pprint.PrettyPrinter(indent=4)
    for i, d in enumerate(list_of_dicts):
        print(f"{i + 1}:")
        pp.pprint(d)
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
def dedup_tuple(tuples_list,i):
    seen = set()  # To track seen first values
    result = []   # To store the result

    for tup in tuples_list:
        first_value = tup[i]
        if first_value not in seen:
            seen.add(first_value)
            result.append(tup)

    return result
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
