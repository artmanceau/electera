import os
import re

import pandas as pd


def load_parquet_into_df(file_path, verbose=False):
    """Loads a Parquet file into a Pandas DataFrame."""
    try:
        df = pd.read_parquet(file_path)
        if verbose:
            print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def load_all_data(directory):
    """Opens every parquet file in the given directory exploring all subdirectories."""
    # Directory containing the election files
    dfs_by_file = {}

    # Recursively explore subfolders and load each CSV file
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".parquet"):
                filepath = os.path.join(dirpath, filename)

                # Load the data
                data = load_parquet_into_df(filepath)

                # Store the resulting DataFrame in the dictionary with the relative path as key
                relative_path = os.path.relpath(filepath, directory)
                dfs_by_file[relative_path] = data

    print("Data successfully loaded onto the Notebook")
    return dfs_by_file


def sort_data_dict(dfs_by_file):
    simplified_keys = {}
    for key, df in dfs_by_file.items():
        match = re.match(r"([a-zA-Z]+)/(\d+)", key)
        if match:
            election_type = match.group(1)
            year = match.group(2)
            simplified_key = f"{election_type}/{year}"
            simplified_keys[simplified_key] = df

    # Step 2: Sort the keys by year
    sorted_simplified_keys = sorted(
        simplified_keys.keys(), key=lambda x: int(x.split("/")[1])
    )

    # Step 3: Reorganize the dictionary based on sorted keys
    sorted_dfs_by_file = {key: simplified_keys[key] for key in sorted_simplified_keys}

    # Now you have your dictionary sorted by year with simplified keys
    dfs_by_file = sorted_dfs_by_file
    return dfs_by_file


def add_data(selected_columns_df, data, criteria_name, criteria, born_inf):
    """
    Function that adds PIB criteria to each election file
    """
    modified_data = {}

    for key, df in selected_columns_df.items():
        # Extract the year from the key (assumes format 'referendum/<year>/<...>')
        year = int((key.split("/")[1])[:4])
        if year < born_inf:
            continue

        # Merge the election data with PIB data based on 'codecommune' and 'year'
        merged_df = pd.merge(
            df,
            data[["codecommune", criteria_name + str(year)]],
            how="left",
            on="codecommune",
        )

        merged_df = merged_df.rename(columns={criteria_name + str(year): criteria})

        merged_df[criteria] = merged_df[criteria].fillna(0)

        # Add the modified DataFrame to the result dictionary
        modified_data[key] = merged_df

    return modified_data
