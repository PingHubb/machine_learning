import os
import numpy as np
import pandas as pd
import shutil
import random

# --- Configuration ---
# You can easily change these paths and the noise level.

# Source folders containing the clean data
CLEAN_DATA_PATHS = [
    'C:/Users/cplam/Downloads/machine_learning/ai/training_data',
    'C:/Users/cplam/Downloads/machine_learning/ai/testing_data'
]

# Destination folder where the noisy data will be saved
NOISY_DATA_BASE_PATH = '//ai/noise_data'

# The standard deviation of the Gaussian noise to add.
# A larger value means more noise. 0.1 is a good starting point.
NOISE_LEVEL = 0.1


# --- Helper Functions ---

def load_data_from_file(file_path):
    """Loads data from a space-separated file, handling potential errors."""
    try:
        data = pd.read_csv(file_path, sep='\s+', header=None, engine='python')
        if data.empty:
            print(f"  - Warning: File is empty, skipping: {os.path.basename(file_path)}")
            return None
        if data.shape[1] != 56:
            print(f"  - Warning: Incorrect column count ({data.shape[1]}), skipping: {os.path.basename(file_path)}")
            return None
        return data.values
    except Exception as e:
        print(f"  - Error reading file {os.path.basename(file_path)}: {e}")
        return None


def add_noise(data, noise_level):
    """Adds Gaussian noise to a numpy array."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def save_data_to_file(data, file_path):
    """Saves a numpy array to a text file with space delimiters."""
    np.savetxt(file_path, data, fmt='%.6f', delimiter=' ')


# --- Main Logic ---

def generate_noisy_dataset():
    """
    Main function to read clean data, add noise, and save it to a single flat
    directory with randomized filenames. No mapping file is created.
    """
    print("--- Starting Anonymized Noisy Data Generation (No Mapping) ---")

    # 1. Clean up and create the base destination folder
    if os.path.exists(NOISY_DATA_BASE_PATH):
        print(f"Removing existing directory: {NOISY_DATA_BASE_PATH}")
        shutil.rmtree(NOISY_DATA_BASE_PATH)
    print(f"Creating new directory: {NOISY_DATA_BASE_PATH}")
    os.makedirs(NOISY_DATA_BASE_PATH)

    total_files_processed = 0
    total_files_failed = 0

    # Generate a pool of unique random numbers for filenames to avoid collisions
    possible_ids = list(range(1_000_000))
    random.shuffle(possible_ids)

    # 2. Iterate through each source directory
    for source_base_path in CLEAN_DATA_PATHS:
        if not os.path.exists(source_base_path):
            print(f"Source path not found, skipping: {source_base_path}")
            continue

        print(f"\nProcessing files from: {source_base_path}")

        # 3. Walk through the directory structure
        for root, dirs, files in os.walk(source_base_path):
            for file in files:
                if file.endswith('.txt'):
                    source_file_path = os.path.join(root, file)

                    # 4. Load the clean data
                    clean_data = load_data_from_file(source_file_path)

                    if clean_data is None:
                        total_files_failed += 1
                        continue

                    # 5. Add noise
                    noisy_data = add_noise(clean_data, NOISE_LEVEL)

                    # 6. Generate Randomized Filename
                    if not possible_ids:
                        raise Exception("Ran out of unique random IDs for filenames!")
                    random_id = possible_ids.pop()
                    noisy_filename = f"{random_id}.txt"
                    destination_file_path = os.path.join(NOISY_DATA_BASE_PATH, noisy_filename)

                    # 7. Save the new noisy file
                    save_data_to_file(noisy_data, destination_file_path)

                    total_files_processed += 1
                    if total_files_processed % 100 == 0:
                        print(f"  ... {total_files_processed} files processed ...")

    print("\n--- Data Generation Complete ---")
    print(f"Successfully processed and created {total_files_processed} noisy files.")
    if total_files_failed > 0:
        print(f"Failed to process or skipped {total_files_failed} files due to errors.")
    print(f"Anonymized noisy data saved in: {NOISY_DATA_BASE_PATH}")
    print("No mapping file was created, as requested.")


# --- Run the Script ---
if __name__ == '__main__':
    # Set seeds for reproducibility of the noise and random filenames
    np.random.seed(73)
    random.seed(64)
    generate_noisy_dataset()