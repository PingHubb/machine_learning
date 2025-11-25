import os
import random
import sys


def random_file_deleter():
    """
    Prompts the user for a directory and a percentage, then randomly deletes that
    percentage of files from the directory after a clear confirmation.
    """
    # --- STEP 0: THE BIG, SCARY WARNING ---
    print("=" * 60)
    print("!!! WARNING: DANGEROUS SCRIPT !!!")
    print("This script PERMANENTLY DELETES files from your computer.")
    print("There is NO UNDO and NO RECYCLE BIN.")
    print("Use with EXTREME CAUTION. The author is not responsible for data loss.")
    print("=" * 60)
    print()

    # --- STEP 1: Get and Validate the Target Directory ---

    # target_directory = input("Enter the full path of the directory to process: ").strip()
    # target_directory = 'C:/Users/cplam/Downloads/machine_learning/ai/training_data_3level/num_finger_1/gesture_groundtruth_4/good/'
    target_directory = 'C:/Users/cplam/Downloads/machine_learning/ai/training_data_3level/num_finger_1/gesture_cylinder_two_finger_up_10/good/'

    if not os.path.isdir(target_directory):
        print(f"\nError: The path '{target_directory}' is not a valid directory.")
        sys.exit("Aborting script.")

    # --- STEP 2: Get and Validate the Percentage ---
    try:
        percentage_str = input("Enter the percentage of files to delete (e.g., 25 for 25%): ")
        percentage_to_delete = float(percentage_str)
        if not (0 <= percentage_to_delete <= 100):
            print("\nError: Percentage must be between 0 and 100.")
            sys.exit("Aborting script.")
    except ValueError:
        print("\nError: Invalid input. Please enter a number for the percentage.")
        sys.exit("Aborting script.")

    # --- STEP 3: List all files in the directory ---
    try:
        # Use a list comprehension to get only files, not subdirectories
        all_files = [f for f in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, f))]
    except OSError as e:
        print(f"\nError reading directory: {e}")
        sys.exit("Aborting script.")

    if not all_files:
        print("\nThere are no files in the specified directory.")
        sys.exit("Nothing to do.")

    # --- STEP 4: Calculate and select files for deletion ---
    total_file_count = len(all_files)
    num_to_delete = int(total_file_count * (percentage_to_delete / 100.0))

    # random.sample is perfect for selecting a unique subset of items from a list
    files_to_delete = random.sample(all_files, num_to_delete)

    # --- STEP 5: FINAL CONFIRMATION (CRITICAL SAFETY STEP) ---
    print("\n--- SUMMARY ---")
    print(f"Target Directory: {target_directory}")
    print(f"Total files found: {total_file_count}")
    print(f"Percentage to delete: {percentage_to_delete}%")
    print(f"Number of files that will be DELETED: {num_to_delete}")
    print("\nThis action is IRREVERSIBLE.")

    # Only proceed if the user types exactly "yes"
    confirmation = input('Are you absolutely sure you want to proceed? Type "yes" to confirm: ')

    if confirmation.lower() != 'yes':
        print("\nOperation cancelled by user.")
        sys.exit("Aborting script.")

    # --- STEP 6: Execute Deletion ---
    print("\nStarting deletion process...")
    deleted_count = 0
    for filename in files_to_delete:
        file_path_to_delete = os.path.join(target_directory, filename)
        try:
            os.remove(file_path_to_delete)
            print(f"  Deleted: {filename}")
            deleted_count += 1
        except OSError as e:
            print(f"  Error deleting {filename}: {e}")

    print("\n--- DELETION COMPLETE ---")
    print(f"Successfully deleted {deleted_count} out of {num_to_delete} targeted files.")


# --- Run the main function when the script is executed ---
if __name__ == "__main__":
    random_file_deleter()