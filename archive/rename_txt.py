import os
import sys

# +-------------------------------------------------------------+
# | EDIT THIS VARIABLE to set the folder containing your files. |
# | Use '.' for the current directory.                          |
# +-------------------------------------------------------------+
TARGET_FOLDER_PATH = "//predict/gesture_99"


# Example for Windows: TARGET_FOLDER_PATH = "C:\\Users\\Bob\\Desktop\\MyNotes"
# Example for macOS/Linux: TARGET_FOLDER_PATH = "/home/bob/documents/notes"


def reorder_numeric_files(target_dir):
    """
    Finds and reorders numeric .txt files in the given directory.
    """
    # --- 1. Validate Path ---
    if not os.path.isdir(target_dir):
        print(f"Error: The specified path does not exist or is not a directory.")
        print(f"Path: '{os.path.abspath(target_dir)}'")
        return

    print(f"Scanning directory: {os.path.abspath(target_dir)}")

    # --- The rest of the script is identical to Option 1 from this point on ---
    # --- just using the 'target_dir' variable passed into the function.     ---

    # 2. Find and Validate Files
    files_to_rename = []
    for filename in os.listdir(target_dir):
        full_path = os.path.join(target_dir, filename)
        if os.path.isfile(full_path) and filename.endswith('.txt'):
            basename = filename[:-4]
            if basename.isdigit():
                files_to_rename.append((int(basename), filename))

    if not files_to_rename:
        print(f"No numeric .txt files found to rename in '{target_dir}'.")
        return

    # 3. Sort Files and Get Confirmation
    files_to_rename.sort()
    print(f"\nFound {len(files_to_rename)} files to rename.")
    print("Files will be processed in this order:")
    for _, filename in files_to_rename:
        print(f"  - {filename}")

    try:
        confirm = input("\nDo you want to proceed with renaming? (y/n): ").lower()
        if confirm != 'y':
            print("Operation cancelled by user.")
            return
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return

    # 4. Get the Starting Number
    start_number = None
    while start_number is None:
        try:
            input_str = input("Enter the number to start reordering from (e.g., 7): ")
            start_number = int(input_str)
        except ValueError:
            print("Invalid input. Please enter a whole number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return

    # 5. Rename Safely
    print("\n--- Starting Renaming Process ---")
    temp_suffix = "_temp_rename"
    temp_files_basenames = []

    # PASS 1
    print("\n[PASS 1/2] Renaming to temporary names...")
    try:
        for _, old_filename in files_to_rename:
            temp_name = f"{old_filename}{temp_suffix}"
            os.rename(os.path.join(target_dir, old_filename), os.path.join(target_dir, temp_name))
            temp_files_basenames.append(temp_name)
            print(f"  '{old_filename}' -> '{temp_name}'")
    except OSError as e:
        print(f"\nAn error occurred during PASS 1: {e}.")
        return

    # PASS 2
    print("\n[PASS 2/2] Renaming to final ordered names...")
    current_number = start_number
    try:
        for temp_name in temp_files_basenames:
            new_filename = f"{current_number}.txt"
            os.rename(os.path.join(target_dir, temp_name), os.path.join(target_dir, new_filename))
            print(f"  '{temp_name}' -> '{new_filename}'")
            current_number += 1
    except OSError as e:
        print(f"\nAn error occurred during PASS 2: {e}. Please check the folder for temporary files.")
        return

    print("\n--- Renaming Complete! ---")


if __name__ == "__main__":
    reorder_numeric_files(TARGET_FOLDER_PATH)