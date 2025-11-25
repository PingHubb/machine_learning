import os
import random
import shutil

def move_files_to_testing(training_data_dir, testing_data_dir, fraction=0.1):
    # Get all gesture folders in training_data
    gesture_folders = [f for f in os.listdir(training_data_dir) if os.path.isdir(os.path.join(training_data_dir, f))]

    for gesture_folder in gesture_folders:
        gesture_path = os.path.join(training_data_dir, gesture_folder)
        # Ensure the corresponding folder exists in testing_data
        target_test_path = os.path.join(testing_data_dir, gesture_folder)
        os.makedirs(target_test_path, exist_ok=True)

        # Get all .txt files in the current gesture folder
        txt_files = [f for f in os.listdir(gesture_path) if f.endswith('.txt')]

        # Select 10% of the files randomly
        num_files_to_move = max(1, int(len(txt_files) * fraction))  # At least move 1 file
        files_to_move = random.sample(txt_files, num_files_to_move)

        # Move selected files to the testing_data folder
        for file_name in files_to_move:
            src_file = os.path.join(gesture_path, file_name)
            dest_file = os.path.join(target_test_path, file_name)
            shutil.move(src_file, dest_file)
            print(f'Moved {file_name} from {gesture_path} to {target_test_path}')

# Example usage
training_data_dir = '//ai/training_data'
testing_data_dir = '//ai/testing_data'

move_files_to_testing(training_data_dir, testing_data_dir)
