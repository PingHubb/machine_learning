import os
import shutil

def move_files_back_to_training(testing_data_dir, training_data_dir):
    # Get all gesture folders in testing_data
    gesture_folders = [f for f in os.listdir(testing_data_dir) if os.path.isdir(os.path.join(testing_data_dir, f))]

    for gesture_folder in gesture_folders:
        gesture_path = os.path.join(testing_data_dir, gesture_folder)
        # Corresponding folder in training_data
        target_train_path = os.path.join(training_data_dir, gesture_folder)

        # Get all .txt files in the current gesture folder
        txt_files = [f for f in os.listdir(gesture_path) if f.endswith('.txt')]

        # Move files back to the training_data folder
        for file_name in txt_files:
            src_file = os.path.join(gesture_path, file_name)
            dest_file = os.path.join(target_train_path, file_name)
            shutil.move(src_file, dest_file)
            print(f'Moved {file_name} from {gesture_path} to {target_train_path}')

# Example usage
testing_data_dir = '//ai/testing_data'
training_data_dir = '//ai/training_data'

move_files_back_to_training(testing_data_dir, training_data_dir)
