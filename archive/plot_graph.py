# import pandas as pd
# import matplotlib.pyplot as plt
#
# def get_number_of_epochs(txt_path):
#     with open(txt_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             if "epochs" in line:
#                 parts = line.split(',')
#                 for part in parts:
#                     if "epochs" in part:
#                         epoch_num = int(part.split('=')[1])
#                         return epoch_num
#
# def plot_combined_first_fold(csv_path, txt_path):
#     # Get the number of epochs from the txt file
#     num_epochs = get_number_of_epochs(txt_path)
#
#     # Load the data from CSV
#     data = pd.read_csv(csv_path)
#
#     # Select the entries for the first fold, based on num_epochs extracted
#     first_fold_data = data.head(num_epochs)
#
#     # Creating a figure and a single subplot
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#
#     # Plotting training loss on the primary y-axis
#     color = 'tab:red'
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss', color=color)
#     ax1.plot(first_fold_data['epoch'], first_fold_data['loss'], color=color, label='Training Loss')
#     ax1.tick_params(axis='y', labelcolor=color)
#
#     # Creating a second y-axis for accuracy
#     ax2 = ax1.twinx()
#     color = 'tab:blue'
#     ax2.set_ylabel('Accuracy', color=color)
#     ax2.plot(first_fold_data['epoch'], first_fold_data['accuracy'], color=color, label='Training Accuracy')
#     ax2.tick_params(axis='y', labelcolor=color)
#
#     # Adding a title and a legend
#     plt.title('Training Loss and Accuracy Over Epochs')
#     fig.tight_layout()  # Adjust the layout to make room for the second y-axis
#
#     # Adding legends
#     lines, labels = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2, loc='upper left')
#
#     plt.show()
#
#
# # Example usage
#
# number = "0"  # 79, 27, 112
# csv_file_path = f'C:/dev/machine_learning/models_3/good_{number}_history.csv'  # CSV file path
# txt_file_path = f'C:/dev/machine_learning/models_3/good_{number}.txt'  # Parameter TXT file path
# plot_combined_first_fold(csv_file_path, txt_file_path)

import pandas as pd
import matplotlib.pyplot as plt
import os


def load_and_average_folds(csv_directory, num_folds):
    # Initialize empty DataFrame to store averages
    average_data = None

    # Load each fold's data
    for fold in range(1, num_folds + 1):
        csv_filename = os.path.join(csv_directory, f'fold_{fold}.csv')
        fold_data = pd.read_csv(csv_filename)

        # Sum fold data or initialize average_data
        if average_data is None:
            average_data = fold_data.set_index('epoch')
        else:
            average_data += fold_data.set_index('epoch')

    # Divide to get the average
    average_data /= num_folds

    # Reset index for plotting
    average_data.reset_index(inplace=True)

    return average_data

def main():
    # Specify the directory where CSV files are stored
    csv_directory = 'C:/dev/machine_learning/csv/8/'
    num_folds = 5  # Total number of folds

    # Load and average data
    average_data = load_and_average_folds(csv_directory, num_folds)

    # Set the style for the plot
    plt.style.use('seaborn-white')  # Set the plot background to white

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Axis 1: Loss
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(average_data['epoch'], average_data['train_loss'], label='Average Training Loss', color=color)
    ax1.plot(average_data['epoch'], average_data['val_loss'], label='Average Validation Loss', color='tab:pink')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Axis 2: Accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(average_data['epoch'], average_data['train_accuracy'], label='Average Training Accuracy', color=color)
    ax2.plot(average_data['epoch'], average_data['val_accuracy'], label='Average Validation Accuracy', color='tab:cyan')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Average Training and Validation Loss and Accuracy Across Folds')
    plt.show()

if __name__ == '__main__':
    main()
