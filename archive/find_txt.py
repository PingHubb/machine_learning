import os

# Specify the directory containing the files
directory = 'C:/dev/machine_learning/models_3/'

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.startswith("good_") and filename.endswith(".txt"):
        # Construct the full file path
        filepath = os.path.join(directory, filename)

        # Open and read the file
        with open(filepath, 'r') as file:
            line = file.readline().strip()
            # Extract the average accuracy from the line
            if 'Trial' in line:
                parts = line.split(',')
                avg_accuracy_part = parts[-1]  # This should be the part containing the accuracy
                avg_accuracy = float(avg_accuracy_part.split('=')[-1])

                # Check if the average accuracy is greater than 90%
                if avg_accuracy > 93:
                    print(f"{filename}: {line}")
#

# import os
#
# # Specify the directory containing the files
# directory = 'C:/dev/machine_learning/txt/'
#
# # Iterate over each file in the directory that matches the criteria
# for filename in os.listdir(directory):
#     if filename.startswith("good_") and filename.endswith(".txt"):
#         filepath = os.path.join(directory, filename)
#
#         # Open and read the file
#         with open(filepath, 'r') as file:
#             for line in file:
#                 line = line.strip()  # Remove any leading/trailing whitespace
#                 # Check if this line contains the average test accuracy
#                 if 'Average Test Accuracy' in line:
#                     # Extract the average accuracy from the line
#                     avg_accuracy = float(line.split('=')[-1].strip('%'))
#
#                     # Check if the average accuracy is greater than 70%
#                     if avg_accuracy > 70:
#                         print(f"{filename}: Average Test Accuracy = {avg_accuracy}%")
#                         break  # Stop reading further as we only need the accuracy