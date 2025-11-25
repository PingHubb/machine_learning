import os
import numpy as np
import random
import pandas as pd
import sys
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


# Custom Dataset class
class GestureDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data_list = data_list
        self.labels_list = labels_list
        self.lengths = [len(seq) for seq in data_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.labels_list[idx]
        length = self.lengths[idx]
        return data, label, length


# Custom Dataset class for prediction (no labels)
class GestureDatasetPredict(Dataset):
    def __init__(self, data_list, filenames):
        self.data_list = data_list
        self.filenames = filenames
        self.lengths = [len(seq) for seq in data_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        filename = self.filenames[idx]
        length = self.lengths[idx]
        return data, length, filename


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)    # Numpy module.
    random.seed(seed_value)       # Python random module.
    torch.manual_seed(seed_value) # PyTorch to set the seed for CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value) # Seed for CUDA
        torch.cuda.manual_seed_all(seed_value) # Seed all GPUs if there are multiple GPUs
        torch.backends.cudnn.deterministic = True  # More reproducibility
        torch.backends.cudnn.benchmark = False     # It can slow down training, disable if not needed


# Function to load data from a single file
def load_data_from_file(file_path):
    data = pd.read_csv(file_path, sep=" ", header=None)
    return data.values


# Function to load all trials for a single gesture
def load_gesture_data(gesture_folder):
    files = sorted([os.path.join(gesture_folder, f) for f in os.listdir(gesture_folder) if f.endswith('.txt')])
    trials = []
    for file in files:
        features = load_data_from_file(file)
        trials.append(features)
    return trials


# Load data from all gestures and prepare for training
def load_all_gestures(base_path):
    gesture_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    all_data = []
    all_labels = []
    for folder in gesture_folders:
        trials = load_gesture_data(folder)
        gesture_label = int(folder.split('_')[-1])
        for trial in trials:
            all_data.append(trial)
            all_labels.append(gesture_label)
    return all_data, all_labels


# Load data for prediction (no labels)
def load_predict_data(predict_path):
    files = sorted([os.path.join(predict_path, f) for f in os.listdir(predict_path) if f.endswith('.txt')])
    all_data = []
    filenames = []
    for file in files:
        features = load_data_from_file(file)
        all_data.append(features)
        filenames.append(os.path.basename(file))
    return all_data, filenames


def train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs, fold, enable_scheduler, is_test=False):
    history = {'epoch': [], 'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for data, labels, lengths in train_loader:
            data, labels, lengths = data.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            # Sort sequences by length in descending order
            lengths, perm_idx = lengths.sort(0, descending=True)
            data = data[perm_idx]
            labels = labels[perm_idx]
            # Pack sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(data, lengths.cpu(), batch_first=True)
            outputs = model(packed_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_accuracy'].append(train_accuracy)

        # Evaluate on validation or test set
        if is_test:
            test_loss, test_accuracy = evaluate(val_loader, model, criterion, device)
            history['test_loss'] = history.get('test_loss', []) + [test_loss]
            history['test_accuracy'] = history.get('test_accuracy', []) + [test_accuracy]
            print(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, '
                f'Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, '
                f'Test Accuracy: {test_accuracy:.2f}%')
        else:
            val_loss, val_accuracy = evaluate(val_loader, model, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            print(
                f'Fold {fold + 1}, Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, '
                f'Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, '
                f'Validation Accuracy: {val_accuracy:.2f}%')

        # Scheduler step after each epoch
        if enable_scheduler:
            scheduler.step()

        history['epoch'].append(epoch + 1)

    return history


def evaluate(data_loader, model, criterion, device):
    total_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, labels, lengths in data_loader:
            data, labels, lengths = data.to(device), labels.to(device), lengths.to(device)
            # Sort sequences by length in descending order
            lengths, perm_idx = lengths.sort(0, descending=True)
            data = data[perm_idx]
            labels = labels[perm_idx]
            # Pack sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(data, lengths.cpu(), batch_first=True)
            outputs = model(packed_input)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(data_loader), 100 * correct / total


class GestureCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, num_layers, dropout_rate):
        super(GestureCNNLSTM, self).__init__()
        # Assuming the sensor grid is 13x10
        self.height = 13
        self.width = 10
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        conv_output_size = self._get_conv_output_size()
        self.lstm = nn.LSTM(conv_output_size, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def _get_conv_output_size(self):
        dummy_input = torch.zeros(1, 1, self.height, self.width)
        output = self.conv(dummy_input)
        output_size = output.view(1, -1).size(1)
        return output_size

    def forward(self, x):
        batch_size = x.batch_sizes[0].item()  # For PackedSequence
        # Unpack sequences
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        seq_len = x.size(1)
        # Reshape to (batch_size * seq_len, 1, H, W)
        x = x.contiguous().view(-1, 1, self.height, self.width)
        x = self.conv(x)
        x = x.view(batch_size, seq_len, -1)
        # Pack sequences again
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(x)
        logits = self.classifier(hidden[-1])
        return logits


def main():
    set_seed(54)  # Setting the seed at the start of the main function

    select = input('Enter option number (Enter 1 to train the model, or t to test the model): ')

    base_path = 'C:/dev/phd/ai'
    train_path = 'C:/dev/phd/ai/training_data'
    test_path = 'C:/dev/phd/ai/testing_data'
    predict_path = 'C:/dev/phd/predict/'  # Added predict_path
    model_save_path = 'C:/dev/phd/models/'
    csv_save_path = 'C:/dev/phd/csv/'
    txt_save_path = 'C:/dev/phd/txt/'

    if select == '1':

        # Load training and testing data
        train_data, train_labels = load_all_gestures(train_path)
        test_data, test_labels = load_all_gestures(test_path)

        # No scaling is performed since data is binary

        # Create datasets
        train_dataset = GestureDataset(train_data, train_labels)
        test_dataset = GestureDataset(test_data, test_labels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')

        output_dim = len(set(train_labels))

        # Hyperparameters (adjusted)
        num_layers_options = [1, 2, 3]
        k_fold_options = [5]  # Keep as is
        hidden_dims = [64, 128, 256]
        epochs_range = [50, 100, 150]
        batch_sizes = [16, 32, 64]
        dropout_rates = [0.2, 0.3, 0.5]
        weight_decays = [0, 0.0001, 0.001]
        learning_rates = [0.001, 0.0005, 0.0001]
        step_sizes = [10, 20, 30]
        gammas = [0.1, 0.5]

        good_model_count = 1
        enable_scheduler = True

        best_accuracy = 0
        best_config = {}

        for num_layers in num_layers_options:
            for k_folds in k_fold_options:
                for hidden_dim in hidden_dims:
                    for epochs in epochs_range:
                        for batch_size in batch_sizes:
                            for lr in learning_rates:
                                for dropout in dropout_rates:
                                    for wd in weight_decays:
                                        for step in step_sizes:
                                            for gamma in gammas:

                                                print(
                                                    f"num_layers={num_layers}, k_folds={k_folds}, hidden_dim={hidden_dim}, epochs={epochs}, "
                                                    f"batch_size={batch_size}, learning_rate={lr}, dropout_rate={dropout}, "
                                                    f"step_size={step}, gamma={gamma}, weight_decay={wd}")

                                                kfold = KFold(n_splits=k_folds, shuffle=True, random_state=47)
                                                fold_train_accuracies = []
                                                fold_val_accuracies = []

                                                for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
                                                    print(f"Fold {fold + 1}/{k_folds}:")
                                                    # Split train_dataset into training and validation sets
                                                    train_subsampler = Subset(train_dataset, train_idx)
                                                    val_subsampler = Subset(train_dataset, val_idx)

                                                    # Create custom collate function to handle variable-length sequences
                                                    def collate_fn(batch):
                                                        batch.sort(key=lambda x: len(x[0]), reverse=True)
                                                        sequences, labels, lengths = zip(
                                                            *[(torch.tensor(seq, dtype=torch.float32),
                                                               label,
                                                               len(seq)) for seq, label, length in batch])
                                                        sequences_padded = nn.utils.rnn.pad_sequence(sequences,
                                                                                                     batch_first=True)
                                                        lengths = torch.tensor(lengths, dtype=torch.long)
                                                        labels = torch.tensor(labels, dtype=torch.long)
                                                        return sequences_padded, labels, lengths

                                                    train_loader = DataLoader(train_subsampler, batch_size=batch_size,
                                                                              shuffle=True, collate_fn=collate_fn)
                                                    val_loader = DataLoader(val_subsampler, batch_size=batch_size,
                                                                            shuffle=False, collate_fn=collate_fn)

                                                    # Adjust input_size based on the CNN
                                                    input_size = train_dataset[0][0].shape[
                                                        1]  # Number of features per time step

                                                    model = GestureCNNLSTM(input_size, hidden_dim, output_dim,
                                                                           num_layers=num_layers,
                                                                           dropout_rate=dropout).to(device)
                                                    criterion = nn.CrossEntropyLoss()
                                                    optimizer = optim.Adam(model.parameters(), lr=lr,
                                                                           weight_decay=wd)
                                                    scheduler = StepLR(optimizer, step_size=step, gamma=gamma)

                                                    history = train_and_evaluate(train_loader, val_loader, model,
                                                                                 criterion, optimizer,
                                                                                 scheduler, device, epochs, fold,
                                                                                 enable_scheduler)
                                                    fold_train_accuracies.append(history['train_accuracy'][-1])
                                                    fold_val_accuracies.append(history['val_accuracy'][-1])

                                                    # Save history to CSV
                                                    model_csv_dir = os.path.join(csv_save_path, str(good_model_count))
                                                    if not os.path.exists(model_csv_dir):
                                                        os.makedirs(model_csv_dir)
                                                    csv_filename = os.path.join(model_csv_dir, f'fold_{fold + 1}.csv')
                                                    pd.DataFrame(history).to_csv(csv_filename, index=False)

                                                avg_train_accuracy = sum(fold_train_accuracies) / len(
                                                    fold_train_accuracies)
                                                avg_val_accuracy = sum(fold_val_accuracies) / len(fold_val_accuracies)

                                                print(f'Average Train Accuracy: {avg_train_accuracy:.2f}%')
                                                print(f'Average Validation Accuracy: {avg_val_accuracy:.2f}%')

                                                # If this configuration performs better, test on the test set
                                                if avg_val_accuracy >= best_accuracy:
                                                    best_accuracy = avg_val_accuracy
                                                    best_config = {
                                                        'num_layers': num_layers,
                                                        'k_folds': k_folds,
                                                        'hidden_dim': hidden_dim,
                                                        'epochs': epochs,
                                                        'batch_size': batch_size,
                                                        'learning_rate': lr,
                                                        'weight_decay': wd,
                                                        'dropout_rate': dropout,
                                                        'step_size': step,
                                                        'gamma': gamma,
                                                        'train_accuracy': avg_train_accuracy,
                                                        'validation_accuracy': avg_val_accuracy,
                                                        'output_dim': output_dim  # Add this line
                                                    }

                                                    # Train on the full training dataset with the best configuration
                                                    full_train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                                                                   shuffle=True, collate_fn=collate_fn)
                                                    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                                                             shuffle=False, collate_fn=collate_fn)

                                                    model = GestureCNNLSTM(input_size, hidden_dim, output_dim,
                                                                           num_layers=num_layers,
                                                                           dropout_rate=dropout).to(device)
                                                    criterion = nn.CrossEntropyLoss()
                                                    optimizer = optim.Adam(model.parameters(), lr=lr,
                                                                           weight_decay=wd)
                                                    scheduler = StepLR(optimizer, step_size=step, gamma=gamma)

                                                    # Train the model on the full training data
                                                    history = train_and_evaluate(full_train_loader, test_loader, model,
                                                                                 criterion, optimizer, scheduler,
                                                                                 device, epochs, 0, enable_scheduler,
                                                                                 is_test=True)
                                                    test_accuracy = history['test_accuracy'][-1]
                                                    best_config['test_accuracy'] = test_accuracy

                                                    # Save the model and parameters
                                                    model_filename = f'best_model_{good_model_count}.pth'
                                                    param_filename = f'best_model_{good_model_count}.txt'
                                                    torch.save(model.state_dict(),
                                                               os.path.join(model_save_path, model_filename))
                                                    with open(os.path.join(txt_save_path, param_filename), 'w') as f:
                                                        for key, value in best_config.items():
                                                            f.write(f"{key}: {value}\n")
                                                    print(f'Model and parameters saved.')
                                                    good_model_count += 1

                                                    # Save best accuracy details
                                                    best_accuracy_file = os.path.join(base_path,
                                                                                      "best_accuracy_details.txt")
                                                    with open(best_accuracy_file, 'w') as f:
                                                        for key, value in best_config.items():
                                                            f.write(f"{key}: {value}\n")

    elif select == 't':
        # List available models
        model_files = [f for f in os.listdir(model_save_path) if f.startswith('best_model_') and f.endswith('.pth')]
        if not model_files:
            print("No models found in the model save path.")
            return

        print("Available models:")
        model_numbers = []
        for model_file in model_files:
            model_num = model_file.split('_')[-1].split('.')[0]
            model_numbers.append(model_num)
            print(f"{model_num}")

        selected_model_num = input('Enter the model number you want to select: ')
        if selected_model_num not in model_numbers:
            print("Invalid model number selected.")
            return

        # Load the selected model and configuration
        model_filename = os.path.join(model_save_path, f'best_model_{selected_model_num}.pth')
        config_filename = os.path.join(txt_save_path, f'best_model_{selected_model_num}.txt')

        if not os.path.exists(model_filename) or not os.path.exists(config_filename):
            print("Selected model or configuration file not found.")
            return

        # Load the best configuration
        best_config = {}
        with open(config_filename, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                best_config[key] = value

        # Extract hyperparameters
        num_layers = int(best_config['num_layers'])
        hidden_dim = int(best_config['hidden_dim'])
        dropout = float(best_config['dropout_rate'])
        input_size = 130  # Adjust based on your data
        output_dim = int(best_config['output_dim'])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')

        # Initialize the model
        model = GestureCNNLSTM(input_size, hidden_dim, output_dim,
                               num_layers=num_layers,
                               dropout_rate=dropout).to(device)
        # Load the model weights
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.eval()

        # Load the data from predict_path
        predict_data, filenames = load_predict_data(predict_path)

        # Create the dataset
        predict_dataset = GestureDatasetPredict(predict_data, filenames)

        # Create DataLoader with collate_fn
        def collate_fn_predict(batch):
            batch.sort(key=lambda x: len(x[0]), reverse=True)
            sequences, lengths, filenames_batch = zip(*[(torch.tensor(seq, dtype=torch.float32), len(seq), filename) for seq, length, filename in batch])
            sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            lengths = torch.tensor(lengths, dtype=torch.long)
            return sequences_padded, lengths, filenames_batch

        predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_predict)

        # Run predictions
        all_predictions = []
        all_filenames = []
        with torch.no_grad():
            for data, lengths, filenames_batch in predict_loader:
                data, lengths = data.to(device), lengths.to(device)
                lengths, perm_idx = lengths.sort(0, descending=True)
                data = data[perm_idx]
                filenames_batch = [filenames_batch[i] for i in perm_idx.cpu().numpy()]
                packed_input = nn.utils.rnn.pack_padded_sequence(data, lengths.cpu(), batch_first=True)
                outputs = model(packed_input)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_filenames.extend(filenames_batch)

        # Print the predictions with filenames
        for filename, pred in zip(all_filenames, all_predictions):
            print(f'File {filename}: Predicted Gesture: {pred}')

    else:
        print('Invalid input.')
        main()


if __name__ == '__main__':
    main()
