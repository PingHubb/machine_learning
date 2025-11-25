import os
import numpy as np
import random
import pandas as pd
import sys
import math
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR


class GestureDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data_list = data_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.labels_list[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class GestureDatasetPredict(Dataset):
    def __init__(self, data_list, filenames):
        self.data_list = data_list
        self.filenames = filenames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        filename = self.filenames[idx]
        return torch.tensor(data, dtype=torch.float32), filename


def set_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# The definitive, robust data loading function. Replace your old one with this.
def load_data_from_file(file_path):
    """
    Loads data from a space-separated file.
    Handles empty files gracefully by returning an empty numpy array.
    """
    try:
        # Use sep='\s+' to handle any amount of whitespace. engine='python' is more robust.
        data = pd.read_csv(file_path, sep='\s+', header=None, engine='python')
        # If the file was empty, pandas creates a DataFrame with no columns.
        # We explicitly check for this and return a correctly shaped empty array.
        if data.empty:
            # Return an empty array with 0 rows but the correct number of columns (56)
            # This is optional but good practice. A simple empty array also works.
            return np.empty((0, 56))
        return data.values
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        # Return an empty array in case of any other reading error
        return np.empty((0, 56))


def load_gesture_data(gesture_folder):
    files = sorted([os.path.join(gesture_folder, f) for f in os.listdir(gesture_folder) if f.endswith('.txt')])
    trials = []
    for file in files:
        features = load_data_from_file(file)
        trials.append(features)
    return trials


def load_all_gestures(base_path):
    gesture_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if
                       os.path.isdir(os.path.join(base_path, d))]
    all_data = []
    all_labels = []
    for folder in gesture_folders:
        trials = load_gesture_data(folder)
        gesture_label = int(folder.split('_')[-1])
        for trial in trials:
            # Add a check to ensure data has the correct shape
            if trial.shape[1] != 56:
                print(f"Warning: File {folder} has incorrect feature count: {trial.shape[1]}. Expected 56.")
                continue  # Skip this malformed file
            all_data.append(trial)
            all_labels.append(gesture_label)
    return all_data, all_labels


# NEW VERSION
def load_predict_data(predict_path):
    # First, get all the file paths from the directory
    file_paths = [os.path.join(predict_path, f) for f in os.listdir(predict_path) if f.endswith('.txt')]
    files = sorted(file_paths, key=lambda p: int(os.path.basename(p).split('.')[0].split('_')[-1]))

    all_data = []
    filenames = []
    for file_path in files:
        features = load_data_from_file(file_path)

        if features.size == 0:
            print(f"Warning: Skipping empty or malformed file: {os.path.basename(file_path)}")
            continue

        all_data.append(features)
        filenames.append(os.path.basename(file_path))
    return all_data, filenames


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# --- FIX #2: UPDATED MODEL DIMENSIONS ---
class GestureConvTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0.5):
        super(GestureConvTransformer, self).__init__()
        self.d_model = d_model

        # --- Part 1: CNN with CORRECTED sensor grid dimensions ---
        self.height = 8  # New dimension
        self.width = 7  # New dimension

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 8x7 to 4x3
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> 4x3 to 2x1
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.height, self.width)
            cnn_output_size = self.conv(dummy_input).view(1, -1).size(1)

        # --- Part 2: Transformer for Temporal Analysis ---
        self.input_proj = nn.Linear(cnn_output_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # --- Part 3: Classifier Head ---
        self.classifier = nn.Linear(d_model, output_dim)

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = src.shape
        src_reshaped = src.contiguous().view(-1, 1, self.height, self.width)
        cnn_out = self.conv(src_reshaped)
        cnn_out_flat = cnn_out.view(batch_size, seq_len, -1)
        trans_input = self.input_proj(cnn_out_flat) * math.sqrt(self.d_model)
        trans_input = self.pos_encoder(trans_input.permute(1, 0, 2)).permute(1, 0, 2)
        output = self.transformer_encoder(trans_input, src_key_padding_mask=src_padding_mask)
        output = output[:, 0, :]
        logits = self.classifier(output)
        return logits


def train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs, fold,
                       enable_scheduler, is_test=False):
    history = {'epoch': [], 'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for data, labels, padding_mask in train_loader:
            data, labels, padding_mask = data.to(device), labels.to(device), padding_mask.to(device)
            optimizer.zero_grad()
            outputs = model(data, padding_mask)
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

        if is_test:
            test_loss, test_accuracy = evaluate(val_loader, model, criterion, device)
            history['test_loss'] = history.get('test_loss', []) + [test_loss]
            history['test_accuracy'] = history.get('test_accuracy', []) + [test_accuracy]
            print(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
        else:
            val_loss, val_accuracy = evaluate(val_loader, model, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            print(
                f'Fold {fold + 1}, Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        if enable_scheduler:
            scheduler.step()
        history['epoch'].append(epoch + 1)
    return history


def evaluate(data_loader, model, criterion, device):
    total_loss, correct, total = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data, labels, padding_mask in data_loader:
            data, labels, padding_mask = data.to(device), labels.to(device), padding_mask.to(device)
            outputs = model(data, padding_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(data_loader), 100 * correct / total


def main():
    set_seed(53)
    select = input('Enter option number (Enter 1 to train the model, or 2 to test the model): ')

    base_path = '//ai'
    train_path = os.path.join(base_path, 'training_data')
    test_path = os.path.join(base_path, 'testing_data')
    predict_path = '//predict/'
    model_save_path = '//models_hybrid/'
    csv_save_path = '//csv_hybrid/'
    txt_save_path = '//txt_hybrid/'

    for path in [model_save_path, csv_save_path, txt_save_path]:
        if not os.path.exists(path): os.makedirs(path)

    if select == '1':
        train_data, train_labels = load_all_gestures(train_path)
        # test_data, test_labels = load_all_gestures(test_path) # You might not need test data during hyperparameter search
        train_dataset = GestureDataset(train_data, train_labels)
        # test_dataset = GestureDataset(test_data, test_labels)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        output_dim = len(set(train_labels))
        if output_dim == 0:
            print("Error: No training data was loaded successfully. Check data paths and file formats.")
            return

        # --- Hyperparameters for Hybrid Model ---
        d_models = [64, 128]  # Adjusted since CNN output is smaller
        n_heads = [4, 8]
        num_enc_layers = [2, 4]
        epochs_range = [50, 100]
        learning_rates = [0.0001, 0.00005]
        k_folds = 5
        batch_size = 32
        dropout = 0.3
        weight_decay = 0.0001

        best_accuracy = 0
        good_model_count = 1

        def collate_fn(batch):
            batch.sort(key=lambda x: len(x[0]), reverse=True)
            sequences, labels = zip(*batch)
            sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
            lengths = [len(seq) for seq in sequences]
            padding_mask = torch.zeros(sequences_padded.shape[0], sequences_padded.shape[1], dtype=torch.bool)
            for i, length in enumerate(lengths): padding_mask[i, length:] = True
            labels = torch.stack(labels)
            return sequences_padded, labels, padding_mask

        for d_model in d_models:
            for nhead in n_heads:
                if d_model % nhead != 0: continue
                for num_layers in num_enc_layers:
                    for epochs in epochs_range:
                        for lr in learning_rates:
                            print(
                                f"Testing config: d_model={d_model}, nhead={nhead}, layers={num_layers}, epochs={epochs}, lr={lr}")

                            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=47)
                            fold_val_accuracies = []
                            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
                                train_subsampler = Subset(train_dataset, train_idx)
                                val_subsampler = Subset(train_dataset, val_idx)
                                train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True,
                                                          collate_fn=collate_fn)
                                val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False,
                                                        collate_fn=collate_fn)

                                model = GestureConvTransformer(
                                    d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                    dim_feedforward=d_model * 4, output_dim=output_dim, dropout=dropout
                                ).to(device)

                                criterion = nn.CrossEntropyLoss()
                                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                                scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

                                history = train_and_evaluate(train_loader, val_loader, model, criterion, optimizer,
                                                             scheduler, device, epochs, fold, True)
                                fold_val_accuracies.append(history['val_accuracy'][-1])

                            avg_val_accuracy = sum(fold_val_accuracies) / len(fold_val_accuracies)
                            print(f'Average Validation Accuracy for config: {avg_val_accuracy:.2f}%')

                            if avg_val_accuracy > best_accuracy:
                                best_accuracy = avg_val_accuracy
                                best_config = {
                                    'd_model': d_model, 'nhead': nhead, 'num_layers': num_layers,
                                    'epochs': epochs, 'learning_rate': lr, 'output_dim': output_dim,
                                    'validation_accuracy': avg_val_accuracy
                                }
                                print(f"*** New best model found with val acc: {best_accuracy:.2f}% ***")

                                torch.save(model.state_dict(),
                                           os.path.join(model_save_path, f'best_hybrid_model_{good_model_count}.pth'))
                                with open(os.path.join(txt_save_path, f'best_hybrid_model_{good_model_count}.txt'),
                                          'w') as f:
                                    for key, value in best_config.items(): f.write(f"{key}: {value}\n")
                                good_model_count += 1

    elif select == '2':
        model_files = [f for f in os.listdir(model_save_path) if
                       f.startswith('best_hybrid_model_') and f.endswith('.pth')]
        if not model_files:
            print("No Hybrid models found.")
            return

        print("Available models:")
        for model_file in sorted(model_files): print(model_file)

        selected_model_file = input('Enter the full model filename for prediction: ')
        model_filename = os.path.join(model_save_path, selected_model_file)
        config_filename = os.path.join(txt_save_path, selected_model_file.replace('.pth', '.txt'))

        if not os.path.exists(model_filename) or not os.path.exists(config_filename):
            print("Selected model or configuration file not found.")
            return

        config = {}
        with open(config_filename, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                try:
                    config[key] = int(value)
                except ValueError:
                    config[key] = float(value)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GestureConvTransformer(
            d_model=config['d_model'], nhead=config['nhead'],
            num_encoder_layers=config['num_layers'], dim_feedforward=config['d_model'] * 4,
            output_dim=config['output_dim'], dropout=0.3
        ).to(device)

        model.load_state_dict(torch.load(model_filename, map_location=device, weights_only=True))

        model.eval()

        predict_data, filenames = load_predict_data(predict_path)
        predict_dataset = GestureDatasetPredict(predict_data, filenames)

        def collate_fn_predict(batch):
            sequences, fns = zip(*batch)
            sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
            padding_mask = torch.zeros(sequences_padded.shape[0], sequences_padded.shape[1], dtype=torch.bool)
            lengths = [len(seq) for seq in sequences]
            for i, length in enumerate(lengths): padding_mask[i, length:] = True
            return sequences_padded, fns, padding_mask

        predict_loader = DataLoader(predict_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_predict)

        print("\n--- Predictions ---")
        with torch.no_grad():
            for data, fns, mask in predict_loader:
                data, mask = data.to(device), mask.to(device)
                outputs = model(data, mask)
                _, predicted = torch.max(outputs.data, 1)
                for filename, pred in zip(fns, predicted):
                    print(f'File {filename}: Predicted Gesture: {pred.item()}')
    else:
        print('Invalid input.')
        main()


if __name__ == '__main__':
    main()