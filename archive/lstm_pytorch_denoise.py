import os
import numpy as np
import random
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Set random seed for reproducibility
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ----------------------------
# Data Preparation Functions
# ----------------------------

def load_data_from_file(file_path):
    data = pd.read_csv(file_path, sep=" ", header=None)
    # Ensure the data has 100 features per time step (10x10 grid)
    if data.shape[1] != 100:
        print(f"Warning: {file_path} has {data.shape[1]} features instead of 100.")
        # Handle accordingly: trim, pad, or skip the file
        return None
    return data.values

def normalize_data(data):
    """
    Normalize data to the range [0, 1].
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return data - min_val  # Avoid division by zero
    return (data - min_val) / (max_val - min_val)

def estimate_clean_signal(noisy_signals):
    """
    Estimate the clean signal by averaging multiple noisy recordings.
    """
    # First, ensure all sequences are of the same length
    lengths = [signal.shape[0] for signal in noisy_signals]
    min_len = min(lengths)
    # Truncate sequences to the minimum length to avoid padding
    truncated_signals = [signal[:min_len] for signal in noisy_signals]
    stacked_signals = np.stack(truncated_signals, axis=0)
    clean_signal = np.mean(stacked_signals, axis=0)
    return clean_signal

# ----------------------------
# Custom Dataset
# ----------------------------

class SensorDenoiseDataset(Dataset):
    def __init__(self, noisy_data_list, clean_data_list):
        self.noisy_data_list = noisy_data_list
        self.clean_data_list = clean_data_list

    def __len__(self):
        return len(self.noisy_data_list)

    def __getitem__(self, idx):
        noisy_input = self.noisy_data_list[idx]
        clean_target = self.clean_data_list[idx]
        if noisy_input.shape[1] != 100 or clean_target.shape[1] != 100:
            raise ValueError(f"Data sample {idx} has incorrect feature size.")
        # Convert to tensors
        noisy_input = torch.tensor(noisy_input, dtype=torch.float32)
        clean_target = torch.tensor(clean_target, dtype=torch.float32)
        return noisy_input, clean_target

# ----------------------------
# Collate Function
# ----------------------------

def collate_fn(batch):
    # Batch is a list of (noisy_input, clean_target)
    sequences_noisy, sequences_clean = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences_noisy]
    # No need to sort if enforce_sorted=False
    return sequences_noisy, sequences_clean, lengths

# ----------------------------
# Model Definition
# ----------------------------

class DenoisingCNNLSTMAutoencoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout_rate):
        super(DenoisingCNNLSTMAutoencoder, self).__init__()
        # Sensor grid dimensions
        self.height = 10
        self.width = 10

        # Encoder CNN
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Input channels = 1
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output size: (16, H/2, W/2)
        )

        # Compute the size after CNN layers
        self.cnn_output_size = 16 * (self.height // 2) * (self.width // 2)

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=self.cnn_output_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Decoder CNN
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),  # Upsample back to original size
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x_packed):
        x, lengths = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        batch_size, seq_len, feature_size = x.size()
        expected_feature_size = self.height * self.width  # Should be 100
        if feature_size != expected_feature_size:
            raise ValueError(f"Expected feature size {expected_feature_size}, but got {feature_size}")
        x = x.contiguous().view(-1, 1, self.height, self.width)
        # Encoder CNN
        x = self.encoder_cnn(x)
        x = x.view(batch_size, seq_len, -1)
        # Pack sequences
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Encoder LSTM
        x_packed, _ = self.encoder_lstm(x_packed)
        # Decoder LSTM
        x_packed, _ = self.decoder_lstm(x_packed)
        # Unpack sequences
        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        # Reshape for decoder CNN
        x = x.contiguous().view(-1, 16, self.height // 2, self.width // 2)
        # Decoder CNN
        x = self.decoder_cnn(x)
        x = x.view(batch_size, seq_len, 1, self.height, self.width)
        return x  # Returns a Tensor

# ----------------------------
# Training Function
# ----------------------------

def train_autoencoder(train_loader, model, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for noisy_input_list, clean_target_list, lengths in train_loader:
            # Move data to device
            noisy_input_list = [seq.to(device) for seq in noisy_input_list]
            clean_target_list = [seq.to(device) for seq in clean_target_list]
            lengths = torch.tensor(lengths).to(device)

            # Pad sequences
            noisy_input_padded = nn.utils.rnn.pad_sequence(noisy_input_list, batch_first=True)
            clean_target_padded = nn.utils.rnn.pad_sequence(clean_target_list, batch_first=True)

            # Pack the noisy inputs
            noisy_input_packed = nn.utils.rnn.pack_padded_sequence(
                noisy_input_padded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            optimizer.zero_grad()
            outputs = model(noisy_input_packed)  # Outputs is a Tensor

            # Flatten outputs and targets
            batch_size, seq_len = outputs.size(0), outputs.size(1)
            outputs_flat = outputs.view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, feature_size]
            clean_target_flat = clean_target_padded.view(batch_size, seq_len, -1).to(device)

            # Create mask
            max_length = seq_len
            mask = torch.arange(max_length).expand(len(lengths), max_length).to(device) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(2)  # Shape: [batch_size, seq_len, 1]

            # Compute loss
            loss = (criterion(outputs_flat, clean_target_flat) * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

# ----------------------------
# Denoising Function
# ----------------------------

def denoise_signal(model, noisy_input, device):
    model.eval()
    with torch.no_grad():
        seq_len = noisy_input.shape[0]
        # Ensure the input has 100 features per time step
        if noisy_input.shape[1] != 100:
            print(f"Input has {noisy_input.shape[1]} features instead of 100.")
            return None
        noisy_input = torch.tensor(noisy_input, dtype=torch.float32).to(device)
        lengths = [seq_len]
        # Pad input (though it's a single sequence, padding isn't necessary)
        noisy_input_padded = noisy_input.unsqueeze(0)  # Shape: [1, seq_len, 100]
        # Pack the input
        noisy_input_packed = nn.utils.rnn.pack_padded_sequence(
            noisy_input_padded, lengths, batch_first=True, enforce_sorted=False
        )
        outputs = model(noisy_input_packed)
        outputs_flat = outputs.view(seq_len, -1).cpu().numpy()  # Shape: [seq_len, 100]
        return outputs_flat

# ----------------------------
# Main Functions
# ----------------------------

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Paths (modify these paths as per your directory structure)
    data_path = 'C:/dev/phd/ai/training_data'  # Directory containing noisy recordings
    model_save_dir = 'C:/dev/phd/models'
    model_filename = 'denoising_autoencoder.pth'
    model_save_path = os.path.join(model_save_dir, model_filename)

    # Ensure the model save directory exists
    os.makedirs(model_save_dir, exist_ok=True)

    # Hyperparameters
    hidden_dim = 128
    num_layers = 2
    dropout_rate = 0.3
    learning_rate = 0.001
    batch_size = 16
    epochs = 50

    # ----------------------------
    # Data Loading and Preparation
    # ----------------------------

    # Load all recordings (assuming files are named appropriately)
    # Each signal should have multiple noisy recordings
    all_recordings = []  # List of lists. Each inner list contains noisy recordings of the same signal.

    signal_folders = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for signal_folder in signal_folders:
        recordings = []
        files = sorted([os.path.join(signal_folder, f) for f in os.listdir(signal_folder) if f.endswith('.txt')])
        for file in files:
            data = load_data_from_file(file)
            if data is None:
                continue  # Skip files with incorrect feature size
            data = normalize_data(data)
            print(f"Loaded {file}: shape {data.shape}")
            recordings.append(data)
        if recordings:
            all_recordings.append(recordings)
        else:
            print(f"No valid recordings found in {signal_folder}")

    if not all_recordings:
        print("No valid recordings found in the data path.")
        return

    # Prepare training pairs
    noisy_data_list = []
    clean_data_list = []

    for recordings_of_same_signal in all_recordings:
        # Estimate clean signal
        clean_estimated = estimate_clean_signal(recordings_of_same_signal)
        for noisy_signal in recordings_of_same_signal:
            # Truncate noisy_signal to match clean_estimated length
            noisy_signal = noisy_signal[:clean_estimated.shape[0]]
            noisy_data_list.append(noisy_signal)
            clean_data_list.append(clean_estimated)

    # Convert lists to numpy arrays
    noisy_data_list = [np.array(seq) for seq in noisy_data_list]
    clean_data_list = [np.array(seq) for seq in clean_data_list]

    # Create dataset and dataloader
    dataset = SensorDenoiseDataset(noisy_data_list, clean_data_list)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # ----------------------------
    # Initialize Model, Loss, Optimizer
    # ----------------------------

    model = DenoisingCNNLSTMAutoencoder(hidden_dim, num_layers, dropout_rate).to(device)
    criterion = nn.MSELoss(reduction='none')  # Use reduction='none' to compute loss per element
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ----------------------------
    # Train the Model
    # ----------------------------

    print("Starting training...")
    train_autoencoder(train_loader, model, criterion, optimizer, device, epochs)

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def denoise_data():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Paths
    model_load_path = 'C:/dev/phd/models/denoising_autoencoder.pth'
    new_noisy_file = 'C:/dev/phd/ai/testing_data/gesture_0/105.txt'  # Replace with your actual file path

    # Load the model
    hidden_dim = 128
    num_layers = 2
    dropout_rate = 0.3
    model = DenoisingCNNLSTMAutoencoder(hidden_dim, num_layers, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    # Load new noisy data
    new_noisy_input = load_data_from_file(new_noisy_file)
    if new_noisy_input is None:
        print("Failed to load new noisy data.")
        return
    new_noisy_input = normalize_data(new_noisy_input)

    # Denoise the signal
    denoised_output = denoise_signal(model, new_noisy_input, device)
    if denoised_output is None:
        print("Denoising failed.")
        return

    # Save or process the denoised output as needed
    # For example, save to a file
    denoised_output_flat = denoised_output.reshape(-1, 100)  # 100 = 10 * 10
    output_file = 'denoised_output.txt'
    np.savetxt(output_file, denoised_output_flat)
    print(f"Denoising completed and saved to {output_file}")

def main():
    print("Please select an option:")
    print("1: Train the model")
    print("2: Denoise new data")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        train_model()
    elif choice == '2':
        denoise_data()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == '__main__':
    main()
