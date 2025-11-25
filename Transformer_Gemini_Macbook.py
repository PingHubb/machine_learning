import os
import numpy as np
import random
import pandas as pd
import sys
import math
import shutil
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  # For plotting
import matplotlib.pyplot as plt  # For plotting
import time

# =============================================================================
# 1. GLOBAL CONSTANTS & CONFIG :)
# =============================================================================
SENSOR_ROWS = 9  # 13 for elbow
SENSOR_COLS = 10  # 10 for elbow
NUM_SENSORS = SENSOR_ROWS * SENSOR_COLS
print(f"--- Using Sensor Dimensions: {SENSOR_ROWS} rows x {SENSOR_COLS} cols ({NUM_SENSORS} total sensors) ---")

# =============================================================================
# 2. CORE UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_data_from_file(file_path):
    try:
        data = pd.read_csv(file_path, sep='\s+', header=None, engine='python')
        if data.empty or data.shape[1] != NUM_SENSORS:
            return None
        return data.values
    except Exception:
        return None

def parse_config_file(file_path):
    config = {}
    if not os.path.exists(file_path):
        print(f"Warning: Config file not found at {file_path}")
        return config
    with open(file_path, 'r') as f:
        for line in f:
            if ': ' in line:
                key, value = line.strip().split(': ', 1)
                try:
                    config[key] = int(value)
                except ValueError:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        config[key] = value
    return config

# =============================================================================
# 3. DATA STANDARDIZATION FUNCTIONS
# =============================================================================
def get_standardization_params(data_list):
    print("Calculating standardization parameters (mean, std) from data...")
    if not data_list: return 0, 1
    full_dataset = np.vstack(data_list)
    global_mean = full_dataset.mean()
    global_std = full_dataset.std()
    if global_std == 0: global_std = 1e-6
    print(f"Global Mean: {global_mean:.4f}, Global Std Dev: {global_std:.4f}")
    return global_mean, global_std

def standardize_data(data_list, global_mean, global_std):
    return [(data - global_mean) / global_std for data in data_list]

# =============================================================================
# 4. PYTORCH DATASET CLASSES
# =============================================================================
class UnlabeledGestureDataset(Dataset):
    def __init__(self, data_list): self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return torch.tensor(self.data_list[idx], dtype=torch.float32)

class ThreeLevelDataset(Dataset):
    def __init__(self, data_list, finger_labels, gesture_labels, quality_labels):
        self.data_list = data_list
        self.finger_labels = finger_labels
        self.gesture_labels = gesture_labels
        self.quality_labels = quality_labels
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        data = torch.tensor(self.data_list[idx], dtype=torch.float32)
        finger_label = torch.tensor(self.finger_labels[idx], dtype=torch.long)
        gesture_label = torch.tensor(self.gesture_labels[idx], dtype=torch.long)
        quality_label = torch.tensor(self.quality_labels[idx], dtype=torch.long)
        return data, finger_label, gesture_label, quality_label

# =============================================================================
# 5. DATA LOADING FUNCTIONS
# =============================================================================
def load_unlabeled_from_flat_dir(path):
    all_data = []
    if not os.path.exists(path): print(f"Warning: Directory not found {path}"); return all_data
    for f in os.listdir(path):
        if f.endswith('.txt'):
            trial = load_data_from_file(os.path.join(path, f))
            if trial is not None: all_data.append(trial)
    print(all_data)
    return all_data

def load_all_clean_data(*paths):
    all_data = []
    for path in paths:
        if not os.path.exists(path): print(f"Warning: Clean data path not found: {path}"); continue
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith('.txt'):
                    trial = load_data_from_file(os.path.join(root, f))
                    if trial is not None: all_data.append(trial)
    return all_data

def load_three_level_data(base_path):
    all_data, finger_labels, gesture_labels, quality_labels = [], [], [], []
    if not os.path.exists(base_path):
        print(f"Warning: Labeled data path not found: {base_path}")
        return all_data, finger_labels, gesture_labels, quality_labels
    for finger_dir in sorted(os.listdir(base_path)):
        try:
            finger_label = int(finger_dir.split('_')[-1])
            finger_path = os.path.join(base_path, finger_dir)
            if not os.path.isdir(finger_path): continue
            for gesture_dir in sorted(os.listdir(finger_path)):
                try:
                    gesture_label = int(gesture_dir.split('_')[-1])
                    gesture_path = os.path.join(finger_path, gesture_dir)
                    if not os.path.isdir(gesture_path): continue
                    for quality_dir in sorted(os.listdir(gesture_path)):
                        quality_label = 0 if quality_dir == 'good' else 1
                        quality_path = os.path.join(gesture_path, quality_dir)
                        if not os.path.isdir(quality_path): continue
                        for f in sorted(os.listdir(quality_path)):
                            if f.endswith('.txt'):
                                trial = load_data_from_file(os.path.join(quality_path, f))
                                if trial is not None:
                                    all_data.append(trial)
                                    finger_labels.append(finger_label)
                                    gesture_labels.append(gesture_label)
                                    quality_labels.append(quality_label)
                except (ValueError, IndexError): continue
        except (ValueError, IndexError): continue
    return all_data, finger_labels, gesture_labels, quality_labels

def load_predict_data(predict_path):
    all_data, filenames = [], []
    if not os.path.exists(predict_path): print(f"Warning: Prediction directory not found {predict_path}"); return all_data, filenames
    file_paths = [os.path.join(predict_path, f) for f in os.listdir(predict_path) if f.endswith('.txt')]
    try:
        files_sorted = sorted(file_paths, key=lambda p: int(os.path.basename(p).split('.')[0].split('_')[-1]))
    except (ValueError, IndexError):
        files_sorted = sorted(file_paths)
    for file_path in files_sorted:
        features = load_data_from_file(file_path)
        if features is not None: all_data.append(features); filenames.append(os.path.basename(file_path))
    return all_data, filenames

# =============================================================================
# 6. DATA AUGMENTATION & PREPARATION FUNCTIONS
# =============================================================================
def _process_and_save_gesture(base_gesture, groundtruth_files, window_size, output_path, filename):
    """
    Helper to handle padding, shifting, and saving a single gesture.
    MODIFIED: The gesture segment is now always placed at the END of the
    fixed-size window, with padding only at the beginning.
    """
    num_sensors = SENSOR_ROWS * SENSOR_COLS

    # Step 1: Create a realistic background template of the correct window_size
    initial_gt = random.choice(groundtruth_files)
    gt_len = len(initial_gt)

    if gt_len == window_size:
        background_template = random.choice(groundtruth_files).copy()
    elif gt_len > window_size:
        start = random.randint(0, gt_len - window_size)
        background_template = initial_gt[start: start + window_size].copy()
    else:
        num_files_needed = math.ceil(window_size / gt_len)
        files_to_stitch = random.choices(groundtruth_files, k=num_files_needed)
        stitched_gt = np.vstack(files_to_stitch)
        stitched_len = len(stitched_gt)
        start = random.randint(0, stitched_len - window_size)
        background_template = stitched_gt[start: start + window_size].copy()

    # Step 2: Prepare the gesture to be placed
    gesture_len = len(base_gesture)
    # if gesture_len > window_size:
    #     # If the gesture is longer than the window, we take the LAST part of it
    #     start_index = gesture_len - window_size
    #     gesture_to_place = base_gesture[start_index:]
    if gesture_len > window_size:
        gesture_to_place = base_gesture[:window_size]
    else:
        gesture_to_place = base_gesture

    # Step 3: Create the final window and paste the gesture
    final_window = background_template
    current_len = len(gesture_to_place)

    # --- THIS IS THE KEY CHANGE ---
    max_start_pos = window_size - current_len
    start_pos = random.randint(0, max_start_pos)

    # Paste the gesture at the calculated end position
    final_window[start_pos: start_pos + current_len] = gesture_to_place

    # Step 4: Save the file
    np.savetxt(os.path.join(output_path, filename), final_window, fmt='%.4f')


# REPLACEMENT FOR: prepare_three_level_data
def prepare_three_level_data(clean_data_path, broken_data_path, groundtruth_data_path, output_base_path,
                             config_save_path,  # The path for the new config file
                             num_good_versions=10, num_damaged_versions=1, window_size=50, min_gesture_len=10):
    """
    Creates a three-level data structure.
    Saves the key data generation parameters to a config file.
    """
    print(f"--- Preparing Data: Slicing random segments (min {min_gesture_len} frames) ---")

    broken_signals = []
    if num_damaged_versions > 0:
        print(f"Loading broken signals from: {broken_data_path}")
        broken_signals = load_unlabeled_from_flat_dir(broken_data_path)
        if not broken_signals:
            print("Error: No broken signal data found, but num_damaged_versions > 0. Aborting.")
            return
    else:
        print("num_damaged_versions is 0, skipping loading of broken signals.")

    print(f"Loading ground truth background from: {groundtruth_data_path}")
    groundtruth_files = load_unlabeled_from_flat_dir(groundtruth_data_path)
    if not groundtruth_files:
        print("Error: No ground truth background data found. Aborting.")
        return
    print(f"Found {len(groundtruth_files)} ground truth files to use as background.")

    if os.path.exists(output_base_path):
        shutil.rmtree(output_base_path)
    os.makedirs(output_base_path)

    print(f"Processing clean data from: {clean_data_path}")
    total_good_files = 0
    total_damaged_files = 0

    for finger_dir in os.listdir(clean_data_path):
        finger_path = os.path.join(clean_data_path, finger_dir)
        if not os.path.isdir(finger_path): continue

        for gesture_dir in os.listdir(finger_path):
            gesture_path = os.path.join(finger_path, gesture_dir)
            if not os.path.isdir(gesture_path): continue

            output_good_path = os.path.join(output_base_path, finger_dir, gesture_dir, 'good')
            os.makedirs(output_good_path)

            if num_damaged_versions > 0:
                output_damaged_path = os.path.join(output_base_path, finger_dir, gesture_dir, 'damaged')
                os.makedirs(output_damaged_path)

            for filename in os.listdir(gesture_path):
                if not filename.endswith('.txt'): continue

                original_clean_gesture = load_data_from_file(os.path.join(gesture_path, filename))
                if original_clean_gesture is None: continue

                original_len = len(original_clean_gesture)
                if original_len < min_gesture_len:
                    continue

                # --- Generate 'good' versions with random slices ---
                for i in range(num_good_versions):
                    segment_len = random.randint(min_gesture_len, original_len)
                    max_start = original_len - segment_len
                    start_index = random.randint(0, max_start)
                    gesture_segment = original_clean_gesture[start_index: start_index + segment_len]

                    _process_and_save_gesture(
                        base_gesture=gesture_segment,
                        groundtruth_files=groundtruth_files,
                        window_size=window_size,
                        output_path=output_good_path,
                        filename=f"good_v{i}_{filename}"
                    )
                    total_good_files += 1

                # --- Generate 'damaged' versions with random slices (only if requested) ---
                if num_damaged_versions > 0:
                    for i in range(num_damaged_versions):
                        segment_len = random.randint(min_gesture_len, original_len)
                        max_start = original_len - segment_len
                        start_index = random.randint(0, max_start)
                        gesture_segment = original_clean_gesture[start_index: start_index + segment_len]

                        broken_signal = random.choice(broken_signals)
                        seg_len, broken_len = len(gesture_segment), len(broken_signal)
                        if broken_len >= seg_len:
                            start = random.randint(0, broken_len - seg_len)
                            broken_part = broken_signal[start: start + seg_len]
                        else:
                            repeats = (seg_len // broken_len) + 1 if broken_len > 0 else 1
                            broken_part = np.tile(broken_signal, (repeats, 1))[:seg_len]

                        damaged_segment = gesture_segment + broken_part

                        _process_and_save_gesture(
                            base_gesture=damaged_segment,
                            groundtruth_files=groundtruth_files,
                            window_size=window_size,
                            output_path=output_damaged_path,
                            filename=f"damaged_v{i}_{filename}"
                        )
                        total_damaged_files += 1

    print("-" * 50)
    print("Data preparation complete.")
    print(f"Total shifted 'good' files generated: {total_good_files}")
    print(f"Total shifted 'damaged' files generated: {total_damaged_files}")

    # --- NEW: Save the configuration used to generate this data ---
    print(f"Saving data preparation configuration to {config_save_path}")
    with open(config_save_path, 'w') as f:
        f.write(f"num_good_versions: {num_good_versions}\n")
        f.write(f"num_damaged_versions: {num_damaged_versions}\n")
        f.write(f"window_size: {window_size}\n")
        f.write(f"min_gesture_len: {min_gesture_len}\n")
    print("Configuration saved.")

# =============================================================================
# 7. MODEL ARCHITECTURE CLASSES
# =============================================================================
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
        # x = x + self.pe[:x.size(0)]
        # return self.dropout(x)
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)

class GestureBackbone(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.height, self.width = SENSOR_ROWS, SENSOR_COLS
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        with torch.no_grad():
            cnn_output_size = self.conv(torch.zeros(1, 1, self.height, self.width)).view(1, -1).size(1)
        self.input_proj = nn.Linear(cnn_output_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = src.shape
        src_reshaped = src.contiguous().view(-1, 1, self.height, self.width)
        cnn_out = self.conv(src_reshaped).view(batch_size, seq_len, -1)
        trans_input = self.input_proj(cnn_out) * math.sqrt(self.d_model)
        trans_input = self.pos_encoder(trans_input)
        return self.transformer_encoder(trans_input, src_key_padding_mask=src_padding_mask)
        # batch_size, seq_len, _ = src.shape;
        # src_reshaped = src.contiguous().view(-1, 1, self.height, self.width)
        # cnn_out = self.conv(src_reshaped).view(batch_size, seq_len, -1);
        # trans_input = self.input_proj(cnn_out) * math.sqrt(self.d_model)
        # trans_input = self.pos_encoder(trans_input.permute(1, 0, 2)).permute(1, 0, 2)
        # return self.transformer_encoder(trans_input, src_key_padding_mask=src_padding_mask)

class ThreeLevelHierarchicalModel(nn.Module):
    def __init__(self, backbone, d_model, num_finger_classes, num_gesture_classes, num_quality_classes):
        super().__init__()
        self.backbone = backbone
        self.finger_classifier = nn.Linear(d_model, num_finger_classes)
        self.gesture_classifier = nn.Linear(d_model, num_gesture_classes)
        self.quality_classifier = nn.Linear(d_model, num_quality_classes)
    def forward(self, src, src_padding_mask):
        representations = self.backbone(src, src_padding_mask)
        cls_representation = representations[:, 0, :]
        finger_logits = self.finger_classifier(cls_representation)
        gesture_logits = self.gesture_classifier(cls_representation)
        quality_logits = self.quality_classifier(cls_representation)
        return finger_logits, gesture_logits, quality_logits

class ThreeLevelLSTMModel(nn.Module):
    """
    An LSTM-based model for hierarchical gesture classification.
    Serves as a baseline comparison to the Transformer model.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_finger_classes, num_gesture_classes,
                 num_quality_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # Bidirectional is often more powerful
        )

        # The input to the classifiers will be twice the hidden size because the LSTM is bidirectional
        classifier_input_size = hidden_size * 2

        self.finger_classifier = nn.Linear(classifier_input_size, num_finger_classes)
        self.gesture_classifier = nn.Linear(classifier_input_size, num_gesture_classes)
        self.quality_classifier = nn.Linear(classifier_input_size, num_quality_classes)

    def forward(self, src, src_padding_mask=None):  # Added padding_mask for compatibility, but it's not used by LSTM
        # LSTM layer
        # The output features from the last time step are used for classification
        # output shape: (batch, seq_len, num_directions * hidden_size)
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(src)

        # We concatenate the final hidden states from the forward and backward passes
        # hidden[-2,:,:] is the last forward hidden state
        # hidden[-1,:,:] is the last backward hidden state
        final_hidden_state = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # Pass the concatenated hidden state to the classifiers
        finger_logits = self.finger_classifier(final_hidden_state)
        gesture_logits = self.gesture_classifier(final_hidden_state)
        quality_logits = self.quality_classifier(final_hidden_state)

        return finger_logits, gesture_logits, quality_logits
# =============================================================================
# 8. MAIN FUNCTION & WORKFLOWS
# =============================================================================
def main():
    set_seed(77)
    # --- PATHS ---
    base_path = '/Users/cplam/Downloads/phd/ai'
    models_path = '/Users/cplam/Downloads/phd/models'
    clean_train_path = os.path.join(base_path, 'training_data_clean')
    broken_data_path = os.path.join(base_path, 'training_data_broken')
    three_level_train_path = os.path.join(base_path, 'training_data_3level')
    groundtruth_data_path = os.path.join(base_path, 'training_data_clean/num_finger_0/gesture_cylinder_groundtruth_0')
    predict_path = '/Users/cplam/Downloads/gesture_newest/gesture_cylinder_two_finger_up'

    # --- Track 3 Paths (Fixed-Length 3-Level) ---
    models_path_3level = os.path.join(models_path, 'models_3level')
    specialized_backbone_path = os.path.join(models_path_3level, 'backbone_3level.pth')

    # specialized_config_path = os.path.join(models_path_3level, 'backbone_3level_config.txt')
    specialized_config_path = os.path.join(models_path_3level, '3level_config_from_scratch.txt')
    # three_level_model_path = os.path.join(models_path_3level, '3level_model.pth')
    three_level_model_path = os.path.join(models_path_3level, '3level_model_from_scratch.pth')

    three_level_output_parameters_path = os.path.join(models_path_3level, '3level_output_parameters.txt')
    specialized_scaler_path = os.path.join(models_path_3level, 'scaler_3level.npz')
    data_prep_config_path = os.path.join(models_path_3level, 'data_prep_config.txt')

    # --- Track 5 Paths (LSTM Comparison) ---
    models_path_lstm = os.path.join(models_path, 'models_lstm')
    lstm_model_path = os.path.join(models_path_lstm, 'lstm_model.pth')
    lstm_config_path = os.path.join(models_path_lstm, 'lstm_config.txt')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'Using device: {device}')

    # --- MENU SYSTEM ---
    print("--- Gesture Recognition Research Pipeline ---")
    main_choice = input(
        "Select a track:\n"
        " 3: 3-Level Hierarchical Classifier (Fixed-Length)\n"
        " 5: LSTM Classifier Comparison\n"
        "Enter option: "
    )

    select = None

    if main_choice == '3':
        select = input(
            "\n--- 3-Level Hierarchical Classifier (Fixed-Length) ---\n"
            " 7: PREPARE 3-Level Fixed-Length Data\n"
            " 8: Pre-train Backbone (Hyperparameter Search)\n"
            " 8.1: Train Transformer from Scratch (Supervised Only)\n" # <-- ADD THIS LINE
            " 8.5: Final Pre-training Run (Longer Epochs)\n"
            " 9: Fine-tune 3-Level Classifier (Uses Pre-trained Model)\n" # <-- Updated description
            " 10: PREDICT with 3-Level Fixed-Length Classifier\n"
            "Enter option number: "
        )

    elif main_choice == '5':
        print("\n--- Starting Track 5: LSTM Classifier Training & Evaluation ---")
        print("Using the same fixed-length data as the Transformer for a direct comparison.")

        # --- This section is a near-direct copy of Option 9's setup ---
        # --- Step 1: Check for prerequisites ---
        if not os.path.exists(three_level_train_path):
            print(f"Error: Fixed-length training data not found at {three_level_train_path}. Run Option 7 first.")
            return
        # Use the same scaler as the transformer for a fair comparison
        if not os.path.exists(specialized_scaler_path):
            print(f"Error: Scaler not found at {specialized_scaler_path}. Run Option 8 first.")
            return

        # --- Step 2: Load Scaler and Data ---
        scaler_data = np.load(specialized_scaler_path)
        global_mean, global_std = scaler_data['mean'], scaler_data['std']
        print(f"Loaded SPECIALIZED scaler: Mean={global_mean:.4f}, Std={global_std:.4f}")

        print("Loading all 3-level data for splitting...")
        data, f_labels, g_labels, q_labels = load_three_level_data(three_level_train_path)
        if not data:
            print(f"Error: No data found in {three_level_train_path}.")
            return
        data_std = standardize_data(data, global_mean, global_std)

        indices = list(range(len(data_std)))
        train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=42, stratify=g_labels)
        train_data = [data_std[i] for i in train_indices]
        val_data = [data_std[i] for i in val_indices]
        train_f_labels = [f_labels[i] for i in train_indices]
        val_f_labels = [f_labels[i] for i in val_indices]
        train_g_labels = [g_labels[i] for i in train_indices]
        val_g_labels = [g_labels[i] for i in val_indices]
        train_q_labels = [q_labels[i] for i in train_indices]
        val_q_labels = [q_labels[i] for i in val_indices]
        print(f"Data split into {len(train_data)} training samples and {len(val_data)} validation samples.")

        train_dataset = ThreeLevelDataset(train_data, train_f_labels, train_g_labels, train_q_labels)
        val_dataset = ThreeLevelDataset(val_data, val_f_labels, val_g_labels, val_q_labels)

        def collate_fn_3level(batch):
            data_b, f_labels_b, g_labels_b, q_labels_b = zip(*batch)
            padded_data = torch.stack(data_b)
            return padded_data, torch.stack(f_labels_b), torch.stack(g_labels_b), torch.stack(q_labels_b), None

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_3level)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_3level)

        num_finger_classes = len(set(f_labels))
        num_gesture_classes = len(set(g_labels))
        num_quality_classes = len(set(q_labels))
        print(
            f"--> Total Detected Classes: {num_finger_classes} finger, {num_gesture_classes} gesture, {num_quality_classes} quality")

        # --- Step 3: LSTM Model, Optimizer, and Scheduler Setup ---
        # Define LSTM hyperparameters
        LSTM_HIDDEN_SIZE = 256
        LSTM_NUM_LAYERS = 2
        LSTM_DROPOUT = 0.5

        model = ThreeLevelLSTMModel(
            input_size=NUM_SENSORS,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            num_finger_classes=num_finger_classes,
            num_gesture_classes=num_gesture_classes,
            num_quality_classes=num_quality_classes,
            dropout=LSTM_DROPOUT
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # LSTMs can often handle a slightly higher starting LR
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

        # --- Step 4: Training and Validation Loop ---
        EPOCHS = 1
        best_val_gesture_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            # ... (The training and validation loop is identical to Option 9) ...
            # [This loop is omitted for brevity, but you would copy it here]
            # --- TRAINING PHASE ---
            total_loss, c_f, c_g, c_q, total_samples = 0, 0, 0, 0, 0
            for data, f_lbl, g_lbl, q_lbl, _ in train_loader:
                data, f_lbl, g_lbl, q_lbl = data.to(device), f_lbl.to(device), g_lbl.to(device), q_lbl.to(device)
                optimizer.zero_grad()
                f_logits, g_logits, q_logits = model(data)
                loss_f = criterion(f_logits, f_lbl)
                loss_g = criterion(g_logits, g_lbl)
                loss_q = criterion(q_logits, q_lbl)
                combined_loss = loss_f + loss_g + loss_q
                combined_loss.backward()
                optimizer.step()
                total_loss += combined_loss.item()
                c_f += (torch.argmax(f_logits, 1) == f_lbl).sum().item()
                c_g += (torch.argmax(g_logits, 1) == g_lbl).sum().item()
                c_q += (torch.argmax(q_logits, 1) == q_lbl).sum().item()
                total_samples += f_lbl.size(0)

            train_acc_f = 100 * c_f / total_samples
            train_acc_g = 100 * c_g / total_samples
            train_acc_q = 100 * c_q / total_samples
            print(
                f"Epoch {epoch + 1:02d} TRAIN | Loss: {total_loss / len(train_loader):.4f} | Acc F:{train_acc_f:.2f}% G:{train_acc_g:.2f}% Q:{train_acc_q:.2f}%")

            # --- VALIDATION PHASE ---
            model.eval()
            val_loss, vc_f, vc_g, vc_q, val_total_samples = 0, 0, 0, 0, 0
            with torch.no_grad():
                for data, f_lbl, g_lbl, q_lbl, _ in val_loader:
                    data, f_lbl, g_lbl, q_lbl = data.to(device), f_lbl.to(device), g_lbl.to(device), q_lbl.to(device)
                    f_logits, g_logits, q_logits = model(data)
                    loss_f = criterion(f_logits, f_lbl)
                    loss_g = criterion(g_logits, g_lbl)
                    loss_q = criterion(q_logits, q_lbl)
                    combined_loss = loss_f + loss_g + loss_q
                    val_loss += combined_loss.item()
                    vc_f += (torch.argmax(f_logits, 1) == f_lbl).sum().item()
                    vc_g += (torch.argmax(g_logits, 1) == g_lbl).sum().item()
                    vc_q += (torch.argmax(q_logits, 1) == q_lbl).sum().item()
                    val_total_samples += f_lbl.size(0)

            val_acc_f = 100 * vc_f / val_total_samples
            val_acc_g = 100 * vc_g / val_total_samples
            val_acc_q = 100 * vc_q / val_total_samples
            avg_val_loss = val_loss / len(val_loader)
            print(
                f"Epoch {epoch + 1:02d} VALID | Loss: {avg_val_loss:.4f} | Acc F:{val_acc_f:.2f}% G:{val_acc_g:.2f}% Q:{val_acc_q:.2f}%")
            scheduler.step(avg_val_loss)

            if val_acc_g > best_val_gesture_acc:
                print(f"*** New best LSTM validation accuracy: {val_acc_g:.2f}%. Saving model. ***")
                best_val_gesture_acc = val_acc_g
                torch.save(model.state_dict(), lstm_model_path)
                with open(lstm_config_path, 'w') as f:
                    f.write(f"NUM_FINGER_CLASSES: {num_finger_classes}\n")
                    f.write(f"NUM_GESTURE_CLASSES: {num_gesture_classes}\n")
                    f.write(f"NUM_QUALITY_CLASSES: {num_quality_classes}\n")

        print(f"LSTM model and config saved. Best validation gesture accuracy: {best_val_gesture_acc:.2f}%")

        # --- Step 5: Final Evaluation for LSTM ---
        # This is a direct copy of the advanced evaluation from Option 9
        # to ensure the outputs are perfectly comparable.
        print("\n" + "=" * 50)
        print("--- Final LSTM Model Evaluation on Validation Set ---")
        if not os.path.exists(lstm_model_path):
            print("No LSTM model was saved. Cannot perform final evaluation.")
            return

        print(f"Loading best LSTM model (Val Acc: {best_val_gesture_acc:.2f}%) for final report...")
        # Re-instantiate the model to load the best weights
        model = ThreeLevelLSTMModel(
            input_size=NUM_SENSORS, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS,
            num_finger_classes=num_finger_classes, num_gesture_classes=num_gesture_classes,
            num_quality_classes=num_quality_classes, dropout=LSTM_DROPOUT
        ).to(device)
        model.load_state_dict(torch.load(lstm_model_path, map_location=device))
        model.eval()

        # --- The entire advanced analysis section is copied here ---
        all_f_labels, all_f_preds, all_g_labels, all_g_preds, all_q_labels, all_q_preds = [], [], [], [], [], []
        with torch.no_grad():
            for data_batch, f_lbl, g_lbl, q_lbl, _ in val_loader:
                data_batch = data_batch.to(device)
                f_logits, g_logits, q_logits = model(data_batch)
                f_pred, g_pred, q_pred = torch.argmax(f_logits, 1), torch.argmax(g_logits, 1), torch.argmax(q_logits, 1)
                all_f_labels.extend(f_lbl.cpu().numpy());
                all_f_preds.extend(f_pred.cpu().numpy())
                all_g_labels.extend(g_lbl.cpu().numpy());
                all_g_preds.extend(g_pred.cpu().numpy())
                all_q_labels.extend(q_lbl.cpu().numpy());
                all_q_preds.extend(q_pred.cpu().numpy())

        finger_names = [f"{i} Finger(s)" for i in range(num_finger_classes)]
        gesture_names = [f"Gesture_{i}" for i in range(num_gesture_classes)]

        print("\n--- LSTM Finger Classification Report ---")
        print(classification_report(all_f_labels, all_f_preds, target_names=finger_names, digits=3))

        print("\n--- LSTM Gesture Classification Report (Overall) ---")
        print(classification_report(all_g_labels, all_g_preds, target_names=gesture_names, digits=3))

        print("\n" + "=" * 50)
        print("--- LSTM Gesture Confusion Matrix (Per Finger Count) ---")
        for finger_idx in range(num_finger_classes):
            indices_for_finger = [i for i, label in enumerate(all_f_labels) if label == finger_idx]
            if not indices_for_finger: continue

            g_labels_finger = [all_g_labels[i] for i in indices_for_finger]
            g_preds_finger = [all_g_preds[i] for i in indices_for_finger]

            print(f"\n--- Confusion Matrix for {finger_names[finger_idx]} ---")
            unique_gestures = sorted(list(set(g_labels_finger) | set(g_preds_finger)))
            unique_gesture_names = [gesture_names[i] for i in unique_gestures]

            cm_gesture = confusion_matrix(g_labels_finger, g_preds_finger, labels=unique_gestures)
            cm_df = pd.DataFrame(cm_gesture, index=unique_gesture_names, columns=unique_gesture_names)

            if not cm_df.empty:
                print(cm_df)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
                plt.title(f'LSTM Gesture CM for {finger_names[finger_idx]} (Validation Set)')
                plt.ylabel('Actual Label');
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                lstm_cm_save_path = os.path.join(models_path_lstm, f'lstm_gesture_cm_{finger_idx}finger.png')
                plt.savefig(lstm_cm_save_path)
                print(f"\nPlot saved to {lstm_cm_save_path}")

    else:
        print("Invalid main choice.")
        return

    if select == '7':
        process_start_time = time.time()

        print("\n--- Starting Stage 7: Preparing 3-Level Fixed-Length Data Structure ---")
        prepare_three_level_data(
            clean_data_path=clean_train_path,
            broken_data_path=broken_data_path,
            groundtruth_data_path=groundtruth_data_path,
            output_base_path=three_level_train_path,
            config_save_path=data_prep_config_path,
            num_good_versions=10,
            num_damaged_versions=0,
            window_size=30,
            min_gesture_len=10
        )

        process_end_time = time.time()
        total_seconds = process_end_time - process_start_time

        # Format the time for readability
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        print("\n" + "=" * 50)
        print(f"Total execution time: {hours} hours, {minutes} minutes, {seconds} seconds.")
        print("=" * 50)

    elif select == '8':
        print("\n--- Pre-training SPECIALIZED Backbone on 3-Level Data ---")

        # --- Step 1: Check for data and add necessary imports ---
        if not os.path.exists(three_level_train_path) or not os.listdir(three_level_train_path):
            print(f"Error: 3-level training data folder is empty. Run Option 7 first.")
            return

        # --- Step 2: Load data and create a Train/Validation Split ---
        unlabeled_data_raw, _, _, _ = load_three_level_data(three_level_train_path)
        if not unlabeled_data_raw:
            print("Error: Failed to load data from 3-level training folder.")
            return

        train_data_raw, val_data_raw = train_test_split(unlabeled_data_raw, test_size=0.15, random_state=42)
        print(f"Pre-train data split into {len(train_data_raw)} training and {len(val_data_raw)} validation samples.")

        # --- Step 3: Create scaler based ONLY on training data ---
        global_mean, global_std = get_standardization_params(train_data_raw)
        np.savez(specialized_scaler_path, mean=global_mean, std=global_std)
        print(f"Specialized scaler saved to {specialized_scaler_path}")

        train_data_std = standardize_data(train_data_raw, global_mean, global_std)
        val_data_std = standardize_data(val_data_raw, global_mean, global_std)

        pretrain_dataset = UnlabeledGestureDataset(train_data_std)
        pretrain_val_dataset = UnlabeledGestureDataset(val_data_std)

        # --- Step 4: Define the new, expanded Hyperparameter Search Space ---
        d_model_options = [1024]
        n_head_options = [8, 16]
        num_layers_options = [4, 6]
        dropout_options = [0.1, 0.2]
        lr_options = [0.00005, 0.00001]
        masking_ratio_options = [0.25]

        PRETRAIN_EPOCHS = 30
        PRETRAIN_BATCH_SIZE = 32

        # --- Step 5: Modified Collate Fn and DataLoaders ---
        def collate_fn_pretrain(batch, masking_ratio):
            sequences = torch.stack(batch)
            batch_size, seq_len, _ = sequences.shape
            inputs, labels = sequences.clone(), sequences.clone()
            loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for i in range(batch_size):
                num_to_mask = int(seq_len * masking_ratio)
                if num_to_mask > 0:
                    indices_to_mask = random.sample(range(seq_len), num_to_mask)
                    inputs[i, indices_to_mask, :] = 0.0
                    loss_mask[i, indices_to_mask, :] = True
            return inputs, None, labels, loss_mask

        best_loss = float('inf')
        best_config = {}

        # --- Step 6: The Full Search Loop with all enhancements ---
        for d_model in d_model_options:
            for n_head in n_head_options:
                if d_model % n_head != 0: continue
                for num_layers in num_layers_options:
                    for dropout in dropout_options:
                        for lr in lr_options:
                            for masking_ratio in masking_ratio_options:
                                current_config = {"D_MODEL": d_model, "N_HEAD": n_head, "NUM_ENC_LAYERS": num_layers,
                                                  "DROPOUT": dropout, "PRETRAIN_LR": lr, "MASKING_RATIO": masking_ratio}
                                print("-" * 50)
                                print(f"Testing SPECIALIZED configuration: {current_config}")

                                # Create DataLoaders with the current masking ratio
                                train_loader = DataLoader(pretrain_dataset, batch_size=PRETRAIN_BATCH_SIZE,
                                                          shuffle=True,
                                                          collate_fn=lambda b: collate_fn_pretrain(b, masking_ratio))
                                val_loader = DataLoader(pretrain_val_dataset, batch_size=PRETRAIN_BATCH_SIZE,
                                                        shuffle=False,
                                                        collate_fn=lambda b: collate_fn_pretrain(b, masking_ratio))

                                backbone = GestureBackbone(d_model, n_head, num_layers, d_model * 4, dropout).to(device)
                                prediction_head = nn.Linear(d_model, NUM_SENSORS).to(device)

                                # +++ NEW: PARAMETER CALCULATION AND PRINTOUT +++
                                print("\n--- Model Parameter Details ---")
                                backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
                                head_params = sum(p.numel() for p in prediction_head.parameters() if p.requires_grad)
                                total_params = backbone_params + head_params

                                print(f"  Backbone Trainable Parameters: {backbone_params:,}")
                                print(f"  Prediction Head Trainable Parameters: {head_params:,}")
                                print(f"  Total Trainable Parameters for Pre-training: {total_params:,}")
                                print("---------------------------------\n")
                                # +++ END OF NEW SECTION +++

                                optimizer = optim.Adam(list(backbone.parameters()) + list(prediction_head.parameters()),
                                                       lr=lr)
                                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
                                criterion = nn.MSELoss()

                                final_val_loss = 0
                                for epoch in range(PRETRAIN_EPOCHS):
                                    # --- Training Phase ---
                                    backbone.train()
                                    prediction_head.train()
                                    total_train_loss = 0
                                    for d_train in train_loader:
                                        inputs, _, labels, loss_mask = d_train
                                        inputs, labels, loss_mask = inputs.to(device), labels.to(device), loss_mask.to(
                                            device)
                                        optimizer.zero_grad()
                                        predictions = prediction_head(backbone(inputs, None))
                                        loss = criterion(predictions[loss_mask], labels[loss_mask])
                                        loss.backward()
                                        optimizer.step()
                                        total_train_loss += loss.item()

                                    # --- Validation Phase ---
                                    backbone.eval()
                                    prediction_head.eval()
                                    total_val_loss = 0
                                    with torch.no_grad():
                                        for d_val in val_loader:
                                            inputs, _, labels, loss_mask = d_val
                                            inputs, labels, loss_mask = inputs.to(device), labels.to(
                                                device), loss_mask.to(device)
                                            predictions = prediction_head(backbone(inputs, None))
                                            loss = criterion(predictions[loss_mask], labels[loss_mask])
                                            total_val_loss += loss.item()

                                    avg_val_loss = total_val_loss / len(val_loader)
                                    final_val_loss = avg_val_loss

                                    print(
                                        f"  Epoch {epoch + 1:02d} | Train Loss: {total_train_loss / len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")

                                    scheduler.step(avg_val_loss)

                                print(f"Final Validation Loss for this configuration: {final_val_loss:.6f}")

                                if final_val_loss < best_loss:
                                    best_loss = final_val_loss
                                    best_config = current_config.copy()
                                    best_config['BEST_VAL_LOSS'] = best_loss
                                    print(f"*** New best SPECIALIZED model found! Val Loss: {best_loss:.6f} ***")
                                    torch.save(backbone.state_dict(), specialized_backbone_path)
                                    with open(specialized_config_path, 'w') as f:
                                        for key, value in best_config.items(): f.write(f"{key}: {value}\n")
        print("\n" + "=" * 50)
        print("Specialized Hyperparameter search finished!")
        print(f"Best model saved to {specialized_backbone_path}")
        print(f"Best configuration (Val Loss: {best_loss:.6f}):\n{best_config}")
        print("=" * 50)

        # In main(), replace the existing "elif select == '8.1':" block with this one

        # In main(), replace the existing "elif select == '8.1':" block with this one

    elif select == '8.1':

        process_start_time = time.time()

        print("\n--- Starting Stage 8.1: Training Transformer from Scratch (Supervised Only) ---")

        # --- Step 1: Check for data ---
        if not os.path.exists(three_level_train_path):
            print(f"Error: Fixed-length training data not found. Run Option 7 first.")
            return

        # Define paths for this model's outputs
        scratch_model_path = os.path.join(models_path_3level, '3level_model_from_scratch.pth')
        scratch_config_path = os.path.join(models_path_3level, '3level_config_from_scratch.txt')

        # --- Step 2: Ask the user for hyperparameter source ---
        use_config_choice = input(
            "\nLoad a single hyperparameter set from the config file (for direct comparison)? (y/n): "
        ).lower()

        # --- Step 3: Load and process data (common to both paths) ---
        print("\nLoading data and calculating scaler...")
        unlabeled_data_raw, _, _, _ = load_three_level_data(three_level_train_path)
        if not unlabeled_data_raw:
            print("Error: Failed to load data from 3-level training folder.");
            return

        train_data_raw, val_data_raw = train_test_split(unlabeled_data_raw, test_size=0.15, random_state=42)
        global_mean, global_std = get_standardization_params(train_data_raw)
        np.savez(specialized_scaler_path, mean=global_mean, std=global_std)
        print(f"Scaler calculated and saved to {specialized_scaler_path}")

        data_std = standardize_data(unlabeled_data_raw, global_mean, global_std)
        all_labels = load_three_level_data(three_level_train_path)[1:]  # Get f_labels, g_labels, q_labels

        # Split indices once to ensure both training paths use the same data split
        indices = list(range(len(data_std)))
        train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=42,
                                                      stratify=all_labels[1])  # Stratify by gesture

        train_data = [data_std[i] for i in train_indices];
        val_data = [data_std[i] for i in val_indices]
        train_f_labels = [all_labels[0][i] for i in train_indices];
        val_f_labels = [all_labels[0][i] for i in val_indices]
        train_g_labels = [all_labels[1][i] for i in train_indices];
        val_g_labels = [all_labels[1][i] for i in val_indices]
        train_q_labels = [all_labels[2][i] for i in train_indices];
        val_q_labels = [all_labels[2][i] for i in val_indices]

        train_dataset = ThreeLevelDataset(train_data, train_f_labels, train_g_labels, train_q_labels)
        val_dataset = ThreeLevelDataset(val_data, val_f_labels, val_g_labels, val_q_labels)

        def collate_fn_3level(batch):
            data_b, f_labels_b, g_labels_b, q_labels_b = zip(*batch)
            return torch.stack(data_b), torch.stack(f_labels_b), torch.stack(g_labels_b), torch.stack(q_labels_b), None

        BATCH_SIZE = 32
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_3level)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_3level)

        num_finger_classes = len(set(all_labels[0]));
        num_gesture_classes = len(set(all_labels[1]));
        num_quality_classes = len(set(all_labels[2]))

        # =============================================================================
        # === PATH A: SINGLE TRAINING RUN USING CONFIG FILE (user chose 'y') ========
        # =============================================================================
        if use_config_choice == 'y':
            print("\n--- Path A: Training single configuration from file ---")
            if not os.path.exists(specialized_config_path):
                print(f"Error: Config file not found at {specialized_config_path}. Cannot proceed.");
                return
            config = parse_config_file(specialized_config_path)
            if not config:
                print("Error: Could not parse the config file. Cannot proceed.");
                return

            D_MODEL = int(config['D_MODEL']);
            N_HEAD = int(config['N_HEAD'])
            NUM_ENC_LAYERS = int(config['NUM_ENC_LAYERS']);
            DROPOUT = float(config['DROPOUT'])
            LR = 0.0001  # A good default LR for single runs
            EPOCHS = 1

            print(f"Training with loaded config: {config}")
            backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
            model = ThreeLevelHierarchicalModel(backbone, D_MODEL, num_finger_classes, num_gesture_classes,
                                                num_quality_classes).to(device)

            criterion = nn.CrossEntropyLoss();
            optimizer = optim.Adam(model.parameters(), lr=LR)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

            best_val_gesture_acc = 0.0
            for epoch in range(EPOCHS):
                # ... (Standard training and validation loop) ...
                model.train();
                total_loss = 0
                for data, f_lbl, g_lbl, q_lbl, _ in train_loader:
                    data, f_lbl, g_lbl, q_lbl = data.to(device), f_lbl.to(device), g_lbl.to(device), q_lbl.to(device)
                    optimizer.zero_grad()
                    f_logits, g_logits, q_logits = model(data, None)
                    loss = criterion(g_logits, g_lbl) + criterion(f_logits, f_lbl) + criterion(q_logits, q_lbl)
                    loss.backward();
                    optimizer.step();
                    total_loss += loss.item()

                model.eval();
                val_loss, vc_g, val_total_samples = 0, 0, 0
                with torch.no_grad():
                    for data, f_lbl, g_lbl, q_lbl, _ in val_loader:
                        data, g_lbl = data.to(device), g_lbl.to(device)
                        _, g_logits, _ = model(data, None)
                        val_loss += criterion(g_logits, g_lbl).item() * g_lbl.size(0)
                        vc_g += (torch.argmax(g_logits, 1) == g_lbl).sum().item()
                        val_total_samples += g_lbl.size(0)

                avg_val_loss = val_loss / val_total_samples
                val_acc_g = 100 * vc_g / val_total_samples
                print(
                    f"  Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc_g:.2f}%")
                scheduler.step(avg_val_loss)

                if val_acc_g > best_val_gesture_acc:
                    print(f"*** New best validation accuracy: {val_acc_g:.2f}%. Saving model. ***")
                    best_val_gesture_acc = val_acc_g
                    torch.save(model.state_dict(), scratch_model_path)

        # =============================================================================
        # === PATH B: HYPERPARAMETER SEARCH (user chose 'n') ========================
        # =============================================================================
        else:
            print("\n--- Path B: Starting hyperparameter search for 'from scratch' training ---")
            d_model_options = [1024]
            n_head_options = [8]
            num_layers_options = [4]
            dropout_options = [0.1]
            lr_options = [0.00005]
            EPOCHS = 30

            best_val_acc = 0.0
            best_config = {}
            for d_model in d_model_options:
                for n_head in n_head_options:
                    if d_model % n_head != 0: continue
                    for num_layers in num_layers_options:
                        for dropout in dropout_options:
                            for lr in lr_options:
                                current_config = {"D_MODEL": d_model, "N_HEAD": n_head, "NUM_ENC_LAYERS": num_layers,
                                                  "DROPOUT": dropout, "LEARNING_RATE": lr}
                                print("-" * 60);
                                print(f"Testing 'From Scratch' Configuration: {current_config}")
                                backbone = GestureBackbone(d_model, n_head, num_layers, d_model * 4, dropout)
                                model = ThreeLevelHierarchicalModel(backbone, d_model, num_finger_classes,
                                                                    num_gesture_classes, num_quality_classes).to(device)
                                criterion = nn.CrossEntropyLoss();
                                optimizer = optim.Adam(model.parameters(), lr=lr)
                                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

                                final_val_acc = 0.0
                                for epoch in range(EPOCHS):
                                    model.train();
                                    train_loss = 0
                                    for data, f_lbl, g_lbl, q_lbl, _ in train_loader:
                                        data, f_lbl, g_lbl, q_lbl = data.to(device), f_lbl.to(device), g_lbl.to(
                                            device), q_lbl.to(device)
                                        optimizer.zero_grad()
                                        f_logits, g_logits, q_logits = model(data, None)
                                        loss = criterion(g_logits, g_lbl) + criterion(f_logits, f_lbl) + criterion(
                                            q_logits, q_lbl)
                                        loss.backward();
                                        optimizer.step();
                                        train_loss += loss.item()

                                    model.eval();
                                    val_loss, vc_g, val_total_samples = 0, 0, 0
                                    with torch.no_grad():
                                        for data, f_lbl, g_lbl, q_lbl, _ in val_loader:
                                            data, g_lbl = data.to(device), g_lbl.to(device)
                                            _, g_logits, _ = model(data, None)
                                            val_loss += criterion(g_logits, g_lbl).item() * g_lbl.size(0)
                                            vc_g += (torch.argmax(g_logits, 1) == g_lbl).sum().item()
                                            val_total_samples += g_lbl.size(0)
                                    avg_val_loss = val_loss / val_total_samples;
                                    val_acc_g = 100 * vc_g / val_total_samples
                                    scheduler.step(avg_val_loss)
                                    final_val_acc = val_acc_g

                                    print(
                                            f"    Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss / len(train_loader):.4f} | Val Acc: {val_acc_g:.2f}%")

                                print(f"  Final Validation Accuracy for this config: {final_val_acc:.2f}%")
                                if final_val_acc > best_val_acc:
                                    best_val_acc = final_val_acc
                                    best_config = current_config.copy()
                                    best_config['BEST_VAL_ACC'] = best_val_acc
                                    print(f"*** New BEST 'from scratch' model found! Val Acc: {best_val_acc:.2f}% ***")
                                    torch.save(model.state_dict(), scratch_model_path)
                                    with open(scratch_config_path, 'w') as f:
                                        for key, value in best_config.items(): f.write(f"{key}: {value}\n")

                                    # +++ NEW: RUN EVALUATION FOR THE NEW BEST MODEL +++
                                    print("\n--- Generating Interim Evaluation Report for New Best Model ---")
                                    model.eval()  # Ensure model is in eval mode
                                    all_f_labels, all_f_preds, all_g_labels, all_g_preds = [], [], [], []
                                    with torch.no_grad():
                                        for data_batch, f_lbl, g_lbl, q_lbl, _ in val_loader:
                                            data_batch = data_batch.to(device)
                                            f_logits, g_logits, _ = model(data_batch, None)
                                            f_pred, g_pred = torch.argmax(f_logits, 1), torch.argmax(g_logits, 1)
                                            all_f_labels.extend(f_lbl.cpu().numpy());
                                            all_f_preds.extend(f_pred.cpu().numpy())
                                            all_g_labels.extend(g_lbl.cpu().numpy());
                                            all_g_preds.extend(g_pred.cpu().numpy())

                                    finger_names = [f"{i} Finger(s)" for i in range(num_finger_classes)]
                                    gesture_names = [f"Gesture_{i}" for i in range(num_gesture_classes)]

                                    print("\n--- Overall Gesture Classification Report ---")
                                    print(classification_report(all_g_labels, all_g_preds, target_names=gesture_names,
                                                                digits=3))

                                    for finger_idx in range(num_finger_classes):
                                        indices_for_finger = [i for i, label in enumerate(all_f_labels) if
                                                              label == finger_idx]
                                        if not indices_for_finger: continue
                                        g_labels_finger = [all_g_labels[i] for i in indices_for_finger]
                                        g_preds_finger = [all_g_preds[i] for i in indices_for_finger]
                                        unique_gestures = sorted(list(set(g_labels_finger) | set(g_preds_finger)))
                                        if not unique_gestures: continue
                                        unique_gesture_names = [gesture_names[i] for i in unique_gestures]

                                        cm_gesture = confusion_matrix(g_labels_finger, g_preds_finger,
                                                                      labels=unique_gestures)
                                        cm_df = pd.DataFrame(cm_gesture, index=unique_gesture_names,
                                                             columns=unique_gesture_names)

                                        if not cm_df.empty:
                                            print(
                                                f"\n--- Confusion Matrix for {finger_names[finger_idx]} (New Best) ---")
                                            print(cm_df)
                                            plt.figure(figsize=(8, 6));
                                            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
                                            plt.title(
                                                f'Gesture CM for {finger_names[finger_idx]} (From Scratch - Best)')
                                            plt.ylabel('Actual Label');
                                            plt.xlabel('Predicted Label');
                                            plt.tight_layout()
                                            # The save path is the same, so it will overwrite the previous best
                                            cm_save_path = os.path.join(models_path_3level,
                                                                        f'gesture_cm_{finger_idx}finger_from_scratch.png')
                                            plt.savefig(cm_save_path)
                                            plt.close()  # Important: close the plot to free up memory
                                            print(f"Plot updated at: {cm_save_path}")
                                    print("-" * 60)
                                    # +++ END OF NEW SECTION +++

            print("\n" + "=" * 50);
            print("Training from scratch search finished!")
            print(f"Best model saved to {scratch_model_path}");
            print(f"Best configuration (Val Acc: {best_val_acc:.2f}%):\n{best_config}")

        # --- FINAL EVALUATION (runs for BOTH paths) ---
        print("\n" + "=" * 50)
        print("--- Final Model Evaluation on Validation Set (Transformer from Scratch) ---")
        if not os.path.exists(scratch_model_path):
            print("No model was saved. Cannot perform final evaluation.")
            return

        print(f"Loading best 'from scratch' model for final report...")

        # We need to know which architecture to build before loading the weights
        # This works whether it was a single run or a search, as the best config is always saved.
        final_config = parse_config_file(scratch_config_path)
        D_MODEL = int(final_config['D_MODEL'])
        N_HEAD = int(final_config['N_HEAD'])
        NUM_ENC_LAYERS = int(final_config['NUM_ENC_LAYERS'])
        DROPOUT = float(final_config['DROPOUT'])

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
        model = ThreeLevelHierarchicalModel(backbone, D_MODEL, num_finger_classes, num_gesture_classes,
                                            num_quality_classes).to(device)
        model.load_state_dict(torch.load(scratch_model_path, map_location=device))
        model.eval()

        # --- Now, run the full advanced evaluation ---
        all_f_labels, all_f_preds = [], []
        all_g_labels, all_g_preds = [], []
        all_q_labels, all_q_preds = [], []

        with torch.no_grad():
            for data_batch, f_lbl, g_lbl, q_lbl, _ in val_loader:
                data_batch = data_batch.to(device)
                f_logits, g_logits, q_logits = model(data_batch, None)

                f_pred = torch.argmax(f_logits, 1)
                g_pred = torch.argmax(g_logits, 1)
                q_pred = torch.argmax(q_logits, 1)

                all_f_labels.extend(f_lbl.cpu().numpy())
                all_f_preds.extend(f_pred.cpu().numpy())
                all_g_labels.extend(g_lbl.cpu().numpy())
                all_g_preds.extend(g_pred.cpu().numpy())
                all_q_labels.extend(q_lbl.cpu().numpy())
                all_q_preds.extend(q_pred.cpu().numpy())

        # --- Standard Reports ---
        print("\n--- Finger Classification Report ---")
        finger_names = [f"{i} Finger(s)" for i in range(num_finger_classes)]
        print(classification_report(all_f_labels, all_f_preds, target_names=finger_names, digits=3))

        print("\n--- Gesture Classification Report (Overall) ---")
        gesture_names = [f"Gesture_{i}" for i in range(num_gesture_classes)]
        print(classification_report(all_g_labels, all_g_preds, target_names=gesture_names, digits=3))

        # Fix for the quality report crash
        if len(set(all_q_labels)) > 1:
            print("\n--- Quality Classification Report ---")
            quality_names = ["Good (0)", "Damaged (1)"]
            print(classification_report(all_q_labels, all_q_preds, target_names=quality_names, digits=3))
        else:
            print("\n--- Quality Classification Report ---")
            print("Only one quality class found in validation set. Report cannot be generated.")

        # --- Advanced Misclassification Analysis ---
        print("\n" + "=" * 50)
        print("--- Advanced Misclassification Analysis (Finger vs. Gesture) ---")

        finger_errors = 0;
        gesture_only_errors = 0
        errors_by_finger = {i: [] for i in range(num_finger_classes)}

        for i in range(len(all_g_labels)):
            true_finger = all_f_labels[i];
            pred_finger = all_f_preds[i]
            true_gesture = all_g_labels[i];
            pred_gesture = all_g_preds[i]

            if true_finger != pred_finger or true_gesture != pred_gesture:
                if true_finger != pred_finger:
                    finger_errors += 1
                elif true_gesture != pred_gesture:
                    gesture_only_errors += 1

                if true_gesture != pred_gesture:
                    error_detail = f"  - True: {true_finger}-Finger/Gesture_{true_gesture}, Predicted: {pred_finger}-Finger/Gesture_{pred_gesture}"
                    errors_by_finger[true_finger].append(error_detail)

        total_errors = finger_errors + gesture_only_errors
        if total_errors > 0:
            print(f"Total incorrect predictions in validation set: {total_errors}")
            print(f"Breakdown:")
            print(
                f"  - Errors due to incorrect FINGER prediction: {finger_errors} ({finger_errors / total_errors:.1%})")
            print(
                f"  - Errors where FINGER was correct but GESTURE was wrong: {gesture_only_errors} ({gesture_only_errors / total_errors:.1%})")
        else:
            print("No incorrect predictions found in the validation set!")
        print("-" * 50)

        # --- SEPARATE CONFUSION MATRICES PER FINGER COUNT ---
        print("\n" + "=" * 50)
        print("--- Gesture Confusion Matrix (Per Finger Count) ---")

        for finger_idx in range(num_finger_classes):
            indices_for_finger = [i for i, label in enumerate(all_f_labels) if label == finger_idx]
            if not indices_for_finger: continue

            g_labels_finger = [all_g_labels[i] for i in indices_for_finger]
            g_preds_finger = [all_g_preds[i] for i in indices_for_finger]

            print(f"\n--- Confusion Matrix for {finger_names[finger_idx]} (From Scratch) ---")

            unique_gestures = sorted(list(set(g_labels_finger) | set(g_preds_finger)))
            unique_gesture_names = [gesture_names[i] for i in unique_gestures]

            cm_gesture = confusion_matrix(g_labels_finger, g_preds_finger, labels=unique_gestures)
            cm_df = pd.DataFrame(cm_gesture, index=unique_gesture_names, columns=unique_gesture_names)

            if not cm_df.empty:
                print(cm_df)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Gesture CM for {finger_names[finger_idx]} (From Scratch)')
                plt.ylabel('Actual Label');
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                cm_save_path = os.path.join(models_path_3level, f'gesture_cm_{finger_idx}finger_from_scratch.png')
                plt.savefig(cm_save_path)
                print(f"\nPlot saved to {cm_save_path}")

            process_end_time = time.time()
            total_seconds = process_end_time - process_start_time

            # Format the time for readability
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)

            print("\n" + "=" * 50)
            print("--- Option 8.1 Complete ---")
            print(f"Total execution time: {hours} hours, {minutes} minutes, {seconds} seconds.")
            print("=" * 50)

    elif select == '8.5':
        print("\n--- Final, Long Pre-training Run for Fixed-Length Backbone ---")

        # --- Step 1: Check for prerequisites and load the BEST config ---
        for p in [specialized_config_path, specialized_scaler_path]:
            if not os.path.exists(p):
                print(f"Error: Required file not found at {p}. Run hyperparameter search (Option 7.5) first.")
                return

        config = parse_config_file(specialized_config_path)
        if not config:
            print("Error: Could not load best configuration file.")
            return

        print(f"Loaded BEST configuration from search: {config}")

        # Extract the winning hyperparameters
        D_MODEL = int(config['D_MODEL'])
        N_HEAD = int(config['N_HEAD'])
        NUM_ENC_LAYERS = int(config['NUM_ENC_LAYERS'])
        DROPOUT = float(config['DROPOUT'])
        LR = float(config['PRETRAIN_LR'])
        MASKING_RATIO = float(config['MASKING_RATIO'])

        # --- Step 2: Set parameters for the long run ---
        FINAL_PRETRAIN_EPOCHS = 80  # Increase epochs for the final training
        PRETRAIN_BATCH_SIZE = 16  # Keep the same batch size that worked in the search

        # --- Step 3: Load and prepare data (same as Option 7.5) ---
        unlabeled_data_raw, _, _, _ = load_three_level_data(three_level_train_path)
        train_data_raw, val_data_raw = train_test_split(unlabeled_data_raw, test_size=0.15, random_state=42)

        scaler_data = np.load(specialized_scaler_path)
        global_mean, global_std = scaler_data['mean'], scaler_data['std']

        train_data_std = standardize_data(train_data_raw, global_mean, global_std)
        val_data_std = standardize_data(val_data_raw, global_mean, global_std)

        pretrain_dataset = UnlabeledGestureDataset(train_data_std)
        pretrain_val_dataset = UnlabeledGestureDataset(val_data_std)

        # The collate function is the same as in Option 7.5
        # Let's assume Random Frame masking based on the best config.
        # A more advanced version could read MASKING_STRATEGY from the config file.
        def collate_fn_pretrain(batch, masking_ratio):
            sequences = torch.stack(batch)
            batch_size, seq_len, _ = sequences.shape
            inputs, labels = sequences.clone(), sequences.clone()
            loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for i in range(batch_size):
                num_to_mask = int(seq_len * masking_ratio)
                if num_to_mask > 0:
                    indices_to_mask = random.sample(range(seq_len), num_to_mask)
                    inputs[i, indices_to_mask, :] = 0.0
                    loss_mask[i, indices_to_mask, :] = True
            return inputs, None, labels, loss_mask

        train_loader = DataLoader(pretrain_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda b: collate_fn_pretrain(b, MASKING_RATIO))
        val_loader = DataLoader(pretrain_val_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=False,
                                collate_fn=lambda b: collate_fn_pretrain(b, MASKING_RATIO))

        # --- Step 4: The Final Training Loop ---
        print("\nStarting final training run with the best hyperparameters...")

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT).to(device)
        prediction_head = nn.Linear(D_MODEL, NUM_SENSORS).to(device)
        optimizer = optim.Adam(list(backbone.parameters()) + list(prediction_head.parameters()), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
        criterion = nn.MSELoss()

        best_loss = float('inf')

        for epoch in range(FINAL_PRETRAIN_EPOCHS):
            backbone.train();
            prediction_head.train()
            total_train_loss = 0
            for d_train in train_loader:
                inputs, _, labels, loss_mask = d_train
                inputs, labels, loss_mask = inputs.to(device), labels.to(device), loss_mask.to(device)
                optimizer.zero_grad()
                predictions = prediction_head(backbone(inputs, None))
                loss = criterion(predictions[loss_mask], labels[loss_mask])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            backbone.eval();
            prediction_head.eval()
            total_val_loss = 0
            with torch.no_grad():
                for d_val in val_loader:
                    inputs, _, labels, loss_mask = d_val
                    inputs, labels, loss_mask = inputs.to(device), labels.to(device), loss_mask.to(device)
                    predictions = prediction_head(backbone(inputs, None))
                    loss = criterion(predictions[loss_mask], labels[loss_mask])
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(
                f"  Epoch {epoch + 1:02d}/{FINAL_PRETRAIN_EPOCHS} | Train Loss: {total_train_loss / len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                print(f"--- New best validation loss: {best_loss:.6f}. Saving best model. ---")
                torch.save(backbone.state_dict(), specialized_backbone_path)

        print("\n" + "=" * 50)
        print("Final Fixed-Length Pre-training Finished!")
        print(f"The best backbone model (Val Loss: {best_loss:.6f}) was saved to {specialized_backbone_path}")
        print("=" * 50)

    elif select == '9':
        print("\n--- Starting Stage 8: Fine-tuning 3-Level Classifier ---")

        # --- Step 1: Check for prerequisites and add necessary imports ---
        for p in [specialized_backbone_path, specialized_config_path, specialized_scaler_path]:
            if not os.path.exists(p):
                print(f"Error: Required file not found at {p}. Run pre-training (Option 7.5) first.")
                return

        # --- Step 2: Load Configurations and Scaler ---
        config = parse_config_file(specialized_config_path)
        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
            config['NUM_ENC_LAYERS']), config['DROPOUT']
        print(f"Loaded SPECIALIZED backbone configuration: {config}")

        scaler_data = np.load(specialized_scaler_path)
        global_mean, global_std = scaler_data['mean'], scaler_data['std']
        print(f"Loaded SPECIALIZED scaler: Mean={global_mean:.4f}, Std={global_std:.4f}")

        # --- Step 3: Load Data and Create Train/Validation Split ---
        print("Loading all 3-level data for splitting...")
        data, f_labels, g_labels, q_labels = load_three_level_data(three_level_train_path)
        if not data:
            print(f"Error: No data found in {three_level_train_path}. Run option 7 first.")
            return
        data_std = standardize_data(data, global_mean, global_std)

        indices = list(range(len(data_std)))
        train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=42, stratify=g_labels)
        train_data = [data_std[i] for i in train_indices]
        val_data = [data_std[i] for i in val_indices]
        train_f_labels = [f_labels[i] for i in train_indices]
        val_f_labels = [f_labels[i] for i in val_indices]
        train_g_labels = [g_labels[i] for i in train_indices]
        val_g_labels = [g_labels[i] for i in val_indices]
        train_q_labels = [q_labels[i] for i in train_indices]
        val_q_labels = [q_labels[i] for i in val_indices]
        print(f"Data split into {len(train_data)} training samples and {len(val_data)} validation samples.")

        train_dataset = ThreeLevelDataset(train_data, train_f_labels, train_g_labels, train_q_labels)
        val_dataset = ThreeLevelDataset(val_data, val_f_labels, val_g_labels, val_q_labels)

        def collate_fn_3level(batch):
            data_b, f_labels_b, g_labels_b, q_labels_b = zip(*batch)
            padded_data = torch.stack(data_b)
            return padded_data, torch.stack(f_labels_b), torch.stack(g_labels_b), torch.stack(q_labels_b), None

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_3level)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_3level)

        num_finger_classes = len(set(f_labels))
        num_gesture_classes = len(set(g_labels))
        num_quality_classes = len(set(q_labels))
        print(
            f"--> Total Detected Classes: {num_finger_classes} finger, {num_gesture_classes} gesture, {num_quality_classes} quality")

        # --- Step 4: Model, Optimizer, and Scheduler Setup ---
        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
        print(f"Loading pre-trained SPECIALIZED weights from {specialized_backbone_path}")
        backbone.load_state_dict(torch.load(specialized_backbone_path, map_location=device, weights_only=True))
        model = ThreeLevelHierarchicalModel(backbone, D_MODEL, num_finger_classes, num_gesture_classes,
                                            num_quality_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # --- NEW: Initialize the Learning Rate Scheduler ---
        # It will monitor the validation loss ('min' mode).
        # If the loss doesn't improve for 'patience' epochs, the LR is reduced by 'factor'.
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

        # --- Step 5: Training and Validation Loop ---
        EPOCHS = 50
        best_val_gesture_acc = 0.0

        for epoch in range(EPOCHS):
            # --- TRAINING PHASE ---
            model.train()
            total_loss, c_f, c_g, c_q, total_samples = 0, 0, 0, 0, 0
            for data, f_lbl, g_lbl, q_lbl, _ in train_loader:
                data, f_lbl, g_lbl, q_lbl = data.to(device), f_lbl.to(device), g_lbl.to(device), q_lbl.to(device)
                optimizer.zero_grad()
                f_logits, g_logits, q_logits = model(data, None)
                loss_f = criterion(f_logits, f_lbl)
                loss_g = criterion(g_logits, g_lbl)
                loss_q = criterion(q_logits, q_lbl)
                combined_loss = loss_f + loss_g + loss_q
                combined_loss.backward()
                optimizer.step()
                total_loss += combined_loss.item()
                c_f += (torch.argmax(f_logits, 1) == f_lbl).sum().item()
                c_g += (torch.argmax(g_logits, 1) == g_lbl).sum().item()
                c_q += (torch.argmax(q_logits, 1) == q_lbl).sum().item()
                total_samples += f_lbl.size(0)

            train_acc_f = 100 * c_f / total_samples
            train_acc_g = 100 * c_g / total_samples
            train_acc_q = 100 * c_q / total_samples
            print(
                f"Epoch {epoch + 1:02d} TRAIN | Loss: {total_loss / len(train_loader):.4f} | Acc F:{train_acc_f:.2f}% G:{train_acc_g:.2f}% Q:{train_acc_q:.2f}%")

            # --- VALIDATION PHASE ---
            model.eval()
            val_loss, vc_f, vc_g, vc_q, val_total_samples = 0, 0, 0, 0, 0
            with torch.no_grad():
                for data, f_lbl, g_lbl, q_lbl, _ in val_loader:
                    data, f_lbl, g_lbl, q_lbl = data.to(device), f_lbl.to(device), g_lbl.to(device), q_lbl.to(device)
                    f_logits, g_logits, q_logits = model(data, None)
                    loss_f = criterion(f_logits, f_lbl)
                    loss_g = criterion(g_logits, g_lbl)
                    loss_q = criterion(q_logits, q_lbl)
                    combined_loss = loss_f + loss_g + loss_q
                    val_loss += combined_loss.item()
                    vc_f += (torch.argmax(f_logits, 1) == f_lbl).sum().item()
                    vc_g += (torch.argmax(g_logits, 1) == g_lbl).sum().item()
                    vc_q += (torch.argmax(q_logits, 1) == q_lbl).sum().item()
                    val_total_samples += f_lbl.size(0)

            val_acc_f = 100 * vc_f / val_total_samples
            val_acc_g = 100 * vc_g / val_total_samples
            val_acc_q = 100 * vc_q / val_total_samples
            avg_val_loss = val_loss / len(val_loader)  # Calculate average validation loss for the scheduler
            print(
                f"Epoch {epoch + 1:02d} VALID | Loss: {avg_val_loss:.4f} | Acc F:{val_acc_f:.2f}% G:{val_acc_g:.2f}% Q:{val_acc_q:.2f}%")

            # --- NEW: Step the scheduler with the validation loss ---
            scheduler.step(avg_val_loss)

            # --- Save the best model based on validation accuracy ---
            if val_acc_g > best_val_gesture_acc:
                print(f"*** New best validation accuracy: {val_acc_g:.2f}%. Saving model. ***")
                best_val_gesture_acc = val_acc_g
                torch.save(model.state_dict(), three_level_model_path)
                with open(three_level_output_parameters_path, 'w') as f:
                    f.write(f"NUM_FINGER_CLASSES: {num_finger_classes}\n")
                    f.write(f"NUM_GESTURE_CLASSES: {num_gesture_classes}\n")
                    f.write(f"NUM_QUALITY_CLASSES: {num_quality_classes}\n")

        print(f"3-Level model and config saved. Best validation gesture accuracy: {best_val_gesture_acc:.2f}%")

        print("\n" + "=" * 50)
        print("--- Final Model Evaluation on Validation Set ---")
        if not os.path.exists(three_level_model_path):
            print("No model was saved. Cannot perform final evaluation.")
            return

        print(f"Loading best model (Val Acc: {best_val_gesture_acc:.2f}%) for final report...")
        model.load_state_dict(torch.load(three_level_model_path, map_location=device))
        model.eval()

        # --- NEW: We need to collect all three sets of predictions and labels ---
        all_f_labels, all_f_preds = [], []
        all_g_labels, all_g_preds = [], []
        all_q_labels, all_q_preds = [], []

        with torch.no_grad():
            for data_batch, f_lbl, g_lbl, q_lbl, _ in val_loader:
                data_batch = data_batch.to(device)
                f_logits, g_logits, q_logits = model(data_batch, None)

                f_pred = torch.argmax(f_logits, 1)
                g_pred = torch.argmax(g_logits, 1)
                q_pred = torch.argmax(q_logits, 1)

                all_f_labels.extend(f_lbl.cpu().numpy())
                all_f_preds.extend(f_pred.cpu().numpy())
                all_g_labels.extend(g_lbl.cpu().numpy())
                all_g_preds.extend(g_pred.cpu().numpy())
                all_q_labels.extend(q_lbl.cpu().numpy())
                all_q_preds.extend(q_pred.cpu().numpy())

        # --- Standard Reports (as before) ---
        print("\n--- Finger Classification Report ---")
        # Assuming your finger folders are num_finger_0, num_finger_1, etc.
        finger_names = [f"{i} Finger(s)" for i in range(num_finger_classes)]
        print(classification_report(all_f_labels, all_f_preds, target_names=finger_names, digits=3))

        print("\n--- Gesture Classification Report (Overall) ---")
        gesture_names = [f"Gesture_{i}" for i in range(num_gesture_classes)]
        print(classification_report(all_g_labels, all_g_preds, target_names=gesture_names, digits=3))

        # Fix for the quality report crash
        if len(set(all_q_labels)) > 1:
            print("\n--- Quality Classification Report ---")
            quality_names = ["Good (0)", "Damaged (1)"]
            print(classification_report(all_q_labels, all_q_preds, target_names=quality_names, digits=3))
        else:
            print("\n--- Quality Classification Report ---")
            print("Only one quality class found in validation set. Report cannot be generated.")

        # +++ NEW: ADVANCED MISCLASSIFICATION ANALYSIS +++
        print("\n" + "=" * 50)
        print("--- Advanced Misclassification Analysis (Finger vs. Gesture) ---")

        misclassified_indices = [i for i, (gl, gp) in enumerate(zip(all_g_labels, all_g_preds)) if gl != gp]

        finger_errors = 0
        gesture_only_errors = 0

        # Dictionary to store gesture errors broken down by the TRUE finger count
        # e.g., errors_by_finger[1] will be a list of errors for 1-finger gestures
        errors_by_finger = {i: [] for i in range(num_finger_classes)}

        for i in range(len(all_g_labels)):
            true_finger = all_f_labels[i]
            pred_finger = all_f_preds[i]
            true_gesture = all_g_labels[i]
            pred_gesture = all_g_preds[i]

            # Check for any kind of error in this sample
            if true_finger != pred_finger or true_gesture != pred_gesture:
                # This is a FINGER error if the predicted finger is wrong
                if true_finger != pred_finger:
                    finger_errors += 1
                # This is a GESTURE-ONLY error if the finger was right but the gesture was wrong
                elif true_gesture != pred_gesture:
                    gesture_only_errors += 1

                # Log the gesture error details, categorized by the true finger count
                if true_gesture != pred_gesture:
                    error_detail = f"  - True: {true_finger}-Finger/Gesture_{true_gesture}, Predicted: {pred_finger}-Finger/Gesture_{pred_gesture}"
                    errors_by_finger[true_finger].append(error_detail)

        total_errors = finger_errors + gesture_only_errors
        print(f"Total incorrect predictions in validation set: {total_errors}")
        print(f"Breakdown:")
        print(f"  - Errors due to incorrect FINGER prediction: {finger_errors} ({finger_errors / total_errors:.1%})")
        print(
            f"  - Errors where FINGER was correct but GESTURE was wrong: {gesture_only_errors} ({gesture_only_errors / total_errors:.1%})")
        print("-" * 50)

        for finger_idx, error_list in errors_by_finger.items():
            if error_list:
                print(f"\nGesture Misclassifications for TRUE {finger_names[finger_idx]}:")
                for error in error_list[:5]:  # Print first 5 examples for brevity
                    print(error)
                if len(error_list) > 5:
                    print(f"  ... and {len(error_list) - 5} more.")

        # +++ NEW: SEPARATE CONFUSION MATRICES PER FINGER COUNT +++
        print("\n" + "=" * 50)
        print("--- Gesture Confusion Matrix (Per Finger Count) ---")

        for finger_idx in range(num_finger_classes):
            # Find all the samples that BELONG to this finger class
            indices_for_finger = [i for i, label in enumerate(all_f_labels) if label == finger_idx]

            if not indices_for_finger:
                continue

            # Filter the gesture labels and predictions for this finger class
            g_labels_finger = [all_g_labels[i] for i in indices_for_finger]
            g_preds_finger = [all_g_preds[i] for i in indices_for_finger]

            print(f"\n--- Confusion Matrix for {finger_names[finger_idx]} ---")

            # Find the unique gesture labels present for this finger count
            unique_gestures = sorted(list(set(g_labels_finger) | set(g_preds_finger)))
            unique_gesture_names = [gesture_names[i] for i in unique_gestures]

            cm_gesture = confusion_matrix(g_labels_finger, g_preds_finger, labels=unique_gestures)
            cm_df = pd.DataFrame(cm_gesture, index=unique_gesture_names, columns=unique_gesture_names)

            # Only plot the matrix if it's not empty
            if not cm_df.empty:
                print(cm_df)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Gesture Confusion Matrix for {finger_names[finger_idx]} (Validation Set)')
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                cm_save_path = os.path.join(models_path_3level, f'gesture_cm_{finger_idx}finger.png')
                plt.savefig(cm_save_path)
                print(f"\nPlot saved to {cm_save_path}")

    elif select == '10':
        print("\n--- Starting Stage 9: Prediction with 3-Level Classifier ---")

        for p in [three_level_model_path, specialized_config_path, specialized_scaler_path, three_level_output_parameters_path]:
            if not os.path.exists(p):
                print(f"Error: Required file not found at {p}.")
                return

        config = parse_config_file(specialized_config_path)
        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
            config['NUM_ENC_LAYERS']), config['DROPOUT']

        config_3level = parse_config_file(three_level_output_parameters_path)
        num_f = int(config_3level['NUM_FINGER_CLASSES'])
        num_g = int(config_3level['NUM_GESTURE_CLASSES'])
        num_q = int(config_3level['NUM_QUALITY_CLASSES'])

        scaler_data = np.load(specialized_scaler_path)
        global_mean, global_std = scaler_data['mean'], scaler_data['std']

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
        model = ThreeLevelHierarchicalModel(backbone, D_MODEL, num_f, num_g, num_q).to(device)
        model.load_state_dict(torch.load(three_level_model_path, map_location=device, weights_only=True))
        model.eval()

        predict_data_raw, filenames = load_predict_data(predict_path)
        if not predict_data_raw:
            print("No valid prediction data found.")
            return
        predict_data_std = standardize_data(predict_data_raw, global_mean, global_std)

        with torch.no_grad():
            for i, seq_np in enumerate(predict_data_std):
                filename = filenames[i]
                data_tensor = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(device)
                mask = torch.zeros(1, data_tensor.shape[1], dtype=torch.bool).to(device)

                f_logits, g_logits, q_logits = model(data_tensor, mask)
                pred_f = torch.argmax(f_logits, dim=1).item()
                pred_g = torch.argmax(g_logits, dim=1).item()
                pred_q = torch.argmax(q_logits, dim=1).item()

                finger_str = f"{pred_f} Finger(s)"
                gesture_map = f"{pred_g} Gesture"
                quality_str = "Good" if pred_q == 0 else "Damaged"

                print(f"File: {filename} -> Predicted: {finger_str}, {gesture_map}, Quality: {quality_str}")

    else:
        print("Invalid option selected.")


if __name__ == '__main__':
    main()