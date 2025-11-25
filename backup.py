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


SENSOR_ROWS = 9  # 13 for elbow
SENSOR_COLS = 10  # 10 for elbow
NUM_SENSORS = SENSOR_ROWS * SENSOR_COLS


# --- UTILITY AND DATA LOADING ---
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
    """
    Robustly parses a configuration file containing key: value pairs.
    Handles integers, floats, and scientific notation correctly.
    """
    config = {}
    if not os.path.exists(file_path):
        print(f"Warning: Config file not found at {file_path}")
        return config  # Return empty dict if file doesn't exist

    with open(file_path, 'r') as f:
        for line in f:
            if ': ' in line:
                key, value = line.strip().split(': ', 1)
                try:
                    # First, try to convert to an integer
                    config[key] = int(value)
                except ValueError:
                    # If that fails, it must be a float
                    try:
                        config[key] = float(value)
                    except ValueError:
                        # If it's neither, store it as a string
                        config[key] = value
    return config


def prepare_three_level_data(clean_data_path, broken_data_path, groundtruth_data_path, output_base_path,
                             num_good_versions=10, num_damaged_versions=1, window_size=50, min_gesture_len=10):
    """
    Creates a three-level data structure.
    NEW: For each augmentation, it takes a RANDOM CONSECUTIVE SEGMENT from the
    original gesture, with a minimum length of `min_gesture_len`.
    CORRECTED: Now handles the case where num_damaged_versions is 0.
    """
    print(f"--- Preparing Data: Slicing random segments (min {min_gesture_len} frames) ---")

    broken_signals = []  # Initialize as an empty list
    # --- THIS IS THE KEY CHANGE ---
    # Only load broken signals if we are actually going to create damaged versions.
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

            # Only create the damaged directory if we need it
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


class ThreeLevelDiskDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform  # For standardization
        self.file_paths = []
        self.labels = []

        print(f"Scanning for files in {base_path}...")
        for finger_dir in sorted(os.listdir(base_path)):
            try:
                finger_label = int(finger_dir.split('_')[-1]) - 1
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
                                    self.file_paths.append(os.path.join(quality_path, f))
                                    self.labels.append({
                                        'finger': finger_label,
                                        'gesture': gesture_label,
                                        'quality': quality_label
                                    })
                    except (ValueError, IndexError):
                        continue
            except (ValueError, IndexError):
                continue
        print(f"Found {len(self.file_paths)} total samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # This is the magic: load ONE file from disk when requested.
        file_path = self.file_paths[idx]
        data = load_data_from_file(file_path)

        if data is None:
            # Handle corrupted file case
            print(f"Warning: Could not load {file_path}. Returning zero tensor.")
            # We need to know the length of other tensors. Let's assume a common length, e.g. 50
            # This is a simplification. A more robust solution would be needed if lengths vary wildly.
            data = np.zeros((50, NUM_SENSORS))

        if self.transform:
            data = self.transform(data)

        labels = self.labels[idx]
        finger_label = torch.tensor(labels['finger'], dtype=torch.long)
        gesture_label = torch.tensor(labels['gesture'], dtype=torch.long)
        quality_label = torch.tensor(labels['quality'], dtype=torch.long)

        return torch.tensor(data, dtype=torch.float32), finger_label, gesture_label, quality_label


def _process_and_save_gesture(base_gesture, groundtruth_files, window_size, output_path, filename):
    """
    Helper to handle padding, shifting, and saving a single gesture.
    NEW: Intelligently creates a background template of `window_size` by
    either slicing or combining ground truth files.
    """
    num_sensors = SENSOR_ROWS * SENSOR_COLS

    # --- Step 1: Create a realistic background template of the correct window_size ---

    # Select a random ground truth file to start with
    initial_gt = random.choice(groundtruth_files)
    gt_len = len(initial_gt)

    if gt_len == window_size:
        # Perfect match, just use a copy of a random ground truth file
        background_template = random.choice(groundtruth_files).copy()
    elif gt_len > window_size:
        # Ground truth is LONGER than window. Take a random slice.
        start = random.randint(0, gt_len - window_size)
        background_template = initial_gt[start: start + window_size].copy()
    else:  # gt_len < window_size
        # Ground truth is SHORTER than window. Stitch multiple files together.

        # Calculate how many files we need to stitch together
        num_files_needed = math.ceil(window_size / gt_len)

        # Randomly select that many files from the pool (with replacement)
        files_to_stitch = random.choices(groundtruth_files, k=num_files_needed)

        # Concatenate them into one long sequence
        stitched_gt = np.vstack(files_to_stitch)

        # Now, take a random slice of the required window_size from the long sequence
        stitched_len = len(stitched_gt)
        start = random.randint(0, stitched_len - window_size)
        background_template = stitched_gt[start: start + window_size].copy()

    # --- The rest of the logic remains the same ---

    # Step 2: Prepare the gesture to be placed
    gesture_len = len(base_gesture)
    if gesture_len > window_size:
        gesture_to_place = base_gesture[:window_size]
    else:
        gesture_to_place = base_gesture

    # Step 3: Create the final window and paste the gesture
    final_window = background_template  # We've already made a copy

    current_len = len(gesture_to_place)
    max_start_pos = window_size - current_len
    start_pos = random.randint(0, max_start_pos)

    final_window[start_pos: start_pos + current_len] = gesture_to_place

    # Step 4: Save the file
    np.savetxt(os.path.join(output_path, filename), final_window, fmt='%.4f')


def prepare_variable_length_data(clean_data_path, broken_data_path, groundtruth_data_path, output_base_path,
                                 num_good_versions=10, num_damaged_versions=1,
                                 min_gesture_len=10, max_gesture_len=40,
                                 min_clip_len=20, max_clip_len=50):
    """
    Creates a three-level data structure with REALISTIC VARIABLE LENGTH gestures.
    Uses new, more explicit parameters for controlling gesture and clip length.
    CORRECTED: Now robustly handles the case where num_damaged_versions is 0.
    """
    print(f"--- Preparing VARIABLE LENGTH Data ---")
    print(f"Gesture slice length will be between: {min_gesture_len}-{max_gesture_len} frames.")
    print(f"Final output clip length will be between: {min_clip_len}-{max_clip_len} frames.")

    broken_signals = [] # Initialize as an empty list
    # --- THIS IS THE KEY CHANGE ---
    # Only load broken signals if we are actually going to create damaged versions.
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

    long_background = []
    shuffled_gt = groundtruth_files * 5
    random.shuffle(shuffled_gt)
    long_background = np.vstack(shuffled_gt)
    print(f"Created a long background sequence of {len(long_background)} frames.")

    if os.path.exists(output_base_path):
        shutil.rmtree(output_base_path)
    os.makedirs(output_base_path)

    print(f"Processing clean data from: {clean_data_path}")
    total_good_files, total_damaged_files = 0, 0

    for finger_dir in os.listdir(clean_data_path):
        finger_path = os.path.join(clean_data_path, finger_dir)
        if not os.path.isdir(finger_path): continue

        for gesture_dir in os.listdir(finger_path):
            gesture_path = os.path.join(finger_path, gesture_dir)
            if not os.path.isdir(gesture_path): continue

            # Create the 'good' directory unconditionally
            output_good_path = os.path.join(output_base_path, finger_dir, gesture_dir, 'good')
            os.makedirs(output_good_path)

            # --- THIS IS THE KEY CHANGE ---
            # Only create the 'damaged' directory if we will be making damaged files
            if num_damaged_versions > 0:
                output_damaged_path = os.path.join(output_base_path, finger_dir, gesture_dir, 'damaged')
                os.makedirs(output_damaged_path)

            for filename in os.listdir(gesture_path):
                if not filename.endswith('.txt'): continue

                original_clean_gesture = load_data_from_file(os.path.join(gesture_path, filename))
                if original_clean_gesture is None or len(original_clean_gesture) < min_gesture_len:
                    continue

                # --- Generate 'good' versions ---
                for i in range(num_good_versions):
                    final_segment = _create_variable_sample(
                        original_clean_gesture, long_background, None,
                        min_gesture_len, max_gesture_len, min_clip_len, max_clip_len
                    )
                    new_filename = f"good_v{i}_{filename}"
                    np.savetxt(os.path.join(output_good_path, new_filename), final_segment, fmt='%.4f')
                    total_good_files += 1

                # --- THIS IS THE KEY CHANGE ---
                # --- Generate 'damaged' versions (only if requested) ---
                if num_damaged_versions > 0:
                    for i in range(num_damaged_versions):
                        broken_signal = random.choice(broken_signals)
                        final_segment = _create_variable_sample(
                            original_clean_gesture, long_background, broken_signal,
                            min_gesture_len, max_gesture_len, min_clip_len, max_clip_len
                        )
                        new_filename = f"damaged_v{i}_{filename}"
                        np.savetxt(os.path.join(output_damaged_path, new_filename), final_segment, fmt='%.4f')
                        total_damaged_files += 1

    print("-" * 50)
    print("Data preparation complete.")
    print(f"Total 'good' variable-length files: {total_good_files}")
    print(f"Total 'damaged' variable-length files: {total_damaged_files}")


def _create_variable_sample(original_gesture, long_background, broken_signal,
                            min_gesture_len, max_gesture_len, min_clip_len, max_clip_len):
    """
    Helper function to create a single, realistic, variable-length training sample with precise control.
    """
    original_len = len(original_gesture)

    # --- Step 1: Select a random gesture SEGMENT of a specific length ---
    # Ensure max_gesture_len doesn't exceed the actual length of the original gesture
    actual_max_gesture_len = min(original_len, max_gesture_len)
    if min_gesture_len > actual_max_gesture_len:
        # If the original gesture is too short for the min slice, use the whole thing
        segment_len = original_len
        start_index = 0
    else:
        segment_len = random.randint(min_gesture_len, actual_max_gesture_len)
        max_start = original_len - segment_len
        start_index = random.randint(0, max_start)

    gesture_segment = original_gesture[start_index: start_index + segment_len]

    # --- Step 2: Prepare the base gesture (good or damaged) ---
    if broken_signal is None:
        # It's a 'good' gesture, use the segment directly
        base_gesture = gesture_segment
    else:
        # It's a 'damaged' gesture, so add a broken signal of the same length
        seg_len, broken_len = len(gesture_segment), len(broken_signal)
        if broken_len >= seg_len:
            start = random.randint(0, broken_len - seg_len)
            broken_part = broken_signal[start: start + seg_len]
        else:
            repeats = (seg_len // broken_len) + 1 if broken_len > 0 else 1
            broken_part = np.tile(broken_signal, (repeats, 1))[:seg_len]
        base_gesture = gesture_segment + broken_part

    # --- Step 3: Create a canvas and paste the gesture ---
    # The canvas needs to be large enough to accommodate the longest possible clip
    canvas_len = max_clip_len * 2
    if len(long_background) < canvas_len:
        repeats = math.ceil(canvas_len / len(long_background))
        long_background = np.tile(long_background, (repeats, 1))

    start = random.randint(0, len(long_background) - canvas_len)
    canvas = long_background[start: start + canvas_len].copy()

    # Paste into the middle of the canvas for maximum slicing flexibility
    paste_start = (canvas_len - len(base_gesture)) // 2
    canvas[paste_start: paste_start + len(base_gesture)] = base_gesture
    paste_end = paste_start + len(base_gesture)

    # --- Step 4: Take the final variable-length CLIP, guaranteeing it contains the gesture ---
    final_clip_len = random.randint(min_clip_len, max_clip_len)

    # Define the valid range for our slice's starting point to guarantee overlap.
    # The latest our slice can start and still catch the end of the gesture.
    latest_start_point = paste_end - 1
    # The earliest our slice can start and still catch the beginning of the gesture.
    earliest_start_point = paste_start - final_clip_len + 1

    # Clamp the start range to the canvas bounds
    valid_start = max(0, earliest_start_point)
    valid_end = min(canvas_len - final_clip_len, latest_start_point)

    if valid_start >= valid_end:
        # This edge case happens if the clip is shorter than the gesture.
        # We'll just take a slice that starts with the gesture to ensure it's included.
        slice_start = paste_start
        final_clip_len = min(final_clip_len, len(base_gesture))
    else:
        slice_start = random.randint(valid_start, valid_end)

    final_segment = canvas[slice_start: slice_start + final_clip_len]

    return final_segment


def collate_fn_variable_length(batch):
    """
    Collates a batch of variable-length tensors for SUPERVISED training.
    Pads sequences and creates a padding mask.
    """
    data_list, finger_labels, gesture_labels, quality_labels = zip(*batch)
    padded_data = nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0.0)
    lengths = [len(seq) for seq in data_list]
    batch_size, max_len, _ = padded_data.shape
    padding_mask = torch.arange(max_len).expand(batch_size, max_len) >= torch.tensor(lengths).unsqueeze(1)

    return (
        padded_data,
        torch.stack(finger_labels),
        torch.stack(gesture_labels),
        torch.stack(quality_labels),
        padding_mask
    )


def load_three_level_data(base_path):
    """
    Loads data from a three-level hierarchical folder structure.
    Structure: base_path/num_finger_X/gesture_Y/[good|broken]/data.txt
    """
    all_data, finger_labels, gesture_labels, quality_labels = [], [], [], []
    if not os.path.exists(base_path):
        print(f"Warning: Labeled data path not found: {base_path}")
        return all_data, finger_labels, gesture_labels, quality_labels

    for finger_dir in sorted(os.listdir(base_path)):
        try:
            finger_label = int(finger_dir.split('_')[-1]) - 1
            finger_path = os.path.join(base_path, finger_dir)
            if not os.path.isdir(finger_path): continue

            for gesture_dir in sorted(os.listdir(finger_path)):
                try:
                    gesture_label = int(gesture_dir.split('_')[-1])
                    gesture_path = os.path.join(finger_path, gesture_dir)
                    if not os.path.isdir(gesture_path): continue

                    for quality_dir in sorted(os.listdir(gesture_path)):
                        # good = 0, broken = 1
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
                except (ValueError, IndexError):
                    continue
        except (ValueError, IndexError):
            continue

    return all_data, finger_labels, gesture_labels, quality_labels


# --- NEW, MEMORY-EFFICIENT DATASET for FIXED-LENGTH DATA ---
class FixedLengthDiskDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform
        self.file_paths = []
        self.labels = []

        print(f"Scanning for fixed-length files in {base_path}...")
        # The file scanning logic is identical to ThreeLevelDiskDataset
        for finger_dir in sorted(os.listdir(base_path)):
            try:
                finger_label = int(finger_dir.split('_')[-1]) - 1
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
                                    self.file_paths.append(os.path.join(quality_path, f))
                                    self.labels.append({
                                        'finger': finger_label,
                                        'gesture': gesture_label,
                                        'quality': quality_label
                                    })
                    except (ValueError, IndexError):
                        continue
            except (ValueError, IndexError):
                continue
        print(f"Found {len(self.file_paths)} total samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load one file from disk when requested by the DataLoader
        file_path = self.file_paths[idx]
        data = load_data_from_file(file_path)

        if data is None:
            print(f"Warning: Could not load {file_path}. Returning zero tensor.")
            # Since this is for fixed-length, we can find the window_size from the config
            # But for simplicity, we hardcode a common value.
            # In a real scenario, you'd pass window_size to the constructor.
            data = np.zeros((30, NUM_SENSORS))

        if self.transform:
            data = self.transform(data)

        labels = self.labels[idx]
        finger_label = torch.tensor(labels['finger'], dtype=torch.long)
        gesture_label = torch.tensor(labels['gesture'], dtype=torch.long)
        quality_label = torch.tensor(labels['quality'], dtype=torch.long)

        # Return a tensor directly, as there's no padding needed later
        return torch.tensor(data, dtype=torch.float32), finger_label, gesture_label, quality_label


class ThreeLevelHierarchicalModel(nn.Module):
    """ A model with a shared backbone and three separate classification heads. """

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


class ThreeLevelDataset(Dataset):
    def __init__(self, data_list, finger_labels, gesture_labels, quality_labels):
        self.data_list = data_list
        self.finger_labels = finger_labels
        self.gesture_labels = gesture_labels
        self.quality_labels = quality_labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = torch.tensor(self.data_list[idx], dtype=torch.float32)
        finger_label = torch.tensor(self.finger_labels[idx], dtype=torch.long)
        gesture_label = torch.tensor(self.gesture_labels[idx], dtype=torch.long)
        quality_label = torch.tensor(self.quality_labels[idx], dtype=torch.long)
        return data, finger_label, gesture_label, quality_label


# --- STANDARDIZATION HELPER FUNCTIONS ---
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


# --- DATASET CLASSES ---
class UnlabeledGestureDataset(Dataset):
    def __init__(self, data_list): self.data_list = data_list

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx): return torch.tensor(self.data_list[idx], dtype=torch.float32)


class DenoisingPairedDataset(Dataset):
    def __init__(self, noisy_file_paths, clean_file_paths):
        self.noisy_paths, self.clean_paths = noisy_file_paths, clean_file_paths
        assert len(self.noisy_paths) == len(self.clean_paths), "Mismatch in number of noisy and clean files"

    def __len__(self): return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy_data = np.load(self.noisy_paths[idx]);
        clean_data = np.load(self.clean_paths[idx])
        return torch.tensor(noisy_data, dtype=torch.float32), torch.tensor(clean_data, dtype=torch.float32)


# --- HIERARCHICAL CLASSIFICATION COMPONENTS ---
def load_hierarchical_data_from_nested_folders(base_path):
    """
    Loads data from a nested hierarchical folder structure.
    Expects top-level folders like 'num_finger_1', 'num_finger_2', etc.
    Inside each, expects gesture direction folders 'gesture_0', 'gesture_1', etc.
    """
    all_data, finger_labels, gesture_labels = [], [], []
    if not os.path.exists(base_path):
        print(f"Warning: Labeled data path not found: {base_path}")
        return all_data, finger_labels, gesture_labels

    # Iterate through the top-level folders (e.g., 'num_finger_1', 'num_finger_2')
    for finger_count_folder in sorted(os.listdir(base_path)):
        finger_folder_path = os.path.join(base_path, finger_count_folder)

        if os.path.isdir(finger_folder_path):
            try:
                # --- THIS IS THE CORRECTED PART ---
                # It now splits the name and subtracts 1 for zero-indexing.
                # 'num_finger_1' -> split -> '1' -> int -> 1 -> -1 -> label 0
                finger_label = int(finger_count_folder.split('_')[-1]) - 1
                if finger_label < 0: # Basic check to skip non-conforming folders
                    continue

                # Iterate through the gesture folders inside (e.g., 'gesture_0', 'gesture_1')
                for gesture_folder_name in sorted(os.listdir(finger_folder_path)):
                    gesture_folder_path = os.path.join(finger_folder_path, gesture_folder_name)

                    if os.path.isdir(gesture_folder_path):
                        try:
                            # The gesture label is the number from the sub-folder name
                            gesture_label = int(gesture_folder_name.split('_')[-1])

                            # Load all .txt files from this final directory
                            for f in sorted(os.listdir(gesture_folder_path)):
                                if f.endswith('.txt'):
                                    trial = load_data_from_file(os.path.join(gesture_folder_path, f))
                                    if trial is not None:
                                        all_data.append(trial)
                                        finger_labels.append(finger_label)
                                        gesture_labels.append(gesture_label)
                        except (ValueError, IndexError):
                            print(f"Warning: Could not extract gesture label from '{gesture_folder_name}'. Skipping.")
                            continue
            except (ValueError, IndexError):
                print(f"Warning: Could not extract finger label from top-level folder '{finger_count_folder}'. Skipping.")
                continue

    return all_data, finger_labels, gesture_labels


class HierarchicalDataset(Dataset):
    """Dataset that returns data and two separate labels."""

    def __init__(self, data_list, finger_labels, gesture_labels):
        self.data_list = data_list
        self.finger_labels = finger_labels
        self.gesture_labels = gesture_labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = torch.tensor(self.data_list[idx], dtype=torch.float32)
        finger_label = torch.tensor(self.finger_labels[idx], dtype=torch.long)
        gesture_label = torch.tensor(self.gesture_labels[idx], dtype=torch.long)
        return data, finger_label, gesture_label


class HierarchicalGestureModel(nn.Module):
    """
    A model with a shared backbone and two separate classification heads.
    """

    def __init__(self, backbone, d_model, num_finger_classes, num_gesture_classes):
        super().__init__()
        self.backbone = backbone

        # Head 1: Predicts the number of fingers
        self.finger_classifier = nn.Linear(d_model, num_finger_classes)

        # Head 2: Predicts the gesture direction
        self.gesture_classifier = nn.Linear(d_model, num_gesture_classes)

    def forward(self, src, src_padding_mask):
        # Get the rich representation from the shared backbone
        representations = self.backbone(src, src_padding_mask)

        # Use the [CLS] token's representation for both classifications
        cls_representation = representations[:, 0, :]

        # Get predictions from both heads
        finger_logits = self.finger_classifier(cls_representation)
        gesture_logits = self.gesture_classifier(cls_representation)

        return finger_logits, gesture_logits


# --- DATA LOADING FUNCTIONS ---
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


def load_predict_data(predict_path):
    all_data, filenames = [], []
    if not os.path.exists(predict_path): print(
        f"Warning: Prediction directory not found {predict_path}"); return all_data, filenames
    file_paths = [os.path.join(predict_path, f) for f in os.listdir(predict_path) if f.endswith('.txt')]
    try:
        files_sorted = sorted(file_paths, key=lambda p: int(os.path.basename(p).split('.')[0].split('_')[-1]))
    except (ValueError, IndexError):
        files_sorted = sorted(file_paths)
    for file_path in files_sorted:
        features = load_data_from_file(file_path)
        if features is not None: all_data.append(features); filenames.append(os.path.basename(file_path))
    return all_data, filenames


# --- MODEL ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__();
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1);
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model);
        pe[:, 0, 0::2] = torch.sin(position * div_term);
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.dropout(x + self.pe[:x.size(0)])


class GestureBackbone(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.5):
        super().__init__();
        self.d_model = d_model;
        self.height, self.width = SENSOR_ROWS, SENSOR_COLS
        self.conv = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        with torch.no_grad(): cnn_output_size = self.conv(torch.zeros(1, 1, self.height, self.width)).view(1, -1).size(
            1)
        self.input_proj = nn.Linear(cnn_output_size, d_model);
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True);
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = src.shape;
        src_reshaped = src.contiguous().view(-1, 1, self.height, self.width)
        cnn_out = self.conv(src_reshaped).view(batch_size, seq_len, -1);
        trans_input = self.input_proj(cnn_out) * math.sqrt(self.d_model)
        trans_input = self.pos_encoder(trans_input.permute(1, 0, 2)).permute(1, 0, 2)
        return self.transformer_encoder(trans_input, src_key_padding_mask=src_padding_mask)


class DenoisingGestureTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.5):
        super().__init__();
        self.d_model = d_model
        self.encoder_backbone = GestureBackbone(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder_input_proj = nn.Linear(NUM_SENSORS, d_model);
        self.decoder_pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True);
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_proj = nn.Linear(d_model, NUM_SENSORS)

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, tgt_subsequent_mask):
        memory = self.encoder_backbone(src, src_padding_mask);
        tgt_embedded = self.decoder_input_proj(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.decoder_pos_encoder(tgt_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        decoder_output = self.decoder(tgt_embedded, memory, tgt_mask=tgt_subsequent_mask,
                                      tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        return self.output_proj(decoder_output)


def create_additive_augmented_data(clean_data_path, broken_data_path, output_path, scaler_save_path,
                                   num_augmentations_per_clean):
    """
    Creates an augmented dataset for denoising by additively blending broken signals
    with clean gestures. It also calculates and saves the standardization scaler.

    Args:
        clean_data_path (str): Path to the folder with clean gesture subdirectories.
        broken_data_path (str): Path to the folder with pure broken gesture files.
        output_path (str): Path to save the new 'noisy_XXX.npy' and 'clean_XXX.npy' files.
        scaler_save_path (str): Path to save the 'denoising_scaler.npz' file.
        num_augmentations_per_clean (int): How many noisy versions to create for each clean gesture.
    """
    print("--- Creating Additive Augmented Data for Denoising ---")

    # 1. Load all clean gestures and all broken gestures
    print(f"Loading clean gestures from: {clean_data_path}")
    clean_gestures_raw = load_all_clean_data(clean_data_path)
    if not clean_gestures_raw:
        print("Error: No clean gestures found. Aborting.")
        return

    print(f"Loading broken signals from: {broken_data_path}")
    broken_gestures = load_unlabeled_from_flat_dir(broken_data_path)
    if not broken_gestures:
        print("Error: No broken signal data found. Aborting.")
        return

    print(f"Found {len(clean_gestures_raw)} clean gestures and {len(broken_gestures)} broken signals.")

    # --- THIS IS THE NEW, CRUCIAL PART ---
    # 2. Calculate and save scaler based on CLEAN data ONLY.
    print("\nCalculating scaler from clean data...")
    global_mean, global_std = get_standardization_params(clean_gestures_raw)
    np.savez(scaler_save_path, mean=global_mean, std=global_std)
    print(f"Standardization parameters for denoising saved to {scaler_save_path}\n")

    # 3. Standardize the clean data using the new scaler
    clean_gestures_std = standardize_data(clean_gestures_raw, global_mean, global_std)
    # Note: We don't need to standardize the broken data, as it's just an additive signal.
    # If we did, we would use the same scaler from the clean data.

    # 4. Prepare output directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # 5. Create the paired dataset
    pair_counter = 0
    # Use the standardized clean gestures now
    for clean_gesture_std in clean_gestures_std:
        clean_len = len(clean_gesture_std)
        if clean_len == 0:
            continue

        for _ in range(num_augmentations_per_clean):
            source_broken_signal = random.choice(broken_gestures)
            broken_len = len(source_broken_signal)

            if broken_len >= clean_len:
                start_index = random.randint(0, broken_len - clean_len)
                broken_segment = source_broken_signal[start_index: start_index + clean_len]
            else:
                num_repeats = (clean_len // broken_len) + 1
                tiled_broken = np.tile(source_broken_signal, (num_repeats, 1))
                broken_segment = tiled_broken[:clean_len]

            # --- THE CORE LOGIC ---
            # Standardize the broken segment ADDITIVELY.
            # We add the raw broken signal to the standardized clean signal.
            # The model will learn to subtract this un-standardized offset.
            # An alternative would be to standardize the broken signal too, but this is simpler.
            damaged_input = clean_gesture_std + broken_segment

            noisy_input_seq = damaged_input
            clean_target_seq = clean_gesture_std

            np.save(os.path.join(output_path, f'noisy_{pair_counter}.npy'), noisy_input_seq)
            np.save(os.path.join(output_path, f'clean_{pair_counter}.npy'), clean_target_seq)
            pair_counter += 1

    print("-" * 50)
    print(f"Data augmentation complete. Created {pair_counter} paired examples in {output_path}")


# --- MAIN SCRIPT ---
def main():
    set_seed(77)
    # --- PATHS ---
    base_path = '//ai'
    traing_data_path = os.path.join(base_path, 'training_data')
    clean_train_path = os.path.join(base_path, 'training_data_clean')
    broken_data_path = os.path.join(base_path, 'training_data_broken')
    groundtruth_data_path = '//ai/training_data_clean/num_finger_1/gesture_cylinder_groundtruth_6'
    pre_train_path = os.path.join(base_path, 'pre_train_data')
    predict_path = '//predict/'

    # --- Track 1 & 2 Paths (Original) ---
    models_path_classify = '//models_classification/'
    pretrained_backbone_path = os.path.join(models_path_classify, 'backbone.pth')
    pretrained_config_path = os.path.join(models_path_classify, 'backbone_config.txt')
    scaler_path_classify = os.path.join(models_path_classify, 'classification_scaler.npz')
    models_path_hierarchical = '//models_classification/'
    hierarchical_model_path = os.path.join(models_path_hierarchical, 'hierarchical_model.pth')
    hierarchical_config_path = os.path.join(models_path_hierarchical, 'hierarchical_config.txt')
    models_path_denoise = '//models_denoising/'
    denoising_preprocessed_path = os.path.join(models_path_denoise, 'preprocessed_data')
    denoising_model_path = os.path.join(models_path_denoise, 'denoising_model.pth')
    denoising_config_path = os.path.join(models_path_denoise, 'denoising_config.txt')
    scaler_path_denoise = os.path.join(models_path_denoise, 'denoising_scaler.npz')

    # --- Track 3 Paths (Fixed-Length 3-Level) ---
    three_level_train_path = os.path.join(base_path, 'training_data_3level')
    models_path_3level_backbone = '//models_3level_backbone/'
    specialized_backbone_path = os.path.join(models_path_3level_backbone, 'backbone_3level.pth')
    specialized_config_path = os.path.join(models_path_3level_backbone, 'backbone_3level_config.txt')
    specialized_scaler_path = os.path.join(models_path_3level_backbone, 'scaler_3level.npz')
    models_path_3level = '//models_3level_backbone/'
    three_level_model_path = os.path.join(models_path_3level, '3level_model.pth')
    three_level_config_path = os.path.join(models_path_3level, '3level_config.txt')

    # --- Track 4 Paths (Variable-Length 3-Level) ---
    variable_length_train_path = os.path.join(base_path, 'training_data_variable')
    models_path_variable = '//models_variable/'
    variable_backbone_path = os.path.join(models_path_variable, 'backbone_variable.pth')
    variable_config_path = os.path.join(models_path_variable, 'backbone_variable_config.txt')
    variable_scaler_path = os.path.join(models_path_variable, 'scaler_variable.npz')
    variable_model_path = os.path.join(models_path_variable, 'model_variable.pth')
    variable_3level_config_path = os.path.join(models_path_variable, '3level_config_variable.txt')

    for path in [models_path_classify, models_path_hierarchical, models_path_denoise, pre_train_path,
                 broken_data_path, groundtruth_data_path, models_path_3level, three_level_train_path,
                 models_path_3level_backbone, models_path_variable, variable_length_train_path]:
        if not os.path.exists(path): os.makedirs(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # --- MENU SYSTEM ---
    print("--- Gesture Recognition Research Pipeline ---")
    main_choice = input(
        "Select a track:\n"
        " 1: Main Workflow (2-Level Hierarchical)\n"
        " 2: Denoising Utility\n"
        " 3: 3-Level Hierarchical Classifier (Fixed-Length)\n"
        " 4: ADVANCED 3-Level Classifier (Variable-Length)\n"
        "Enter option: "
    )

    select = None

    if main_choice == '1':
        # --- Main Workflow Sub-Menu ---
        select = input(
            "\n--- Main Workflow: Hierarchical Conv-Transformer ---\n"
            "Select a step (run in order):\n"
            " 0: PREPARE Pre-training Data (from all training data)\n"
            " 1: Pre-train Conv-Transformer Backbone\n"
            " 2: Fine-tune Hierarchical Classifier (on clean data)\n"
            " 3: PREDICT with Hierarchical Classifier\n"
            "Enter option: "
        )

    elif main_choice == '2':
        # --- Denoising Utility Menu (CLEANED UP) ---
        select = input(
            "\n--- Denoising Utility ---\n"
            "Select an option:\n"
            " 4: PREPARE Denoising Data (Additive Augmentation)\n"
            " 5: TRAIN Denoising Autoencoder\n"
            " 6: DENOISE a new gesture sequence\n\n"
            "Enter option number: "
        )

    elif main_choice == '3':
        select = input(
            "\n--- 3-Level Hierarchical Classifier (Fixed-Length) ---\n"
            " 7: PREPARE 3-Level Fixed-Length Data\n"
            " 8: Pre-train Backbone (Hyperparameter Search)\n"
            " 8.5: Final Pre-training Run (Longer Epochs) <-- NEW\n"
            " 9: Fine-tune 3-Level Classifier\n"
            " 10: PREDICT with 3-Level Fixed-Length Classifier\n"
            "Enter option number: "
        )

    elif main_choice == '4':
        select = input(
            "\n--- ADVANCED 3-Level Classifier (Variable-Length) ---\n"
            " 11: PREPARE 3-Level Variable-Length Data\n"
            " 12: Pre-train Backbone (Hyperparameter Search)\n"
            " 12.5: Final Pre-training Run (Longer Epochs) <-- NEW\n"
            " 13: Fine-tune Classifier on Variable-Length Data\n"
            " 14: PREDICT with 3-Level Variable-Length Classifier\n"
            "Enter option number: "
        )

    else:
        print("Invalid main choice.")
        return

    if select == '0':
        print(f"\n--- Preparing Pre-training Data ---")
        if os.path.exists(pre_train_path): shutil.rmtree(pre_train_path)
        os.makedirs(pre_train_path)
        all_files_to_copy = []
        for path in [traing_data_path]:
            if not os.path.exists(path): print(f"Warning: Path not found, skipping: {path}"); continue
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith('.txt'): all_files_to_copy.append(os.path.join(root, f))
        if not all_files_to_copy: print("Error: No source files found."); return
        print(f"Found {len(all_files_to_copy)} files to copy and anonymize.")
        for i, src_path in enumerate(all_files_to_copy):
            dest_path = os.path.join(pre_train_path, f"data_{i + 1}.txt")
            shutil.copyfile(src_path, dest_path)
        print(f"Successfully copied {len(all_files_to_copy)} files to {pre_train_path}")

    elif select == '1':
        print("\n--- Pre-training Conv-Transformer Backbone ---")
        if not os.path.exists(pre_train_path) or not os.listdir(pre_train_path): print(
            f"Error: Pre-training data folder is empty. Run Option 0 first."); return
        unlabeled_data_raw = load_unlabeled_from_flat_dir(pre_train_path)
        if not unlabeled_data_raw: print("Error: Failed to load data from pre-training folder."); return

        global_mean, global_std = get_standardization_params(unlabeled_data_raw)
        np.savez(scaler_path_classify, mean=global_mean, std=global_std)
        data_for_pretraining = standardize_data(unlabeled_data_raw, global_mean, global_std)
        pretrain_dataset = UnlabeledGestureDataset(data_for_pretraining)

        d_model_options = [768, 1024]
        n_head_options = [4, 8]
        num_layers_options = [2, 4, 6]
        dropout_options = [0.1, 0.2, 0.3]
        lr_options = [0.0001, 0.00005, 0.00001]
        PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, MASKING_RATIO = 50, 16, 0.25

        def collate_fn_pretrain(batch):
            sequences = nn.utils.rnn.pad_sequence(batch, batch_first=True);
            batch_size, seq_len, _ = sequences.shape
            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool);
            inputs, labels = sequences.clone(), sequences.clone()
            for i, seq in enumerate(batch): padding_mask[i, len(seq):] = True
            loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for i in range(batch_size):
                seq_len_unpadded = seq_len - padding_mask[i].sum();
                num_to_mask = int(seq_len_unpadded * MASKING_RATIO)
                if num_to_mask > 0:
                    mask_start = random.randint(0, seq_len_unpadded - num_to_mask)
                    inputs[i, mask_start:mask_start + num_to_mask, :] = 0.0
                    loss_mask[i, mask_start:mask_start + num_to_mask, :] = True
            return inputs, padding_mask, labels, loss_mask

        pretrain_loader = DataLoader(pretrain_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True,
                                     collate_fn=collate_fn_pretrain)
        best_loss = float('inf');
        best_config = {}

        for d_model in d_model_options:
            for n_head in n_head_options:
                if d_model % n_head != 0: continue
                for num_layers in num_layers_options:
                    for dropout in dropout_options:
                        for lr in lr_options:
                            current_config = {"D_MODEL": d_model, "N_HEAD": n_head, "NUM_ENC_LAYERS": num_layers,
                                              "DROPOUT": dropout, "PRETRAIN_LR": lr}
                            print("-" * 50);
                            print(f"Testing configuration: {current_config}")
                            backbone = GestureBackbone(d_model, n_head, num_layers, d_model * 4, dropout).to(device)
                            prediction_head = nn.Linear(d_model, NUM_SENSORS).to(device)
                            optimizer = optim.Adam(list(backbone.parameters()) + list(prediction_head.parameters()),
                                                   lr=lr)
                            criterion = nn.MSELoss()
                            final_epoch_loss = 0
                            for epoch in range(PRETRAIN_EPOCHS):
                                backbone.train();
                                prediction_head.train();
                                total_loss = 0
                                for d in pretrain_loader:
                                    inputs, padding_mask, labels, loss_mask = [x.to(device) for x in d]
                                    optimizer.zero_grad();
                                    predictions = prediction_head(backbone(inputs, padding_mask))
                                    loss = criterion(predictions[loss_mask], labels[loss_mask]);
                                    loss.backward();
                                    optimizer.step()
                                    total_loss += loss.item()
                                final_epoch_loss = total_loss / len(pretrain_loader)
                                if (epoch + 1) % 10 == 0: print(
                                    f"  Epoch [{epoch + 1}/{PRETRAIN_EPOCHS}], Loss: {final_epoch_loss:.6f}")
                            print(f"Final loss for this configuration: {final_epoch_loss:.6f}")
                            if final_epoch_loss < best_loss:
                                best_loss = final_epoch_loss;
                                best_config = current_config.copy();
                                best_config['BEST_LOSS'] = best_loss
                                print(f"*** New best model found! Loss: {best_loss:.6f} ***");
                                print("Saving backbone model and configuration...")
                                torch.save(backbone.state_dict(), pretrained_backbone_path)
                                with open(pretrained_config_path, 'w') as f:
                                    for key, value in best_config.items(): f.write(f"{key}: {value}\n")
        print("\n" + "=" * 50);
        print("Hyperparameter search finished!");
        print(f"Best model saved to {pretrained_backbone_path}");
        print(f"Best configuration (Loss: {best_loss:.6f}):\n{best_config}");
        print("=" * 50)

    elif select == '2':
        print("\n--- Fine-tuning Hierarchical Classifier (on clean data) ---")
        for p in [pretrained_backbone_path, pretrained_config_path, scaler_path_classify]:
            if not os.path.exists(p): print(
                f"Error: Required file not found at {p}. Run pre-training (Option 1) first."); return

        config = parse_config_file(pretrained_config_path)
        if not config: print("Error loading config."); return

        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
            config['NUM_ENC_LAYERS']), config['DROPOUT']
        print(f"Loaded backbone configuration: {config}")

        scaler_data = np.load(scaler_path_classify);
        global_mean, global_std = scaler_data['mean'], scaler_data['std']
        print(f"Loaded standardization parameters: Mean={global_mean:.4f}, Std={global_std:.4f}")

        train_data_raw, finger_labels, gesture_labels = load_hierarchical_data_from_nested_folders(clean_train_path)
        if not train_data_raw: print("Error: No hierarchical training data found in clean data path."); return
        train_data_std = standardize_data(train_data_raw, global_mean, global_std)

        num_finger_classes = len(set(finger_labels))
        num_gesture_classes = len(set(gesture_labels))
        if num_finger_classes == 0 or num_gesture_classes == 0:
            print("Error: Could not detect any classes from the training data. Check folder structure.")
            return
        print("-" * 50)
        print(f"--> Automatically detected {num_finger_classes} finger classes.")
        print(f"--> Automatically detected {num_gesture_classes} gesture classes.")
        print("-" * 50)

        dataset = HierarchicalDataset(train_data_std, finger_labels, gesture_labels)

        def collate_fn_hierarchical(batch):
            data, f_labels, g_labels = zip(*batch);
            padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True)
            mask = torch.zeros(padded_data.shape[0], padded_data.shape[1], dtype=torch.bool)
            for i, seq in enumerate(data): mask[i, len(seq):] = True
            return padded_data, torch.stack(f_labels), torch.stack(g_labels), mask

        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_hierarchical)

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
        print(f"Loading pre-trained weights from {pretrained_backbone_path}")
        backbone.load_state_dict(torch.load(pretrained_backbone_path, map_location=device, weights_only=True))

        model = HierarchicalGestureModel(backbone, d_model=D_MODEL,
                                         num_finger_classes=num_finger_classes,
                                         num_gesture_classes=num_gesture_classes).to(device)
        criterion = nn.CrossEntropyLoss();
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        EPOCHS = 50
        for epoch in range(EPOCHS):
            model.train();
            total_loss, correct_finger, correct_gesture, total_samples = 0, 0, 0, 0
            for data, f_labels, g_labels, mask in train_loader:
                data, f_labels, g_labels, mask = data.to(device), f_labels.to(device), g_labels.to(device), mask.to(
                    device)
                optimizer.zero_grad();
                finger_logits, gesture_logits = model(data, mask)
                loss_f = criterion(finger_logits, f_labels);
                loss_g = criterion(gesture_logits, g_labels)
                combined_loss = loss_f + loss_g;
                combined_loss.backward();
                optimizer.step()
                total_loss += combined_loss.item()
                _, pred_f = torch.max(finger_logits, 1);
                _, pred_g = torch.max(gesture_logits, 1)
                correct_finger += (pred_f == f_labels).sum().item();
                correct_gesture += (pred_g == g_labels).sum().item()
                total_samples += f_labels.size(0)
            acc_f = 100 * correct_finger / total_samples;
            acc_g = 100 * correct_gesture / total_samples
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f} | Finger Acc: {acc_f:.2f}% | Gesture Acc: {acc_g:.2f}%")

        print(f"Hierarchical training finished. Saving model to {hierarchical_model_path}");
        torch.save(model.state_dict(), hierarchical_model_path)

        with open(hierarchical_config_path, 'w') as f:
            f.write(f"NUM_FINGER_CLASSES: {num_finger_classes}\n")
            f.write(f"NUM_GESTURE_CLASSES: {num_gesture_classes}\n")
        print(f"Hierarchical model configuration saved to {hierarchical_config_path}")

    elif select == '3':
        print("\n--- Starting Stage 3: Prediction with Hierarchical Classifier ---")
        for p in [hierarchical_model_path, pretrained_config_path, scaler_path_classify, hierarchical_config_path]:
            if not os.path.exists(p): print(f"Error: Required file not found at {p}."); return

        config = parse_config_file(pretrained_config_path)
        if not config: print("Error loading backbone config."); return

        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
            config['NUM_ENC_LAYERS']), config['DROPOUT']
        print(f"Loaded backbone configuration: {config}")

        hierarchical_config = parse_config_file(hierarchical_config_path)
        if not hierarchical_config: print("Error loading hierarchical config."); return
        num_finger_classes = int(hierarchical_config['NUM_FINGER_CLASSES'])
        num_gesture_classes = int(hierarchical_config['NUM_GESTURE_CLASSES'])
        print(
            f"Loaded hierarchical config: {num_finger_classes} finger classes, {num_gesture_classes} gesture classes.")

        scaler_data = np.load(scaler_path_classify);
        global_mean, global_std = scaler_data['mean'], scaler_data['std']

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
        model = HierarchicalGestureModel(backbone, D_MODEL, num_finger_classes, num_gesture_classes).to(device)
        model.load_state_dict(torch.load(hierarchical_model_path, map_location=device, weights_only=True));
        model.eval()

        predict_data_raw, filenames = load_predict_data(predict_path)
        if not predict_data_raw: print("No valid prediction data found."); return
        predict_data_std = standardize_data(predict_data_raw, global_mean, global_std)

        with torch.no_grad():
            for i, seq_np in enumerate(predict_data_std):
                filename = filenames[i]
                data_tensor = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(device)
                mask = torch.zeros(1, data_tensor.shape[1], dtype=torch.bool).to(device)

                finger_logits, gesture_logits = model(data_tensor, mask)

                pred_f = torch.argmax(finger_logits, dim=1).item()
                pred_g = torch.argmax(gesture_logits, dim=1).item()

                finger_str = f"{pred_f + 1} Finger(s)"
                gesture_map = {0: "Left", 1: "Right", 2: "Down", 3: "Up"}
                gesture_str = gesture_map.get(pred_g, f"Unknown Gesture ({pred_g})")

                print(f"File: {filename} -> Predicted: {finger_str}, moving {gesture_str}")

    elif select == '4':
        # This new option calls the function we just created.
        create_additive_augmented_data(
            clean_data_path=clean_train_path,
            broken_data_path=broken_data_path,
            output_path=denoising_preprocessed_path,
            scaler_save_path=scaler_path_denoise,  # Pass the path here
            num_augmentations_per_clean=5
        )

    elif select == '5':
        print("\n--- Starting Stage 5: Training Denoising Autoencoder with Hyperparameter Search ---")
        if not os.path.exists(denoising_preprocessed_path) or not os.listdir(denoising_preprocessed_path):
            print(
                f"Error: Preprocessed denoising data not found at {denoising_preprocessed_path}. Please run option 4 first.");
            return

        # --- Hyperparameter Search Space for the Denoiser ---
        d_model_options = [128, 256]
        n_head_options = [4, 8]
        num_layers_options = [2, 4]
        dropout_options = [0.1, 0.2, 0.3]
        lr_options = [0.0001, 0.00005]
        EPOCHS, BATCH_SIZE = 50, 32

        noisy_paths = [os.path.join(denoising_preprocessed_path, f) for f in
                       sorted(os.listdir(denoising_preprocessed_path)) if f.startswith('noisy_')]
        clean_paths = [os.path.join(denoising_preprocessed_path, f) for f in
                       sorted(os.listdir(denoising_preprocessed_path)) if f.startswith('clean_')]
        dataset = DenoisingPairedDataset(noisy_paths, clean_paths)

        def collate_fn_denoising(batch):
            noisy_seqs, clean_seqs = zip(*batch)
            noisy_padded = nn.utils.rnn.pad_sequence(noisy_seqs, batch_first=True);
            clean_padded = nn.utils.rnn.pad_sequence(clean_seqs, batch_first=True)
            noisy_mask = torch.all(noisy_padded == 0, dim=2);
            clean_mask = torch.all(clean_padded == 0, dim=2)
            return noisy_padded, clean_padded, noisy_mask, clean_mask

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_denoising)

        best_loss = float('inf')
        best_config = {}

        for d_model in d_model_options:
            for n_head in n_head_options:
                if d_model % n_head != 0: continue
                for num_layers in num_layers_options:
                    for dropout in dropout_options:
                        for lr in lr_options:
                            current_config = {
                                "D_MODEL": d_model, "N_HEAD": n_head,
                                "NUM_ENC_LAYERS": num_layers, "NUM_DEC_LAYERS": num_layers,
                                "DROPOUT": dropout, "LEARNING_RATE": lr
                            }
                            print("-" * 50)
                            print(f"Testing Denoiser configuration: {current_config}")

                            model = DenoisingGestureTransformer(d_model, n_head, num_layers, num_layers, d_model * 4,
                                                                dropout).to(device)
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            criterion = nn.MSELoss()

                            final_epoch_loss = 0
                            for epoch in range(EPOCHS):
                                model.train()
                                total_loss = 0
                                for noisy_seq, clean_seq, noisy_mask, clean_mask in train_loader:
                                    noisy_seq, clean_seq, noisy_mask = noisy_seq.to(device), clean_seq.to(
                                        device), noisy_mask.to(device)
                                    start_token = torch.zeros(clean_seq.size(0), 1, clean_seq.size(2), device=device)
                                    shifted_clean_seq = torch.cat([start_token, clean_seq[:, :-1, :]], dim=1)
                                    shifted_clean_mask = torch.all(shifted_clean_seq == 0, dim=2)

                                    tgt_subsequent_mask = nn.Transformer.generate_square_subsequent_mask(
                                        shifted_clean_seq.size(1)).to(device)

                                    optimizer.zero_grad()
                                    predicted_sequence = model(noisy_seq, shifted_clean_seq, noisy_mask,
                                                               shifted_clean_mask, tgt_subsequent_mask)

                                    loss = criterion(predicted_sequence[~clean_mask], clean_seq[~clean_mask])
                                    loss.backward()
                                    optimizer.step()
                                    total_loss += loss.item()

                                final_epoch_loss = total_loss / len(train_loader)
                                if (epoch + 1) % 10 == 0:
                                    print(f"  Epoch [{epoch + 1}/{EPOCHS}], Loss: {final_epoch_loss:.6f}")

                            print(f"Final loss for this configuration: {final_epoch_loss:.6f}")
                            if final_epoch_loss < best_loss:
                                best_loss = final_epoch_loss
                                best_config = current_config.copy()
                                best_config['BEST_LOSS'] = best_loss
                                print(f"*** New best Denoiser model found! Loss: {best_loss:.6f} ***")
                                print("Saving Denoiser model and configuration...")
                                torch.save(model.state_dict(), denoising_model_path)
                                with open(denoising_config_path, 'w') as f:
                                    for k, v in best_config.items():
                                        f.write(f"{k}: {v}\n")

        print("\n" + "=" * 50)
        print("Denoiser Hyperparameter search finished!")
        print(f"Best Denoiser model saved to {denoising_model_path}")
        print(f"Best configuration (Loss: {best_loss:.6f}):\n{best_config}")
        print("=" * 50)

    elif select == '6':
        print("\n--- Starting Stage 6: Denoising a new gesture sequence ---")


        for p in [denoising_model_path, scaler_path_denoise, denoising_config_path]:
            if not os.path.exists(p): print(
                f"Error: Required file not found at {p}. Please run previous stages first."); return

        config = parse_config_file(denoising_config_path)
        if not config: print("Error loading denoiser config."); return

        D_MODEL = int(config['D_MODEL'])
        N_HEAD = int(config['N_HEAD'])
        NUM_ENC_LAYERS = int(config['NUM_ENC_LAYERS'])
        NUM_DEC_LAYERS = int(config['NUM_DEC_LAYERS'])
        DROPOUT = float(config['DROPOUT'])
        print(f"Loaded best denoiser configuration: {config}")

        scaler_data = np.load(scaler_path_denoise);
        global_mean, global_std = scaler_data['mean'], scaler_data['std']

        model = DenoisingGestureTransformer(D_MODEL, N_HEAD, NUM_ENC_LAYERS, NUM_DEC_LAYERS, D_MODEL * 4, DROPOUT).to(
            device)
        model.load_state_dict(torch.load(denoising_model_path, map_location=device, weights_only=True));
        model.eval()

        predict_data_raw, filenames = load_predict_data(predict_path)
        if not predict_data_raw: print("No prediction data found."); return
        predict_data_std = standardize_data(predict_data_raw, global_mean, global_std)

        with torch.no_grad():
            for i, noisy_seq_np in enumerate(predict_data_std):
                filename = filenames[i];
                print(f"\nDenoising file: {filename}")

                noisy_seq = torch.tensor(noisy_seq_np, dtype=torch.float32).unsqueeze(0).to(device)
                src_padding_mask = torch.all(noisy_seq == 0, dim=2)

                output_seq = torch.zeros(1, 1, NUM_SENSORS, device=device)

                memory = model.encoder_backbone(noisy_seq, src_padding_mask)

                for _ in range(noisy_seq.size(1)):
                    tgt_subsequent_mask = nn.Transformer.generate_square_subsequent_mask(output_seq.size(1)).to(device)

                    tgt_embedded = model.decoder_pos_encoder(model.decoder_input_proj(output_seq))
                    decoder_output = model.decoder(tgt_embedded, memory, tgt_mask=tgt_subsequent_mask)

                    next_token_logits = model.output_proj(decoder_output[:, -1, :])

                    output_seq = torch.cat([output_seq, next_token_logits.unsqueeze(1)], dim=1)

                final_clean_seq = output_seq[:, 1:, :].squeeze(0).cpu().numpy()

                final_clean_seq_original_scale = (final_clean_seq * global_std) + global_mean

                save_filename = f"DENOISED_{filename}"
                save_path = os.path.join(models_path_denoise, save_filename)
                np.savetxt(save_path, final_clean_seq_original_scale, fmt='%.4f')
                print(f"Saved cleaned gesture to: {save_path}")

    elif select == '7':
        print("\n--- Starting Stage 7: Preparing 3-Level Fixed-Length Data Structure ---")
        prepare_three_level_data(
            clean_data_path=clean_train_path,
            broken_data_path=broken_data_path,
            groundtruth_data_path=groundtruth_data_path,
            output_base_path=three_level_train_path,
            num_good_versions=50,
            num_damaged_versions=0,
            window_size=30,
            min_gesture_len=10
        )

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
        d_model_options = [768]
        n_head_options = [16]
        num_layers_options = [6]
        dropout_options = [0.1]
        lr_options = [0.00005]
        # masking_ratio_options = [0.25, 0.5, 0.75]  # NEW: Search over masking ratios
        masking_ratio_options = [0.25]  # NEW: Search over masking ratios

        PRETRAIN_EPOCHS = 30
        PRETRAIN_BATCH_SIZE = 16

        # --- Step 5: Modified Collate Fn and DataLoaders ---
        # The collate_fn needs the masking ratio as an argument now
        def collate_fn_pretrain(batch, masking_ratio):
            # All data is fixed length, so we can use torch.stack
            sequences = torch.stack(batch)
            batch_size, seq_len, _ = sequences.shape

            inputs, labels = sequences.clone(), sequences.clone()
            loss_mask = torch.zeros_like(inputs, dtype=torch.bool)

            for i in range(batch_size):
                num_to_mask = int(seq_len * masking_ratio)
                if num_to_mask > 0:
                    # # Version 2 with RANDOM FRAME MASKING
                    # Using random frame masking for a harder task
                    indices_to_mask = random.sample(range(seq_len), num_to_mask)
                    inputs[i, indices_to_mask, :] = 0.0
                    loss_mask[i, indices_to_mask, :] = True

                    # #  Version 1: with SPAN MASKING,
                    # Choose a single starting point for the block
                    # mask_start = random.randint(0, seq_len - num_to_mask)
                    # # Apply the mask as a continuous block (span)
                    # inputs[i, mask_start: mask_start + num_to_mask, :] = 0.0
                    # loss_mask[i, mask_start: mask_start + num_to_mask, :] = True

            # No padding mask needed as all data is fixed length
            return inputs, None, labels, loss_mask

        # We will create the DataLoaders inside the loop, as the collate_fn depends on the masking ratio

        best_loss = float('inf')
        best_config = {}

        # --- Step 6: The Full Search Loop with all enhancements ---
        for d_model in d_model_options:
            for n_head in n_head_options:
                if d_model % n_head != 0: continue
                for num_layers in num_layers_options:
                    for dropout in dropout_options:
                        for lr in lr_options:
                            for masking_ratio in masking_ratio_options:  # NEW loop
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
                                optimizer = optim.Adam(list(backbone.parameters()) + list(prediction_head.parameters()),
                                                       lr=lr)
                                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3,
                                                              verbose=True)  # NEW scheduler
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

                                    if (epoch + 1) % 5 == 0:
                                        print(
                                            f"  Epoch {epoch + 1:02d} | Train Loss: {total_train_loss / len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")

                                    # Step the scheduler with the validation loss
                                    scheduler.step(avg_val_loss)

                                print(f"Final Validation Loss for this configuration: {final_val_loss:.6f}")

                                # --- Decision is now based on VALIDATION loss ---
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
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
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
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

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
                with open(three_level_config_path, 'w') as f:
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

        all_g_labels, all_g_preds = [], []
        all_q_labels, all_q_preds = [], []
        with torch.no_grad():
            for data_batch, _, g_lbl, q_lbl, _ in val_loader:
                data_batch = data_batch.to(device)
                _, g_logits, q_logits = model(data_batch, None)
                g_pred = torch.argmax(g_logits, 1)
                q_pred = torch.argmax(q_logits, 1)
                all_g_labels.extend(g_lbl.cpu().numpy())
                all_g_preds.extend(g_pred.cpu().numpy())
                all_q_labels.extend(q_lbl.cpu().numpy())
                all_q_preds.extend(q_pred.cpu().numpy())

        print("\n--- Gesture Classification Report ---")
        gesture_names = [f"Gesture_{i}" for i in range(num_gesture_classes)]
        # You can manually create a more descriptive map if you want
        # gesture_map = {0: "Push", 1: "Pull", 2: "Down", 3: "Up", 4: "Do Nothing"}
        # gesture_names = [gesture_map.get(i, f"Unknown_{i}") for i in range(num_gesture_classes)]
        print(classification_report(all_g_labels, all_g_preds, target_names=gesture_names, digits=3))

        print("\n--- Quality Classification Report ---")
        quality_names = ["Good (0)", "Damaged (1)"]
        print(classification_report(all_q_labels, all_q_preds, target_names=quality_names, digits=3))

        print("\n--- Gesture Confusion Matrix ---")
        cm_gesture = confusion_matrix(all_g_labels, all_g_preds)
        cm_df = pd.DataFrame(cm_gesture, index=gesture_names, columns=gesture_names)
        print(cm_df)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Gesture Confusion Matrix (Validation Set)')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_save_path = os.path.join(models_path_3level, 'gesture_confusion_matrix.png')
        plt.savefig(cm_save_path)
        print(f"\nGesture confusion matrix plot saved to {cm_save_path}")

    elif select == '10':
        print("\n--- Starting Stage 9: Prediction with 3-Level Classifier ---")

        for p in [three_level_model_path, specialized_config_path, specialized_scaler_path, three_level_config_path]:
            if not os.path.exists(p):
                print(f"Error: Required file not found at {p}.")
                return

        config = parse_config_file(specialized_config_path)
        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
            config['NUM_ENC_LAYERS']), config['DROPOUT']

        config_3level = parse_config_file(three_level_config_path)
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

                finger_str = f"{pred_f + 1} Finger(s)"
                gesture_map = {0: "Push", 1: "Pull", 2: "Left", 3: "Right", 4:"Down", 5:"Up"}
                gesture_str = gesture_map.get(pred_g, "Unknown")
                quality_str = "Good" if pred_q == 0 else "Damaged"

                print(f"File: {filename} -> Predicted: {finger_str}, {gesture_str}, Quality: {quality_str}")

    elif select == '11':
        print("\n--- Preparing Variable-Length 3-Level Data ---")
        prepare_variable_length_data(
            clean_data_path=clean_train_path,
            broken_data_path=broken_data_path,
            groundtruth_data_path=groundtruth_data_path,
            output_base_path=variable_length_train_path,
            num_good_versions=100,
            num_damaged_versions=0,
            # Controls the length of the GESTURE segment itself
            min_gesture_len=10,
            max_gesture_len=30,
            # Controls the length of the FINAL output .txt file
            min_clip_len=30,
            max_clip_len=30
        )

    elif select == '12':
        print("\n--- Pre-training Backbone on Variable-Length Data (Search) ---")

        if not os.path.exists(variable_length_train_path) or not os.listdir(variable_length_train_path):
            print(f"Error: Variable-length training data folder is empty. Run Option 11 first.")
            return

        unlabeled_data_raw, _, _, _ = load_three_level_data(variable_length_train_path)
        if not unlabeled_data_raw:
            print("Error: Failed to load data from variable-length training folder.")
            return

        train_data_raw, val_data_raw = train_test_split(unlabeled_data_raw, test_size=0.15, random_state=42)
        print(f"Pre-train data split into {len(train_data_raw)} training and {len(val_data_raw)} validation samples.")

        global_mean, global_std = get_standardization_params(train_data_raw)
        np.savez(variable_scaler_path, mean=global_mean, std=global_std)
        print(f"Variable-length scaler saved to {variable_scaler_path}")

        train_data_std = standardize_data(train_data_raw, global_mean, global_std)
        val_data_std = standardize_data(val_data_raw, global_mean, global_std)

        pretrain_dataset = UnlabeledGestureDataset(train_data_std)
        pretrain_val_dataset = UnlabeledGestureDataset(val_data_std)

        d_model_options = [768, 1024]
        n_head_options = [8, 16]
        num_layers_options = [4, 6]
        dropout_options = [0.1, 0.2]
        lr_options = [0.00005, 0.00001]
        masking_ratio_options = [0.25]
        PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE = 30, 16

        def collate_fn_pretrain_variable(batch, masking_ratio):
            data_list = [torch.tensor(item, dtype=torch.float32) for item in batch]
            padded_data = nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0.0)
            lengths = [len(seq) for seq in data_list]
            batch_size, max_len, _ = padded_data.shape
            padding_mask = torch.arange(max_len).expand(batch_size, max_len) >= torch.tensor(lengths).unsqueeze(1)
            inputs, labels = padded_data.clone(), padded_data.clone()
            loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for i, length in enumerate(lengths):
                num_to_mask = int(length * masking_ratio)
                if num_to_mask > 0:
                    indices_to_mask = random.sample(range(length), num_to_mask)
                    inputs[i, indices_to_mask, :] = 0.0
                    loss_mask[i, indices_to_mask, :] = True
            return inputs, padding_mask, labels, loss_mask

        best_loss = float('inf')
        best_config = {}

        for d_model in d_model_options:
            for n_head in n_head_options:
                if d_model % n_head != 0: continue
                for num_layers in num_layers_options:
                    for dropout in dropout_options:
                        for lr in lr_options:
                            for masking_ratio in masking_ratio_options:
                                current_config = {
                                    "D_MODEL": d_model, "N_HEAD": n_head, "NUM_ENC_LAYERS": num_layers,
                                    "DROPOUT": dropout, "PRETRAIN_LR": lr, "MASKING_RATIO": masking_ratio,
                                    "MASKING_STRATEGY": "Random Frame"
                                }
                                print("-" * 50);
                                print(f"Testing VARIABLE-LENGTH configuration: {current_config}")

                                train_loader = DataLoader(pretrain_dataset, batch_size=PRETRAIN_BATCH_SIZE,
                                                          shuffle=True,
                                                          collate_fn=lambda b: collate_fn_pretrain_variable(b,
                                                                                                            masking_ratio))
                                val_loader = DataLoader(pretrain_val_dataset, batch_size=PRETRAIN_BATCH_SIZE,
                                                        shuffle=False,
                                                        collate_fn=lambda b: collate_fn_pretrain_variable(b,
                                                                                                          masking_ratio))

                                backbone = GestureBackbone(d_model, n_head, num_layers, d_model * 4, dropout).to(device)
                                prediction_head = nn.Linear(d_model, NUM_SENSORS).to(device)
                                optimizer = optim.Adam(list(backbone.parameters()) + list(prediction_head.parameters()),
                                                       lr=lr)
                                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=False)
                                criterion = nn.MSELoss()

                                final_val_loss = float('inf')
                                for epoch in range(PRETRAIN_EPOCHS):
                                    backbone.train();
                                    prediction_head.train()
                                    total_train_loss = 0
                                    for inputs, padding_mask, labels, loss_mask in train_loader:
                                        inputs, padding_mask, labels, loss_mask = inputs.to(device), padding_mask.to(
                                            device), labels.to(device), loss_mask.to(device)
                                        optimizer.zero_grad()
                                        predictions = prediction_head(backbone(inputs, padding_mask))
                                        final_loss_mask = loss_mask & ~padding_mask.unsqueeze(-1)
                                        loss = criterion(predictions[final_loss_mask], labels[final_loss_mask])
                                        loss.backward()
                                        optimizer.step()
                                        total_train_loss += loss.item()

                                    backbone.eval();
                                    prediction_head.eval()
                                    total_val_loss = 0
                                    with torch.no_grad():
                                        for inputs, padding_mask, labels, loss_mask in val_loader:
                                            inputs, padding_mask, labels, loss_mask = inputs.to(
                                                device), padding_mask.to(device), labels.to(device), loss_mask.to(
                                                device)
                                            predictions = prediction_head(backbone(inputs, padding_mask))
                                            final_loss_mask = loss_mask & ~padding_mask.unsqueeze(-1)
                                            loss = criterion(predictions[final_loss_mask], labels[final_loss_mask])
                                            total_val_loss += loss.item()

                                    avg_val_loss = total_val_loss / len(val_loader)
                                    final_val_loss = avg_val_loss
                                    if (epoch + 1) % 5 == 0:
                                        print(
                                            f"  Epoch {epoch + 1:02d} | Train Loss: {total_train_loss / len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")
                                    scheduler.step(avg_val_loss)

                                print(f"Final Validation Loss for this configuration: {final_val_loss:.6f}")
                                if final_val_loss < best_loss:
                                    best_loss = final_val_loss
                                    best_config = current_config.copy();
                                    best_config['BEST_VAL_LOSS'] = best_loss
                                    print(f"*** New best VARIABLE-LENGTH model found! Val Loss: {best_loss:.6f} ***")
                                    torch.save(backbone.state_dict(), variable_backbone_path)
                                    with open(variable_config_path, 'w') as f:
                                        for key, value in best_config.items(): f.write(f"{key}: {value}\n")
        print("\n" + "=" * 50);
        print("Variable-Length Pre-training Search Finished!");
        print(f"Best config:\n{best_config}")

    elif select == '12.5':
        print("\n--- Final, Long Pre-training Run for Variable-Length Backbone ---")

        for p in [variable_config_path, variable_scaler_path]:
            if not os.path.exists(p):
                print(f"Error: Required file not found at {p}. Run hyperparameter search (Option 12) first.")
                return

        config = parse_config_file(variable_config_path)
        if not config:
            print("Error: Could not load best configuration file.")
            return

        print(f"Loaded BEST configuration from search: {config}")

        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT, LR, MASKING_RATIO = int(config['D_MODEL']), int(
            config['N_HEAD']), int(config['NUM_ENC_LAYERS']), float(config['DROPOUT']), float(
            config['PRETRAIN_LR']), float(config['MASKING_RATIO'])

        FINAL_PRETRAIN_EPOCHS = 80
        PRETRAIN_BATCH_SIZE = 32

        unlabeled_data_raw, _, _, _ = load_three_level_data(variable_length_train_path)
        train_data_raw, val_data_raw = train_test_split(unlabeled_data_raw, test_size=0.15, random_state=42)

        scaler_data = np.load(variable_scaler_path)
        global_mean, global_std = scaler_data['mean'], scaler_data['std']

        train_data_std = standardize_data(train_data_raw, global_mean, global_std)
        val_data_std = standardize_data(val_data_raw, global_mean, global_std)

        pretrain_dataset = UnlabeledGestureDataset(train_data_std)
        pretrain_val_dataset = UnlabeledGestureDataset(val_data_std)

        def collate_fn_pretrain_variable(batch, masking_ratio):
            data_list = [torch.tensor(item, dtype=torch.float32) for item in batch]
            padded_data = nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0.0)
            lengths = [len(seq) for seq in data_list]
            batch_size, max_len, _ = padded_data.shape
            padding_mask = torch.arange(max_len).expand(batch_size, max_len) >= torch.tensor(lengths).unsqueeze(1)
            inputs, labels = padded_data.clone(), padded_data.clone()
            loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for i, length in enumerate(lengths):
                num_to_mask = int(length * masking_ratio)
                if num_to_mask > 0:
                    indices_to_mask = random.sample(range(length), num_to_mask)
                    inputs[i, indices_to_mask, :] = 0.0
                    loss_mask[i, indices_to_mask, :] = True
            return inputs, padding_mask, labels, loss_mask

        train_loader = DataLoader(pretrain_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda b: collate_fn_pretrain_variable(b, MASKING_RATIO))
        val_loader = DataLoader(pretrain_val_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=False,
                                collate_fn=lambda b: collate_fn_pretrain_variable(b, MASKING_RATIO))

        print("\nStarting final training run with the best hyperparameters...")

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT).to(device)
        prediction_head = nn.Linear(D_MODEL, NUM_SENSORS).to(device)
        optimizer = optim.Adam(list(backbone.parameters()) + list(prediction_head.parameters()), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
        criterion = nn.MSELoss()

        best_loss = float('inf')

        for epoch in range(FINAL_PRETRAIN_EPOCHS):
            backbone.train();
            prediction_head.train()
            total_train_loss = 0
            for inputs, padding_mask, labels, loss_mask in train_loader:
                inputs, padding_mask, labels, loss_mask = inputs.to(device), padding_mask.to(device), labels.to(
                    device), loss_mask.to(device)
                optimizer.zero_grad()
                predictions = prediction_head(backbone(inputs, padding_mask))
                final_loss_mask = loss_mask & ~padding_mask.unsqueeze(-1)
                loss = criterion(predictions[final_loss_mask], labels[final_loss_mask])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            backbone.eval();
            prediction_head.eval()
            total_val_loss = 0
            with torch.no_grad():
                for inputs, padding_mask, labels, loss_mask in val_loader:
                    inputs, padding_mask, labels, loss_mask = inputs.to(device), padding_mask.to(device), labels.to(
                        device), loss_mask.to(device)
                    predictions = prediction_head(backbone(inputs, padding_mask))
                    final_loss_mask = loss_mask & ~padding_mask.unsqueeze(-1)
                    loss = criterion(predictions[final_loss_mask], labels[final_loss_mask])
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(
                f"  Epoch {epoch + 1:02d}/{FINAL_PRETRAIN_EPOCHS} | Train Loss: {total_train_loss / len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                print(f"--- New best validation loss: {best_loss:.6f}. Saving best model. ---")
                torch.save(backbone.state_dict(), variable_backbone_path)

        print("\n" + "=" * 50)
        print("Final Variable-Length Pre-training Finished!")
        print(f"The best backbone model (Val Loss: {best_loss:.6f}) was saved to {variable_backbone_path}")
        print("=" * 50)

    elif select == '13':
        print("\n--- Fine-tuning Classifier on Variable-Length Data ---")

        for p in [variable_backbone_path, variable_config_path, variable_scaler_path]:
            if not os.path.exists(p):
                print(f"Error: Required file not found at {p}. Run pre-training (Option 12.5) first.")
                return

        config = parse_config_file(variable_config_path)
        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
            config['NUM_ENC_LAYERS']), config['DROPOUT']
        print(f"Loaded VARIABLE-LENGTH backbone configuration: {config}")

        scaler_data = np.load(variable_scaler_path)
        global_mean, global_std = scaler_data['mean'], scaler_data['std']
        print(f"Loaded VARIABLE-LENGTH scaler: Mean={global_mean:.4f}, Std={global_std:.4f}")

        data, f_labels, g_labels, q_labels = load_three_level_data(variable_length_train_path)
        if not data:
            print(f"Error: No data found in {variable_length_train_path}. Run option 11 first.")
            return
        data_std = standardize_data(data, global_mean, global_std)

        indices = list(range(len(data_std)))
        train_indices, val_indices = train_test_split(indices, test_size=0.15, random_state=42, stratify=g_labels)
        train_data = [data_std[i] for i in train_indices];
        val_data = [data_std[i] for i in val_indices]
        train_f_labels = [f_labels[i] for i in train_indices];
        val_f_labels = [f_labels[i] for i in val_indices]
        train_g_labels = [g_labels[i] for i in train_indices];
        val_g_labels = [g_labels[i] for i in val_indices]
        train_q_labels = [q_labels[i] for i in train_indices];
        val_q_labels = [q_labels[i] for i in val_indices]
        print(f"Data split into {len(train_data)} training samples and {len(val_data)} validation samples.")

        train_dataset = ThreeLevelDataset(train_data, train_f_labels, train_g_labels, train_q_labels)
        val_dataset = ThreeLevelDataset(val_data, val_f_labels, val_g_labels, val_q_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_variable_length)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_variable_length)

        num_finger_classes, num_gesture_classes, num_quality_classes = len(set(f_labels)), len(set(g_labels)), len(
            set(q_labels))
        print(
            f"--> Total Detected Classes: {num_finger_classes} finger, {num_gesture_classes} gesture, {num_quality_classes} quality")

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
        print(f"Loading pre-trained VARIABLE-LENGTH weights from {variable_backbone_path}")
        backbone.load_state_dict(torch.load(variable_backbone_path, map_location=device, weights_only=True))
        model = ThreeLevelHierarchicalModel(backbone, D_MODEL, num_finger_classes, num_gesture_classes,
                                            num_quality_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

        EPOCHS = 50
        best_val_gesture_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            total_loss, c_f, c_g, c_q, total_samples = 0, 0, 0, 0, 0
            for data_batch, f_lbl, g_lbl, q_lbl, padding_mask in train_loader:
                data_batch, f_lbl, g_lbl, q_lbl, padding_mask = data_batch.to(device), f_lbl.to(device), g_lbl.to(
                    device), q_lbl.to(device), padding_mask.to(device)
                optimizer.zero_grad()
                f_logits, g_logits, q_logits = model(data_batch, padding_mask)
                loss_f, loss_g, loss_q = criterion(f_logits, f_lbl), criterion(g_logits, g_lbl), criterion(q_logits,
                                                                                                           q_lbl)
                combined_loss = loss_f + loss_g + loss_q
                combined_loss.backward()
                optimizer.step()
                total_loss += combined_loss.item()
                c_f += (torch.argmax(f_logits, 1) == f_lbl).sum().item()
                c_g += (torch.argmax(g_logits, 1) == g_lbl).sum().item()
                c_q += (torch.argmax(q_logits, 1) == q_lbl).sum().item()
                total_samples += f_lbl.size(0)

            train_acc_f = 100 * c_f / total_samples;
            train_acc_g = 100 * c_g / total_samples;
            train_acc_q = 100 * c_q / total_samples
            print(
                f"Epoch {epoch + 1:02d} TRAIN | Loss: {total_loss / len(train_loader):.4f} | Acc F:{train_acc_f:.2f}% G:{train_acc_g:.2f}% Q:{train_acc_q:.2f}%")

            model.eval()
            val_loss, vc_f, vc_g, vc_q, val_total_samples = 0, 0, 0, 0, 0
            with torch.no_grad():
                for data_batch, f_lbl, g_lbl, q_lbl, padding_mask in val_loader:
                    data_batch, f_lbl, g_lbl, q_lbl, padding_mask = data_batch.to(device), f_lbl.to(device), g_lbl.to(
                        device), q_lbl.to(device), padding_mask.to(device)
                    f_logits, g_logits, q_logits = model(data_batch, padding_mask)
                    loss_f, loss_g, loss_q = criterion(f_logits, f_lbl), criterion(g_logits, g_lbl), criterion(q_logits,
                                                                                                               q_lbl)
                    combined_loss = loss_f + loss_g + loss_q
                    val_loss += combined_loss.item()
                    vc_f += (torch.argmax(f_logits, 1) == f_lbl).sum().item()
                    vc_g += (torch.argmax(g_logits, 1) == g_lbl).sum().item()
                    vc_q += (torch.argmax(q_logits, 1) == q_lbl).sum().item()
                    val_total_samples += f_lbl.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc_f = 100 * vc_f / val_total_samples;
            val_acc_g = 100 * vc_g / val_total_samples;
            val_acc_q = 100 * vc_q / val_total_samples
            print(
                f"Epoch {epoch + 1:02d} VALID | Loss: {avg_val_loss:.4f} | Acc F:{val_acc_f:.2f}% G:{val_acc_g:.2f}% Q:{val_acc_q:.2f}%")
            scheduler.step(avg_val_loss)

            if val_acc_g > best_val_gesture_acc:
                print(f"*** New best validation accuracy: {val_acc_g:.2f}%. Saving model. ***")
                best_val_gesture_acc = val_acc_g
                torch.save(model.state_dict(), variable_model_path)
                with open(variable_3level_config_path, 'w') as f:
                    f.write(f"NUM_FINGER_CLASSES: {num_finger_classes}\n")
                    f.write(f"NUM_GESTURE_CLASSES: {num_gesture_classes}\n")
                    f.write(f"NUM_QUALITY_CLASSES: {num_quality_classes}\n")

        print("\n" + "=" * 50)
        print("--- Final Model Evaluation on Validation Set ---")
        if not os.path.exists(variable_model_path):
            print("No model was saved. Cannot perform final evaluation.")
            return

        print(f"Loading best model (Val Acc: {best_val_gesture_acc:.2f}%) for final report...")
        model.load_state_dict(torch.load(variable_model_path, map_location=device))
        model.eval()

        all_g_labels, all_g_preds, all_q_labels, all_q_preds = [], [], [], []
        with torch.no_grad():
            for data_batch, _, g_lbl, q_lbl, padding_mask in val_loader:
                data_batch, padding_mask = data_batch.to(device), padding_mask.to(device)
                _, g_logits, q_logits = model(data_batch, padding_mask)
                g_pred, q_pred = torch.argmax(g_logits, 1), torch.argmax(q_logits, 1)
                all_g_labels.extend(g_lbl.cpu().numpy());
                all_g_preds.extend(g_pred.cpu().numpy())
                all_q_labels.extend(q_lbl.cpu().numpy());
                all_q_preds.extend(q_pred.cpu().numpy())

        print("\n--- Gesture Classification Report ---")
        gesture_names = [f"Gesture_{i}" for i in range(num_gesture_classes)]
        print(classification_report(all_g_labels, all_g_preds, target_names=gesture_names, digits=3))

        print("\n--- Quality Classification Report ---")
        quality_names = ["Good (0)", "Damaged (1)"]
        print(classification_report(all_q_labels, all_q_preds, target_names=quality_names, digits=3))

        print("\n--- Gesture Confusion Matrix ---")
        cm_gesture = confusion_matrix(all_g_labels, all_g_preds)
        cm_df = pd.DataFrame(cm_gesture, index=gesture_names, columns=gesture_names)
        print(cm_df)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Gesture Confusion Matrix (Validation Set)')
        plt.ylabel('Actual Label');
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_save_path = os.path.join(models_path_variable, 'gesture_confusion_matrix.png')
        plt.savefig(cm_save_path)
        print(f"\nGesture confusion matrix plot saved to {cm_save_path}")

    elif select == '14':
        print("\n--- Prediction with 3-Level Variable-Length Classifier ---")

        for p in [variable_model_path, variable_config_path, variable_scaler_path, variable_3level_config_path]:
            if not os.path.exists(p):
                print(f"Error: Required file not found at {p}.")
                return

        config = parse_config_file(variable_config_path)
        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
            config['NUM_ENC_LAYERS']), config['DROPOUT']

        config_3level = parse_config_file(variable_3level_config_path)
        num_f, num_g, num_q = int(config_3level['NUM_FINGER_CLASSES']), int(config_3level['NUM_GESTURE_CLASSES']), int(
            config_3level['NUM_QUALITY_CLASSES'])

        scaler_data = np.load(variable_scaler_path)
        global_mean, global_std = scaler_data['mean'], scaler_data['std']

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
        model = ThreeLevelHierarchicalModel(backbone, D_MODEL, num_f, num_g, num_q).to(device)
        model.load_state_dict(torch.load(variable_model_path, map_location=device, weights_only=True))
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
                padding_mask = torch.zeros(1, data_tensor.shape[1], dtype=torch.bool).to(device)

                f_logits, g_logits, q_logits = model(data_tensor, padding_mask)
                pred_f, pred_g, pred_q = torch.argmax(f_logits, dim=1).item(), torch.argmax(g_logits,
                                                                                            dim=1).item(), torch.argmax(
                    q_logits, dim=1).item()

                finger_str = f"{pred_f + 1} Finger(s)"
                gesture_map = {0: "Left", 1: "Right", 2: "Down", 3: "Up", 4: "Do Nothing"}
                gesture_str = gesture_map.get(pred_g, "Unknown")
                quality_str = "Good" if pred_q == 0 else "Damaged"
                print(f"File: {filename} -> Predicted: {finger_str}, {gesture_str}, Quality: {quality_str}")

    else:
        print("Invalid option selected.")


if __name__ == '__main__':
    main()