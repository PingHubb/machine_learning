import os
import numpy as np
import random
import pandas as pd
import sys
import math
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader


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
        if data.empty:
            return np.empty((0, 56))
        return data.values
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return np.empty((0, 56))


class LabeledGestureDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data_list = data_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx], dtype=torch.float32), torch.tensor(self.labels_list[idx],
                                                                                    dtype=torch.long)


class UnlabeledGestureDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx], dtype=torch.float32)


class PredictionGestureDataset(Dataset):
    def __init__(self, data_list, filenames):
        self.data_list = data_list
        self.filenames = filenames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx], dtype=torch.float32), self.filenames[idx]


def load_labeled_data(base_path):
    gesture_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if
                       os.path.isdir(os.path.join(base_path, d))]
    all_data, all_labels = [], []
    for folder in gesture_folders:
        gesture_label = int(folder.split('_')[-1])
        for f in sorted(os.listdir(folder)):
            if f.endswith('.txt'):
                trial = load_data_from_file(os.path.join(folder, f))
                if trial.size > 0 and trial.shape[1] == 56:
                    all_data.append(trial)
                    all_labels.append(gesture_label)
    return all_data, all_labels


def load_unlabeled_data(*paths):
    all_data = []
    for path in paths:
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith('.txt'):
                    trial = load_data_from_file(os.path.join(root, f))
                    if trial.size > 0 and trial.shape[1] == 56:
                        all_data.append(trial)
    return all_data


def load_predict_data(predict_path):
    file_paths = [os.path.join(predict_path, f) for f in os.listdir(predict_path) if f.endswith('.txt')]
    files_sorted = sorted(file_paths, key=lambda p: int(os.path.basename(p).split('.')[0].split('_')[-1]))
    all_data, filenames = [], []
    for file_path in files_sorted:
        features = load_data_from_file(file_path)
        if features.size > 0:
            all_data.append(features)
            filenames.append(os.path.basename(file_path))
    return all_data, filenames


# --- NORMALIZATION (STANDARDIZATION) HELPER FUNCTIONS ---
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


# --- MODEL ARCHITECTURE ---
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


class GestureBackbone(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.height, self.width = 8, 7
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
        trans_input = self.pos_encoder(trans_input.permute(1, 0, 2)).permute(1, 0, 2)
        return self.transformer_encoder(trans_input, src_key_padding_mask=src_padding_mask)


class FineTunedGestureModel(nn.Module):
    def __init__(self, backbone, d_model, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, src_padding_mask):
        representations = self.backbone(src, src_padding_mask)
        cls_representation = representations[:, 0, :]
        logits = self.classifier(cls_representation)
        return logits


# --- MAIN SCRIPT ---
def main():
    set_seed(53)
    # --- PATHS ---
    base_path = '/ai'
    train_path = os.path.join(base_path, 'training_data')
    test_path = os.path.join(base_path, 'testing_data')
    predict_path = '/predict/'

    models_path = '//models_ssl_standardized/'  # Using a new folder for this experiment
    pretrained_backbone_path = os.path.join(models_path, 'pretrained_backbone.pth')
    finetuned_model_path = os.path.join(models_path, 'finetuned_model.pth')
    scaler_path = os.path.join(models_path, 'standardization_scaler.npz')
    pretrained_config_path = os.path.join(models_path, 'pretrained_backbone_config.txt')

    for path in [models_path]:
        if not os.path.exists(path): os.makedirs(path)

    print("--- Self-Supervised Learning Pipeline with Standardization ---")
    select = input("Select an option:\n"
                   "1: Pre-train the model backbone (Hyperparameter Search)\n"
                   "2: Fine-tune the pre-trained model\n"
                   "3: Predict with the fine-tuned model\n"
                   "Enter option number: ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    if select == '1':
        print("\n--- Starting Stage 1: Pre-training with Hyperparameter Search ---")
        d_model_options = [64, 128]
        n_head_options = [4, 8]
        num_layers_options = [2, 4]
        dropout_options = [0.1, 0.2]
        lr_options = [0.0001, 0.00005]  # Standardized data often benefits from a well-tuned LR

        PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, MASKING_RATIO = 50, 32, 0.25

        unlabeled_data_raw = load_unlabeled_data(train_path, test_path)
        if not unlabeled_data_raw: print("Error: No data found for pre-training."); return

        global_mean, global_std = get_standardization_params(unlabeled_data_raw)
        np.savez(scaler_path, mean=global_mean, std=global_std)
        print(f"Standardization parameters saved to {scaler_path}")
        unlabeled_data_standardized = standardize_data(unlabeled_data_raw, global_mean, global_std)
        print(f"Loaded and standardized {len(unlabeled_data_standardized)} sequences for pre-training.")
        pretrain_dataset = UnlabeledGestureDataset(unlabeled_data_standardized)

        def collate_fn_pretrain(batch):
            sequences = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
            batch_size, seq_len, _ = sequences.shape;
            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            for i, seq in enumerate(batch): padding_mask[i, len(seq):] = True
            inputs, labels = sequences.clone(), sequences.clone();
            loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for i in range(batch_size):
                seq_len_unpadded = seq_len - padding_mask[i].sum()
                num_to_mask = int(seq_len_unpadded * MASKING_RATIO)
                if num_to_mask > 0:
                    mask_start = random.randint(0, seq_len_unpadded - num_to_mask)
                    inputs[i, mask_start: mask_start + num_to_mask, :] = 0.0
                    loss_mask[i, mask_start: mask_start + num_to_mask, :] = True
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
                            prediction_head = nn.Linear(d_model, 56).to(device)
                            optimizer = optim.Adam(list(backbone.parameters()) + list(prediction_head.parameters()),
                                                   lr=lr)
                            criterion = nn.MSELoss()
                            final_epoch_loss = 0
                            for epoch in range(PRETRAIN_EPOCHS):
                                backbone.train();
                                prediction_head.train();
                                total_loss = 0
                                for inputs, padding_mask, labels, loss_mask in pretrain_loader:
                                    inputs, padding_mask, labels, loss_mask = inputs.to(device), padding_mask.to(
                                        device), labels.to(device), loss_mask.to(device)
                                    optimizer.zero_grad()
                                    predictions = prediction_head(backbone(inputs, padding_mask))
                                    loss = criterion(predictions[loss_mask], labels[loss_mask])
                                    loss.backward();
                                    optimizer.step()
                                    total_loss += loss.item()
                                final_epoch_loss = total_loss / len(pretrain_loader)
                                if (epoch + 1) % 10 == 0: print(
                                    f"  Epoch [{epoch + 1}/{PRETRAIN_EPOCHS}], Loss: {final_epoch_loss:.6f}")
                            print(f"Final loss for this configuration: {final_epoch_loss:.6f}")
                            if final_epoch_loss < best_loss:
                                best_loss, best_config = final_epoch_loss, current_config
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
        print("\n--- Starting Stage 2: Fine-tuning ---")
        # Check for required files
        for p in [pretrained_backbone_path, scaler_path, pretrained_config_path]:
            if not os.path.exists(p): print(
                f"Error: Required file not found at {p}. Please run option 1 first."); return

        FINETUNE_EPOCHS_HEAD, FINETUNE_EPOCHS_FULL, FINETUNE_LR_HEAD, FINETUNE_LR_FULL, FINETUNE_BATCH_SIZE = 10, 40, 0.001, 0.00005, 32

        config = {};
        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = 0, 0, 0, 0
        with open(pretrained_config_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ');
                config[key] = float(value) if '.' in value else int(value)
        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
            config['NUM_ENC_LAYERS']), config['DROPOUT']
        print(f"Loaded best backbone configuration: {config}")

        scaler_data = np.load(scaler_path);
        global_mean, global_std = scaler_data['mean'], scaler_data['std']
        print(f"Loaded standardization parameters: Mean={global_mean:.4f}, Std={global_std:.4f}")

        train_data_raw, train_labels = load_labeled_data(train_path)
        if not train_data_raw: print("Error: No labeled training data found."); return
        train_data_standardized = standardize_data(train_data_raw, global_mean, global_std)
        output_dim = len(set(train_labels))
        print(f"Loaded and standardized {len(train_data_standardized)} labeled sequences. Num classes: {output_dim}")

        def collate_fn_finetune(batch):
            sequences, labels = zip(*batch)
            sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
            padding_mask = torch.zeros(sequences_padded.shape[0], sequences_padded.shape[1], dtype=torch.bool)
            for i, seq in enumerate(sequences): padding_mask[i, len(seq):] = True
            return sequences_padded, torch.tensor(labels, dtype=torch.long), padding_mask

        dataset = LabeledGestureDataset(train_data_standardized, train_labels)
        train_loader = DataLoader(dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_finetune)

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
        print(f"Loading pre-trained weights from {pretrained_backbone_path}")
        backbone.load_state_dict(torch.load(pretrained_backbone_path, map_location=device, weights_only=True))

        model = FineTunedGestureModel(backbone, d_model=D_MODEL, num_classes=output_dim).to(device)
        criterion = nn.CrossEntropyLoss()

        print("\n--- Fine-tuning Phase 1: Training classifier head only ---")
        for param in model.backbone.parameters(): param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=FINETUNE_LR_HEAD)
        for epoch in range(FINETUNE_EPOCHS_HEAD):
            model.train();
            total_loss, total_correct, total_samples = 0, 0, 0
            for seqs, labels, mask in train_loader:
                seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)
                optimizer.zero_grad();
                outputs = model(seqs, mask);
                loss = criterion(outputs, labels)
                loss.backward();
                optimizer.step()
                total_loss += loss.item();
                _, predicted = torch.max(outputs, 1);
                total_correct += (predicted == labels).sum().item();
                total_samples += labels.size(0)
            print(
                f"Head-tuning Epoch [{epoch + 1}/{FINETUNE_EPOCHS_HEAD}], Loss: {total_loss / len(train_loader):.4f}, Acc: {100 * total_correct / total_samples:.2f}%")

        print("\n--- Fine-tuning Phase 2: Training full model (end-to-end) ---")
        for param in model.backbone.parameters(): param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR_FULL)
        for epoch in range(FINETUNE_EPOCHS_FULL):
            model.train();
            total_loss, total_correct, total_samples = 0, 0, 0
            for seqs, labels, mask in train_loader:
                seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)
                optimizer.zero_grad();
                outputs = model(seqs, mask);
                loss = criterion(outputs, labels)
                loss.backward();
                optimizer.step()
                total_loss += loss.item();
                _, predicted = torch.max(outputs, 1);
                total_correct += (predicted == labels).sum().item();
                total_samples += labels.size(0)
            print(
                f"Full-tuning Epoch [{epoch + 1}/{FINETUNE_EPOCHS_FULL}], Loss: {total_loss / len(train_loader):.4f}, Acc: {100 * total_correct / total_samples:.2f}%")

        print(f"Fine-tuning finished. Saving final model to {finetuned_model_path}")
        torch.save(model.state_dict(), finetuned_model_path)

    elif select == '3':
        print("\n--- Starting Stage 3: Prediction ---")
        for p in [finetuned_model_path, scaler_path, pretrained_config_path]:
            if not os.path.exists(p): print(
                f"Error: Required file not found at {p}. Please run previous stages first."); return

        config = {};
        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = 0, 0, 0, 0
        with open(pretrained_config_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ');
                config[key] = float(value) if '.' in value else int(value)
        D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
            config['NUM_ENC_LAYERS']), config['DROPOUT']
        print(f"Loaded best backbone configuration: {config}")

        scaler_data = np.load(scaler_path);
        global_mean, global_std = scaler_data['mean'], scaler_data['std']
        print(f"Loaded standardization parameters: Mean={global_mean:.4f}, Std={global_std:.4f}")

        _, train_labels = load_labeled_data(train_path);
        output_dim = len(set(train_labels))
        predict_data_raw, filenames = load_predict_data(predict_path)
        if not predict_data_raw: print("No valid prediction data found."); return

        predict_data_standardized = standardize_data(predict_data_raw, global_mean, global_std)
        predict_dataset = PredictionGestureDataset(predict_data_standardized, filenames)

        backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT)
        model = FineTunedGestureModel(backbone, d_model=D_MODEL, num_classes=output_dim).to(device)
        print(f"Loading fine-tuned model from {finetuned_model_path}")
        model.load_state_dict(torch.load(finetuned_model_path, map_location=device, weights_only=True))
        model.eval()

        def collate_fn_predict(batch):
            sequences, fns = zip(*batch)
            sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
            padding_mask = torch.zeros(sequences_padded.shape[0], sequences_padded.shape[1], dtype=torch.bool)
            for i, seq in enumerate(sequences): padding_mask[i, len(seq):] = True
            return sequences_padded, fns, padding_mask

        predict_loader = DataLoader(predict_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_predict)
        print("\n--- Predictions ---")
        with torch.no_grad():
            for data, fns, mask in predict_loader:
                data, mask = data.to(device), mask.to(device)
                outputs = model(data, mask)
                _, predicted = torch.max(outputs.data, 1)
                for filename, pred in zip(fns, predicted): print(f'File {filename}: Predicted Gesture: {pred.item()}')

    else:
        print("Invalid option selected.")


if __name__ == '__main__':
    main()