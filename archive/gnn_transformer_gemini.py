import os
import numpy as np
import random
import pandas as pd
import math
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# --- GNN-SPECIFIC IMPORTS ---
# Make sure you have installed PyTorch Geometric correctly!
import torch_geometric.nn as gnn


# --- UTILITY AND DATA LOADING (Mostly unchanged) ---

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
        if data.empty: return np.empty((0, 56))
        return data.values
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return np.empty((0, 56))


class GestureDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data_list = data_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx], dtype=torch.float32), torch.tensor(self.labels_list[idx],
                                                                                    dtype=torch.long)


def load_all_gestures(base_path):
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


# --- GRAPH-SPECIFIC UTILITY ---
def create_grid_graph_edge_index(height, width):
    """ Creates the edge_index for a grid graph (4-connectivity). """
    edges = []
    for r in range(height):
        for c in range(width):
            node_id = r * width + c
            # Connect to right neighbor
            if c < width - 1:
                edges.append([node_id, node_id + 1])
                edges.append([node_id + 1, node_id])  # Undirected edge
            # Connect to bottom neighbor
            if r < height - 1:
                edges.append([node_id, node_id + width])
                edges.append([node_id + width, node_id])  # Undirected edge
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# --- MODEL ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    # (Identical to previous versions, a standard component for Transformers)
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


class GestureGNNTransformer(nn.Module):
    def __init__(self, edge_index, num_nodes, gnn_hidden_dim, d_model, nhead, num_encoder_layers, dim_feedforward,
                 output_dim, dropout=0.5):
        super().__init__()
        self.num_nodes = num_nodes

        # --- Part 1: GNN for Spatial Feature Extraction ---
        # The edge_index is constant for all data, so we register it as a buffer.
        self.register_buffer('edge_index', edge_index)

        # We use Graph Attention Convolution (GATConv) which is often more powerful than GCNConv.
        # It takes the raw sensor value (1 feature) and learns a richer representation.
        self.gnn1 = gnn.GATConv(in_channels=1, out_channels=gnn_hidden_dim, heads=4, dropout=dropout)
        self.gnn2 = gnn.GATConv(in_channels=gnn_hidden_dim * 4, out_channels=d_model, dropout=dropout)

        # --- Part 2: Transformer for Temporal Analysis ---
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # --- Part 3: Classifier Head ---
        self.classifier = nn.Linear(d_model, output_dim)

    def forward(self, x, src_padding_mask):
        # x shape: [batch_size, seq_len, num_nodes]
        B, S, N = x.shape

        # --- GNN Spatial Processing (applied to each time step) ---
        # Reshape to treat all time steps across the batch as one big batch for the GNN.
        # x becomes [batch_size * seq_len, num_nodes]
        x = x.reshape(-1, N)
        # Add a feature dimension for the GNN: [batch_size * seq_len, num_nodes, 1]
        x = x.unsqueeze(-1)

        # PyG's GNN layers expect input of shape [num_total_nodes, num_features].
        # We reshape and then apply the GNN layers.
        # x becomes [ (batch_size * seq_len) * num_nodes, 1]
        x_gnn = self.gnn1(x.reshape(-1, 1), self.edge_index)
        x_gnn = self.gnn2(x_gnn, self.edge_index)

        # The output x_gnn has shape [(batch_size * seq_len) * num_nodes, d_model].
        # We need to aggregate node features to get a graph-level representation for each time step.
        # We simply take the mean of all node features.
        # Reshape to [batch_size * seq_len, num_nodes, d_model]
        x_graph = x_gnn.reshape(-1, N, self.gnn2.out_channels)
        # Mean aggregation: [batch_size * seq_len, d_model]
        x_graph = x_graph.mean(dim=1)

        # --- Transformer Temporal Processing ---
        # Reshape back into a sequence for the transformer: [batch_size, seq_len, d_model]
        trans_input = x_graph.reshape(B, S, -1)

        trans_input = self.pos_encoder(trans_input.permute(1, 0, 2)).permute(1, 0, 2)
        output = self.transformer_encoder(trans_input, src_key_padding_mask=src_padding_mask)

        # Use the output of the first token for classification
        output = output[:, 0, :]

        # --- Final Classification ---
        logits = self.classifier(output)
        return logits


def main():
    set_seed(53)
    # --- PATHS ---
    base_path = '//ai'
    train_path = os.path.join(base_path, 'training_data')
    predict_path = '//predict/'
    model_save_path = '//models_gnn/gnn_model.pth'

    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    # --- USER SELECTION ---
    select = input(
        "Select an option:\n1: Train GNN-Transformer model\n2: Predict with trained model\nEnter option number: ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # --- GRAPH and MODEL HYPERPARAMETERS ---
    HEIGHT, WIDTH = 8, 7
    NUM_NODES = HEIGHT * WIDTH
    GNN_HIDDEN_DIM = 32
    D_MODEL = 128
    N_HEAD = 8
    NUM_ENC_LAYERS = 4
    DROPOUT = 0.2

    # Create the graph structure once. It's the same for all data.
    edge_index = create_grid_graph_edge_index(HEIGHT, WIDTH).to(device)

    # --- Training ---
    if select == '1':
        print("\n--- Starting Model Training ---")
        EPOCHS, LR, BATCH_SIZE = 100, 0.0001, 32  # Increased epochs for more training time

        all_data, all_labels = load_all_gestures(train_path)
        output_dim = len(set(all_labels))
        full_dataset = GestureDataset(all_data, all_labels)

        # --- NEW: Create a Train/Validation Split (80/20 split) ---
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        print(
            f"Total samples: {len(full_dataset)}. Training on {len(train_dataset)}, validating on {len(val_dataset)}.")

        def collate_fn(batch):
            sequences, labels = zip(*batch)
            sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
            padding_mask = torch.zeros(sequences_padded.shape[0], sequences_padded.shape[1], dtype=torch.bool)
            for i, seq in enumerate(sequences): padding_mask[i, len(seq):] = True
            return sequences_padded, torch.tensor(labels, dtype=torch.long), padding_mask

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn)  # No shuffle for validation

        model = GestureGNNTransformer(
            edge_index=edge_index, num_nodes=NUM_NODES, gnn_hidden_dim=GNN_HIDDEN_DIM,
            d_model=D_MODEL, nhead=N_HEAD, num_encoder_layers=NUM_ENC_LAYERS,
            dim_feedforward=D_MODEL * 4, output_dim=output_dim, dropout=DROPOUT
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # --- NEW: Training loop with validation ---
        best_val_acc = 0.0
        for epoch in range(EPOCHS):
            model.train()
            train_loss, train_correct, train_samples = 0, 0, 0
            for seqs, labels, mask in train_loader:
                seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)
                optimizer.zero_grad();
                outputs = model(seqs, mask)
                loss = criterion(outputs, labels)
                loss.backward();
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_samples += labels.size(0)

            # --- Validation Phase ---
            model.eval()
            val_loss, val_correct, val_samples = 0, 0, 0
            with torch.no_grad():
                for seqs, labels, mask in val_loader:
                    seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)
                    outputs = model(seqs, mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_samples += labels.size(0)

            train_acc = 100 * train_correct / train_samples
            val_acc = 100 * val_correct / val_samples
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

            # Save the model only if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  -> New best validation accuracy! Saving model to {model_save_path}")
                torch.save(model.state_dict(), model_save_path)

    # --- Prediction ---
    elif select == '2':
        print("\n--- Starting Prediction ---")
        if not os.path.exists(model_save_path):
            print("Error: Trained model not found. Please run option 1 first.");
            return

        _, train_labels = load_all_gestures(train_path)
        output_dim = len(set(train_labels))

        model = GestureGNNTransformer(
            edge_index=edge_index, num_nodes=NUM_NODES, gnn_hidden_dim=GNN_HIDDEN_DIM,
            d_model=D_MODEL, nhead=N_HEAD, num_encoder_layers=NUM_ENC_LAYERS,
            dim_feedforward=D_MODEL * 4, output_dim=output_dim, dropout=DROPOUT
        ).to(device)

        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
        model.eval()

        predict_data, filenames = load_predict_data(predict_path)
        if not predict_data: print("No valid prediction data found."); return

        # For prediction, we process one file at a time for simplicity
        print("\n--- Predictions ---")
        with torch.no_grad():
            for data, filename in zip(predict_data, filenames):
                seq = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
                # Create a padding mask (all False since it's a single, unpadded sequence)
                mask = torch.zeros(1, seq.shape[1], dtype=torch.bool).to(device)

                output = model(seq, mask)
                _, predicted = torch.max(output.data, 1)
                print(f'File {filename}: Predicted Gesture: {predicted.item()}')

    else:
        print("Invalid option selected.")


if __name__ == '__main__':
    main()