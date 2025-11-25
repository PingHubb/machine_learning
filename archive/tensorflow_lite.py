import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import onnx

################################################################
# 1) Dataset: Fixed-sequence approach (no variable-length logic)
################################################################

class GestureDatasetFixed(Dataset):
    """
    Loads each sample, which is a sequence of frames shaped [num_frames, 130]
    (or [num_frames, 13, 10]). We truncate/pad each to max_seq_len frames.
    Then each frame is reshaped to [13, 10] if needed.
    """
    def __init__(self, data_paths, labels, max_seq_len=40):
        """
        Args:
            data_paths: list of file paths, or you can store raw arrays directly
                        here. For demonstration, we assume they're paths
                        to .txt containing space-separated data.
            labels:     list of integer labels (same length as data_paths)
            max_seq_len: fixed number of frames
        """
        self.data_paths = data_paths
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.height = 13
        self.width = 10

        assert len(self.data_paths) == len(self.labels), "Mismatch data vs. labels"

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        label = self.labels[idx]

        # Load your data from file (assuming shape = [num_frames, 130])
        # Example: each line has 130 floats, space-separated
        data_np = np.loadtxt(file_path, delimiter=" ")  # shape: (num_frames, 130)

        # If data_np is already (num_frames, 13,10), skip the reshape:
        # If it's (num_frames, 130), reshape each frame to (13,10)
        if data_np.shape[1] == 130:
            num_frames = data_np.shape[0]
            data_np = data_np.reshape(num_frames, self.height, self.width)
        else:
            # if your file is already 13 x 10, do nothing
            pass

        # Now we have shape: [num_frames, 13, 10].
        num_frames = data_np.shape[0]

        # Truncate or pad to max_seq_len
        if num_frames > self.max_seq_len:
            data_np = data_np[:self.max_seq_len]
        elif num_frames < self.max_seq_len:
            pad_shape = (self.max_seq_len - num_frames, self.height, self.width)
            pad_zeros = np.zeros(pad_shape, dtype=np.float32)
            data_np = np.concatenate((data_np, pad_zeros), axis=0)

        # final shape: [max_seq_len, 13, 10]
        seq_tensor = torch.tensor(data_np, dtype=torch.float32)
        return seq_tensor, label

################################################################
# 2) Model: CNN + LSTM with fixed-sequence forward
################################################################

class GestureCNNLSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=1, dropout_rate=0.0):
        super().__init__()
        # We fix the input "image" size as 13x10
        self.height = 13
        self.width = 10

        # A simple 2-layer CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Determine how many features come out of the CNN
        conv_output_size = self._get_conv_output_size()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_output_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )

        # Final classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def _get_conv_output_size(self):
        # Pass a dummy input of shape (1, 1, 13, 10) through the CNN
        dummy_input = torch.zeros(1, 1, self.height, self.width)
        out = self.conv(dummy_input)
        return out.view(1, -1).size(1)  # flatten

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, 13, 10]
        Steps:
          1) reshape to (batch_size*seq_len, 1, 13, 10)
          2) CNN
          3) reshape to (batch_size, seq_len, conv_output_size)
          4) LSTM (take last hidden state)
          5) classifier
        """
        bsz, seq_len, H, W = x.size()
        # Insert channel=1
        x = x.view(bsz * seq_len, 1, H, W)
        x = self.conv(x)
        x = x.view(bsz, seq_len, -1)  # [B, seq_len, conv_output_size]

        # LSTM => hidden shape: (num_layers, B, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the last layer's hidden for classification
        logits = self.classifier(hidden[-1])  # shape: [B, output_dim]
        return logits


class GestureCNNLSTM_ManualUnroll(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
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
        self.conv_out = self._get_conv_output_size()  # e.g. 192
        self.hidden_dim = hidden_dim

        # Instead of nn.LSTM, define separate gates:
        self.W_ih = nn.Linear(self.conv_out, 4*hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, 4*hidden_dim, bias=False)
        self.b_ih = nn.Parameter(torch.zeros(4*hidden_dim))
        self.b_hh = nn.Parameter(torch.zeros(4*hidden_dim))

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def _get_conv_output_size(self):
        dummy_input = torch.zeros(1, 1, self.height, self.width)
        out = self.conv(dummy_input)
        return out.view(1, -1).size(1)

    def forward(self, x):
        # x: [batch_size, seq_len, 13, 10]
        B, T, _, _ = x.shape
        # 1) CNN over each frame
        x = x.view(B*T, 1, self.height, self.width)
        x = self.conv(x)  # shape [B*T, 32, 3, 2] => flatten => [B*T, 192]
        x = x.view(B, T, -1)

        # 2) Manually unroll the LSTM for T timesteps
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        for t in range(T):
            x_t = x[:, t, :]  # shape [B, conv_out=192]
            gates = self.W_ih(x_t) + self.b_ih + self.W_hh(h) + self.b_hh
            # gates is [B, 4*hidden_dim]
            i, f, g, o = torch.chunk(gates, 4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f*c + i*g
            h = o*torch.tanh(c)

        # 3) Final hidden state => classifier
        logits = self.classifier(h)  # [B, output_dim]
        return logits


################################################################
# 3) Training and Evaluation Loops
################################################################

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(seqs)  # shape [B, output_dim]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy

def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy

################################################################
# 4) Main: Train, Evaluate, Export to ONNX
################################################################

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fixed-sequence CNN+LSTM for TFLite")
    parser.add_argument("--train_dir", type=str, default="C:/dev/machine_learning/ai/training_data",
                        help="Path to training data (folder with .txt subfolders).")
    parser.add_argument("--test_dir", type=str, default="C:/dev/machine_learning/ai/testing_data",
                        help="Path to testing data (folder with .txt subfolders).")
    parser.add_argument("--max_seq_len", type=int, default=40, help="Fixed max sequence length.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden dim.")
    parser.add_argument("--num_layers", type=int, default=1, help="LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in LSTM if num_layers>1.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--save_onnx", type=str, default="gesture_model.onnx",
                        help="Where to save the ONNX file.")
    args = parser.parse_args()

    # -----------------------------
    # 4.1) Load Data
    # -----------------------------
    def load_all_gestures_as_files(base_path):
        """
        Expects subfolders named something like 'gesture_0', 'gesture_1', etc.
        Each subfolder has .txt files of shape [num_frames x 130].
        We'll collect the file paths and labels.
        """
        gesture_dirs = [d for d in os.listdir(base_path)
                        if os.path.isdir(os.path.join(base_path, d))]
        all_paths = []
        all_labels = []
        for d in gesture_dirs:
            # Suppose folder name ends with "_N" where N is the label
            label = int(d.split("_")[-1])
            gpath = os.path.join(base_path, d)
            txt_files = [f for f in os.listdir(gpath) if f.endswith('.txt')]
            for tf in txt_files:
                path = os.path.join(gpath, tf)
                all_paths.append(path)
                all_labels.append(label)
        return all_paths, all_labels

    train_paths, train_labels = load_all_gestures_as_files(args.train_dir)
    test_paths,  test_labels  = load_all_gestures_as_files(args.test_dir)

    print(f"Found {len(train_paths)} training samples, {len(test_paths)} testing samples.")

    # Create Datasets
    train_dataset = GestureDatasetFixed(train_paths, train_labels, max_seq_len=args.max_seq_len)
    test_dataset  = GestureDatasetFixed(test_paths,  test_labels,  max_seq_len=args.max_seq_len)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # -----------------------------
    # 4.2) Build Model
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    output_dim = len(set(train_labels))  # # of unique gestures

    # model = GestureCNNLSTM(
    #     hidden_dim=args.hidden_dim,
    #     output_dim=output_dim,
    #     num_layers=args.num_layers,
    #     dropout_rate=args.dropout
    # ).to(device)

    model = GestureCNNLSTM_ManualUnroll(
        hidden_dim=args.hidden_dim,
        output_dim=output_dim
    ).to(device)

    # -----------------------------
    # 4.3) Train
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc  = evaluate_one_epoch(model, test_loader, criterion, device)

        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    # Final classification report
    model.eval()
    all_preds, all_labels_gt = [], []
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            outputs = model(seqs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels_gt.extend(labels.tolist())

    print("Classification Report:")
    print(classification_report(all_labels_gt, all_preds))

    # -----------------------------
    # 4.4) Export to ONNX
    # -----------------------------
    # We'll do a dummy_input: shape [1, max_seq_len, 13, 10]
    dummy_input = torch.randn(1, args.max_seq_len, 13, 10, device=device)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        args.save_onnx,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None  # We keep shapes fixed
    )
    print(f"Exported model to ONNX => {args.save_onnx}")

    # Check the exported ONNX
    onnx_model = onnx.load(args.save_onnx)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    print("\nTo convert this ONNX model to TFLite, you can use onnx2tf:\n"
          "  pip install onnx2tf\n"
          f"  onnx2tf --model_path {args.save_onnx} --output_no_quant_float32_tflite\n"
          "This will produce a model_float32.tflite file.\n"
          "Then run that on an ESP32-S3 using TensorFlow Lite Micro.\n")

if __name__ == "__main__":
    main()
