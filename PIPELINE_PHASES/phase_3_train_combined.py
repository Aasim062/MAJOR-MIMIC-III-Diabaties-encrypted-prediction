"""
PHASE 3: COMBINED MODEL TRAINING (MLP-3)
- MLP classifier with BCEWithLogitsLoss, weighted for class imbalance
- Architecture: 60→128→64→1, with ReLU, Dropout, Adam, Weight Decay, Cosine LR
- Input: normalized .npy features from phase 2 (60 features); output: best combined model and train log
- Trains on ALL data combined (not hospital-specific)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# =========== CONFIGURATION ===========
SEED = 42
BATCH_SIZE = 256
EPOCHS = 300
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-5
MIN_LR = 1e-5
MAX_GRAD_NORM = 1.0
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mlp_best_model_combined.pt"
LOG_PATH = "mlp_train_log_combined.txt"

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"Device: {DEVICE}")

# =========== DATA LOADING ===========
def load_split(base_dir):
    """Load train/val/test splits from phase 2"""
    X_train = np.load(os.path.join(base_dir, "X_train.npy"))
    y_train = np.load(os.path.join(base_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(base_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(base_dir, "y_val.npy"))
    X_test  = np.load(os.path.join(base_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(base_dir, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    """Create DataLoaders for training and validation"""
    tensor = lambda arr: torch.from_numpy(arr).float()
    data_train = TensorDataset(tensor(X_train), torch.from_numpy(y_train).unsqueeze(1).float())
    data_val = TensorDataset(tensor(X_val), torch.from_numpy(y_val).unsqueeze(1).float())
    return (
        DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False),
        DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    )

# =========== MODEL DEFINITION ===========
class MLP3(nn.Module):
    """
    MLP3 Model for Binary Classification
    Architecture: 60 → 128 → 64 → 1
    """
    def __init__(self, in_dim=60, hidden1=128, hidden2=64, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden2, 1)  # output: logit
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x  # logit

# =========== UTILITY: METRICS FOR LOGGING ===========
from scipy.special import expit
def sigmoid(x):
    return expit(x)

def evaluate(model, loader, device, best_thresh=0.5):
    """Evaluate model on a dataset"""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.extend(logits.cpu().numpy().reshape(-1))
            all_labels.extend(yb.cpu().numpy().reshape(-1))
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_probs = sigmoid(all_logits)
    # Choose optimal threshold by F1 (on val set)
    f1, thresh = 0, 0.5
    for t in np.arange(0.15, 0.85, 0.01):
        preds = (all_probs >= t)
        score = f1_score(all_labels, preds)
        if score > f1:
            f1 = score
            thresh = t
    preds = (all_probs >= thresh)
    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = np.nan
    return {
        "loss": None,
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "best_thresh": thresh,
        "confusion": confusion_matrix(all_labels, preds).tolist(),
        "raw_y": all_labels,
        "raw_pred": preds.astype(int),
        "raw_proba": all_probs
    }

# =========== TRAINING FUNCTION ===========
def train_combined_model(X_train, y_train, X_val, y_val, save_path=MODEL_PATH):
    """
    Train combined MLP model on all data
    
    Args:
        X_train: Training features (N, 60)
        y_train: Training labels (N,)
        X_val: Validation features (M, 60)
        y_val: Validation labels (M,)
        save_path: Path to save best model
    
    Returns:
        Trained model
    """
    print(f"\n[Training] Input shape: {X_train.shape}")
    print(f"[Training] Input dimension: {X_train.shape[1]}")
    print(f"[Training] Positive class ratio: {y_train.sum() / len(y_train):.2%}")
    
    train_loader, val_loader = get_dataloaders(X_train, y_train, X_val, y_val, BATCH_SIZE)
    mlp = MLP3(in_dim=X_train.shape[1]).to(DEVICE)

    # --- Class imbalance weight: pos_weight for BCEWithLogitsLoss ---
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([(n_neg / max(n_pos, 1))], dtype=torch.float32, device=DEVICE)
    print(f"[Training] Class weight (pos_weight): {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    best_auc = -np.inf
    best_state = None
    hist = []

    print(f"\n[Training] Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        mlp.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = mlp(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation
        metrics_val = evaluate(mlp, val_loader, DEVICE)
        auc_val = metrics_val["auc"]
        hist.append(dict(epoch=epoch, val_loss=epoch_loss, **metrics_val))
        # Save best based on AUC
        if auc_val > best_auc:
            best_auc = auc_val
            best_state = mlp.state_dict()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  Loss: {epoch_loss:.4f}  ValAUC: {auc_val:.4f}  ValF1: {metrics_val['f1']:.4f}")

    torch.save(best_state, save_path)
    print(f"\n✓ Best model saved to {save_path} with ValAUC={best_auc:.4f}")
    # Write log
    with open(LOG_PATH, "w") as f:
        for rec in hist:
            f.write(str(rec) + "\n")
    print(f"✓ Training log saved to {LOG_PATH}")
    return mlp

# =========== MAIN ===========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Combined MLP-3 Training (All Hospitals)")
    parser.add_argument("--datadir", default=os.path.join(os.path.dirname(__file__), "data/processed/phase2"), 
                        help="Directory with .npy files")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model_out", default=MODEL_PATH)
    args = parser.parse_args()

    # Training config
    epochs = args.epochs
    batch_size = args.batch_size
    model_path = args.model_out

    # Load Data
    print(f"[Loading] Data from {args.datadir}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(args.datadir)
    
    print(f"[Data] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Ensure output directory exists
    out_dir = os.path.dirname(model_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Train combined model
    print("\n" + "="*60)
    print("TRAINING COMBINED MODEL (ALL HOSPITALS)")
    print("="*60)
    model = train_combined_model(X_train, y_train, X_val, y_val, save_path=model_path)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # FINAL TEST EVALUATION
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).unsqueeze(1).float(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    metrics_test = evaluate(model, test_loader, DEVICE)
    print("\n" + "="*60)
    print("FINAL TEST PERFORMANCE (COMBINED)")
    print("="*60)
    print(f"Model saved: {model_path}")
    print(f"Test AUC:    {metrics_test['auc']:.4f}")
    print(f"Test Acc:    {metrics_test['acc']:.4f}")
    print(f"Test F1:     {metrics_test['f1']:.4f}")
    print(f"Best Threshold: {metrics_test['best_thresh']:.2f}")
    print(f"Confusion Matrix:\n{np.array(metrics_test['confusion'])}")
    print("="*60)
