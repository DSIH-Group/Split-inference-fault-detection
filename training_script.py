import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix

# 1. DATASET: Multi-Modal Fusion (Sensors + Actuators)
class DroneMultiModalDataset(Dataset):
    def __init__(self, manifest_file):
        self.samples = []
        if not os.path.exists(manifest_file):
            print(f"Warning: {manifest_file} not found.")
            return

        print(f"Loading data from {manifest_file}...")
        with open(manifest_file, 'r') as f:
            for line in f:
                base_path = line.strip()
                if not base_path or 'TestInfo.csv' in base_path: continue
                
                sensor_file = self._find_file(base_path, 'sensor_combined_0.csv')
                actuator_file = self._find_file(base_path, 'actuator_outputs_0.csv')
                
                if sensor_file and actuator_file:
                    if 'HIL-NoFault' in base_path: label = 0
                    elif 'HIL_Motor' in base_path: label = 1
                    elif 'HIL_Prop' in base_path: label = 2
                    else: continue
                    self.samples.append((sensor_file, actuator_file, label))

    def _find_file(self, directory, suffix):
        log_dir = os.path.join(directory, 'Log')
        if not os.path.exists(log_dir): log_dir = directory 
        if os.path.isdir(log_dir):
            for file in os.listdir(log_dir):
                if file.endswith(suffix): return os.path.join(log_dir, file)
        return None

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s_path, a_path, label = self.samples[idx]
        s_data = pd.read_csv(s_path).iloc[:12, :10].values
        a_data = pd.read_csv(a_path).iloc[:12, :8].values
        combined = np.hstack([s_data, a_data])
        
        # Z-Score Normalization for numerical stability
        mean = combined.mean(axis=0)
        std = combined.std(axis=0) + 1e-6
        norm_data = (combined - mean) / std
        
        return torch.tensor(norm_data.T, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 2. ARCHITECTURE
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(18, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12, 16) 
        )
    def forward(self, x): return self.conv(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(16, 64 * 12), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 18, kernel_size=3, padding=1) 
        )
    def forward(self, x):
        return self.deconv(self.fc(x).view(-1, 64, 12))

class SentryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    def forward(self, x): return self.fc(x)

class SpecialistHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
    def forward(self, x): return self.fc(x)

# 3. PIPELINE SETUP
train_ds = DroneMultiModalDataset('train_split.txt')
test_ds = DroneMultiModalDataset('test_split.txt')
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

encoder, sentry = SharedEncoder(), SentryHead()
specialist, decoder = SpecialistHead(), Decoder()

# --- DYNAMIC WEIGHT CALCULATION ---
print("\nCalculating Class Weights to fix imbalance...")
counts = [0, 0, 0]
for _, y in train_ds: counts[y.item()] += 1
total_tr = sum(counts)
spec_weights = torch.tensor([total_tr/c if c > 0 else 1.0 for c in counts], dtype=torch.float32)

# Sentry weight: Ratio of Normal to Faults
s_weight = torch.tensor([counts[0] / (counts[1] + counts[2])], dtype=torch.float32)

# --- PHASE 1: JOINT TRAINING (30 Epochs) ---
print("\n" + "="*70)
print(f"{'Epoch':<8} | {'Total Loss':<12} | {'Diag Acc %':<12} | {'Recon MSE':<12}")
print("-" * 70)

opt1 = optim.Adam(list(encoder.parameters()) + list(specialist.parameters()) + list(decoder.parameters()), lr=1e-4)
criterion_diag = nn.CrossEntropyLoss(weight=spec_weights)
criterion_recon = nn.MSELoss()

for epoch in range(30):
    encoder.train(); specialist.train(); decoder.train()
    r_loss, r_acc, r_recon = 0.0, 0, 0.0
    for x, y in train_loader:
        opt1.zero_grad()
        feat = encoder(x)
        out_diag, out_recon = specialist(feat), decoder(feat)
        loss = criterion_diag(out_diag, y) + (0.5 * criterion_recon(out_recon, x))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt1.step()
        r_loss += loss.item(); r_recon += criterion_recon(out_recon, x).item()
        r_acc += (out_diag.argmax(1) == y).sum().item()
    print(f"{epoch+1:<8} | {r_loss/len(train_loader):<12.4f} | {100*r_acc/len(train_ds):<12.2f} | {r_recon/len(train_loader):<12.6f}")

# --- PHASE 2: SENTRY CALIBRATION (10 Epochs) ---
print("\nPHASE 2: Calibrating Sentry Gatekeeper...")
for p in encoder.parameters(): p.requires_grad = False
opt2 = optim.Adam(sentry.parameters(), lr=1e-4)
criterion_sentry = nn.BCEWithLogitsLoss(pos_weight=s_weight)

for epoch in range(10):
    sentry.train()
    for x, y in train_loader:
        opt2.zero_grad()
        binary_y = (y > 0).float().unsqueeze(1)
        loss = criterion_sentry(sentry(encoder(x)), binary_y)
        loss.backward(); opt2.step()

# --- PHASE 3: EVALUATION ---
print("\n" + "="*70)
print("PHASE 3: SPLIT INFERENCE DEPLOYMENT")
print("="*70)
encoder.eval(); sentry.eval(); specialist.eval()

triggers, diag_ok, total = 0, 0, 0
sentry_true, sentry_pred = [], []
system_true, system_pred = [], []
SENTRY_THRESHOLD = 0.5 

with torch.no_grad():
    for x, y in test_loader:
        total += 1
        feat = encoder(x)
        
        y_val = y.item()
        sentry_true.append(1 if y_val > 0 else 0)
        system_true.append(y_val)

        prob = torch.sigmoid(sentry(feat)).item()
        if prob > SENTRY_THRESHOLD:
            triggers += 1
            sentry_pred.append(1)
            pred = specialist(feat).argmax(1).item()
            system_pred.append(pred)
            if pred == y_val: diag_ok += 1
        else:
            sentry_pred.append(0)
            system_pred.append(0)

# --- RESULTS ---
print(f"Bandwidth Saved: {100 * (1 - (triggers/total)):.2f}%")
print(f"Final Accuracy : {100 * diag_ok/triggers if triggers > 0 else 0:.2f}%")

print("\n" + "="*50)
print(" SENTRY CONFUSION MATRIX (Edge Layer)")
print("="*50)
cm_s = confusion_matrix(sentry_true, sentry_pred, labels=[0, 1])
print(f"{'':<15} | {'Pred Norm':<15} | {'Pred Fault':<15}")
print(f"{'Actual Norm':<15} | {cm_s[0,0]:<15} | {cm_s[0,1]:<15}")
print(f"{'Actual Fault':<15} | {cm_s[1,0]:<15} | {cm_s[1,1]:<15}")

print("\n" + "="*50)
print(" SYSTEM CONFUSION MATRIX (Overall)")
print("="*50)
cm_sys = confusion_matrix(system_true, system_pred, labels=[0, 1, 2])
print(f"{'':<15} | {'P: Norm':<10} | {'P: Motor':<10} | {'P: Prop':<10}")
print(f"{'Actual Norm':<15} | {cm_sys[0,0]:<10} | {cm_sys[0,1]:<10} | {cm_sys[0,2]:<10}")
print(f"{'Actual Motor':<15} | {cm_sys[1,0]:<10} | {cm_sys[1,1]:<10} | {cm_sys[1,2]:<10}")
print(f"{'Actual Prop':<15} | {cm_sys[2,0]:<10} | {cm_sys[2,1]:<10} | {cm_sys[2,2]:<10}")
