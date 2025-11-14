# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

from models.snn import BaselineSNN
from utils import load_shd_sample, save_events_npz


# -------------------------
# helper: convert events -> [T, C] spikes tensor
# -------------------------
def events_to_spike_tensor(events, T=100, dt=0.01, C=700, time_unit='s'):
    """
    events: dict {'times'(float), 'x'(int channel), ...}
    T: number of time bins
    dt: bin width in seconds (e.g., 0.01s -> 10ms). Choose so that T*dt covers sample length.
    C: number of channels (SHD ~700)
    返回: tensor shape [T, C] of 0/1 spikes
    """
    times = np.asarray(events['times'], dtype=float)
    channels = np.asarray(events['x'], dtype=int)
    # time origin assumed 0; compute bin index
    bin_idx = np.floor(times / dt).astype(int)
    # clip
    bin_idx = np.clip(bin_idx, 0, T-1)
    spikes = np.zeros((T, C), dtype=np.float32)
    for b, ch in zip(bin_idx, channels):
        if 0 <= ch < C:
            spikes[b, ch] = 1.0  # if multiple events fall in same bin keep 1.0
    return torch.from_numpy(spikes)  # [T, C]

# -------------------------
# Simple PyTorch Dataset wrapper for SHD npz files or tonic loading
# -------------------------
class SHDDataset(Dataset):
    def __init__(self, root='./data', split='train', indices=None, T=100, dt=0.01, C=700, n_classes=20):
        self.root = root
        self.split = split
        self.T = T
        self.dt = dt
        self.C = C
        self.n_classes = n_classes
        # for simplicity we call tonic loader each time (or you can preload npz)
        # We'll use tonic SHD via load_shd_sample wrapper
        # Precompute length via tonic
        import tonic
        from tonic.datasets import SHD
        ds = SHD(save_to=root, train=(split=='train'))
        # subset = ds.train if split == 'train' else ds.test
        self.length = len(ds) if indices is None else len(indices)
        self.indices = indices
        self.subset = ds

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        # Tonic returns (events, label) as a tuple
        events, label = self.subset[real_idx]
        
        evt = {
            'times': events['t'],
            'x': events['x']
        }
        
        # Convert events to spike tensor
        spikes = events_to_spike_tensor(evt, T=self.T, dt=self.dt, C=self.C)

        # convert label to zero-based and validate
        label = int(label)
        # if label >= self.n_classes:
        #     label = label - 1  # convert 1-based -> 0-based for typical SHD labeling
        assert 0 <= label < self.n_classes, f"label {label} out of range [0,{self.n_classes})"
        return spikes, label

# -------------------------
# training loop
# -------------------------
def train_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaselineSNN(in_channels=args['C'], hidden_neurons=args['hidden'],
                        n_classes=args['n_classes'], tau_mem=args['tau_mem'], dt=args['dt'], device=device).to(device)
    opt = optim.Adam(model.parameters(), lr=args['lr'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     opt,
    #     T_max=args['epochs'],   # 运行多少 epoch 就衰减完一个周期
    #     eta_min=1e-5            # 最低学习率，防止变成 0
    # )
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='max',        # maximize val_acc
        factor=0.5,        # reduce LR by 0.5
        patience=5,        # if no improvement for 5 epochs
        threshold=1e-3,
    )

    train_ds = SHDDataset(root=args['root'], split='train', T=args['T'], dt=args['dt'], C=args['C'],n_classes=args['n_classes'])
    val_ds = SHDDataset(root=args['root'], split='test', T=args['T'], dt=args['dt'], C=args['C'],n_classes=args['n_classes'])
    train_loader = DataLoader(train_ds, batch_size=args['batch'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args['batch'], shuffle=False, num_workers=0)

    best_val = 0.0
    wait = 0
    patience = 10  # early stopping patience
    logs = []

    for epoch in range(args['epochs']):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for spikes, labels in train_loader:
            # spikes: [T, C] per sample -> dataset returns (T,C) for each sample; need to stack into [T,B,C]
            # Our Dataset returns per-item tensor shape [T,C], so DataLoader gives [B,T,C]; transpose
            # But our model expects [T,B,C]
            # spikes = spikes.permute(1,0,2) if spikes.dim() == 3 else spikes  # safe guard
            # After typical DataLoader, shape: [B, T, C]
            if spikes.dim() == 3:
                spikes = spikes.permute(1,0,2)  # -> [T, B, C]
            spikes = spikes.to(device).float()
            labels = labels.to(device).long()
            logits = model(spikes)  # [B, n_classes]
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        train_acc = correct / total
        avg_loss = total_loss / total

        # validation
        model.eval()
        v_total = 0
        v_correct = 0
        with torch.no_grad():
            for spikes, labels in val_loader:
                if spikes.dim() == 3:
                    spikes = spikes.permute(1,0,2)  # [T,B,C]
                spikes = spikes.to(device).float()
                labels = labels.to(device).long()
                logits = model(spikes)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        val_acc = v_correct / v_total
        # current_lr = scheduler.get_last_lr()[0]
        # print(f"\nEpoch {epoch+1}/{args['epochs']} - Current learning rate: {current_lr:.6f}") 
        print(f"Epoch {epoch+1}/{args['epochs']}  Train loss {avg_loss:.4f} Train acc {train_acc:.4f}  Val acc {val_acc:.4f}")
        
        # logging
        logs.append({'epoch': epoch+1, 'train_loss': avg_loss, 'train_acc': train_acc, 'val_acc': val_acc})
        with open(os.path.join(args['save_dir'], 'train_log.json'), 'w') as f:
            json.dump(logs, f, indent=2)
        
        # step scheduler
        scheduler.step(val_acc)
        current_lr = opt.param_groups[0]['lr']
        print(f"→ Current LR: {current_lr:.6e}")
        #early stopping
        if val_acc > best_val + 1e-3:
            best_val = val_acc
            wait = 0
            torch.save({'model_state': model.state_dict(), 'args': args},
                       os.path.join(args['save_dir'], 'best.pth'))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

        # save best
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state': model.state_dict(), 'args': args}, os.path.join(args['save_dir'], 'best.pth'))
    print("Training finished. Best val acc:", best_val)

if __name__ == '__main__':
    args = {
        'root': './data',
        'C': 700,
        'T': 400,     # 时间步数（你需要根据 sample 时长与 dt 调整）
        'dt': 0.002,   # 10 ms bins -> 100 steps cover 1s
        'hidden': 512,
        'n_classes': 20,
        'tau_mem': 20.0,
        'lr': 1e-3,
        'batch': 16,
        'epochs': 20,
        'save_dir': './checkpoints'
    }
    os.makedirs(args['save_dir'], exist_ok=True)
    train_loop(args)
