from train import SHDDataset, events_to_spike_tensor, train_loop
import collections
from tonic.datasets import SHD
import torch

# print overall label distribution in train subset (sanity)
ds_train = SHD(save_to='./data', train=True)
labels = [int(ds_train[i][1]) for i in range(len(ds_train))]
print("train label min/max:", min(labels), max(labels))
print("label counts:", collections.Counter(labels))

# now small overfit subset (first 50 samples)
from torch.utils.data import DataLoader
from models.snn_baseline import BaselineSNN
small_ds = SHDDataset(root='./data', split='train', indices=list(range(50)), T=100, dt=0.01, C=700)
loader = DataLoader(small_ds, batch_size=8, shuffle=True)

def overfit_loop(model, loader, epochs=20, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        for spikes, labels in loader:
            if spikes.dim() == 3:
                spikes = spikes.permute(1,0,2)  # [T,B,C]
            spikes = spikes.to(device).float()
            labels = labels.to(device).long()
            opt.zero_grad()
            logits = model(spikes)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Acc: {acc:.4f}")

model = BaselineSNN(in_channels=700, hidden_neurons=512, n_classes=20, tau_mem=20.0, dt=1.0)
overfit_loop(model, loader, epochs=30, device='cpu')

