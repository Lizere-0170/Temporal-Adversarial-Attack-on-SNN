# src/eval.py
import torch, os
from src.models.snn_baseline import BaselineSNN
from src.train import SHDDataset, events_to_spike_tensor
from src.utils import load_events_npz

def evaluate_checkpoint(checkpoint_path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint_path, map_location=device)
    model = BaselineSNN(in_channels=args['C'], hidden_neurons=args['hidden'],
                        n_classes=args['n_classes'], tau_mem=args['tau_mem'], dt=args['dt']).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()

    ds = SHDDataset(root=args['root'], split='test', T=args['T'], dt=args['dt'], C=args['C'])
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args['batch'], shuffle=False)
    total = 0
    correct = 0
    with torch.no_grad():
        for spikes, labels in loader:
            if spikes.dim() == 3:
                spikes = spikes.permute(1,0,2)  # [T,B,C]
            spikes = spikes.to(device)
            labels = labels.to(device)
            logits = model(spikes)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print("Test acc:", correct / total)

if __name__ == '__main__':
    args = {
        'root': './data',
        'C': 700,
        'T': 100,
        'dt': 0.01,
        'hidden': 512,
        'n_classes': 10,
        'tau_mem': 20.0,
        'batch': 8
    }
    evaluate_checkpoint('./checkpoints/best.pth', args)
