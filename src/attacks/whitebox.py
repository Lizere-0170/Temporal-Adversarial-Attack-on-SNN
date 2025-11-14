# src/attack_pgd.py
"""
White-box temporal attack (PGD/Adam) for SHD using differentiable time-kernel encoding.

Usage:
    from src.attack_pgd import attack_sample_pgd, evaluate_attack_on_set
    adv = attack_sample_pgd(model, events_dict, label, args_attack)
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from tqdm import trange

from src.utils import plot_raster, save_events_npz, load_events_npz, _normalize_events_dict

# -------------------------
# helper: convert times -> differentiable input tensor [T, C]
# -------------------------
def events_to_continuous_input(events, T, dt, C, device='cpu', sigma=None):
    """
    Convert events (times, x channels) into continuous input tensor I[t, c].
    I[t,c] = sum_k exp( - (t*dt - (time_k))^2 / (2*sigma^2) ) for events on channel c.

    Parameters:
      events: dict with 'times' (N,), 'x' (N,)
      T: number of time bins
      dt: bin width in seconds
      C: number of channels
      sigma: kernel width in seconds. If None, sigma = dt.
      device: torch device
    Returns:
      I: torch.tensor shape [T, C], dtype=float32 on device
      bin_times: numpy array shape (T,) the center time of each bin
    """
    times = np.asarray(events['times'], dtype=float)
    channels = np.asarray(events['x'], dtype=int)
    if sigma is None:
        sigma = dt  # default kernel width

    bin_times = np.arange(0, T) * dt + dt * 0.5  # center of bins
    # compute distances: shape (T, N)
    # We'll compute in vectorized numpy then convert to torch
    if len(times) == 0:
        I = np.zeros((T, C), dtype=np.float32)
        return torch.from_numpy(I).to(device), bin_times

    # for memory: do accumulation per channel
    I = np.zeros((T, C), dtype=np.float32)
    # compute contribution per event
    two_sigma2 = 2.0 * (sigma ** 2)
    for t_k, ch in zip(times, channels):
        if ch < 0 or ch >= C:
            continue
        dt_vec = bin_times - t_k  # (T,)
        kernel = np.exp(- (dt_vec ** 2) / two_sigma2).astype(np.float32)
        I[:, ch] += kernel
    # normalize per-channel peak to <=1 (optional)
    max_per_channel = I.max(axis=0, keepdims=True)
    max_per_channel[max_per_channel == 0] = 1.0
    # I = I / max_per_channel
    return torch.from_numpy(I).to(device), bin_times

# -------------------------
# attack on a single sample (white-box)
# -------------------------
def attack_sample_pgd(model, events, true_label,
                      device='cpu',
                      T=100, dt=0.001, C=700,
                      tau_max=0.02,         
                      iters=200,
                      lr=1e-2,
                      sigma=0.00025,
                      optimizer_name='adam',
                      clip_to_range=(0.0, None),  # clip times to >=0 and optionally <=max_time
                      verbose=True):
    """
    White-box attack that optimizes per-spike delta (continuous) using Adam/PGD.

    Inputs:
      model: PyTorch model taking input [T,B,C] with B=1
      events: standard dict {'times','x','y','polarity'}
      true_label: int
      device: 'cpu' or 'cuda'
      T, dt, C: time bins, bin width, channel count (must match model training setting)
      tau_max: maximum absolute shift in seconds
      iters: optimization steps
      lr: optimizer lr
      sigma: kernel width (if None, set to dt)
      clip_to_range: (min_time, max_time) clip after applying delta
    Returns:
      adv_events: dict with perturbed 'times','x','y','polarity'
      stats: dict with keys 'success'(bool), 'mean_abs_delta', 'max_abs_delta', 'loss_trace'
    """
    model = model.to(device)
    model.eval()

    # normalize events (defensive)
    events = _normalize_events_dict(events, label=true_label, time_scale=1.0)
    orig_times = np.asarray(events['times'], dtype=float)
    channels = np.asarray(events['x'], dtype=int)
    N = orig_times.shape[0]
    if N == 0:
        raise ValueError("Empty event list")

    # init delta as zeros (torch param)
    delta = torch.zeros(N, device=device, dtype=torch.float32, requires_grad=True)

    # set kernel width
    if sigma is None:
        sigma = dt

    # optional time bounds
    min_time = clip_to_range[0] if clip_to_range is not None else None
    max_time = clip_to_range[1] if clip_to_range is not None else None

    # optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam([delta], lr=lr)
    else:
        optimizer = torch.optim.SGD([delta], lr=lr)

    loss_trace = []
    success = False

    # precompute channel indices as torch for one-hot gather if needed
    channels_t = torch.from_numpy(channels).long().to(device)

    for it in range(iters):
        optimizer.zero_grad()
        # compute perturbed times
        # ensure float32 to match I (which is float32)
        t_adv = torch.from_numpy(orig_times).to(device).float() + delta  # (N,)

        # clip to [min_time, max_time]
        if min_time is not None:
            t_adv = torch.clamp(t_adv, min=min_time)
        if max_time is not None:
            t_adv = torch.clamp(t_adv, max=max_time)

        # Create continuous input I[t,c] differentiably
        # We'll implement kernel evaluation in torch for autograd, vectorized:
        bin_times = torch.arange(0, T, device=device, dtype=torch.float32) * dt + dt*0.5  # (T,)
        # compute pairwise diffs: (T, N) = bin_times.unsqueeze(1) - t_adv.unsqueeze(0)
        diffs = bin_times.unsqueeze(1) - t_adv.unsqueeze(0)  # [T, N]
        two_sigma2 = 2.0 * (sigma ** 2)
        K = torch.exp(- (diffs ** 2) / two_sigma2)  # [T, N]
        # zero-out events with invalid channels (already filtered)
        # accumulate into channels -> I [T, C]
        I = torch.zeros((T, C), dtype=K.dtype, device=device)
        I = I.scatter_add(1, channels_t.unsqueeze(0).expand(T, -1), K)  # sums per channel
        # normalize per channel peak to <=1 to keep input scale similar to training
        max_per_channel, _ = I.max(dim=0, keepdim=True)  # [1, C]
        max_per_channel[max_per_channel == 0] = 1.0
        I = I / max_per_channel

        # model forward expects [T, B, C], B=1
        I_batch = I.unsqueeze(1)  # [T, 1, C]
        logits = model(I_batch)    # [1, n_classes]
        loss = F.cross_entropy(logits, torch.tensor([true_label], device=device, dtype=torch.long))

        # we maximize loss (untargeted), so minimize -loss
        loss_to_min = -loss
        loss_to_min.backward()
        # step
        optimizer.step()

        # projection: clamp delta to [-tau_max, +tau_max]
        with torch.no_grad():
            delta.clamp_(-tau_max, +tau_max)

        loss_val = loss.item()
        loss_trace.append(loss_val)

        # check success (prediction not equal true_label)
        pred = logits.argmax(dim=1).item()
        if pred != int(true_label):
            success = True
            if verbose:
                print(f"[it {it+1}/{iters}] success at step {it+1}, pred {pred}, loss {loss_val:.4f}")
            # optionally break early to save queries/time
            # break

        if verbose and (it % max(1, iters//10) == 0 or success):
            print(f"it {it+1}/{iters}: loss {loss_val:.4f}, pred {pred}, mean|delta| {(delta.abs().mean().item()*1000):.3f} ms")

    # final stats
    delta_final = delta.detach().cpu().numpy()
    mean_abs = float(np.mean(np.abs(delta_final)))
    max_abs = float(np.max(np.abs(delta_final)))

    # build adv events dict
    adv_times = orig_times + delta_final
    if min_time is not None:
        adv_times = np.clip(adv_times, min_time, None)
    if max_time is not None:
        adv_times = np.clip(adv_times, None, max_time)
    adv_events = {'times': adv_times, 'x': channels.copy(), 'y': events.get('y', np.zeros_like(channels)), 'polarity': events.get('polarity', np.ones_like(channels)), 'label': events.get('label', true_label)}

    stats = {'success': success, 'mean_abs_delta': mean_abs, 'max_abs_delta': max_abs, 'loss_trace': loss_trace}
    stats['delta'] = delta_final
    return adv_events, stats

# -------------------------
# batch evaluation / wrapper
# -------------------------
def evaluate_attack_on_set(model, dataset_indices, dataset_loader_factory, save_dir,
                           attack_args, device='cuda'):
    """
    dataset_loader_factory: callable that returns an iterable of (events, label) for given index list
      (we keep it generic: you can implement a function that given index returns events dict)
    dataset_indices: list of sample indices to attack (or 'all' to iterate full set)
    attack_args: dict for attack parameters passed to attack_sample_pgd
    save_dir: where to save adv npz and stats
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []
    for idx in dataset_indices:
        events, label = dataset_loader_factory(idx)  # should return (events_dict, label)
        adv_events, stats = attack_sample_pgd(model, events, label, device=device, **attack_args)
        # compute final prediction on adv
        # compute mean|delta| in ms
        stats['mean_abs_delta_ms'] = stats['mean_abs_delta'] * 1000.0
        stats['max_abs_delta_ms'] = stats['max_abs_delta'] * 1000.0
        success_flag = stats['success']
        stats['index'] = idx
        results.append(stats)
        # save adv sample
        save_path = os.path.join(save_dir, f"adv_{idx}.npz")
        save_events_npz(adv_events, save_path)
    return results

# -------------------------
# small demo runner
# -------------------------
if __name__ == '__main__':
    # quick demo (requires model and src.utils)
    import argparse
    from src.utils import load_shd_sample
    from src.models.snn import BaselineSNN

    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--C', type=int, default=700)
    parser.add_argument('--tau_max', type=float, default=20)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model
    ckpt = torch.load('./checkpoints/best.pth', map_location=device)
    model_args = ckpt.get('args', {})
    model = BaselineSNN(in_channels=args.C, hidden_neurons=model_args.get('hidden', 512),
                        n_classes=model_args.get('n_classes', 20),
                        tau_mem=model_args.get('tau_mem', 20.0),
                        dt=model_args.get('dt', 0.01)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # load one sample
    sample = load_shd_sample(root='./data', split='test', index=args.idx, time_unit='s')
    events = sample
    label = sample.get('label', 0)
    print("original label", label)
    plot_raster(events, title='clean sample')

    adv, stats = attack_sample_pgd(model, events, label, device=device,
                                   T=args.T, dt=args.dt, C=args.C,
                                   tau_max=args.tau_max, iters=200, lr=5e-3, sigma=args.dt/4)
    print("attack stats:", stats)
    print(stats['mean_abs_delta']*1000, "ms")
    print(stats['max_abs_delta']*1000, "ms")
    plot_raster(adv, title='adv sample')
    save_events_npz(adv, f'./results/adv_{args.idx}.npz')
