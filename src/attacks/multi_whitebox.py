import os
import torch
import numpy as np
from tqdm import tqdm

# === Import your existing attack + model utilities ===
from .whitebox import attack_sample_pgd
from src.models.snn import BaselineSNN
from src.utils import save_events_npz

# === Import SHD Dataset from tonic ===
from tonic.datasets import SHD


# -----------------------------------------------------
# 1. Dataset Loader for SHD
# -----------------------------------------------------
def load_shd_sample(root, split, index, time_unit='s'):
    """
    Load one SHD sample using tonic, in the same format as your attack uses.
    """
    dataset = SHD(save_to=root, train=(split == 'train'))

    events, label = dataset[index]
    # tonic gives events as dict with keys: t, x, y, p
    # Convert to your expected format
    t = events['t'].astype(np.float32)

    # Convert tonic timestamps (μs) to seconds if needed
    if time_unit == 's':
        t = t / 1e6

    sample = {
        'times': t,
        'x': events['x'].astype(np.int32),
        # 'y': events['y'].astype(np.int32),
        # 'polarity': events['p'].astype(np.int32),
        'label': int(label),
    }
    return sample, int(label)


# -----------------------------------------------------
# 2. Load Model from Checkpoint
# -----------------------------------------------------
def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get('args', {})

    model = BaselineSNN(
        in_channels=model_args.get("in_channels", 700),
        hidden_neurons=model_args.get("hidden", 512),
        n_classes=model_args.get("n_classes", 20),
        tau_mem=model_args.get("tau_mem", 20.0),
        dt=model_args.get("dt", 0.01)
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# -----------------------------------------------------
# 3. Main Attack Runner
# -----------------------------------------------------
def main():
    # --------------------------
    # Settings
    # --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = "./checkpoints/best.pth"
    data_root = "./data"
    save_dir = "./results_whitebox"
    os.makedirs(save_dir, exist_ok=True)

    # Attack parameters
    attack_args = {
        "T": 100,
        "dt": 0.01,
        "C": 700,
        "iters": 200,
        "lr": 5e-3,
        "tau_max": 0.005,     # 5 ms timing budget
        "sigma": 0.01,
        "verbose": False,
    }

    # Number of test samples to attack
    NUM_SAMPLES = 100
    indices = list(range(NUM_SAMPLES))

    # --------------------------
    # Load model
    # --------------------------
    print("Loading model...")
    model = load_model(checkpoint, device)

    # --------------------------
    # Attack Loop
    # --------------------------
    print(f"Running white-box PGD attack on {NUM_SAMPLES} samples...")
    results = []

    for idx in tqdm(indices):
        # Load clean sample
        events, label = load_shd_sample(data_root, "test", idx)

        # Generate adversarial version
        adv_events, stats = attack_sample_pgd(
            model, events, label, device=device, **attack_args
        )

        # Save adversarial events
        out_path = os.path.join(save_dir, f"adv_{idx}.npz")
        save_events_npz(adv_events, out_path)

        # Collect stats
        stats["idx"] = idx
        stats["mean_abs_delta_ms"] = stats["mean_abs_delta"] * 1000
        stats["max_abs_delta_ms"] = stats["max_abs_delta"] * 1000
        results.append(stats)

    # --------------------------
    # Summary Statistics
    # --------------------------
    successes = sum(r["success"] for r in results)
    avg_delta = np.mean([r["mean_abs_delta_ms"] for r in results])
    max_delta = np.mean([r["max_abs_delta_ms"] for r in results])

    print("\n=====================================")
    print("WHITE-BOX ATTACK SUMMARY")
    print("=====================================")
    print(f"Total samples attacked : {NUM_SAMPLES}")
    print(f"Attack success rate    : {successes / NUM_SAMPLES:.2f}")
    print(f"Avg |δ| (ms)            : {avg_delta:.3f}")
    print(f"Avg max |δ| (ms)        : {max_delta:.3f}")
    print("Results saved to       :", save_dir)
    print("=====================================\n")


if __name__ == "__main__":
    main()
