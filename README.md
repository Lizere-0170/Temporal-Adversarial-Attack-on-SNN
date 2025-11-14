Temporal Adversarial Attacks on Spiking Neural Networks

This project investigates temporal adversarial attacks on Spiking Neural Networks (SNNs).
Unlike conventional adversarial attacks that perturb input values, this work focuses on manipulating the timing of spikes (Δt) while preserving spike count and spatial structure. The objective is to assess the robustness of SNNs trained on event-based data and determine whether small temporal shifts can reliably change model predictions.

This repository contains:

A baseline SNN classifier trained with surrogate gradients

A differentiable temporal encoding pipeline

Single-sample and multi-sample white-box PGD attacks on spike timestamps

Visualization tools for spike rasters and perturbations

Evaluation scripts and analysis tools

1. Project Motivation

Event-based and neuromorphic learning systems rely heavily on precise spike timing. This temporal coding introduces a potential attack surface that is fundamentally different from amplitude-based attacks in classical neural networks.

The central questions addressed in this project:

Can small, realistic perturbations in spike timing mislead an SNN?

How sensitive is a trained SNN to millisecond-scale temporal noise?

Are such vulnerabilities systemic across the dataset, or limited to isolated samples?

How do temporal constraints (e.g., jitter budgets) affect attack success?

2. Dataset
Spiking Heidelberg Digits (SHD)

Event-based audio classification dataset

~10–20K samples split into train/test

Each sample is a set of events:

{times, x, y, polarity, label}


Timestamps are on microsecond scale

Used widely for benchmarking SNN temporal classification

In this project, events are normalized to seconds and then discretized using a Gaussian time-kernel to produce continuous input tensors for differentiable optimization.

3. Method
3.1 Baseline SNN Model

The SNN is implemented with:

Feedforward spiking layers

Learnable membrane time constants

Surrogate-gradient backpropagation

Time-discretized input using dt = 1–10 ms

Cross-entropy loss

Adam optimizer

Validation accuracy reaches 0.40–0.60 on SHD, confirming correct training and data processing.

3.2 Differentiable Temporal Encoding

To enable temporal attacks, raw events are converted to a continuous tensor I[t, c] using:

I[t, c] = Σ exp( - (t*dt - t_k)^2 / (2σ^2) )


where:

t_k is the spike timestamp

σ controls smoothing

discretization: T time bins, each of width dt

normalization is optional but helps stabilize training

This representation is fully differentiable with respect to spike timings, enabling gradient-based attacks.

3.3 White-Box PGD Timing Attack

For each input sample, we optimize a set of per-event time offsets:

t_adv = t_original + δ


with the constraints:

|δ| ≤ τ_max (temporal budget)

δ is optimized using Adam or PGD

τ_max typically 5–20 ms

Objective: maximize the loss (untargeted attack)

The attack pipeline includes:

Forward pass through SNN

Backprop through time-kernel

Update δ by gradient ascent

Clip deltas within allowed range

Reconstruct adversarial event stream

4. Installation
conda create -n temporal_snn python=3.10 -y
conda activate temporal_snn

pip install torch numpy matplotlib tqdm

5. Usage
Train the SNN
python -m src.train


Saves checkpoint to: ./checkpoints/best.pth

Run a single-sample white-box attack
python -m src.attacks.whitebox --idx 0


Outputs:

clean raster

adversarial raster

δ statistics

adversarial .npz file

Run multi-sample white-box attack (recommended)
python -m src.attacks.multi_whitebox


Outputs:

progress bar over N samples

summary statistics

all adversarial event files in results_whitebox/

6. Results Summary
Single-Sample Attack

Clean vs adversarial rasters appear nearly identical

Mean δ ≈ 0.0054 ms

One spike shifted by 61 ms, flipping the prediction

Indicates strong sensitivity to isolated influential spikes

Multi-Sample Attack (N = 100)
Attack success rate : 98%
Avg |δ| (ms)        : 4.468
Avg max |δ| (ms)    : 4.950


Interpretation:

Vulnerability is systemic, not anecdotal

Millisecond-scale timing noise consistently flips classifications

Perturbations are small enough to resemble realistic sensor jitter

Attack does not rely on extreme shifts; perturbations are distributed

Confirms temporal brittleness in the SNN decision boundary

7. Directory Structure
src/
│
├── models/
│   └── snn.py                 # Baseline SNN
│
├── attacks/
│   ├── whitebox.py            # Single-sample PGD attack
│   └── multi_whitebox.py      # Multi-sample batch attack
│
├── utils.py                   # Event I/O, raster plots, encodings
├── train.py                   # Model training
├── debug_overfit.py           # Overfitting tests
│
checkpoints/                   # Saved models
results_whitebox/              # Multi-attack outputs
data/                          # SHD dataset
README.md

8. Key Findings

SNNs trained on SHD are highly sensitive to spike timing

Millisecond-scale perturbations can reliably induce misclassification

Temporal adversarial attacks remain visually stealthy

Vulnerability persists across the dataset (98% success)

Attack requires no spike additions or deletions

Highlights a security gap for neuromorphic computing systems

9. Contact

For project-related questions:

Lizere Yun
NYU Shanghai
Email: jy4195@nyu.edu
