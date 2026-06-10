"""
preprocess.py — ECG Data Preprocessing for BiLSTM-CNN GAN

Dataset: PhysioNet/CinC Challenge 2017
Paper:   Zhu et al., Scientific Reports, 2019

Pipeline: Load → Bandpass Filter (0.5-40Hz) → Min-Max [0,1] → Segment (3120pts)
Output:   11,589 segments of shape (3120,), batch size 100, 115 batches/epoch
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import TensorDataset, DataLoader

# Settings (match paper)
SEQ_LENGTH = 3120
BATCH_SIZE = 100
SAMPLING_RATE = 300
RANDOM_SEED = 42


def load_ecg(filepath):
    """Load one ECG signal from a .mat file."""
    mat = sio.loadmat(filepath)
    return mat['val'].flatten().astype(np.float64)


def bandpass_filter(signal, lowcut=0.5, highcut=40.0):
    """Remove noise outside 0.5-40 Hz (keeps PQRST complex)."""
    nyquist = SAMPLING_RATE / 2.0
    b, a = butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, signal)


def normalize(signal):
    """Min-max normalisation to [0, 1] (Equation 22 in paper)."""
    sig_min, sig_max = signal.min(), signal.max()
    if sig_max - sig_min == 0:
        return np.zeros_like(signal)
    return (signal - sig_min) / (sig_max - sig_min)


def segment(signal, length=SEQ_LENGTH):
    """Slice signal into fixed-length windows of 3120 samples, no overlap."""
    segments = []
    start = 0
    while start + length <= len(signal):
        segments.append(signal[start:start + length])
        start += length
    return segments


def preprocess(data_dir='.'):
    """
    Full pipeline: Load Normal class → Filter → Normalise → Segment.
    Returns numpy array of shape (num_segments, 3120) with values in [0, 1].
    """
    import pandas as pd

    labels = pd.read_csv(os.path.join(data_dir, 'REFERENCE.csv'),
                         header=None, names=['recording', 'label'])
    normal_labels = labels[labels['label'] == 'N']

    print(f"Dataset: {data_dir}")
    print(f"Class: Normal (N) — {len(normal_labels)} recordings")
    print(f"Bandpass filter: 0.5-40 Hz")
    print(f"Normalisation: min-max [0, 1]")
    print(f"Segment length: {SEQ_LENGTH} samples")
    print()

    all_segments = []
    skipped = 0

    for i, row in normal_labels.iterrows():
        signal = load_ecg(os.path.join(data_dir, row['recording'] + '.mat'))
        signal = bandpass_filter(signal)
        signal = normalize(signal)
        segs = segment(signal)

        if len(segs) > 0:
            all_segments.extend(segs)
        else:
            skipped += 1

        count = len(normal_labels[normal_labels.index <= i])
        if count % 1000 == 0:
            print(f"  Processed {count}/{len(normal_labels)} recordings...")

    all_segments = np.array(all_segments)
    np.random.seed(RANDOM_SEED)
    all_segments = all_segments[np.random.permutation(len(all_segments))]

    print()
    print(f"--- Done ---")
    print(f"  Total segments: {len(all_segments)}")
    print(f"  Segment shape: {all_segments[0].shape}")
    print(f"  Skipped (too short): {skipped}")
    print(f"  Value range: [{all_segments.min():.4f}, {all_segments.max():.4f}]")

    return all_segments


def get_dataloader(data_dir='.'):
    """
    Returns a PyTorch DataLoader with batches of shape (100, 3120, 1).

    Usage:
        from preprocess import get_dataloader
        dataloader = get_dataloader(data_dir='./training2017')
        for batch in dataloader:
            real_data = batch[0]  # shape: (100, 3120, 1)
    """
    segments = preprocess(data_dir)
    tensor = torch.FloatTensor(segments).unsqueeze(-1)
    dataset = TensorDataset(tensor)

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True, generator=g)

    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Batch shape: ({BATCH_SIZE}, {SEQ_LENGTH}, 1)")
    return dataloader


if __name__ == '__main__':
    segments = preprocess()
    os.makedirs('processed_data', exist_ok=True)
    save_path = os.path.join('processed_data', 'ecg_segments.npy')
    np.save(save_path, segments)
    print(f"  Saved to {save_path}")
