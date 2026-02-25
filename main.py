"""
Detect and interpolate bad EEG channels in epoched data (app-detect-bad-channels-eeg-v2).

Runs BEFORE noise covariance to ensure clean data.
Uses variance (MAD z-score), correlation, and flat channel detection.

Input:
    - config.json:
      - epochs: Path to MNE epochs .fif file
      - z_thresh: MAD z-score threshold for noisy channel detection (default: 3.0)
      - corr_thresh: Min mean correlation for low-correlation detection (default: 0.4)
      - extra_bads: Optional comma-separated list of channels to force-mark bad

Output:
    - out_dir/meg-epo.fif: Cleaned epochs with bad channels interpolated
    - out_figs/channel_diagnostics.png: Variance z-scores and mean correlations
    - product.json: Summary of pre-existing and detected/interpolated channels
"""

# Copyright (c) 2026 brainlife.io
#
# Authors:
# - Maximilien Chaumon (https://github.com/dnacombo)

import sys
import os

# When deployed on Brainlife: brainlife_utils/ is in this directory.
# When running locally in the monorepo: it's in the parent directory.
_app_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_app_dir)
for _path in [_app_dir, _parent_dir]:
    if os.path.isdir(os.path.join(_path, 'brainlife_utils')):
        sys.path.insert(0, _path)
        break

import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brainlife_utils import (
    load_config,
    setup_matplotlib_backend,
    ensure_output_dirs,
    create_product_json,
    add_info_to_product,
    add_image_to_product,
)

# Set up matplotlib for headless execution
setup_matplotlib_backend()

# Ensure output directories exist
ensure_output_dirs('out_dir', 'out_figs')

# Load configuration
config = load_config()


def detect_bad_channels(epochs, z_thresh=3.0, corr_thresh=0.4):
    """
    Detect bad EEG channels using z-score (MAD) and correlation heuristics.

    Parameters
    ----------
    epochs : mne.Epochs
        The epoched data (must be preloaded).
    z_thresh : float
        Z-score threshold for abnormal variance (default: 3.0).
        Uses MAD (median absolute deviation) for robust z-scoring.
    corr_thresh : float
        Channels with mean correlation to all others below this are flagged (default: 0.4).

    Returns
    -------
    bad_channels : dict
        Keys: 'abnormal_var', 'low_corr', 'flat'. Values: lists of channel names.
    diagnostics : dict
        Diagnostic data for plotting.
    """
    # Only process EEG channels
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    ch_names = [epochs.ch_names[p] for p in picks]
    data = epochs.get_data()[:, picks, :]  # (n_epochs, n_eeg, n_times)

    # 1. Abnormal variance (MAD-based z-score)
    ch_var = np.var(data, axis=(0, 2))  # variance per channel across all epochs+times
    median_var = np.median(ch_var)
    mad = np.median(np.abs(ch_var - median_var))
    if mad > 0:
        z_var = (ch_var - median_var) / (1.4826 * mad)  # 1.4826 = MAD-to-std factor
    else:
        z_var = np.zeros_like(ch_var)
    bad_var = [ch for ch, z in zip(ch_names, z_var) if np.abs(z) > z_thresh]

    # 2. Low correlation with all other channels
    mean_data = data.mean(axis=0)  # (n_eeg, n_times) — average across epochs
    corr_mat = np.corrcoef(mean_data)
    np.fill_diagonal(corr_mat, 0)
    mean_corr = corr_mat.mean(axis=1)
    bad_corr = [ch for ch, c in zip(ch_names, mean_corr) if c < corr_thresh]

    # 3. Flat channels: zero variance in >50% of epochs
    var_per_epoch = np.var(data, axis=2)  # (n_epochs, n_eeg)
    zero_var_frac = np.mean(var_per_epoch < 1e-30, axis=0)
    flat = [ch for ch, f in zip(ch_names, zero_var_frac) if f > 0.5]

    bad_channels = {
        'abnormal_var': bad_var,
        'low_corr': bad_corr,
        'flat': flat,
    }
    diagnostics = {
        'variances': ch_var,
        'z_scores': z_var,
        'mean_corr': mean_corr,
        'median_var': median_var,
        'channel_names': ch_names,
        'z_thresh': z_thresh,
        'corr_thresh': corr_thresh,
    }
    return bad_channels, diagnostics


def plot_channel_diagnostics(diagnostics, bad_channels, out_path):
    """Plot channel variance z-scores and mean correlation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ch_names = diagnostics['channel_names']
    n_ch = len(ch_names)
    all_bads = set(bad_channels['abnormal_var'] + bad_channels['low_corr'] + bad_channels['flat'])

    # Panel 1: Variance z-scores
    z_scores = diagnostics['z_scores']
    colors1 = ['red' if ch in bad_channels['abnormal_var'] else 'gray' for ch in ch_names]
    ax1.bar(range(n_ch), np.abs(z_scores), color=colors1, alpha=0.7, width=1.0)
    ax1.axhline(diagnostics['z_thresh'], color='red', linestyle='--',
                label=f"Z-threshold ({diagnostics['z_thresh']})")
    ax1.set_xlabel('Channel index')
    ax1.set_ylabel('|Z-score| (MAD)')
    ax1.set_title('Variance Z-scores (red = flagged)')
    ax1.legend(loc='upper right')

    # Panel 2: Mean correlation
    mean_corr = diagnostics['mean_corr']
    colors2 = ['red' if ch in bad_channels['low_corr'] else 'gray' for ch in ch_names]
    ax2.bar(range(n_ch), mean_corr, color=colors2, alpha=0.7, width=1.0)
    ax2.axhline(diagnostics['corr_thresh'], color='red', linestyle='--',
                label=f"Correlation threshold ({diagnostics['corr_thresh']})")
    ax2.set_xlabel('Channel index')
    ax2.set_ylabel('Mean correlation')
    ax2.set_title('Mean Correlation with All Channels (red = flagged)')
    ax2.legend(loc='lower right')

    fig.suptitle(f'Bad Channel Detection — {len(all_bads)} channels flagged', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# == MAIN ==
# Load epochs
fname = config.get('epochs') or config.get('fif') or config.get('mne')
if not fname:
    print("ERROR: No epochs file specified in config.json (key: 'epochs')")
    sys.exit(1)

# Brainlife may pass a directory instead of a file — resolve to meg-epo.fif inside it
if os.path.isdir(fname):
    fname = os.path.join(fname, 'meg-epo.fif')

if not os.path.exists(fname):
    print(f"ERROR: Epochs file not found: {fname!r}")
    sys.exit(1)

epochs = mne.read_epochs(fname, preload=True)
print(f"Loaded: {len(epochs)} epochs, {len(epochs.ch_names)} channels")

# Check for pre-existing bads from upstream (e.g. set during epoching or ICA)
existing_bads = list(epochs.info['bads'])
if existing_bads:
    print(f"Pre-existing bads in epochs.info['bads'] ({len(existing_bads)}): {existing_bads}")
else:
    print("No pre-existing bads in epochs.info['bads']")

# Parse config
z_thresh = float(config.get('z_thresh', 3.0))
corr_thresh = float(config.get('corr_thresh', 0.4))
extra_bads_str = config.get('extra_bads', '')
extra_bads = []
if extra_bads_str and extra_bads_str != 'None':
    extra_bads = [ch.strip() for ch in extra_bads_str.split(',')]
    extra_bads = [ch for ch in extra_bads if ch in epochs.ch_names]

# Detect bad channels
print(f"Detecting bad channels (z_thresh={z_thresh}, corr_thresh={corr_thresh})...")
bad_channels, diagnostics = detect_bad_channels(epochs, z_thresh, corr_thresh)

# Combine all bad channels (detected + extra user-specified)
all_bads = list(set(
    bad_channels['abnormal_var'] + bad_channels['low_corr'] + bad_channels['flat'] + extra_bads
))

# Report
print(f"\nDetected bad channels:")
print(f"  Abnormal variance ({len(bad_channels['abnormal_var'])}): {bad_channels['abnormal_var']}")
print(f"  Low correlation ({len(bad_channels['low_corr'])}): {bad_channels['low_corr']}")
print(f"  Flat ({len(bad_channels['flat'])}): {bad_channels['flat']}")
if extra_bads:
    print(f"  Extra (user-specified): {extra_bads}")
print(f"  Total newly detected: {len(all_bads)} bad channels")

# Plot diagnostics
var_plot = plot_channel_diagnostics(diagnostics, bad_channels, 'out_figs/channel_diagnostics.png')

# Apply: mark bad and interpolate
if all_bads:
    epochs.info['bads'] = list(set(existing_bads + all_bads))
    print(f"\nInterpolating {len(all_bads)} bad channels...")
    epochs.interpolate_bads(reset_bads=True)
    print("Interpolation complete.")
else:
    print("\nNo new bad channels detected. Passing through unchanged.")

# Save
out_path = os.path.join('out_dir', 'meg-epo.fif')
epochs.save(out_path, overwrite=True)
print(f"Saved: {out_path}")

# Product JSON
product_items = []
add_info_to_product(product_items,
    f"Bad channel detection (z_thresh={z_thresh}, corr_thresh={corr_thresh})")
if existing_bads:
    add_info_to_product(product_items,
        f"Pre-existing bads from upstream ({len(existing_bads)}): "
        f"{', '.join(existing_bads)}")
else:
    add_info_to_product(product_items, "Pre-existing bads from upstream: none")
add_info_to_product(product_items,
    f"Abnormal variance ({len(bad_channels['abnormal_var'])}): "
    f"{', '.join(bad_channels['abnormal_var']) or 'none'}")
add_info_to_product(product_items,
    f"Low correlation ({len(bad_channels['low_corr'])}): "
    f"{', '.join(bad_channels['low_corr']) or 'none'}")
add_info_to_product(product_items,
    f"Flat ({len(bad_channels['flat'])}): "
    f"{', '.join(bad_channels['flat']) or 'none'}")
if extra_bads:
    add_info_to_product(product_items,
        f"User-specified bads: {', '.join(extra_bads)}")
add_info_to_product(product_items,
    f"Total interpolated: {len(all_bads)}")
add_image_to_product(product_items, 'Channel Diagnostics', filepath=var_plot)

create_product_json(product_items)
print("Done!")

