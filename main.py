"""
app-detect-bad-channels-eeg-v2: Detect bad EEG channels in epoched data.

Marks detected channels in info['bads'] and saves epochs unchanged.
Runs BEFORE ICA so bad channels are known before decomposition.
Uses variance (MAD z-score) and flat channel detection.

Inputs:  epochs FIF file
Outputs: out_dir/meg-epo.fif, out_figs/*.png, out_report/report.html, product.json
"""

# Copyright (c) 2026 brainlife.io

import sys
import os
import base64
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'brainlife_utils'))

import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from brainlife_utils import (
    load_config,
    setup_matplotlib_backend,
    ensure_output_dirs,
)

setup_matplotlib_backend()
config = load_config()
ensure_output_dirs('out_dir', 'out_figs', 'out_report')

# ── Config ────────────────────────────────────────────────────────────────────
fname = config.get('epo') or config.get('epochs') or config.get('fif') or config.get('mne')

if not fname:
    print("ERROR: No epochs file specified in config.json (key: 'epochs')")
    sys.exit(1)

if os.path.isdir(fname):
    fname = os.path.join(fname, 'meg-epo.fif')

if not os.path.exists(fname):
    print(f"ERROR: Epochs file not found: {fname!r}")
    sys.exit(1)

var_thresh     = float(config.get('var_thresh', 5.0))   # noisy: var > N × median
extra_bads_str = config.get('extra_bads', '') or ''
extra_bads = []
if extra_bads_str and extra_bads_str != 'None':
    extra_bads = [ch.strip() for ch in extra_bads_str.split(',')]

# ── Load epochs ───────────────────────────────────────────────────────────────
epochs = mne.read_epochs(fname, preload=True)
print(f"Loaded: {len(epochs)} epochs, {len(epochs.ch_names)} channels")

existing_bads = list(epochs.info['bads'])
if existing_bads:
    print(f"Pre-existing bads in info['bads'] ({len(existing_bads)}): {existing_bads}")
else:
    print("No pre-existing bads in info['bads']")

extra_bads = [ch for ch in extra_bads if ch in epochs.ch_names]

# ── Detection ─────────────────────────────────────────────────────────────────
picks    = mne.pick_types(epochs.info, eeg=True, exclude=[])
ch_names = [epochs.ch_names[p] for p in picks]
data_arr = epochs.get_data()[:, picks, :]   # (n_epochs, n_eeg, n_times)

# Use baseline only (tmin to 0s) for variance — avoids flagging channels with
# strong evoked responses as bad (e.g. occipital channels in visual tasks).
baseline_mask = epochs.times <= 0
data_baseline = data_arr[:, :, baseline_mask]   # (n_epochs, n_eeg, n_baseline_times)

# 1. Variance — MAD z-score on baseline
ch_var     = np.var(data_baseline, axis=(0, 2))
median_var = np.median(ch_var)
mad        = np.median(np.abs(ch_var - median_var))
z_var      = (ch_var - median_var) / (1.4826 * mad) if mad > 0 else np.zeros_like(ch_var)
bad_var    = [ch for ch, z in zip(ch_names, z_var) if np.abs(z) > var_thresh]

# 2. Flat channels — zero variance in >50% of epochs (baseline only)
var_per_epoch = np.var(data_baseline, axis=2)   # (n_epochs, n_eeg)
zero_var_frac = np.mean(var_per_epoch < 1e-30, axis=0)
bad_flat      = [ch for ch, f in zip(ch_names, zero_var_frac) if f > 0.5]

all_detected = list(set(bad_var + bad_flat + extra_bads))

print(f"\nDetected bad channels:")
print(f"  MAD z-score ({len(bad_var)}): {bad_var}")
print(f"  Flat        ({len(bad_flat)}): {bad_flat}")
if extra_bads:
    print(f"  User-specified ({len(extra_bads)}): {extra_bads}")
print(f"  Total newly detected: {len(all_detected)}")

# ── Plot 1: Variance z-scores + topomap ───────────────────────────────────────
bad_set = set(bad_var + bad_flat)
fig_var, (ax_bar, ax_topo) = plt.subplots(
    1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [4, 1]}
)

colors = ['red' if ch in bad_set else 'steelblue' for ch in ch_names]
ax_bar.bar(range(len(ch_names)), np.abs(z_var), color=colors, alpha=0.8, width=1.0)
ax_bar.axhline(var_thresh, color='red', linestyle='--', linewidth=1.2,
               label=f'Z-threshold ({var_thresh})')
ax_bar.set_yscale('log')
ax_bar.set_xlabel('Channel index')
ax_bar.set_ylabel('|Z-score| (MAD, log scale)')
ax_bar.set_title(f'Variance Z-scores — {len(bad_set)} flagged')
ax_bar.legend(loc='upper right', fontsize=8)

try:
    info_tmp = epochs.info.copy()
    info_tmp['bads'] = list(bad_set)
    mne.viz.plot_sensors(info_tmp, ch_type='eeg', axes=ax_topo, show=False, show_names=False)
    ax_topo.set_aspect('equal', adjustable='datalim')
    ax_topo.set_title(f'EEG sensors\n({len(bad_set)} bad)', fontsize=9)
    ax_topo.legend(handles=[
        Patch(facecolor='steelblue', label='Good'),
        Patch(facecolor='red', label=f'Bad ({len(bad_set)})'),
    ], loc='lower center', fontsize=7, framealpha=0.8)
except Exception as e:
    ax_topo.axis('off')
    ax_topo.text(0.5, 0.5, f'No topomap\n{e}', ha='center', va='center',
                 transform=ax_topo.transAxes, fontsize=7)
    print(f"Could not draw topomap: {e}")

fig_var.suptitle('Bad Channel Detection — Variance & Flat', fontsize=13)
fig_var.tight_layout()
var_path = os.path.join('out_figs', 'variance_zscores.png')
fig_var.savefig(var_path, dpi=150, bbox_inches='tight')
plt.close(fig_var)

# ── Plot 2: Time traces of detected bad channels ───────────────────────────────
traces_path = None
if all_detected:
    times = epochs.times
    mean_epoch = data_arr.mean(axis=0)   # (n_eeg, n_times) — mean across epochs
    bad_indices = [ch_names.index(ch) for ch in all_detected if ch in ch_names]

    n_bad = len(bad_indices)
    ncols = min(4, n_bad)
    nrows = int(np.ceil(n_bad / ncols))
    fig_tr, axes_tr = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows),
                                   squeeze=False)

    for i, idx in enumerate(bad_indices):
        row, col = divmod(i, ncols)
        ax = axes_tr[row][col]
        ax.plot(times, mean_epoch[idx] * 1e6, color='red', linewidth=0.8)
        ax.set_title(ch_names[idx], fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=7)
        ax.set_ylabel('μV', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--')

    # hide unused subplots
    for i in range(n_bad, nrows * ncols):
        row, col = divmod(i, ncols)
        axes_tr[row][col].axis('off')

    fig_tr.suptitle(f'Mean Epoch Traces — {n_bad} Detected Bad Channels', fontsize=12)
    fig_tr.tight_layout()
    traces_path = os.path.join('out_figs', 'bad_channel_traces.png')
    fig_tr.savefig(traces_path, dpi=150, bbox_inches='tight')
    plt.close(fig_tr)

# ── Mark bads in info — no interpolation ─────────────────────────────────────
epochs.info['bads'] = list(set(existing_bads + all_detected))
if all_detected:
    print(f"\nMarked {len(all_detected)} channels as bad in info['bads']")
else:
    print("\nNo new bad channels detected.")

# ── Save FIF ──────────────────────────────────────────────────────────────────
out_path = os.path.join('out_dir', 'meg-epo.fif')
epochs.save(out_path, overwrite=True)
print(f"Saved: {out_path}")

# ── MNE Report ────────────────────────────────────────────────────────────────
report = mne.Report(title='Bad Channel Detection Report')
if os.path.exists(var_path):
    report.add_image(var_path, title='Variance Z-scores (MAD)')
if traces_path and os.path.exists(traces_path):
    report.add_image(traces_path, title='Bad Channel Traces')
report.save(os.path.join('out_report', 'report.html'), overwrite=True)

# ── product.json ──────────────────────────────────────────────────────────────
product = {'brainlife': []}

def _info(msg):
    product['brainlife'].append({'type': 'info', 'msg': msg})

_info(f"Bad channel detection (var_thresh={var_thresh})")
if existing_bads:
    _info(f"Pre-existing bads from upstream ({len(existing_bads)}): {', '.join(existing_bads)}")
else:
    _info("Pre-existing bads from upstream: none")
_info(f"MAD z-score ({len(bad_var)}): {', '.join(bad_var) or 'none'}")
_info(f"Flat        ({len(bad_flat)}): {', '.join(bad_flat) or 'none'}")
if extra_bads:
    _info(f"User-specified bads: {', '.join(extra_bads)}")
_info(f"Total marked as bad: {len(all_detected)}")

for img_name, img_path in [
    ('Variance Z-scores', var_path),
    ('Bad Channel Traces', traces_path),
]:
    if img_path and os.path.exists(img_path):
        data_uri = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
        product['brainlife'].append({'type': 'image/png', 'name': img_name, 'base64': data_uri})

with open('product.json', 'w') as f:
    json.dump(product, f)

print("Done!")
