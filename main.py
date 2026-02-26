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

z_thresh         = float(config.get('z_thresh', 5.0))
var_median_thresh = float(config.get('var_median_thresh', 5.0))
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

# 1. Variance — MAD z-score
ch_var     = np.var(data_arr, axis=(0, 2))
median_var = np.median(ch_var)
mad        = np.median(np.abs(ch_var - median_var))
z_var      = (ch_var - median_var) / (1.4826 * mad) if mad > 0 else np.zeros_like(ch_var)
bad_var    = [ch for ch, z in zip(ch_names, z_var) if np.abs(z) > z_thresh]

# 2. Variance — N× median threshold
bad_var_median = [ch for ch, v in zip(ch_names, ch_var) if v > var_median_thresh * median_var]

# 3. Flat channels — zero variance in >50% of epochs
var_per_epoch = np.var(data_arr, axis=2)   # (n_epochs, n_eeg)
zero_var_frac = np.mean(var_per_epoch < 1e-30, axis=0)
bad_flat      = [ch for ch, f in zip(ch_names, zero_var_frac) if f > 0.5]

all_detected = list(set(bad_var + bad_var_median + bad_flat + extra_bads))

print(f"\nDetected bad channels:")
print(f"  MAD z-score      ({len(bad_var)}): {bad_var}")
print(f"  N×median         ({len(bad_var_median)}): {bad_var_median}")
print(f"  Flat             ({len(bad_flat)}): {bad_flat}")
if extra_bads:
    print(f"  User-specified   ({len(extra_bads)}): {extra_bads}")
print(f"  Total newly detected: {len(all_detected)}")

# ── Plot: Variance z-scores + topomap ─────────────────────────────────────────
bad_set = set(bad_var + bad_flat)
fig_var, (ax_bar, ax_topo) = plt.subplots(
    1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [4, 1]}
)

colors = ['red' if ch in bad_set else 'steelblue' for ch in ch_names]
ax_bar.bar(range(len(ch_names)), np.abs(z_var), color=colors, alpha=0.8, width=1.0)
ax_bar.axhline(z_thresh, color='red', linestyle='--', linewidth=1.2,
               label=f'Z-threshold ({z_thresh})')
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

# ── Plot 2: Raw variance + N×median threshold + topomap ───────────────────────
bad_median_set = set(bad_var_median)
fig_med, (ax_med, ax_topo_med) = plt.subplots(
    1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [4, 1]}
)

colors_med = ['red' if ch in bad_median_set else 'steelblue' for ch in ch_names]
ax_med.bar(range(len(ch_names)), ch_var * 1e12, color=colors_med, alpha=0.8, width=1.0)
ax_med.axhline(var_median_thresh * median_var * 1e12, color='red', linestyle='--',
               linewidth=1.2, label=f'{var_median_thresh}× median')
ax_med.set_yscale('log')
ax_med.set_xlabel('Channel index')
ax_med.set_ylabel('Variance (μV², log scale)')
ax_med.set_title(f'Raw Variance — {len(bad_median_set)} flagged')
ax_med.legend(loc='upper right', fontsize=8)

try:
    info_tmp_med = epochs.info.copy()
    info_tmp_med['bads'] = list(bad_median_set)
    mne.viz.plot_sensors(info_tmp_med, ch_type='eeg', axes=ax_topo_med, show=False, show_names=False)
    ax_topo_med.set_aspect('equal', adjustable='datalim')
    ax_topo_med.set_title(f'EEG sensors\n({len(bad_median_set)} bad)', fontsize=9)
    ax_topo_med.legend(handles=[
        Patch(facecolor='steelblue', label='Good'),
        Patch(facecolor='red', label=f'Bad ({len(bad_median_set)})'),
    ], loc='lower center', fontsize=7, framealpha=0.8)
except Exception as e:
    ax_topo_med.axis('off')
    ax_topo_med.text(0.5, 0.5, f'No topomap\n{e}', ha='center', va='center',
                     transform=ax_topo_med.transAxes, fontsize=7)
    print(f"Could not draw median topomap: {e}")

fig_med.suptitle(f'Bad Channel Detection — N×Median Variance ({var_median_thresh}×)', fontsize=13)
fig_med.tight_layout()
med_path = os.path.join('out_figs', 'variance_median.png')
fig_med.savefig(med_path, dpi=150, bbox_inches='tight')
plt.close(fig_med)

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
if os.path.exists(med_path):
    report.add_image(med_path, title=f'Variance N×Median ({var_median_thresh}×)')
report.save(os.path.join('out_report', 'report.html'), overwrite=True)

# ── product.json ──────────────────────────────────────────────────────────────
product = {'brainlife': []}

def _info(msg):
    product['brainlife'].append({'type': 'info', 'msg': msg})

_info(f"Bad channel detection (z_thresh={z_thresh}, var_median_thresh={var_median_thresh})")
if existing_bads:
    _info(f"Pre-existing bads from upstream ({len(existing_bads)}): {', '.join(existing_bads)}")
else:
    _info("Pre-existing bads from upstream: none")
_info(f"MAD z-score      ({len(bad_var)}): {', '.join(bad_var) or 'none'}")
_info(f"N×median         ({len(bad_var_median)}): {', '.join(bad_var_median) or 'none'}")
_info(f"Flat             ({len(bad_flat)}): {', '.join(bad_flat) or 'none'}")
if extra_bads:
    _info(f"User-specified bads: {', '.join(extra_bads)}")
_info(f"Total marked as bad: {len(all_detected)}")

for img_name, img_path in [
    ('Variance Z-scores (MAD)', var_path),
    (f'Variance N×Median ({var_median_thresh}×)', med_path),
]:
    if os.path.exists(img_path):
        data_uri = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
        product['brainlife'].append({'type': 'image/png', 'name': img_name, 'base64': data_uri})

with open('product.json', 'w') as f:
    json.dump(product, f)

print("Done!")
