# Detect Bad EEG Channels (v2)

[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)

Brainlife App to detect bad EEG channels in epoched data using variance-based methods. Marks detected channels in `info['bads']` and saves epochs unchanged (no interpolation). Designed to run **before ICA** so bad channels are known before decomposition.

Detection uses **baseline-only variance** (pre-stimulus window, `tmin` to 0 s) to avoid falsely flagging channels with strong evoked responses (e.g., occipital channels in visual tasks).

---

## Detection Methods

| Method | Description |
|--------|-------------|
| **MAD z-score** | Channel variance z-scored using the median absolute deviation. Flags channels with `|z| > z_thresh` (default 5). |
| **Dead channels** | Channel variance < median variance / 100. Identifies physically disconnected or dead electrodes. |
| **Flat channels** | Zero variance (< 1e-30) in > 50% of epochs. Identifies completely flat/disconnected channels. |
| **User-specified** | Additional channels passed via `extra_bads` config key. |

Bad channels are **marked in `info['bads']`** — the epochs file is saved with bads marked but channels are not removed or interpolated. Interpolation should be done after ICA, once the full bad channel list is finalised.

---

## Inputs

| Key | Type | Description |
|-----|------|-------------|
| `epo` | neuro/eeg/mne | Epoched EEG FIF file (output of app-epoch or similar) |

---

## Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `z_thresh` | number | 5.0 | MAD z-score threshold for noisy channel detection. Higher = more conservative (fewer flagged). Typical range: 5–10. |
| `extra_bads` | string | `""` | Comma-separated list of channel names to mark as bad regardless of detection (e.g. `"A13,B5"`). |

---

## Outputs

| Directory | File | Description |
|-----------|------|-------------|
| `out_dir/` | `meg-epo.fif` | Epochs with bad channels marked in `info['bads']` |
| `out_figs/` | `variance_zscores.png` | Bar chart of per-channel MAD z-scores + sensor topomap with bad channels highlighted |
| `out_figs/` | `bad_channel_traces.png` | Mean epoch time traces for each detected bad channel (only if bads found) |
| `out_report/` | `report.html` | MNE HTML report with all figures |
| `product.json` | — | Brainlife UI summary with detection counts and base64 figure thumbnails |

---

## Usage Notes

- **Run after bandpass filtering and epoching, before ICA.**
- The app reads `z_thresh` or `var_thresh` config keys (both accepted for backwards compatibility).
- For BioSemi data without an average reference, correlation-based detection is intentionally excluded — it produces false positives on unreferenced data.
- Baseline window is `tmin` to `0 s`. If your epochs have no pre-stimulus period, all variance is used.
- Sub-threshold bad channels in BIDS `channels.tsv` may not be detected at `z_thresh=10`. Lower to 5–7 for more sensitive detection.

---

## Authors

- Maximilien Chaumon (maximilien.chaumon@icm-institute.org)

## Citations

*MNE-Python:*

Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Goj R, Jas M, Brooks T, Parkkonen L, and Hämäläinen MS. **MEG and EEG data analysis with MNE-Python**. Frontiers in Neuroscience, 7(267):1–13, 2013. https://doi.org/10.3389/fnins.2013.00267

*Brainlife.io:*

Avesani, P., McPherson, B., Hayashi, S. et al. **The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services**. Sci Data 6, 69 (2019). https://doi.org/10.1038/s41597-019-0073-y

## Funding Acknowledgement

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://grantome.com/grant/NIH/R01-EB030896-01)

#### MIT Copyright (c) 2026 brainlife.io The University of Texas at Austin and Indiana University
