<h1 align="center">app-detect-bad-channels-eeg-v2</h1>

<p align="center">
  Detects bad EEG channels in epoched data from baseline variance (MAD z-score, dead and flat channels) with MNE-Python and marks them in info['bads'] without interpolation.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.897"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.897-blue.svg" alt="Run on brainlife.io"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Detects bad EEG channels in epoched data from baseline variance (MAD z-score, dead and flat channels) with MNE-Python and marks them in info['bads'] without interpolation.**

Designed to run before ICA: variance is computed in the pre-stimulus window to avoid flagging channels with strong evoked responses. Channels are flagged if their MAD z-score exceeds z_thresh, their variance is under 1/100 of the median, or they are flat in over half of the epochs; extra channels can be added by name. Epochs are saved unchanged except for the bads list; interpolate after ICA.

| | |
|---|---|
| **Inputs** | `neuro/meeg/mne/epochs` |
| **Outputs** | `neuro/meeg/mne/epochs` |
| **Container** | `brainlife/mne:1.0.2` |
| **Resources** | 1 CPUs · 20 GB memory · 0.5 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.897](https://doi.org/10.25663/brainlife.app.897) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `epochs` | `neuro/meeg/mne/epochs` | yes | Epoched EEG data (MNE -epo.fif). |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `out_dir` | `neuro/meeg/mne/epochs` | The same epochs with detected bad channels listed in info['bads']. |

<details>
<summary><b>Parameters</b> (2)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `z_thresh` | number | `5` | MAD z-score threshold. Channels flagged if variance deviates more than this many robust SDs from median |
| `extra_bads` | string |  | Comma-separated channel names to mark bad in addition to those detected. |

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.897](https://doi.org/10.25663/brainlife.app.897), choose `neuro/meeg/mne/epochs` from your project, set the parameters and press **Execute**. The results are saved to your project as `neuro/meeg/mne/epochs`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `brainlife/mne:1.0.2`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "epo": "/path/to/epochs/epo",
       "z_thresh": 5,
       "extra_bads": ""
   }
   ```

3. Run `./main`.

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- app-detect-bad-channels-eeg-v2. brainlife.io App. [doi:10.25663/brainlife.app.897](https://doi.org/10.25663/brainlife.app.897)
- Gramfort et al. MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience* 7 (2013). [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

Maintained by [@obVdo](https://github.com/obVdo).
