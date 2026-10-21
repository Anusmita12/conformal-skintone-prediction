# Skin-tone-conditional conformal prediction for dermatology under dataset shift

Standard (marginal) conformal prediction guarantees e.g. 90% coverage *on
average*. In dermatology that average can hide under-coverage for darker skin
tones, and the guarantee breaks completely when a model trained on one
population is deployed on another. This repository studies:

1. **In-domain (DDI):** does marginal conformal prediction under-cover some
   Fitzpatrick skin-tone (FST) groups, and what does per-tone (Mondrian)
   calibration cost in set size?
2. **Shift (PAD-UFES-20 → DDI):** a malignancy classifier trained on
   PAD-UFES-20 (mostly FST I–III) and deployed on DDI (balanced FST I–VI).
   Does source-calibrated conformal prediction still hold per tone, and how
   many labelled target images per tone are needed to repair it?
3. **Tone-blind conformal prediction (ours):** keep "at least 1 − α of
   cancers caught in *every* skin tone" when the patient's skin tone is **not
   known at test time**. The tone-set union method splits the error budget
   α = α_label + β between a conformal set over skin tones and per-(tone,
   label) thresholds, giving a valid per-tone guarantee by a union bound.

## Repository layout

| File | Purpose |
|---|---|
| `conformal.py` | LAC, APS, Mondrian / class-conditional conformal, metrics |
| `tone_blind.py` | Tone-blind methods: tone-set union, tone-adaptive, plug-in, soft threshold, worst-case union |
| `extract_features.py` | DDI / PAD-UFES-20 loading and frozen-backbone embeddings (ResNet-50, DINOv2 ViT-S/14) |
| `extract_extra.py` | Fitzpatrick17k loading, MONET and DINOv2 ViT-B/14 backbones, ITA skin-colour features |
| `run_experiment.py` | Experiment 1: in-domain DDI, repeated random splits |
| `run_shift.py` | Experiment 2: PAD-UFES-20 → DDI shift and few-shot recalibration |
| `run_tone_blind.py` | Experiment 3: tone-blind conformal prediction under shift |
| `check_guarantee.py` | Synthetic sanity check of the guarantee (no data needed) |
| `download_fitz17k.py` | Downloads Fitzpatrick17k images from the URLs in its CSV |
| `run_all.ps1`, `run_tone_blind.ps1` | Run every experiment in the paper (PowerShell) |
| `results/` | Aggregated results (per-trial and summary CSVs, figures) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a CUDA GPU, install PyTorch from https://pytorch.org first.
`transformers` is only needed for the MONET backbone.

## Quick check (no data needed)

```bash
python check_guarantee.py --tone_signal 2
python check_guarantee.py --tone_signal 0.5
```

This simulates three skin-tone groups with known ground truth and checks which
methods keep ≥ 1 − α coverage in every group. It runs in seconds.

## Data

**No images, metadata or image-level features are included in this
repository.** The datasets are distributed under their own licences and must
be obtained from the original sources.

| Dataset | Role | Access | Licence |
|---|---|---|---|
| DDI (Diverse Dermatology Images) | target | [Stanford AIMI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images) (free account, research use agreement) | Stanford research use agreement, no redistribution |
| PAD-UFES-20 | source | [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | CC BY 4.0 |
| Fitzpatrick17k | tone estimator / second target | [github.com/mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) | annotations CC BY-NC-SA 3.0; images from their original sources |

Place the data as follows (the `data/` folder is git-ignored):

```
data/
├── ddi/            # unzipped DDI: ddi_metadata.csv + images
├── pad_ufes/       # unzipped PAD-UFES-20: metadata.csv + images
└── fitz17k/        # fitzpatrick17k.csv (+ images/ after running the script below)
```

Fitzpatrick17k images are fetched from their original URLs:

```bash
python download_fitz17k.py
```

Some URLs are no longer reachable, so the number of downloaded images may be
lower than in the original release.

## Reproducing the paper

The PowerShell scripts run everything end to end and skip steps whose output
already exists:

```bash
powershell -ExecutionPolicy Bypass -File run_all.ps1          # experiments 1 and 2
powershell -ExecutionPolicy Bypass -File run_tone_blind.ps1   # experiment 3
```

Equivalent individual commands (Linux / macOS):

```bash
# 1. features (frozen backbones)
python extract_features.py --dataset ddi      --backbone dinov2_vits14
python extract_features.py --dataset ddi      --backbone resnet50
python extract_features.py --dataset pad_ufes --backbone dinov2_vits14
python extract_features.py --dataset pad_ufes --backbone resnet50
python extract_features.py --dataset pad_ufes --backbone dinov2_vits14 --ack exclude

# 2. in-domain DDI
python run_experiment.py --backbone dinov2_vits14 --alpha 0.1 --n_trials 100 --out_dir results/alpha_0.1

# 3. shift PAD-UFES-20 -> DDI
python run_shift.py --backbone dinov2_vits14 --alpha 0.1 --n_trials 100

# 4. tone-blind conformal prediction
python run_tone_blind.py --backbone dinov2_vits14 --alpha 0.1
```

| Result | Command | Output |
|---|---|---|
| In-domain DDI coverage per tone | `run_experiment.py --alpha {0.05,0.1,0.2}` | `results/alpha_*/dinov2_vits14/` |
| Shift: source vs target calibration | `run_shift.py --alpha {0.05,0.1,0.2}` | `results/shift/dinov2_vits14/alpha_*/shift_coverage.png` |
| Few-shot recalibration | `run_shift.py` | `results/shift/dinov2_vits14/alpha_*/fewshot.png` |
| Backbone robustness (ResNet-50) | `run_shift.py --backbone resnet50` | `results/shift/resnet50/` |
| Label robustness (AK excluded) | `run_shift.py --source pad_ufes_ackexcl` | `results/shift/dinov2_vits14_pad_ufes_ackexcl/` |
| Tone-blind methods | `run_tone_blind.py` | `results/tone_blind/*/alpha_0.1/main.png`, `summary.csv` |
| β sweep | `run_tone_blind.py` | `results/tone_blind/*/alpha_0.1/beta_sweep.png` |

All experiments average over 100 random splits; summaries report mean ± std.

## Limitations

- DDI has 656 images from 570 patients and no patient ID in the metadata, so a
  few patients may appear in both calibration and test splits.
- Per-tone (and per tone × label) calibration sets in DDI are small, so
  per-group coverage has high variance across splits.
- Fitzpatrick types in DDI are clinician-assigned and grouped into three bins
  (I–II, III–IV, V–VI).

## Datasets and models used

Please cite the original work if you use this code:

- DDI: Daneshjou et al., "Disparities in dermatology AI performance on a diverse, curated clinical image set," *Science Advances*, 2022.
- PAD-UFES-20: Pacheco et al., "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones," *Data in Brief*, 2020.
- Fitzpatrick17k: Groh et al., "Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset," CVPR Workshops, 2021.
- DINOv2: Oquab et al., "DINOv2: Learning robust visual features without supervision," *TMLR*, 2024.
- MONET: Kim et al., "Transparent medical image AI via an image–text foundation model grounded in medical literature," *Nature Medicine*, 2024.

## Licence

Code is released under the MIT licence (see `LICENSE`). The licence covers
the code only, not the datasets.
