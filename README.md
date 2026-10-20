# conformal-skintone-prediction

Group-conditional conformal prediction for dermatology images across
Fitzpatrick skin types, on the **DDI (Diverse Dermatology Images)** dataset.

**Question:** standard (marginal) conformal prediction guarantees e.g. 90%
coverage *on average*. Does that average hide under-coverage for some skin
tones, and what does calibrating per skin-tone group (Mondrian conformal)
cost in prediction-set size?

## Tasks
| Task | Label | Groups for evaluation | Methods |
|---|---|---|---|
| `skin_tone` | FST I-II / III-IV / V-VI | true skin tone | LAC, APS, class-conditional LAC |
| `malignancy` | benign / malignant | skin tone | LAC, APS, Mondrian LAC (per tone) |

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```
For a CUDA GPU, install torch from https://pytorch.org first.

## Data
Download DDI from Stanford AIMI (requires a free account) and unzip it into
`data/ddi/` so that `ddi_metadata.csv` and the images are somewhere inside it.
The data folder is git-ignored. Do not commit images.

## Run
```bash
# 1. embed all images once with a frozen pretrained backbone
python extract_features.py --ddi_dir data/ddi --backbone resnet50
python extract_features.py --ddi_dir data/ddi --backbone dinov2_vits14   # optional, usually stronger

# 2. conformal experiments over 50 random train/cal/test splits (50/25/25)
python run_experiment.py --backbone resnet50
python run_experiment.py --backbone dinov2_vits14 --alpha 0.1 --n_trials 100
```
Results go to `results/<backbone>/`: `trials.csv`, `summary.csv` (mean ± std
of coverage and set size per method x skin-tone group) and coverage/set-size plots.

## Files
- `conformal.py` - LAC, APS, Mondrian / class-conditional conformal, metrics
- `extract_features.py` - DDI loading + frozen-backbone embeddings
- `run_experiment.py` - repeated-split experiments, tables, plots

## Notes / limitations
- DDI has 656 images from 570 patients. The metadata has no patient ID, so a
  few patients may appear in both calibration and test splits.
- Per-group calibration sets are small (~50 images), so per-group coverage
  has high variance. Always report mean ± std over many splits.