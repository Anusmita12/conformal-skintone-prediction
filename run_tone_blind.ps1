# Tone-blind experiments. From the repo root, with the venv active:
#     powershell -ExecutionPolicy Bypass -File run_tone_blind.ps1
# Needs DDI + PAD-UFES features from run_all.ps1, and Fitzpatrick17k in data/fitz17k.
# Steps whose output already exists are skipped.

$ErrorActionPreference = "Stop"
$bb = "dinov2_vits14"

function Run($script, $argList) {
    Write-Host "`n>>> python $script $($argList -join ' ')" -ForegroundColor Cyan
    python $script @argList
    if ($LASTEXITCODE -ne 0) { throw "$script failed" }
}
function Once($file, $script, $argList) {
    if (Test-Path $file) { Write-Host "[skip] $file exists" } else { Run $script $argList }
}

# ---------- 0. sanity check on synthetic data ----------
Run "check_guarantee.py" @("--tone_signal", 2)
Run "check_guarantee.py" @("--tone_signal", 0.5)

# ---------- 1. extra features ----------
Once "features\ita_.csv"                       "extract_extra.py" @("--dataset", "ddi", "--ita")
Once "features\features_fitz17k_all_$bb.npy"   "extract_extra.py" @("--dataset", "fitz17k", "--data_dir", "data/fitz17k", "--backbone", $bb, "--keep_non_neoplastic")
Once "features\ita_fitz17k_all_.csv"           "extract_extra.py" @("--dataset", "fitz17k", "--data_dir", "data/fitz17k", "--ita", "--keep_non_neoplastic")
Once "features\features_fitz17k_$bb.npy"       "extract_extra.py" @("--dataset", "fitz17k", "--data_dir", "data/fitz17k", "--backbone", $bb)
Once "features\ita_fitz17k_.csv"               "extract_extra.py" @("--dataset", "fitz17k", "--data_dir", "data/fitz17k", "--ita")

# ---------- 2. PAD-UFES -> DDI, tone unknown at test time ----------
foreach ($a in @(0.05, 0.1, 0.2)) {
    Run "run_tone_blind.py" @("--backbone", $bb, "--alpha", $a)
}
# tone model variants (alpha = 0.1)
Run "run_tone_blind.py" @("--backbone", $bb, "--tone_features", "ita")
Run "run_tone_blind.py" @("--backbone", $bb, "--tone_train", "fitz17k", "--tone_features", "both")
# per-tone guarantee only (bigger cells, use if tone x label cells are too small)
Run "run_tone_blind.py" @("--backbone", $bb, "--cells", "tone")

# ---------- 3. second target: PAD-UFES -> Fitzpatrick17k (large calibration) ----------
Run "run_tone_blind.py" @("--backbone", $bb, "--target", "fitz17k_", "--tone_features", "both")

# ---------- 4. optional: dermatology backbone (needs: pip install transformers) ----------
# Once "features\features_monet.npy"          "extract_extra.py" @("--dataset", "ddi", "--backbone", "monet")
# Once "features\features_pad_ufes_monet.npy" "extract_extra.py" @("--dataset", "pad_ufes", "--backbone", "monet")
# Run "run_tone_blind.py" @("--backbone", "monet")

Write-Host "`nTone-blind experiments finished. Results in results\tone_blind\" -ForegroundColor Green
