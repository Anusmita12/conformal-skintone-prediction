# Tone-blind experiments. From the repo root, with the venv active:
#     powershell -ExecutionPolicy Bypass -File run_tone_blind.ps1
# Needs DDI + PAD-UFES features from run_all.ps1.
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

# ---------- 1. PAD-UFES -> DDI, tone unknown at test time ----------
Run "run_tone_blind.py" @("--backbone", $bb, "--alpha", 0.1)

# ---------- 2. dermatology backbone (needs: pip install transformers) ----------
Once "features\features_monet.npy"          "extract_extra.py" @("--dataset", "ddi", "--backbone", "monet")
Once "features\features_pad_ufes_monet.npy" "extract_extra.py" @("--dataset", "pad_ufes", "--backbone", "monet")
Run "run_tone_blind.py" @("--backbone", "monet", "--alpha", 0.1)

Write-Host "`nTone-blind experiments finished. Results in results\tone_blind\" -ForegroundColor Green
