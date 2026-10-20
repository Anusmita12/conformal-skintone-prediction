# Runs every experiment for the paper. From the repo root, with the venv active:
#     powershell -ExecutionPolicy Bypass -File run_all.ps1
# Feature extraction is skipped when the features already exist.
# Results land in results/ (in-domain DDI) and results/shift/ (PAD-UFES -> DDI).

$ErrorActionPreference = "Stop"
$alphas = @(0.05, 0.1, 0.2)

function Extract($dataset, $backbone, $file, $extra = @()) {
    if (Test-Path "features\$file") {
        Write-Host "[skip] features\$file already exists"
    } else {
        python extract_features.py --dataset $dataset --backbone $backbone @extra
        if ($LASTEXITCODE -ne 0) { throw "extract_features failed ($dataset, $backbone)" }
    }
}

function Run($script, $argList) {
    Write-Host "`n>>> python $script $($argList -join ' ')" -ForegroundColor Cyan
    python $script @argList
    if ($LASTEXITCODE -ne 0) { throw "$script failed" }
}

# ---------------- 1. features ----------------
Extract "ddi"      "dinov2_vits14" "features_dinov2_vits14.npy"
Extract "ddi"      "resnet50"      "features_resnet50.npy"
Extract "pad_ufes" "dinov2_vits14" "features_pad_ufes_dinov2_vits14.npy"
Extract "pad_ufes" "resnet50"      "features_pad_ufes_resnet50.npy"
Extract "pad_ufes" "dinov2_vits14" "features_pad_ufes_ackexcl_dinov2_vits14.npy" @("--ack", "exclude")

# ---------------- 2. in-domain DDI baseline (no shift) ----------------
foreach ($a in $alphas) {
    Run "run_experiment.py" @("--backbone", "dinov2_vits14", "--alpha", $a, "--n_trials", 100, "--out_dir", "results/alpha_$a")
}

# ---------------- 3. shift: PAD-UFES -> DDI (main results) ----------------
foreach ($a in $alphas) {
    Run "run_shift.py" @("--backbone", "dinov2_vits14", "--alpha", $a, "--n_trials", 100)
}

# ---------------- 4. robustness checks (alpha = 0.1) ----------------
Run "run_shift.py" @("--backbone", "resnet50", "--alpha", 0.1, "--n_trials", 100)
Run "run_shift.py" @("--backbone", "dinov2_vits14", "--source", "pad_ufes_ackexcl", "--alpha", 0.1, "--n_trials", 100)

Write-Host "`nAll experiments finished." -ForegroundColor Green