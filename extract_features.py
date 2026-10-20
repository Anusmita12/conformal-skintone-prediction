"""
Step 1 - Extract image embeddings with a frozen pretrained backbone.

Supported datasets
  ddi       Diverse Dermatology Images (656 images, biopsy-proven, balanced FST)
  pad_ufes  PAD-UFES-20 (2,298 smartphone images, Brazil, mostly FST I-III)

Small datasets -> we embed every image once and train lightweight classifiers
on the embeddings. Fast on a laptop and cheap to repeat over many splits.

Usage (from the repo root):
    python extract_features.py --dataset ddi      --data_dir data/ddi
    python extract_features.py --dataset pad_ufes --data_dir data/pad_ufes
    (add --backbone dinov2_vits14 for DINOv2 features)

Outputs (in --out_dir, default features/):
    DDI:       features_<backbone>.npy,          meta_<backbone>.csv
    PAD-UFES:  features_pad_ufes_<backbone>.npy, meta_pad_ufes_<backbone>.csv
Meta columns: file, skin_tone (12/34/56, 0 = unknown), malignant (0/1),
              disease, patient_id
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# PAD-UFES-20 diagnoses: BCC, SCC (incl. Bowen's), MEL are skin cancers;
# NEV, SEK are benign; ACK (actinic keratosis) is pre-cancerous -> see --ack
PAD_MALIGNANT = {"BCC", "SCC", "MEL"}
PAD_BENIGN = {"NEV", "SEK"}


def index_images(root: Path) -> dict:
    """file name -> full path, searching all sub-folders."""
    return {p.name: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}


def find_csv(root: Path, pattern: str) -> Path:
    csvs = sorted(root.rglob(pattern))
    if not csvs:
        raise FileNotFoundError(f"No '{pattern}' found under {root}.")
    return csvs[0]


def attach_paths(meta: pd.DataFrame, root: Path) -> pd.DataFrame:
    paths = index_images(root)
    meta["path"] = meta["file"].map(paths)
    n_missing = meta["path"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} images listed in metadata were not found; skipping them.")
        meta = meta.dropna(subset=["path"]).reset_index(drop=True)
    return meta


def fst_to_group(v):
    """Fitzpatrick 1-6 -> DDI-style group 12 / 34 / 56 (0 if unknown)."""
    try:
        v = int(float(v))
    except (TypeError, ValueError):
        return 0
    return {1: 12, 2: 12, 3: 34, 4: 34, 5: 56, 6: 56}.get(v, 0)


# ---------------------------------------------------------------------------
# Dataset loaders -> common metadata format
# ---------------------------------------------------------------------------
def load_ddi(root: Path) -> pd.DataFrame:
    csv = find_csv(root, "*metadata*.csv")
    raw = pd.read_csv(csv)
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    print(f"Metadata: {csv}  ({len(raw)} rows)")
    for col in ["DDI_file", "skin_tone", "malignant"]:
        if col not in raw.columns:
            raise ValueError(f"DDI metadata missing '{col}'. Found: {list(raw.columns)}")
    meta = pd.DataFrame({
        "file": raw["DDI_file"],
        "skin_tone": raw["skin_tone"].astype(int),
        "malignant": raw["malignant"].astype(str).str.lower().isin(["true", "1"]).astype(int),
        "disease": raw.get("disease", ""),
        "patient_id": raw.get("DDI_ID", raw.index),   # DDI has no patient id
    })
    return attach_paths(meta, root)


def load_pad_ufes(root: Path, ack: str) -> pd.DataFrame:
    csv = find_csv(root, "metadata*.csv")
    raw = pd.read_csv(csv)
    raw.columns = raw.columns.str.strip().str.lower()
    print(f"Metadata: {csv}  ({len(raw)} rows)")
    fst_col = "fitspatrick" if "fitspatrick" in raw.columns else "fitzpatrick"
    for col in ["img_id", "diagnostic", "patient_id", fst_col]:
        if col not in raw.columns:
            raise ValueError(f"PAD-UFES metadata missing '{col}'. Found: {list(raw.columns)}")

    dx = raw["diagnostic"].str.upper().str.strip()
    benign = PAD_BENIGN | ({"ACK"} if ack == "benign" else set())
    keep = dx.isin(PAD_MALIGNANT | benign)
    print(f"Diagnoses: {dx.value_counts().to_dict()}  | ACK -> {ack}")
    raw, dx = raw[keep].reset_index(drop=True), dx[keep].reset_index(drop=True)

    meta = pd.DataFrame({
        "file": raw["img_id"],
        "skin_tone": raw[fst_col].map(fst_to_group),
        "malignant": dx.isin(PAD_MALIGNANT).astype(int),
        "disease": dx,
        "patient_id": raw["patient_id"],
    })
    return attach_paths(meta, root)


class ImageDataset(Dataset):
    def __init__(self, paths, tfm):
        self.paths, self.tfm = list(paths), tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tfm(Image.open(self.paths[i]).convert("RGB"))


def build_backbone(name: str, pretrained: bool):
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        net.fc = torch.nn.Identity()                       # 2048-d
    elif name == "dinov2_vits14":
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                             pretrained=pretrained)        # 384-d
    else:
        raise ValueError(name)
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ddi", choices=["ddi", "pad_ufes"])
    ap.add_argument("--data_dir", type=Path, default=None,
                    help="dataset folder (default: data/ddi or data/pad_ufes)")
    ap.add_argument("--ddi_dir", type=Path, default=None, help="alias of --data_dir for DDI")
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "dinov2_vits14"])
    ap.add_argument("--ack", default="benign", choices=["benign", "exclude"],
                    help="PAD-UFES: treat actinic keratosis as benign or drop it")
    ap.add_argument("--out_dir", type=Path, default=Path("features"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_pretrained", action="store_true",
                    help="random weights; only for smoke-testing the pipeline")
    args = ap.parse_args()

    root = args.data_dir or args.ddi_dir or Path("data") / args.dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | dataset: {args.dataset} | folder: {root}")

    meta = load_ddi(root) if args.dataset == "ddi" else load_pad_ufes(root, args.ack)
    print(meta.groupby(["skin_tone", "malignant"]).size().unstack(fill_value=0)
              .rename(columns={0: "benign", 1: "malignant"},
                      index={0: "unknown", 12: "FST I-II", 34: "FST III-IV", 56: "FST V-VI"}))

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(ImageDataset(meta["path"], tfm),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = build_backbone(args.backbone, pretrained=not args.no_pretrained).to(device)
    feats = []
    with torch.inference_mode():
        for x in tqdm(loader, desc=f"Embedding with {args.backbone}"):
            feats.append(net(x.to(device)).float().cpu().numpy())
    feats = np.concatenate(feats).astype(np.float32)

    prefix = "" if args.dataset == "ddi" else f"{args.dataset}_"
    if args.dataset == "pad_ufes" and args.ack == "exclude":
        prefix = "pad_ufes_ackexcl_"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / f"features_{prefix}{args.backbone}.npy", feats)
    meta.drop(columns="path").to_csv(
        args.out_dir / f"meta_{prefix}{args.backbone}.csv", index=False)
    print(f"Saved {feats.shape} embeddings to {args.out_dir}/")


if __name__ == "__main__":
    main()