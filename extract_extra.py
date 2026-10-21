"""
Extra feature extraction for the tone-blind experiments.
Builds on extract_features.py (imports its loaders) and does not change it.

Adds
  1. Fitzpatrick17k as a dataset   (--dataset fitz17k)
       - used to train the skin-tone estimator on 16k images with FST labels
       - can also serve as a second deployment target for malignancy
  2. Two more backbones             (--backbone dinov2_vitb14 | monet)
       - monet = MONET, a CLIP model trained on dermatology images
         (huggingface.co/suinleelab/monet). Needs: pip install transformers
  3. Skin-colour features (ITA)     (--ita)
       - Individual Typology Angle from the image BORDER, which is mostly
         healthy skin around the lesion. Cheap, interpretable tone estimate.

Usage (repo root):
    # Fitzpatrick17k embeddings (same backbone as your DDI/PAD features!)
    python extract_extra.py --dataset fitz17k --data_dir data/fitz17k --backbone dinov2_vits14

    # dermatology backbone for DDI and PAD-UFES
    python extract_extra.py --dataset ddi      --backbone monet
    python extract_extra.py --dataset pad_ufes --backbone monet

    # ITA skin-colour features (no backbone involved)
    python extract_extra.py --dataset ddi     --ita
    python extract_extra.py --dataset fitz17k --data_dir data/fitz17k --ita

Fitzpatrick17k layout expected under --data_dir:
    fitzpatrick17k.csv  (or any *fitzpatrick*.csv)
    images named <md5hash>.jpg/.png anywhere below (how most download scripts save them)

Outputs in features/:
    features_<prefix><backbone>.npy, meta_<prefix><backbone>.csv   (same format as before)
    ita_<prefix>.csv  (file, ita_median, ita_q25, ita_q75, L_median, b_median)
    prefix: ""  for DDI, "pad_ufes_" for PAD-UFES,
            "fitz17k_" for Fitzpatrick17k benign+malignant (malignancy target),
            "fitz17k_all_" with --keep_non_neoplastic (tone-model training)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from extract_features import (ImageDataset, attach_paths, build_backbone, find_csv,
                              fst_to_group, load_ddi, load_pad_ufes)
from torchvision import transforms


# ---------------------------------------------------------------------------
# Fitzpatrick17k
# ---------------------------------------------------------------------------
def load_fitz17k(root: Path, keep_non_neoplastic: bool) -> pd.DataFrame:
    csv = find_csv(root, "*fitzpatrick*.csv")
    raw = pd.read_csv(csv)
    raw.columns = raw.columns.str.strip().str.lower()
    print(f"Metadata: {csv}  ({len(raw)} rows)")
    need = ["md5hash", "fitzpatrick_scale", "three_partition_label", "label"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"Fitzpatrick17k metadata missing {missing}. Found: {list(raw.columns)}")

    part = raw["three_partition_label"].astype(str).str.lower().str.strip()
    keep = part.isin(["benign", "malignant"] + (["non-neoplastic"] if keep_non_neoplastic else []))
    raw, part = raw[keep].reset_index(drop=True), part[keep].reset_index(drop=True)
    print(f"Partition counts kept: {part.value_counts().to_dict()}")

    meta = pd.DataFrame({
        "file": raw["md5hash"].astype(str),        # matched to image stem below
        "skin_tone": raw["fitzpatrick_scale"].map(fst_to_group),   # -1 -> 0 (unknown)
        "malignant": (part == "malignant").astype(int),
        "disease": raw["label"].astype(str),
        "patient_id": raw["md5hash"].astype(str),  # no patient ids in Fitzpatrick17k
    })
    # images are usually saved as <md5hash>.<ext>; match on stem
    stems = {p.stem: p.name for p in root.rglob("*")
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    meta["file"] = meta["file"].map(stems)
    n_missing = meta["file"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} rows have no downloaded image (dead links are common); skipping.")
        meta = meta.dropna(subset=["file"]).reset_index(drop=True)
    return attach_paths(meta, root)


def load_dataset(name, root, ack, keep_non_neoplastic):
    if name == "ddi":
        return load_ddi(root), ""
    if name == "pad_ufes":
        prefix = "pad_ufes_ackexcl_" if ack == "exclude" else "pad_ufes_"
        return load_pad_ufes(root, ack), prefix
    # all-diagnosis version (for training the tone model) gets its own prefix
    return load_fitz17k(root, keep_non_neoplastic), ("fitz17k_all_" if keep_non_neoplastic
                                                     else "fitz17k_")


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------
IMAGENET_TFM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def embed_torchvision_like(paths, backbone, device, batch_size):
    if backbone == "dinov2_vitb14":
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()   # 768-d
    else:
        net = build_backbone(backbone, pretrained=True)
    net = net.to(device)
    loader = DataLoader(ImageDataset(paths, IMAGENET_TFM), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    out = []
    with torch.inference_mode():
        for x in tqdm(loader, desc=f"Embedding with {backbone}"):
            out.append(net(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def embed_monet(paths, device, batch_size):
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
    processor = AutoProcessor.from_pretrained("suinleelab/monet")
    model = AutoModelForZeroShotImageClassification.from_pretrained("suinleelab/monet")
    model = model.to(device).eval()
    paths, out = list(paths), []
    with torch.inference_mode():
        for i in tqdm(range(0, len(paths), batch_size), desc="Embedding with MONET"):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            if not torch.is_tensor(feats):   # newer transformers return an output object
                emb = getattr(feats, "image_embeds", None)
                feats = emb if emb is not None else feats.pooler_output
            out.append(feats.float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# ITA skin-colour features
# ---------------------------------------------------------------------------
def srgb_to_lab(rgb):
    """rgb in [0, 1], shape [..., 3] -> CIE Lab (D65)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def ita_features(path, size=224, ring=0.15):
    """
    ITA = atan((L - 50) / b) in degrees, over the outer `ring` fraction of the
    image (mostly healthy skin). Higher ITA = lighter skin. Rough guide:
    > 55 very light, 28-41 intermediate, 10-28 tan, < -30 dark.
    Very dark / very bright pixels (shadows, glare, background) are dropped.
    """
    img = np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float64) / 255
    lab = srgb_to_lab(img)
    w = int(size * ring)
    mask = np.zeros((size, size), bool)
    mask[:w], mask[-w:], mask[:, :w], mask[:, -w:] = True, True, True, True
    px = lab[mask]
    px = px[(px[:, 0] > 15) & (px[:, 0] < 95)]
    if len(px) < 50:                      # almost nothing usable: fall back to whole image
        px = lab.reshape(-1, 3)
    ita = np.degrees(np.arctan2(px[:, 0] - 50, px[:, 2]))
    return dict(ita_median=np.median(ita), ita_q25=np.quantile(ita, 0.25),
                ita_q75=np.quantile(ita, 0.75),
                L_median=np.median(px[:, 0]), b_median=np.median(px[:, 2]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ddi", choices=["ddi", "pad_ufes", "fitz17k"])
    ap.add_argument("--data_dir", type=Path, default=None)
    ap.add_argument("--backbone", default="dinov2_vits14",
                    choices=["resnet50", "dinov2_vits14", "dinov2_vitb14", "monet"])
    ap.add_argument("--ita", action="store_true", help="compute ITA skin-colour features instead")
    ap.add_argument("--ack", default="benign", choices=["benign", "exclude"])
    ap.add_argument("--keep_non_neoplastic", action="store_true",
                    help="Fitzpatrick17k: keep inflammatory/non-neoplastic images "
                         "(useful for training the tone estimator: more images)")
    ap.add_argument("--out_dir", type=Path, default=Path("features"))
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    root = args.data_dir or Path("data") / args.dataset
    meta, prefix = load_dataset(args.dataset, root, args.ack, args.keep_non_neoplastic)
    print(meta.groupby(["skin_tone", "malignant"]).size().unstack(fill_value=0))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.ita:
        rows = [dict(file=f, **ita_features(p)) for f, p in
                tqdm(zip(meta["file"], meta["path"]), total=len(meta), desc="ITA")]
        out = pd.DataFrame(rows)
        out.to_csv(args.out_dir / f"ita_{prefix}.csv", index=False)
        print(out.merge(meta[["file", "skin_tone"]], on="file")
                 .groupby("skin_tone").ita_median.describe().round(1))
        print(f"Saved {args.out_dir / f'ita_{prefix}.csv'}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.backbone == "monet":
        feats = embed_monet(meta["path"], device, args.batch_size)
    else:
        feats = embed_torchvision_like(meta["path"], args.backbone, device, args.batch_size)
    np.save(args.out_dir / f"features_{prefix}{args.backbone}.npy", feats.astype(np.float32))
    meta.drop(columns="path").to_csv(args.out_dir / f"meta_{prefix}{args.backbone}.csv", index=False)
    print(f"Saved {feats.shape} embeddings with prefix '{prefix}' to {args.out_dir}/")


if __name__ == "__main__":
    main()
