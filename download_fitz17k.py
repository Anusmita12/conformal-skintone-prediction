import pandas as pd, requests
from pathlib import Path

out = Path("data/fitz17k/images")
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/fitz17k/fitzpatrick17k.csv")

ok = 0
for i, (h, url) in enumerate(zip(df.md5hash, df.url)):
    f = out / f"{h}.jpg"
    if f.exists():
        ok += 1
        continue
    if not isinstance(url, str):
        continue
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            f.write_bytes(r.content)
            ok += 1
    except Exception:
        pass
    if i % 500 == 0:
        print(f"{i}/{len(df)} checked, {ok} images saved")

print(f"Done: {ok} images in {out}")