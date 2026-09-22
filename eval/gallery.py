"""Build review sheets grouped by verdict so each class can be judged apart."""
import csv, subprocess, sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
gallery = out / "gallery"
gallery.mkdir(exist_ok=True)
rows = list(csv.DictReader((out / "report.csv").open()))

buckets = defaultdict(list)
for row in rows:
    key = row["verdict"]
    if key == "review" and "merged_pair" in row["flags"]:
        key = "review_merged_pair"
    elif key == "review" and row["anchor"] == "outer":
        key = "review_outer_only"
    buckets[key].append(row["file"])

made = []
for name, files in buckets.items():
    seen, picked = set(), []
    for f in files:                       # spread across the capture, dedup
        if f not in seen:
            seen.add(f); picked.append(f)
    step = max(len(picked) // 12, 1)
    picked = picked[::step][:12]
    paths = [str(out / "overlays" / f) for f in picked
             if (out / "overlays" / f).exists()]
    if not paths:
        continue
    sheet = gallery / f"{name}.jpg"
    subprocess.run(["montage", *paths, "-tile", "4x3",
                    "-geometry", "620x349+3+3", str(sheet)],
                   stderr=subprocess.DEVNULL)
    if sheet.exists():
        made.append((name, len(files), len(paths), sheet))

for name, total, shown, sheet in sorted(made):
    print(f"{name:22s} {total:5d} instances  ->  {shown} shown  {sheet}")
