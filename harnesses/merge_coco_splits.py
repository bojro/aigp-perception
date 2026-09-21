"""Merge a Roboflow multi-split COCO export into one annotations file.

`build_ab_datasets.py` reads a single COCO json and decides its own splits --
contiguous blocks of the capture, because the frames are 0.5 s apart along a
walk and a random split leaks near-duplicate neighbours between train and val
(the same model reads 0.517 box mAP50-95 on a random split and 0.283 on
blocks). Roboflow, which does not know that, ships its own train/valid/test
directories. Taking only `train/` from such an export silently throws away
every label a human placed in the other two.

Merging is not concatenation. Roboflow numbers `images[].id` and
`annotations[].id` from zero **inside each split**, so appending the lists
directly makes annotations in `valid` point at images in `train` -- which does
not error, it just attaches the wrong boxes to the wrong pictures. Ids are
therefore reassigned, and category ids are remapped by *name* rather than
trusted to agree across splits.

    python harnesses/merge_coco_splits.py <export-dir> -o merged.coco.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLITS = ("train", "valid", "test")


def merge(export: Path, splits=SPLITS) -> dict:
    categories: list[dict] = []
    by_name: dict[str, int] = {}
    images: list[dict] = []
    annotations: list[dict] = []
    seen_files: dict[str, str] = {}

    for split in splits:
        path = export / split / "_annotations.coco.json"
        if not path.is_file():
            print(f"  {split:6s} (absent)")
            continue
        d = json.loads(path.read_text())

        # Categories by name. A merge that trusted the ids would relabel
        # everything the moment one split ordered them differently.
        local_cat: dict[int, int] = {}
        for c in d.get("categories", []):
            if c["name"] not in by_name:
                by_name[c["name"]] = len(categories)
                categories.append({**c, "id": by_name[c["name"]]})
            local_cat[c["id"]] = by_name[c["name"]]

        local_img: dict[int, int] = {}
        dupes = 0
        for im in d.get("images", []):
            name = im["file_name"]
            if name in seen_files:
                # The same picture in two splits would be trained on twice and
                # weighted twice. Keep the first and say so.
                dupes += 1
                continue
            seen_files[name] = split
            new_id = len(images)
            local_img[im["id"]] = new_id
            images.append({**im, "id": new_id})

        kept = 0
        for a in d.get("annotations", []):
            if a["image_id"] not in local_img:
                continue
            annotations.append({
                **a,
                "id": len(annotations),
                "image_id": local_img[a["image_id"]],
                "category_id": local_cat.get(a["category_id"], a["category_id"]),
            })
            kept += 1

        note = f", {dupes} duplicate file names skipped" if dupes else ""
        print(f"  {split:6s} {len(local_img):4d} images, {kept:4d} annotations{note}")

    return dict(images=images, annotations=annotations, categories=categories,
                info={"description": "merged Roboflow splits"}, licenses=[])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path, help="directory holding train/valid/test")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--splits", default=",".join(SPLITS))
    args = ap.parse_args()

    merged = merge(args.export, tuple(s for s in args.splits.split(",") if s))
    args.out.write_text(json.dumps(merged))

    kp = [a for a in merged["annotations"] if a.get("keypoints")]
    widths = {len(a["keypoints"]) // 3 for a in kp}
    print(f"\n  merged {len(merged['images'])} images, "
          f"{len(merged['annotations'])} annotations")
    print(f"  categories {[c['name'] for c in merged['categories']]}")
    print(f"  keypoints per instance {sorted(widths)}")

    # Ids must be dense and annotations must all resolve, or the merge is worse
    # than the thing it replaced.
    assert [i["id"] for i in merged["images"]] == list(range(len(merged["images"])))
    valid_ids = set(range(len(merged["images"])))
    assert all(a["image_id"] in valid_ids for a in merged["annotations"])
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
