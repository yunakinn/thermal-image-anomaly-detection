"""Prepare COCO thermal data, train YOLO, and evaluate a held-out test set."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

CLASS_NAMES = ("normal", "anomaly")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def build_ann_map(json_path):
    with Path(json_path).open(encoding="utf-8") as source:
        data = json.load(source)
    category_map = {}
    for category in data["categories"]:
        name = str(category["name"]).strip().lower()
        if name not in CLASS_NAMES:
            raise ValueError(f"Unsupported category {name!r}; expected {CLASS_NAMES}.")
        category_map[category["id"]] = CLASS_NAMES.index(name)
    if set(category_map.values()) != {0, 1}:
        raise ValueError("Both normal and anomaly categories must be declared.")
    annotations = {}
    for annotation in data["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append({
            "bbox": annotation["bbox"], "cls": category_map[annotation["category_id"]]
        })
    return data["images"], annotations


def write_yolo_items(images, annotations, image_root, output_root, split):
    """Use one conversion path for both training and test annotations."""
    image_root, output_root = Path(image_root), Path(output_root)
    image_dest = output_root / "images" / split
    label_dest = output_root / "labels" / split
    image_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)
    seen = set()
    for image in images:
        relative = Path(str(image["file_name"]).strip())
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Image name must be relative to image_root: {relative}")
        label_relative = relative.with_suffix(".txt")
        if label_relative in seen:
            raise ValueError(f"Duplicate label destination: {label_relative}")
        seen.add(label_relative)
        source = image_root / relative
        image_data = imread_unicode(source)
        if image_data is None:
            raise ValueError(f"Cannot decode image: {source}")
        height, width = image_data.shape[:2]
        lines = []
        for annotation in annotations.get(image["id"], []):
            x, y, box_width, box_height = map(float, annotation["bbox"])
            if not all(np.isfinite([x, y, box_width, box_height])) or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Invalid bounding box in {relative}: {annotation['bbox']}")
            left, top = max(0.0, x), max(0.0, y)
            right, bottom = min(float(width), x + box_width), min(float(height), y + box_height)
            if right <= left or bottom <= top:
                raise ValueError(f"Bounding box is outside image {relative}")
            values = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                      (right - left) / width, (bottom - top) / height)
            lines.append(f"{annotation['cls']} " + " ".join(map(str, values)))
        (image_dest / relative).parent.mkdir(parents=True, exist_ok=True)
        (label_dest / label_relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_dest / relative)
        (label_dest / label_relative).write_text("\n".join(lines) + ("\n" if lines else ""))


def create_yaml(root, test=False):
    root = Path(root).resolve()
    lines = [f"path: {json.dumps(str(root))}"]
    if test:
        lines += ["train: images/test", "val: images/test", "test: images/test"]
    else:
        lines += ["train: images/train", "val: images/val"]
    lines += ["nc: 2", 'names: ["normal", "anomaly"]']
    path = root / "data.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def train_yolo(model_pt, epochs, data_yaml, run_dir, seed):
    model = YOLO(model_pt)
    return model.train(
        data=str(data_yaml), epochs=epochs, seed=seed, deterministic=True,
        imgsz=640, batch=8, optimizer="SGD", lr0=0.0008,
        momentum=0.95, weight_decay=0.0005, warmup_epochs=3,
        mosaic=0.0, mixup=0.0, hsv_h=0.02, hsv_s=0.03, hsv_v=0.03,
        translate=0.05, scale=0.1, fliplr=0.2, flipud=0.05, multi_scale=False,
        project=str(run_dir), name="training", exist_ok=False,
    )


def evaluate_test(weights, data_yaml, run_dir):
    model = YOLO(weights)
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=640, conf=0.001,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(run_dir), name="test", exist_ok=False,
    )
    result = {"mAP50-95": float(metrics.box.map), "Precision": float(metrics.box.mp),
              "Recall": float(metrics.box.mr)}
    result["FinalScore"] = .8 * result["mAP50-95"] + .1 * result["Precision"] + .1 * result["Recall"]
    (Path(run_dir) / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    import csv
    with (Path(run_dir) / "test_metrics.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--workspace", type=Path, default=Path("data/workspace"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-dir", type=Path, help="Must be absent or empty; existing results are never deleted.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0, help="YOLO trainer seed.")
    parser.add_argument("--split-seed", type=int, default=42, help="Train/validation allocation seed.")
    parser.add_argument("--weights")
    parser.add_argument("--model", default="yolov8x.pt")
    args = parser.parse_args(argv)
    if args.mode == "test" and not args.weights:
        parser.error("--weights is required in test mode")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    run_dir = args.run_dir or args.output_dir / (args.mode + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["run_dir"] = str(run_dir.resolve())
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    data_root = run_dir / "dataset"
    if args.mode == "train":
        images, annotations = build_ann_map(args.workspace / "train.json")
        labels = [int(any(a["cls"] == 1 for a in annotations.get(image["id"], []))) for image in images]
        train, val = train_test_split(images, test_size=.1, stratify=labels, random_state=args.split_seed)
        for split, items in [("train", train), ("val", val)]:
            write_yolo_items(items, annotations, args.workspace / "images/train", data_root, split)
        train_yolo(args.model, args.epochs, create_yaml(data_root), run_dir, args.seed)
    else:
        images, annotations = build_ann_map(args.workspace / "test.json")
        write_yolo_items(images, annotations, args.workspace / "images/test", data_root, "test")
        print(json.dumps(evaluate_test(args.weights, create_yaml(data_root, test=True), run_dir), indent=2))


if __name__ == "__main__":
    main()
