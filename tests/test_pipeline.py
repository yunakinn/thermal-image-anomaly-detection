import json
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
import main as pipeline


def coco(categories):
    return {"categories": categories, "images": [{"id": 1, "file_name": "nested/sample.png"}], "annotations": [{"image_id": 1, "category_id": 20, "bbox": [0, 0, 5, 5]}]}


def test_class_mapping_does_not_depend_on_json_order(tmp_path):
    cats = [{"id": 10, "name": "normal"}, {"id": 20, "name": "anomaly"}]
    for index, categories in enumerate([cats, cats[::-1]]):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(coco(categories)))
        assert pipeline.build_ann_map(path)[1][1][0]["cls"] == 1
    path.write_text(json.dumps(coco([{"id": 10, "name": "unknown"}])))
    with pytest.raises(ValueError, match="Unsupported"):
        pipeline.build_ann_map(path)


def test_shared_conversion_handles_subdirectories(tmp_path):
    source = tmp_path / "images/nested"
    source.mkdir(parents=True)
    cv2.imwrite(str(source / "sample.png"), np.zeros((10, 10, 3), np.uint8))
    config = tmp_path / "input.json"
    config.write_text(json.dumps(coco([{"id": 10, "name": "normal"}, {"id": 20, "name": "anomaly"}])))
    images, annotations = pipeline.build_ann_map(config)
    pipeline.write_yolo_items(images, annotations, tmp_path / "images", tmp_path / "out", "test")
    assert (tmp_path / "out/labels/test/nested/sample.txt").read_text().startswith("1 ")


def test_training_seed_is_forwarded(monkeypatch, tmp_path):
    model = Mock()
    monkeypatch.setattr(pipeline, "YOLO", Mock(return_value=model))
    pipeline.train_yolo("model.pt", 1, "data.yaml", tmp_path, 19)
    assert model.train.call_args.kwargs["seed"] == 19
    assert model.train.call_args.kwargs["exist_ok"] is False


def test_test_split_and_metrics(monkeypatch, tmp_path):
    model = Mock()
    model.val.return_value = SimpleNamespace(box=SimpleNamespace(map=.7, mp=.8, mr=.6))
    monkeypatch.setattr(pipeline, "YOLO", Mock(return_value=model))
    result = pipeline.evaluate_test("model.pt", "data.yaml", tmp_path)
    assert model.val.call_args.kwargs["split"] == "test"
    assert model.val.call_args.kwargs["conf"] == .001
    assert result["FinalScore"] == pytest.approx(.7)
    assert (tmp_path / "test_metrics.csv").is_file()


def test_existing_run_is_not_deleted(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        pipeline.main(["--mode", "train", "--run-dir", str(tmp_path)])
    assert marker.read_text() == "original"
