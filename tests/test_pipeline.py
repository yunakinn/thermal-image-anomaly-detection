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


@pytest.mark.parametrize("case,message", [
    ("duplicate_image_id", "Duplicate image id"),
    ("duplicate_image_file", "Duplicate image file"),
    ("duplicate_category_id", "Duplicate category id"),
    ("unknown_image", "unknown image id"),
    ("unknown_category", "unknown category id"),
])
def test_invalid_coco_identity_is_rejected_before_splitting(tmp_path, case, message):
    data = coco([{"id": 10, "name": "normal"}, {"id": 20, "name": "anomaly"}])
    if case == "duplicate_image_id":
        data["images"].append({"id": 1, "file_name": "other.png"})
    elif case == "duplicate_image_file":
        data["images"].append({"id": 2, "file_name": " nested/./sample.png "})
    elif case == "duplicate_category_id":
        data["categories"].append({"id": 20, "name": "anomaly"})
    elif case == "unknown_image":
        data["annotations"][0]["image_id"] = 999
    else:
        data["annotations"][0]["category_id"] = 999
    path = tmp_path / "input.json"
    original = json.dumps(data)
    path.write_text(original)
    with pytest.raises(ValueError, match=message):
        pipeline.build_ann_map(path)
    assert path.read_text() == original


def test_distinct_images_and_unannotated_background_are_preserved(tmp_path):
    data = coco([{"id": 10, "name": "normal"}, {"id": 20, "name": "anomaly"}])
    data["images"].append({"id": 2, "file_name": "other/sample.png"})
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))
    images, annotations = pipeline.build_ann_map(path)
    assert images == data["images"]
    assert annotations == {1: [{"bbox": [0, 0, 5, 5], "cls": 1}]}
    assert annotations.get(2, []) == []


def conversion_input(tmp_path, bbox):
    source = tmp_path / "source"
    source.mkdir()
    cv2.imwrite(str(source / "sample.png"), np.zeros((10, 20, 3), np.uint8))
    images = [{"id": 1, "file_name": "sample.png"}]
    annotations = {1: [{"bbox": bbox, "cls": 1}]}
    return source, images, annotations


@pytest.mark.parametrize("bbox", ["1234", {"x": 0, "y": 0, "w": 5, "h": 5}, None,
                                  [0, 0, 5], [0, 0, 5, 5, 1], [0, 0, "bad", 5],
                                  [0, 0, 0, 5], [0, 0, -1, 5], [0, 0, 5, float("nan")],
                                  [0, 0, float("inf"), 5], [25, 0, 5, 5]])
def test_invalid_boxes_do_not_produce_image_or_label_files(tmp_path, bbox):
    source, images, annotations = conversion_input(tmp_path, bbox)
    with pytest.raises(ValueError, match="bounding box|Bounding box"):
        pipeline.write_yolo_items(images, annotations, source, tmp_path / "out", "test")
    assert not list((tmp_path / "out").rglob("*.txt"))
    assert not list((tmp_path / "out").rglob("*.png"))


@pytest.mark.parametrize("bbox,expected", [
    ([-5, -2, 15, 10], [.25, .4, .5, .8]),
    ([15, 7, 10, 5], [.875, .85, .25, .3]),
    ([0, 0, 20, 10], [.5, .5, 1, 1]),
])
def test_clipped_boxes_have_correct_yolo_coordinates(tmp_path, bbox, expected):
    source, images, annotations = conversion_input(tmp_path, bbox)
    pipeline.write_yolo_items(images, annotations, source, tmp_path / "out", "test")
    values = (tmp_path / "out/labels/test/sample.txt").read_text().split()
    assert values[0] == "1"
    assert list(map(float, values[1:])) == pytest.approx(expected)


def test_label_collisions_are_rejected_before_any_copy(tmp_path):
    source, images, annotations = conversion_input(tmp_path, [0, 0, 5, 5])
    images.append({"id": 2, "file_name": "sample.jpg"})
    with pytest.raises(ValueError, match="Duplicate label destination"):
        pipeline.write_yolo_items(images, annotations, source, tmp_path / "out", "test")
    assert not (tmp_path / "out").exists()
