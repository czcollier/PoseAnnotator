import datetime
import json


def load_annotations(annotations_file):
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)

    id2name = {}
    name2id = {}
    ann_dict = {}
    for img in coco_data["images"]:
        id2name[img["id"]] = img["file_name"]
        name2id[img["file_name"]] = img["id"]
        ann_dict[img["id"]] = []

    for ann in coco_data["annotations"]:
        ann_dict[ann["image_id"]].append(ann)

    return coco_data, id2name, name2id, ann_dict


def save_annotations(annotations_file, annotations, update_date=False, save=True):
    if not save:
        return
    if update_date:
        annotations["info"]["date_created"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=2)


def increment_idx(idx, len_annotations, increment):
    idx += increment
    if idx >= len_annotations:
        idx = 0
    elif idx < 0:
        idx = len_annotations - 1
    return idx
