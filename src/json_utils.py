import datetime
import json


def load_annotations(annotations_file):
    """
    Load annotations from a JSON file.

    Args:
        annotations_file (str): The path to the JSON file containing the annotations.

    Returns:
        tuple: A tuple containing the following:
            - coco_data (dict): The loaded JSON data.
            - id2name (dict): A dictionary mapping image IDs to file names.
            - name2id (dict): A dictionary mapping file names to image IDs.
            - ann_dict (dict): A dictionary mapping image IDs to a list of annotations.
    """
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
    """
    Save the annotations to a JSON file.

    Args:
        annotations_file (str): The path to the JSON file where the annotations will be saved.
        annotations (dict): The annotations to be saved.
        update_date (bool, optional): Whether to update the "date_created" field in the annotations. Defaults to False.
        save (bool, optional): Whether to save the annotations. If set to False, the function will return without saving. Defaults to True.
    """
    if not save:
        return
    if update_date:
        annotations["info"]["date_created"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=2)


def increment_idx(idx, len_annotations, increment):
    """
    Increments the given index by the specified increment, taking into account the length of the annotations.

    Args:
        idx (int): The current index.
        len_annotations (int): The length of the annotations.
        increment (int): The amount by which to increment the index.

    Returns:
        int: The updated index value.
    """
    idx += increment
    if idx >= len_annotations:
        idx = 0
    elif idx < 0:
        idx = len_annotations - 1
    return idx
