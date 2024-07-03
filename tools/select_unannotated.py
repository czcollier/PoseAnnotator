# Description: Select images that do not have any annotations in the OCHuman dataset.
# Usage: python tools/select_unannotated.py

import json

import numpy as np

bboxes = json.load(open("data/OCHuman/annotations/person_keypoints_val2017.json", "r"))

new_bboxes = []
images = {}
for ann in bboxes["annotations"]:
    kpts = np.array(ann["keypoints"])
    if np.allclose(kpts, np.zeros_like(kpts)):
        new_bboxes.append(ann)
        images[ann["image_id"]] = True

new_images = []
for img in bboxes["images"]:
    if img["id"] in images:
        new_images.append(img)

print(
    "Selected {:d}/{:d} ({:.2f} %) images".format(
        len(new_images), len(bboxes["images"]), 100 * len(new_images) / len(bboxes["images"])
    )
)
print(
    "Selected {:d}/{:d} ({:.2f} %) annotations".format(
        len(new_bboxes),
        len(bboxes["annotations"]),
        100 * len(new_bboxes) / len(bboxes["annotations"]),
    )
)

bboxes["annotations"] = new_bboxes
bboxes["images"] = new_images

json.dump(
    bboxes,
    open("data/OCHuman/annotations/person_keypoints_val2017_unannotated.json", "w"),
    indent=2,
)
