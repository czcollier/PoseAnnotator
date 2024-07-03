import os
import datetime
import argparse
import json
import cv2
import numpy as np

from interactive_annotator import InteractiveAnnotator


def load_annotations(annotations_file):
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)

    id2name = {}
    name2id = {}
    for img in coco_data["images"]:
        id2name[img["id"]] = img["file_name"]
        name2id[img["file_name"]] = img["id"]
    return coco_data, id2name, name2id


def save_annotations(annotations_file, annotations, update_date=False, save=True):
    if not save:
        return
    if update_date:
        annotations["info"]["date_created"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=2)


def parse_imdir(annotations_file):
    ann_filename = ".".join(os.path.basename(annotations_file).split(".")[::-1])
    ann_filename = ann_filename.replace("_new", "")
    ann_type = ann_filename.split("_")[-1]
    if ann_type not in ["train2017", "val2017"]:
        print(
            "Could not determine image directory from annotations file name. Using 'val2017' as default."
        )
        ann_type = "val2017"
    coco_ann_root = os.path.dirname(annotations_file)

    return os.path.join(os.path.dirname(coco_ann_root), ann_type)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "coco_filepath",
        type=str,
        help="Filename of the coco annotations file",
    )
    parser.add_argument("--img-path", type=str, help="Path to the folder with images", default=None)

    parser.add_argument("--without-hands", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if not (os.path.exists(args.coco_filepath) and os.path.isfile(args.coco_filepath)):
        old_fname = os.path.join(
            "data", args.coco_filepath, "annotations", "person_keypoints_val2017.json"
        )
        new_fname = os.path.join(
            "data", args.coco_filepath, "annotations", "person_keypoints_val2017_new.json"
        )
        if os.path.exists(new_fname):
            args.coco_filepath = new_fname
        else:
            args.coco_filepath = old_fname

    assert os.path.exists(args.coco_filepath), "COCO annotations file ({:s}) not found".format(
        args.coco_filepath
    )
    assert os.path.isfile(args.coco_filepath), "COCO annotations file ({:s}) is not a file".format(
        args.coco_filepath
    )

    if args.img_path is None:
        args.img_path = parse_imdir(args.coco_filepath)

    return args


def increment_idx(idx, len_annotations, increment):
    idx += increment
    if idx >= len_annotations:
        idx = 0
    elif idx < 0:
        idx = len_annotations - 1
    return idx


def main(args):
    # Load the data
    coco_data, id2name, name2id = load_annotations(args.coco_filepath)
    ann_idx = 0

    new_coco_filepath = args.coco_filepath
    if "_new.json" not in args.coco_filepath:
        new_coco_filepath = args.coco_filepath.replace(".json", "_new.json")

    annotations = coco_data["annotations"]

    cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
    ia = InteractiveAnnotator(
        annotations[ann_idx],
        os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
        is_start=ann_idx == 0,
        with_hand=not args.without_hands,
    )
    cv2.setMouseCallback("Image", ia.mouse_callback)

    while cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) > 0:
        # The function waitKey waits for a key event infinitely (when delay<=0)
        k = cv2.waitKey(100)
        if k == ord("m") or k == 83:  # toggle current image
            annotations[ann_idx] = ia.get_annotation(json_compatible=True)
            ann_idx = increment_idx(ann_idx, len(annotations), 1)

            ia = InteractiveAnnotator(
                annotations[ann_idx],
                os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
                is_start=ann_idx == 0,
                with_hand=not args.without_hands,
            )
            coco_data["annotations"] = annotations
            save_annotations(new_coco_filepath, coco_data, update_date=True, save=args.save)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord("."):  # jump 10
            annotations[ann_idx] = ia.get_annotation(json_compatible=True)
            ann_idx = increment_idx(ann_idx, len(annotations), 10)

            ia = InteractiveAnnotator(
                annotations[ann_idx],
                os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
                is_start=ann_idx == 0,
            )
            coco_data["annotations"] = annotations
            save_annotations(new_coco_filepath, coco_data, update_date=True, save=args.save)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord(","):  # jump -10
            annotations[ann_idx] = ia.get_annotation(json_compatible=True)
            ann_idx = increment_idx(ann_idx, len(annotations), -10)

            ia = InteractiveAnnotator(
                annotations[ann_idx],
                os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
                is_start=ann_idx == 0,
            )
            coco_data["annotations"] = annotations
            save_annotations(new_coco_filepath, coco_data, update_date=True, save=args.save)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord("n") or k == 81:
            annotations[ann_idx] = ia.get_annotation(json_compatible=True)
            ann_idx = increment_idx(ann_idx, len(annotations), -1)

            ia = InteractiveAnnotator(
                annotations[ann_idx],
                os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
                is_start=ann_idx == 0,
                with_hand=not args.without_hands,
            )
            coco_data["annotations"] = annotations
            save_annotations(new_coco_filepath, coco_data, update_date=True, save=args.save)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord("x"):
            annotations[ann_idx] = ia.get_annotation(json_compatible=True)
            ann_idx = np.random.randint(len(annotations))

            ia = InteractiveAnnotator(
                annotations[ann_idx],
                os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
                is_start=ann_idx == 0,
                with_hand=not args.without_hands,
            )
            coco_data["annotations"] = annotations
            save_annotations(new_coco_filepath, coco_data, update_date=True, save=args.save)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord("q"):
            annotations[ann_idx] = ia.get_annotation(json_compatible=True)
            break
        elif k == ord("u"):
            annotations[ann_idx] = ia.get_annotation(json_compatible=True)
            while "checked" in annotations[ann_idx].keys():
                ann_idx = increment_idx(ann_idx, len(annotations), 1)

            ia = InteractiveAnnotator(
                annotations[ann_idx],
                os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
                is_start=ann_idx == 0,
                with_hand=not args.without_hands,
            )
            coco_data["annotations"] = annotations
            save_annotations(new_coco_filepath, coco_data, update_date=True, save=args.save)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        else:
            ia.key_pressed(k)

    cv2.destroyAllWindows()
    coco_data["annotations"] = annotations
    save_annotations(new_coco_filepath, coco_data, update_date=True, save=args.save)


if __name__ == "__main__":
    args = parse_args()
    main(args)
