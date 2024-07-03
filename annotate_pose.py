import argparse
import os

import cv2
import numpy as np

from src.interactive_annotator import InteractiveAnnotator
from src.json_utils import increment_idx, load_annotations, save_annotations


def parse_imdir(annotations_file):
    ann_filename = ".".join(os.path.basename(annotations_file).split(".")[::-1])
    ann_filename = ann_filename.replace("_kpts", "")
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

    # Optional arguments
    parser.add_argument(
        "--pose-format", type=str, default="coco", help="Format of the annotated skeleton"
    )
    parser.add_argument("--without-hands", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if not (os.path.exists(args.coco_filepath) and os.path.isfile(args.coco_filepath)):
        old_fname = os.path.join(args.coco_filepath, "annotations", "person_keypoints_val2017.json")
        new_fname = os.path.join(
            args.coco_filepath, "annotations", "person_keypoints_val2017_kpts.json"
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

    args.pose_format = args.pose_format.lower()
    implemented_formats = ["coco", "coco_with_thumbs"]
    if args.pose_format not in implemented_formats:
        raise NotImplementedError(
            "Format {:s} not implemented. Use one of the following: {}".format(
                args.pose_format, implemented_formats
            )
        )

    return args


def main(args):
    # Load the data
    coco_data, id2name, _, _ = load_annotations(args.coco_filepath)
    ann_idx = 0

    new_coco_filepath = args.coco_filepath
    if "_kpts.json" not in args.coco_filepath:
        new_coco_filepath = args.coco_filepath.replace(".json", "_kpts.json")

    annotations = coco_data["annotations"]

    cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
    ia = InteractiveAnnotator(
        annotations[ann_idx],
        os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
        is_start=ann_idx == 0,
        pose_format=args.pose_format,
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
                pose_format=args.pose_format,
            )
            coco_data["annotations"] = annotations
            if args.save:
                save_annotations(new_coco_filepath, coco_data, update_date=True)

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
            if args.save:
                save_annotations(new_coco_filepath, coco_data, update_date=True)

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
            if args.save:
                save_annotations(new_coco_filepath, coco_data, update_date=True)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord("n") or k == 81:
            annotations[ann_idx] = ia.get_annotation(json_compatible=True)
            ann_idx = increment_idx(ann_idx, len(annotations), -1)

            ia = InteractiveAnnotator(
                annotations[ann_idx],
                os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
                is_start=ann_idx == 0,
                pose_format=args.pose_format,
            )
            coco_data["annotations"] = annotations
            if args.save:
                save_annotations(new_coco_filepath, coco_data, update_date=True)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        elif k == ord("x"):
            annotations[ann_idx] = ia.get_annotation(json_compatible=True)
            ann_idx = np.random.randint(len(annotations))

            ia = InteractiveAnnotator(
                annotations[ann_idx],
                os.path.join(args.img_path, id2name[annotations[ann_idx]["image_id"]]),
                is_start=ann_idx == 0,
                pose_format=args.pose_format,
            )
            coco_data["annotations"] = annotations
            if args.save:
                save_annotations(new_coco_filepath, coco_data, update_date=True)

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
                pose_format=args.pose_format,
            )
            coco_data["annotations"] = annotations
            if args.save:
                save_annotations(new_coco_filepath, coco_data, update_date=True)

            cv2.setMouseCallback("Image", ia.mouse_callback)
        else:
            ia.key_pressed(k)

    cv2.destroyAllWindows()
    coco_data["annotations"] = annotations
    if args.save:
        save_annotations(new_coco_filepath, coco_data, update_date=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
