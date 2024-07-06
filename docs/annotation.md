# The annotation guide

The task of annotating 2D human pose involves identifying and marking key points on a person's body in an image. These key points correspond to important joints such as shoulders, elbows, wrists, hips, knees, and ankles. Your goal as an annotator is to accurately label these keypoints, enabling the understanding of the body's pose and movement for various applications like motion analysis, fitness tracking, and animation.

## Examples

There are some examples in this [file](examples.md)

## How to start

**Bboxes annotation:** use script [annotate_bboxes](../annotate_bboxes.py) to take batch of images and annotate bboxes.

> __Example usage:__ `python annotate_bboxes.py test_data/RePoGen_bbox_test`

**Keypoints annotation:** use script [annotate_keypoints](../annotate_pose.py) to take images with annotated bboxes and add keypoints.

> __Example usage:__ `python annotate_pose.py test_data/RePoGen_kpts_test/annotations/person_keypoints_val2017.json`


### Folder structure

The tool is implemented primarily for COCO-style annotation and therefore expects its file structure. Each dataset should have 3 subfolders `annotations`, `val2017` and optionally `train2017`. Example datasets are in folder [test_data](../test_data/).

You don't need any annotation file if you want to annotate bboxes. If you want to annotate both, first annotate bboxes and then use the other script for keypoints annotation.


## Keyboard Shortcuts

The pose annotation tool offers several keyboard shortcuts to improve your annotation workflow:

| Shortcut  | Description                                                                                              |
| --------- | -------------------------------------------------------------------------------------------------------- |
| `u`       | **Next unannotated** pose                                                                                |
| `m`       | Save the current annotation progress and move to the **next one**                                        |
| `n`       | Save the current annotation progress and move to the **previous one**                                    |
| `,/.`     | **Jump by 10** backward/forward.                                                                         |
| `x`       | **Random** annotation. Useful for random quality check.                                                  |
| `v`       | Change the **visibility** of the currently selected keypoint (works only while holding the mouse button) |
| `d`       | **Delete** the currently selected keypoint (works only while holding the mouse button)                   |
| `a`       | **Add** a new kypoint, if any is missing                                                                 |
| `z`       | **Undo** the previous step (works only untill save)                                                      |
| `r`       | **Reset** the pose to the original (in the last save)                                                    |
| `g`       | **Generate** a new pose                                                                                  |
| `l`       | **Flip L/R** of the pose                                                                                 |
| `o` / `p` | **Zoom** the image in and out pose                                                                       |
| `q`       | **Quit** the program (with save)                                                                         |


### Annotated vs. Un-annotated images

If a person has already been annotated in the application, their bounding box will be displayed in a darker shade of green. You can still view previously annotated persons for later reference. To skip directly to the next unfinished task, press the 'u' key.

### Memory

The program remembers the steps you've taken, so you can go back to the way things were before. However, if you change the image or exit the program, this memory will be lost, and you won't be able to undo or reset your actions.

### Visibility

The visibility of keypoints is indicated by their transparency. Keypoints that appear with semi-transparent lines and a grey marker are annotated as non-visible (v=1). If you're unsure whether a keypoint is marked as visible or not, select it with the mouse and press 'v' to toggle its visibility. You will notice the difference in visibility afterward. 

There is also settings to allow for 3 levels of visibility. The least visible (we call it a guess; v=3) is visualized without any marker, just intersection of bones.

### Pose format

The tool works with COCO-format but we created some custom pose formats for our use-cases. You can see example of one such custom format if you use option `--pose-format coco_with_thumbs`. Visualization of the format is in [this image](../example_images/coco_with_thumbs.png).

### Exporting results

The application automatically saves your progress when you switch to another image or exit the application. The keypoints are saved in `*_kpts.json` file, bboxes in `*.json` (without name change) by default. 

### Syncing results to Google Drive:
Annotation results can be synced with your google drive for backup and progress monitoring. Drive API requires OAuth2.0 for authentication which requires the following steps.

1. Go to APIs Console and make your own project.
2. Search for 'Google Drive API', select the entry, and click 'Enable'.
3. Select 'Credentials' from the left menu, click 'Create Credentials', select 'OAuth client ID'.
4. Now, the product name and consent screen need to be set -> click 'Configure consent screen' and follow the instructions. 
5. Once finished, create the credentials for OAuth2:
    - Select 'Application type' to be Web application.
    - Enter an appropriate name.
    - Input http://localhost:8080/ for 'Authorized redirect URIs'.
    - Click 'Save'.
    - Click 'Download JSON' on the right side of Client ID to download client_secret_<really long ID>.json.
7. Rename the file to 'client_secrets.json' and place it in your working directory.

Your Google Drive setup is completed. Add the `cloud-upload` flag to the script to save the results.

> __Example usage:__ `python annotate_pose.py test_data/RePoGen_kpts_test/annotations/person_keypoints_val2017.json --cloud-upload`

To save the results in specific folder in your google drive, provide the folder id as the `cloud-folder` argument.

> __Example usage:__ `python annotate_pose.py test_data/RePoGen_kpts_test/annotations/person_keypoints_val2017.json --cloud-upload --cloud-folder XXXX`
