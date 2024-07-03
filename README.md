# PoseAnnotator
PoseAnnotator is a simple Python tool with a GUI for annotating 2D human poses from images. It allows users to easily label key points on human figures for pose estimation tasks.

### Why PoseAnnotator

There are nice free annotation tools like [CVAT](https://www.cvat.ai) or [LabelStudio](https://labelstud.io) but they do not specialize for 2D Human Pose Annotation. It is possible to annotate images with keypoints or landmarks but there is no structure visualized for annotator to better understand the scene.

We created this tiny tool to annotate our own 2D Human Pose Estimation dataset ([RePoGen](https://mirapurkrabek.github.io/RePoGen-paper/)). From our experience, having visualization of annotated Human Pose minimize common errors like switching left and right side, switching keypoints etc.


We publish this project for other researchers and people focusing on 2D Human Pose Estimation.
It runs localy on any Linux-based computer and allows you to annotate both bounding boxes and keypoints in any images.
If you find this tool useful, let us or other people who might be interested, know.

## Installation

The tool does not need any installation apart from common Python libraries defined in [requirements.txt](requirements.txt). You can use either pip or Conda and provided [environment file](environment.yml).

If you would like to contribute to the project, please use the pre-commit which automatically format the code. To install, just run 
`pre-commit install` and the code will be automatically formatted after each commit.

## Usage

The script might need minor adaptations to use for your specific use-case. 

For details like keyboard shortcuts, see [annotation how to](docs/annotation.md)