# The installation guide


## How to install

The app requires following:
- Linux machine (developed on Ubuntu)
- Python3 (developed with version 3.10)
- NumPy
- OpenCV bindings for Python

If you have Linux machine with Python already, you can install required packages with `pip install -r requirements.txt`.

### Setting up bash alias

The user might want to set up his own bash alias to run the annotation smoothly. We recomend (and use it in the docs) to set up _annot_ as the main system-wide command with additional arguments as names of sequences.

`export annot='cd $pose_annotation_tool && python main.py'`

Then you can run the command as follows:

`annot seq1` or `annot seq2`

