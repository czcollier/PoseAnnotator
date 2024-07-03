## Tool for processing anotations

This folder contains simple scripts used for processing annotation files. Adapt to your use-case as needed. Usage example is in each of the scripts.

[**annalyze_annot**](./analyze_annot.py) - Print simple statistics for given JSON file. Used when chacking annotators.

[**merge_annot**](./merge_annot.py) - Merge two annotation files into one. 

[**select_subset**](./select_subset.py) - Select subset of the JSON file to make smaller datasets.

[**select_unannotated**](./select_unannotated.py) - Select all images that has no annotation.

[**transform_to_coco**](./transform_to_coco.py) - Transform given JSON file to COCO format (visibility levels etc).
