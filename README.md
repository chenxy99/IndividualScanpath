# IndividualScanpath

This code implements the prediction of individualized visual scanpath in three different tasks (4 different datasets) with two different architecture:

- Free-viewing: the prediction of scanpath for looking at some salient or important object in the given image. (OSIE, OSIE-ASD)
- Visual Question Answering:  the prediction of scanpath during human performing general tasks, e.g., visual question answering, to reflect their attending and reasoning processes. (AiR-D)
- Visual search: the prediction of scanpath during the search of the given target object to reflect the goal-directed behavior. (COCO-Search18)

Reference
------------------
If you use our code or data, please cite our paper:
```text
Anonymous submission for CVPR 2024, paper ID 532.
```

Disclaimer
------------------
For the ScanMatch evaluation metric, we adopt the part of [`GazeParser`](http://gazeparser.sourceforge.net/) package. 
We adopt the implementation of SED and STDE from [`VAME`](https://github.com/dariozanca/VAME) as two of our evaluation metrics mentioned in the [`Visual Attention Models`](https://ieeexplore.ieee.org/document/9207438). 
More specific, we adopt the evaluation metrics provided in [`Scanpath`](https://github.com/chenxy99/Scanpaths).
For ChemLSTM and Gazeformer, we adopt the released code in [`Scanpath`](https://github.com/chenxy99/Scanpaths) and [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer), respectively.
Based on the [`checkpoint`](https://github.com/nocaps-org/updown-baseline/blob/master/updown/utils/checkpointing.py) implementation from [`updown-baseline`](https://github.com/nocaps-org/updown-baseline), we slightly modify it to accommodate our pipeline.

Requirements
------------------

- Python 3.9
- PyTorch 1.12.1 (along with torchvision)

- We also provide the conda environment ``user_scanpath.yml``, you can directly run

```bash
$ conda env create -f user_scanpath.yml
```

to create the same environment where we successfully run our codes.

Tasks
------------------

We provide the corresponding codes for the aforementioned four different datasets.

- OSIE
- OSIE-ASD
- COCOSearch
- AiR-D

We would provide more details for these tasks in their corresponding folders.
