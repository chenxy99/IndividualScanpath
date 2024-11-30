# Beyond Average: Individualized Visual Scanpath Prediction

This code implements the prediction of individualized visual scanpath in three different tasks (4 different datasets) with two different architecture:

- Free-viewing: the prediction of scanpath for looking at some salient or important object in the given image. (OSIE, OSIE-ASD)
- Visual Question Answering:  the prediction of scanpath during human performing general tasks, e.g., visual question answering, to reflect their attending and reasoning processes. (AiR-D)
- Visual search: the prediction of scanpath during the search of the given target object to reflect the goal-directed behavior. (COCO-Search18)

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2404.12235)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=6-TZbsV8Qig)

:mega: Overview
------------------
![overall_structure](./asset/teaser.png)
Understanding how attention varies across individuals has significant scientific and societal impacts. However, existing visual scanpath models treat attention uniformly, neglecting individual differences. To bridge this gap, this paper focuses on individualized scanpath prediction (ISP), a new attention modeling task that aims to accurately predict how different individuals shift their attention in diverse visual tasks. It proposes an ISP method featuring three novel technical components: (1) an observer encoder to characterize and integrate an observer's unique attention traits, (2) an observer-centric feature integration approach that holistically combines visual features, task guidance, and observer-specific characteristics, and (3) an adaptive fixation prioritization mechanism that refines scanpath predictions by dynamically prioritizing semantic feature maps based on individual observers' attention traits. These novel components allow scanpath models to effectively address the attention variations across different observers. Our method is generally applicable to different datasets, model architectures, and visual tasks, offering a comprehensive tool for transforming general scanpath models into individualized ones. Comprehensive evaluations using value-based and ranking-based metrics verify the method's effectiveness and generalizability. 

:bowing_man: Disclaimer
------------------
For the ScanMatch evaluation metric, we adopt the part of [`GazeParser`](http://gazeparser.sourceforge.net/) package. 
We adopt the implementation of SED and STDE from [`VAME`](https://github.com/dariozanca/VAME) as two of our evaluation metrics mentioned in the [`Visual Attention Models`](https://ieeexplore.ieee.org/document/9207438). 
More specific, we adopt the evaluation metrics provided in [`Scanpath`](https://github.com/chenxy99/Scanpaths).
For ChemLSTM and Gazeformer, we adopt the released code in [`Scanpath`](https://github.com/chenxy99/Scanpaths) and [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer), respectively.
Based on the [`checkpoint`](https://github.com/nocaps-org/updown-baseline/blob/master/updown/utils/checkpointing.py) implementation from [`updown-baseline`](https://github.com/nocaps-org/updown-baseline), we slightly modify it to accommodate our pipeline.

:white_check_mark: Requirements
------------------

- Python 3.9
- PyTorch 1.12.1 (along with torchvision)

- We also provide the conda environment ``user_scanpath.yml``, you can directly run

```bash
$ conda env create -f user_scanpath.yml
```

to create the same environment where we successfully run our codes.

:bookmark_tabs: Tasks
------------------

We provide the corresponding codes for the aforementioned four different datasets and the [pretrained models](https://drive.google.com/file/d/1WnU6EYsl339gJzD3LpF7TSDC0TPxqDal/view?usp=sharing).

- OSIE
- OSIE-ASD
- COCOSearch
- AiR-D

More details of these tasks are provided in their corresponding folders.

:black_nib: Citation
------------------
If you use our code or data, please cite our paper:
```text
@InProceedings{xianyu:2024:individualscanpath,
    author={Xianyu Chen and Ming Jiang and Qi Zhao},
    title = {Beyond Average: Individualized Visual Scanpath Prediction},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2024}
}
```
