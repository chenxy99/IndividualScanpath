# Individualized Visual Scanpath Prediction in COCO Search18 Dataset

This code implements the prediction of individualized scanpaths in visual search task, e.g., COCO Search18 Dataset. 
There are two different individualized visual scanpath prediction architectures (e.g., ChenLSTM and Gazeformer).
You can go to the corresponding sub-folder `ChenLSTMISP` and `GazeformerISP` to run the code, respectively.

Datasets
------------------
The description of how to handle the original data is in [`COCO-Search18`](https://sites.google.com/view/cocosearch/home) and [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformerguidance). 
The data split, the bounding box annotations, and the original implementation of Gazeformer can be downloaded from [`link`](https://github.com/cvlab-stonybrook/Gazeformerguidance). 
The pre-processed object bounding boxes from the object detector is obtained by [`CenterNet`](https://github.com/xingyizhou/CenterNet), and alternatively, you can download it from [`link`](https://drive.google.com/file/d/1f_Ha5ppPKCngARg7_W5AlqvP6Q_N8LRu/view?usp=sharing).

We structure `<dataset_root>` as follows
```
<dataset_root>
    -- ./detectors
        -- coco_search18_detector.json                  # bounding box annotation from an object detector
    -- ./fixations                                      # fixation and the training and validation splits
        -- coco_search18_fixations_TP_train.json        # COCO-Search18 training fixations
        -- coco_search18_fixations_TP_validation.json   # COCO-Search18 validation fixations
        -- coco_search18_fixations_TP_test.json         # COCO-Search18 test fixations
    -- ./images                                         # image stimuli
    -- ./bbox_annos.npy                                 # bounding box annotation for each image (available at COCO)
    -- ./embeddings.npy                                 # sentence semantic embedding for Gazeformer
```

Training your own network on COCO Search18 dataset
------------------

We set all the corresponding hyper-parameters in ``opt.py``. 

The `train.py` script will dump checkpoints into the folder specified by `--log_root` (default = `./runs/`). You can also set the other hyper-parameters in `opt.py` or define them in the `bash/train.sh`.

- `--img_dir` Folder to the image data (stimuli), e.g., `<dataset_root>/stimuli`.
- `--fix_dir` Folder to the raw fixations, e.g., `<dataset_root>/fixations`.
- `--detector_dir` Folder to the detector results, e.g., `<dataset_root>/fixations`.
- `--epoch` The number of total epochs.
- `--start_rl_epoch` Start to use reinforcement learning at the predefined epoch.
- `--detector_threshold` We would only use the detection results if its confidence is larger than the given `detector_threshold`.
- `--subject_num` The number of different subject in the dataset.
- `--action_map_num` The hyper-parameter to determine the number of the maps that combine to the action map.

You can also use the following commands to train your own network. Then you can run the following commands to evaluate the performance of your trained model on test split.
```bash
$ sh bash/train.sh
```

Evaluate on test split
------------------
```bash
$ CUDA_VISIBLE_DEVICES=0,1 python test.py --evaluation_dir "./runs/your_model"
```
