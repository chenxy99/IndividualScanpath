# Individualized Visual Scanpath Prediction in OSIE Dataset

This code implements the prediction of individualized scanpaths in free-viewing task. There are two different individualized visual scanpath prediction architectures (e.g., ChenLSTM and Gazeformer).
You can go to the corresponding sub-folder `ChenLSTMISP` and `GazeformerISP` to run the code, respectively.

Datasets
------------------

This dataset is mainly based on [`predicting-human-gaze-beyond-pixels`](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels). You need to download [`stimuli`](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/tree/master/data/stimuli) and [`fixations`](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/tree/master/data/eye) and put them in a proper location. Then you can get the splits of this dataset by the execution of the following command 

```bash
$ python ./preprocess/preprocess_fixations.py
```

```bash
$ python ./preprocess/feature_extractor.py
```

Alternatively, we provide the pre-processed fixation files, and you can directly download them from [`link`](https://drive.google.com/drive/folders/1yOJtb5wk7h-NqvVr7vG5GPDP4PqELsZ_?usp=drive_link). The sentence semantic embedding for Gazeformer is in [`link`](https://drive.google.com/file/d/1lLFaoTgo2WVYYL7Mr71clrmGUQ926HEc/view?usp=drive_link).

The typical `<dataset_root>` should be structured as follows

```
<dataset_root>
    -- ./processed                                   # fixation and the training, validation and test splits
        fixations.json
    -- ./stimuli                                     # image stimuli
    -- ./embeddings.npy                              # sentence semantic embedding for Gazeformer
```

Training your own network on OSIE dataset
------------------

We have set all the corresponding hyper-parameters in ``opt.py``. 

The `train.py` script will dump checkpoints into the folder specified by `--log_root` (default = `./runs/`). You can also set the other hyper-parameters in `opt.py`.

- `--img_dir` Directory to the image data (stimuli), e.g., `<dataset_root>/stimuli`.
- `--fix_dir` Directory to the raw fixations, e.g., `<dataset_root>/fixations`.
- `--epoch` The number of total epochs.
- `--start_rl_epoch` Start to use reinforcement learning when reaching this given epoch.
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



