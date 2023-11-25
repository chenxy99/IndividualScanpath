# Individualized Visual Scanpath Prediction in AiR-D Dataset

This code implements the prediction of human scanpaths in visual question-answering task.  There are two different individualized visual scanpath prediction architectures (e.g., ChenLSTM and Gazeformer).
You can go to the corresponding sub-folder `ChenLSTMISP` and `GazeformerISP` to run the code, respectively.

Datasets
------------------

To process the data, you can follow the instructions provided in [`Scanpaths`](https://github.com/chenxy99/Scanpaths/tree/main/AiR) and you can run the following scripts to process the data

```bash
$ python ./preprocess/preprocess_fixations.py
```

```bash
$ python ./preprocess/feature_extractor.py
```

The typical `<dataset_root>` should be structured as follows

```
<dataset_root>
    -- ./attention_reasoning                        # machine attention from AiR
    -- ./fixations                                  # fixation and the training, validation and test splits
        AiR_fixations_test.json
        AiR_fixations_train.json
        AiR_fixations_validation.json
    -- ./stimuli                                    # image stimuli
    -- ./embeddings.npy                             # sentence semantic embedding for Gazeformer
```

Training your own network on AiR dataset
------------------

We have set all the corresponding hyper-parameters in ``opt.py``. 

The `train.py` script will dump checkpoints into the folder specified by `--log_root` (default = `./assets/`). You can also set the other hyper-parameters in `opt.py`.

- `--img_dir` Directory to the image data (stimuli), e.g., `<dataset_root>/stimuli`.
- `--fix_dir` Directory to the raw fixations, e.g., `<dataset_root>/fixations`.
- `--att_dir` Directory to the attention maps, e.g., `<dataset_root>/attention_reasoning`.
- `--epoch` The number of total epochs.
- `--start_rl_epoch` Start to use reinforcement learning when reaching this given epoch.
- `--supervised_save` The default parameter is `True`. It can save a whole checkpoint before we start to use reinforcement learning to train our model. The saved checkpoint can be treated as an ablation study of self-critical sequential training. We would add `_supervised_save` as a suffix for the checkpoint document.
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
