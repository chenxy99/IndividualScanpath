# The name of this experiment.
DATASET_NAME='COCO_Search'
MODEL_NAME='baseline'

# Save logs and models under snap/; make backup.
output=runs/${DATASET_NAME}_${MODEL_NAME}
mkdir -p $output/src
mkdir -p $output/bash
rsync -av  src/* $output/src/
cp $0 $output/bash/run.bash

CUDA_VISIBLE_DEVICES=0,1 python src/train.py \
  --log_root runs/${DATASET_NAME}_${MODEL_NAME} --seed 10 --epoch 40 --start_rl_epoch 20
