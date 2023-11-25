import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import numpy as np
import scipy.stats

import time
import os
import argparse
from os.path import join
from tqdm import tqdm
import datetime
import json
import sys

from dataset.dataset import COCO_Search18, COCO_Search18_evaluation, COCO_Search18_rl
from models.baseline_attention import baseline

from utils.evaluation import comprehensive_evaluation_by_subject
from utils.logger import Logger
from models.sampling import Sampling

parser = argparse.ArgumentParser(description="Scanpath prediction for images")
parser.add_argument("--mode", type=str, default="test", help="Selecting running mode (default: test)")
parser.add_argument("--img_dir", type=str, default="/home/COCOSearch18/TP/images",
                    help="Directory to the image data (stimuli)")
parser.add_argument("--fix_dir", type=str, default="/home/COCOSearch18/TP/processed",
                    help="Directory to the raw fixation file")
parser.add_argument("--detector_dir", type=str, default="/home/COCOSearch18/TP/detectors",
                    help="Directory to detector results")
parser.add_argument("--width", type=int, default=320, help="Width of input data")
parser.add_argument("--height", type=int, default=240, help="Height of input data")
parser.add_argument("--map_width", type=int, default=40, help="Height of output data")
parser.add_argument("--map_height", type=int, default=30, help="Height of output data")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--seed", type=int, default=10, help="Random seed")
parser.add_argument("--detector_threshold", type=float, default=0.8, help="threshold for the detector")
parser.add_argument("--gpu_ids", type=list, default=[0], help="Used gpu ids")
parser.add_argument("--evaluation_dir", type=str, default="../runs/COCOSearch_baseline",
                    help="Resume from a specific directory")
parser.add_argument("--eval_repeat_num", type=int, default=1, help="Repeat number for evaluation")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
parser.add_argument("--max_length", type=int, default=7, help="Maximum length of the generated scanpath")
parser.add_argument("--subject_num", type=int, default=10, help="The number of the subject in OSIE")
parser.add_argument("--embedding_dim", type=int, default=128, help="The dim of embedding for each subject")
parser.add_argument("--dropout", type=float, default=0.2, help="The dropout rate applied to the model")
parser.add_argument("--action_map_num", type=int, default=4, help="The dim of action map")
args = parser.parse_args()

# For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
# These five lines control all the major sources of randomness.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

transform = transforms.Compose([
                                transforms.Resize((args.height, args.width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def main():

    # load logger
    log_dir = args.evaluation_dir
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    log_file = os.path.join(log_dir, "log_test.txt")
    logger = Logger(log_file)

    logger.info("The args corresponding to test process are: ")
    for (key, value) in vars(args).items():
        logger.info("{key:20}: {value:}".format(key=key, value=value))

    test_dataset = COCO_Search18_evaluation(args.img_dir, args.fix_dir, args.detector_dir,
                                                  type="test", split="split3", transform=transform,
                                                  detector_threshold=args.detector_threshold)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.collate_func
    )


    model = baseline(embed_size=512, convLSTM_length=args.max_length, min_length=args.min_length, dropout=args.dropout,
                     subject_num=args.subject_num, map_width=args.map_width, map_height=args.map_height,
                     embedding_dim=args.embedding_dim).cuda()
    sampling = Sampling(convLSTM_length=args.max_length, min_length=args.min_length,
                        map_width=args.map_width, map_height=args.map_height,
                        width=args.width, height=args.height)

    # Load checkpoint to start evaluation.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    test_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint_best.pth"))
    for key in test_checkpoint:
        if key == "optimizer":
            continue
        else:
            model.load_state_dict(test_checkpoint[key], strict=False)

    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)

    model.eval()
    repeat_num = args.eval_repeat_num
    all_gt_fix_vectors = []
    all_predict_fix_vectors = []
    with tqdm(total=len(test_loader) * repeat_num) as pbar_test:
        for i_batch, batch in enumerate(test_loader):
            tmp = [batch["images"], batch["subjects"], batch["fix_vectors"], batch["attention_maps"],
                   batch["taskints"]]
            tmp = [_ if not torch.is_tensor(_) else _.cuda() for _ in tmp]
            # merge the first two dim
            tmp = [_.view(-1, *_.shape[2:]) if torch.is_tensor(_) else _ for _ in tmp]
            images, subjects, gt_fix_vectors, attention_maps, taskints = tmp
            N, C, H, W = images.shape

            with torch.no_grad():
                predict = model(images, subjects, attention_maps, taskints)

            log_normal_mu = predict["log_normal_mu"]
            log_normal_sigma2 = predict["log_normal_sigma2"]
            all_actions_prob = predict["all_actions_prob"]

            image_prediction_dict = {_: [] for _ in range(len(batch["img_names"]))}
            all_gt_fix_vectors.extend(gt_fix_vectors)
            for trial in range(repeat_num):
                samples = sampling.random_sample(all_actions_prob, log_normal_mu, log_normal_sigma2)
                prob_sample_actions = samples["selected_actions_probs"]
                durations = samples["durations"]
                sample_actions = samples["selected_actions"]
                sampling_random_predict_fix_vectors, _, _ = sampling.generate_scanpath(
                    images, prob_sample_actions, durations, sample_actions)

                for idx in range(len(batch["img_names"])):
                    image_prediction_dict[idx].extend(
                        sampling_random_predict_fix_vectors[idx * args.subject_num:(idx + 1) * args.subject_num])

                pbar_test.update(1)

            all_predict_fix_vectors.extend(list(image_prediction_dict.values()))

        cur_metrics, cur_metrics_std, score_details = comprehensive_evaluation_by_subject(all_gt_fix_vectors,
                                                                                          all_predict_fix_vectors,
                                                                                          args)

        # Print and log all evaluation metrics to tensorboard.
        logger.info("The metrics for best model performance are: ")
        for metrics_key in cur_metrics.keys():
            for (metric_name, metric_value) in cur_metrics[metrics_key].items():
                logger.info("{metrics_key:10}-{metric_name:15}: {metric_value:.4f}".format
                            (metrics_key=metrics_key, metric_name=metric_name, metric_value=metric_value))


if __name__ == "__main__":
    main()
