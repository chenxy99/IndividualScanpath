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

from dataset.dataset import OSIE, OSIE_evaluation
# from models.baseline_attention import baseline
# from models.baseline_attention_with_user_map import baseline
# from models.baseline_attention_with_user_embedding_with_map import baseline
from models.baseline_attention_with_user_embedding_with_map_plus_attention import baseline
from utils.evaluation import human_evaluation, evaluation, evaluation_by_subject, human_evaluation_by_subject, \
    comprehensive_evaluation_by_subject
from utils.logger import Logger
from models.sampling import Sampling

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Scanpath prediction for images")
parser.add_argument("--mode", type=str, default="test", help="Selecting running mode (default: test)")
parser.add_argument("--img_dir", type=str, default="/home/CVPR_2024/OSIE_autism/stimuli",
                    help="Directory to the image data (stimuli)")
parser.add_argument("--fix_dir", type=str, default="/home/CVPR_2024/OSIE_autism/processed",
                    help="Directory to the raw fixation file")
parser.add_argument("--width", type=int, default=320, help="Width of input data")
parser.add_argument("--height", type=int, default=240, help="Height of input data")
parser.add_argument("--map_width", type=int, default=40, help="Height of output data")
parser.add_argument("--map_height", type=int, default=30, help="Height of output data")
parser.add_argument("--batch", type=int, default=16, help="Batch size")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--gpu_ids", type=list, default=[0], help="Used gpu ids")
parser.add_argument("--evaluation_dir", type=str, default="../../runs/OSIE_Autism_baseline",
                    help="Resume from a specific directory")
parser.add_argument("--eval_repeat_num", type=int, default=1, help="Repeat number for evaluation")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")
parser.add_argument("--subject_num", type=int, default=39, help="The number of the subject in OSIE")
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

    model = baseline(embed_size=512, convLSTM_length=args.max_length, min_length=args.min_length, dropout=args.dropout,
                     subject_num=args.subject_num, embedding_dim=args.embedding_dim)

    # Load checkpoint to start evaluation.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    test_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint_best.pth"))
    for key in test_checkpoint:
        if key == "optimizer":
            continue
        else:
            model.load_state_dict(test_checkpoint[key])

    subject_embedding = model.subject_embedding.weight.detach()

    spatial_subject_embedding = model.spatial_subj_embed(subject_embedding).detach().numpy()

    semantic_subject_embedding = model.semantic_subj_embed(subject_embedding).detach().numpy()

    head_proj_subject_embedding = model.subject_attention.proj_subject(subject_embedding).detach().numpy()

    proj_subject_embedding = model.object_head.attention_module.proj_subject(subject_embedding).detach().numpy()

    concat = np.concatenate([spatial_subject_embedding, semantic_subject_embedding, head_proj_subject_embedding, proj_subject_embedding], 1)

    concat = (concat - concat.mean(0, keepdims=True)) / (concat.std(0, keepdims=True) + 1e-16)

    variables = [concat]
    titles = ["concat"]

    # Color palette dictionary
    colors = {"ASD": "red", "Controls": "blue"}
    alpha = 0.5
    for variable, title in zip(variables, titles):
        tsne = TSNE(
            n_components=2,
            n_iter=500,
            n_iter_without_progress=150,
            n_jobs=2,
            random_state=5,
        )
        # plt.figure(figsize=(6, 6), dpi=300)
        tsne_result = tsne.fit_transform(variable)
        group = ["ASD"] * 20 + ["Controls"] * 19
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': group})
        tsne_result_df.to_csv('lstm_tsne_result.csv', index=False)
        fig, ax = plt.subplots(1, figsize=(6, 6), dpi=300)
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=200, palette=colors,
                        alpha=alpha)
        lim = (tsne_result[:, 0].min() - 0.2, tsne_result[:, 0].max() + 0.2)
        ax.set_xlim(lim)
        lim = (tsne_result[:, 1].min() - 0.2, tsne_result[:, 1].max() + 0.2)
        ax.set_ylim(lim)
        plt.ylabel("")
        plt.xlabel("")
        plt.yticks([])
        plt.xticks([])
        ax.set_aspect('equal')
        plt.title("ChenLSTM-ISP")
        for idx in range(39):
            if len(str(idx + 1)) == 1:
                idx_str = " " + str(idx + 1)
            else:
                idx_str = str(idx + 1)

            plt.text(tsne_result[idx, 0] - 0.04, tsne_result[idx, 1] - 0.03, s=idx_str, fontsize=10)
        plt.legend(ncol=2, columnspacing=2.0)
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    main()
