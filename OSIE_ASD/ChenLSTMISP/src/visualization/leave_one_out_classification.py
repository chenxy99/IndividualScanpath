import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
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

from models.baseline_attention.py import baseline
from utils.logger import Logger
from models.sampling import Sampling

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from tqdm import tqdm

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

    concat = np.concatenate(
        [spatial_subject_embedding, semantic_subject_embedding, head_proj_subject_embedding, proj_subject_embedding], 1)


    X = concat
    Y = np.array([0] * 20 + [1] * 19)

    loo = LeaveOneOut()
    prediction = []
    soft_prediction = []
    GT_Y = []
    for i, (train_index, test_index) in tqdm(enumerate(loo.split(X))):
        model = MLPClassifier(random_state=1, hidden_layer_sizes=(100, ), max_iter=300)

        train_X = X[train_index]
        train_Y = Y[train_index]

        test_X = X[test_index]
        test_Y = Y[test_index]
        model.fit(train_X, train_Y)

        pred_Y = model.predict(test_X)
        prediction.append(pred_Y)
        GT_Y.append(test_Y)

        soft_prediction.append(model.predict_proba(test_X)[0, 1])

    prediction = np.array(prediction)
    GT_Y = np.array(GT_Y)
    Acc = (prediction == GT_Y).mean()
    auc = roc_auc_score(GT_Y, soft_prediction)
    print("ACC", Acc)
    print("auc", auc)

if __name__ == "__main__":
    main()
