import torch
import torch.nn as nn
import torch.optim as optim
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

from models.subject_encoder_decoder_models import Transformer
from models.gazeformer_subject_encoder_decoder import gazeformer
from utils.evaluation import human_evaluation, evaluation, evaluation_by_subject, human_evaluation_by_subject, \
    comprehensive_evaluation_by_subject
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
parser.add_argument("--img_dir", type=str, default="/srv/CVPR_2024/data/OSIE_autism/images", help="Directory to the image data (stimuli)")
parser.add_argument("--feat_dir", type=str, default="/srv/CVPR_2024/data/OSIE_autism/image_features",
                    help="Directory to the image feature data (stimuli)")
parser.add_argument("--fix_dir", type=str, default="/srv/CVPR_2024/data/OSIE_autism/processed", help="Directory to the raw fixation file")
parser.add_argument("--width", type=int, default=512, help="Width of input data")
parser.add_argument("--height", type=int, default=384, help="Height of input data")
parser.add_argument("--origin_width", type=int, default=800, help="original Width of input data")
parser.add_argument("--origin_height", type=int, default=600, help="original Height of input data")
parser.add_argument('--im_h', default=24, type=int, help="Height of feature map input to encoder")
parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--gpu_ids", type=list, default=[0], help="Used gpu ids")
parser.add_argument("--evaluation_dir", type=str, default="../../runs/Autism_baseline",
                    help="Resume from a specific directory")
parser.add_argument("--eval_repeat_num", type=int, default=1, help="Repeat number for evaluation")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")

parser.add_argument('--patch_size', default=16, type=int,
                        help="Patch size of feature map input with respect to fixation image dimensions (320X512)")
parser.add_argument('--num_encoder', default=6, type=int, help="Number of transformer encoder layers")
parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimensionality of transformer layers")
parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
parser.add_argument('--img_hidden_dim', default=2048, type=int, help="Channel size of initial ResNet feature map")
parser.add_argument('--lm_hidden_dim', default=768, type=int,
                    help="Dimensionality of target embeddings from language model")
parser.add_argument('--encoder_dropout', default=0.1, type=float, help="Encoder dropout rate")
parser.add_argument('--decoder_dropout', default=0.2, type=float, help="Decoder and fusion step dropout rate")
parser.add_argument('--cls_dropout', default=0.4, type=float, help="Final scanpath prediction dropout rate")

parser.add_argument('--cuda', default=3, type=int, help="CUDA core to load models and data")
parser.add_argument("--subject_num", type=int, default=39, help="The number of the subject in OSIE")
parser.add_argument("--subject_feature_dim", type=int, default=128, help="The dim of the subject feature")
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

    device = torch.device('cuda:{}'.format(args.cuda))

    transformer = Transformer(num_encoder_layers=args.num_encoder, nhead=args.nhead,
                              subject_feature_dim=args.subject_feature_dim, d_model=args.hidden_dim,
                              num_decoder_layers=args.num_decoder, encoder_dropout=args.encoder_dropout,
                              decoder_dropout=args.decoder_dropout, dim_feedforward=args.hidden_dim,
                              img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device,
                              args=args).cpu()

    model = gazeformer(transformer, spatial_dim=(args.im_h, args.im_w), args=args,
                       subject_num=args.subject_num, subject_feature_dim=args.subject_feature_dim,
                       action_map_num=args.action_map_num,
                       dropout=args.cls_dropout, max_len=args.max_length).cpu()

    # Load checkpoint to start evaluation.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    test_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint.pth"))
    for key in test_checkpoint:
        if key == "optimizer":
            continue
        else:
            model.load_state_dict(test_checkpoint[key])

    subject_embedding = model.subject_embed.weight.detach()

    spatial_subject_embedding = model.transformer.integration_module.subj_spatial_feature(subject_embedding).cpu().detach().numpy()

    semantic_subject_embedding = model.transformer.integration_module.subj_semantic_feature(subject_embedding).cpu().detach().numpy()

    head_proj_subject_embedding = model.transformer.subject_attention_module.proj_subject(subject_embedding).cpu().detach().numpy()

    proj_subject_embedding = model.attention_module.proj_subject(subject_embedding).cpu().detach().numpy()

    concat = np.concatenate([spatial_subject_embedding, semantic_subject_embedding, head_proj_subject_embedding, proj_subject_embedding], 1)

    concat = (concat - concat.mean(0, keepdims=True)) / (concat.std(0, keepdims=True) + 1e-16)

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
