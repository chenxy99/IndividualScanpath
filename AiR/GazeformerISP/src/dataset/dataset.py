import argparse

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from os.path import join
import json
from PIL import Image
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import scipy.ndimage as filters
from torchvision.transforms import transforms
from tqdm import tqdm
from scipy.io import loadmat

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class AiR(Dataset):
    """
    get AiR data
    """

    def __init__(self,
                 stimuli_dir,
                 feature_dir,
                 fixations_dir,
                 action_map=(30, 40),
                 origin_size=(1680, 1050),
                 resize=(240, 320),
                 max_length=16,
                 blur_sigma=1,
                 type="train",
                 transform=None):
        self.stimuli_dir = stimuli_dir
        self.feature_dir = feature_dir
        self.fixations_dir = fixations_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.max_length = max_length
        self.blur_sigma = blur_sigma
        self.type = type
        self.transform = transform
        self.PAD = [-3, -3, -3]

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        # target embeddings
        self.embedding_dict = np.load(open(join("/".join(stimuli_dir.split("/")[:-1]), 'embeddings.npy'), mode='rb'),
                                      allow_pickle=True).item()

        self.fixations_file = join(self.fixations_dir, "AiR_fixations_{}.json".format(type))
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        self.fixations = fixations

        self.qid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.qid_to_sub.setdefault(fixation['question_id'], []).append(index)
        self.qid = list(self.qid_to_sub.keys())


    def __len__(self):
        return len(self.qid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        qid = self.qid[idx]
        fixation = self.fixations[self.qid_to_sub[qid][0]]
        img_name = fixation["image_id"]
        img_path = join(self.feature_dir, img_name.replace('jpg', 'pth'))
        image_ftrs = torch.load(img_path).unsqueeze(0)

        images = []
        subjects = []
        performances = []
        task_embeddings = []
        target_scanpaths = []
        durations = []
        action_masks = []
        duration_masks = []
        for ids in self.qid_to_sub[qid]:
            fixation = self.fixations[ids]

            origin_size_y, origin_size_x = fixation["height"], fixation["width"]
            self.downscale_x = origin_size_x / self.action_map[1]
            self.downscale_y = origin_size_y / self.action_map[0]

            scanpath = np.zeros((self.max_length, self.action_map[0], self.action_map[1]), dtype=np.float32)
            # the first element denotes the termination action
            target_scanpath = np.zeros((self.max_length, self.action_map[0] * self.action_map[1] + 1), dtype=np.float32)
            duration = np.zeros(self.max_length, dtype=np.float32)
            action_mask = np.zeros(self.max_length, dtype=np.float32)
            duration_mask = np.zeros(self.max_length, dtype=np.float32)

            pos_x = np.array(fixation["X"]).astype(np.float32)
            pos_y = np.array(fixation["Y"]).astype(np.float32)
            duration_raw = np.array(fixation["T_end"]).astype(np.float32) - np.array(fixation["T_start"]).astype(
                np.float32)

            pos_x_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
            pos_y_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
            for index in range(len(pos_x)):
                # only preserve the max_length ground-truth
                if index == self.max_length:
                    break
                pos_x_discrete[index] = (pos_x[index] / self.downscale_x).astype(np.int32)
                pos_y_discrete[index] = (pos_y[index] / self.downscale_y).astype(np.int32)
                duration[index] = duration_raw[index] / 1000.0
                action_mask[index] = 1
                duration_mask[index] = 1
            if action_mask.sum() <= self.max_length - 1:
                action_mask[int(action_mask.sum())] = 1

            for index in range(self.max_length):
                if pos_x_discrete[index] == -1 or pos_y_discrete[index] == -1:
                    target_scanpath[index, 0] = 1
                else:
                    scanpath[index, pos_y_discrete[index], pos_x_discrete[index]] = 1
                    if self.blur_sigma:
                        scanpath[index] = filters.gaussian_filter(scanpath[index], self.blur_sigma)
                        scanpath[index] /= scanpath[index].sum()
                    target_scanpath[index, 1:] = scanpath[index].reshape(-1)

            performance = fixation["subject_answer"] == fixation["answer"] and fixation["subject_answer"] != "faild"

            images.append(image_ftrs)
            durations.append(duration)
            action_masks.append(action_mask)
            duration_masks.append(duration_mask)
            subjects.append(fixation["subject_idx"])
            performances.append(performance)
            task_embedding = self.embedding_dict[qid]
            task_embeddings.append(task_embedding)
            target_scanpaths.append(target_scanpath)


        images = torch.cat(images)
        subjects = np.array(subjects)
        task_embeddings = np.array(task_embeddings)
        target_scanpaths = np.array(target_scanpaths)
        durations = np.array(durations)
        performances = np.array(performances)
        action_masks = np.array(action_masks)
        duration_masks = np.array(duration_masks)

        # self.show_image(image/255)
        # self.show_image(image_resized/255)

        return {
            "image": images,
            "subject": subjects,
            "img_name": img_name,
            "qid": qid,
            "duration": durations,
            "action_mask": action_masks,
            "duration_mask": duration_masks,
            "performance": performances,
            "task_embedding": task_embeddings,
            "target_scanpath": target_scanpaths
        }

    def collate_func(self, batch):

        img_batch = []
        subject_batch = []
        img_name_batch = []
        duration_batch = []
        action_mask_batch = []
        duration_mask_batch = []
        qid_batch = []
        task_embedding_batch = []
        target_scanpath_batch = []
        performance_batch = []

        for sample in batch:
            tmp_img, tmp_subject, tmp_img_name, tmp_duration, tmp_action_mask, tmp_duration_mask, \
                tmp_qid, tmp_task_embedding, tmp_target_scanpath, tmp_performance =\
                sample["image"], sample["subject"], sample["img_name"],\
                sample["duration"], sample["action_mask"], sample["duration_mask"], sample["qid"], sample["task_embedding"], sample["target_scanpath"], sample["performance"]
            img_batch.append(tmp_img)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            qid_batch.append(tmp_qid)
            task_embedding_batch.append(tmp_task_embedding)
            target_scanpath_batch.append(tmp_target_scanpath)
            performance_batch.append(tmp_performance)

        data = dict()
        data["images"] = torch.cat(img_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["durations"] = np.concatenate(duration_batch)
        data["action_masks"] = np.concatenate(action_mask_batch)
        data["duration_masks"] = np.concatenate(duration_mask_batch)
        data["qids"] = qid_batch
        data["task_embeddings"] = np.concatenate(task_embedding_batch)
        data["target_scanpaths"] = np.concatenate(target_scanpath_batch)
        data["performances"] = np.concatenate(performance_batch)

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}
        return data


class AiR_rl(Dataset):
    """
    get AiR data for evaluation
    """

    def __init__(self,
                 stimuli_dir,
                 feature_dir,
                 fixations_dir,
                 action_map=(30, 40),
                 origin_size=(600, 800),
                 resize=(240, 320),
                 type="validation",
                 transform=None):
        self.stimuli_dir = stimuli_dir
        self.feature_dir = feature_dir
        self.fixations_dir = fixations_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.type = type
        self.transform = transform

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        # target embeddings
        self.embedding_dict = np.load(open(join("/".join(stimuli_dir.split("/")[:-1]), 'embeddings.npy'), mode='rb'),
                                      allow_pickle=True).item()


        self.fixations_file = join(self.fixations_dir, "AiR_fixations_{}.json".format(type))
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        self.fixations = fixations

        self.qid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.qid_to_sub.setdefault(fixation['question_id'], []).append(index)
        self.qid = list(self.qid_to_sub.keys())

    def __len__(self):
        # return len(self.imgid) * 15
        # return len(self.fixations)
        return len(self.qid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        qid = self.qid[idx]
        fixation = self.fixations[self.qid_to_sub[qid][0]]
        img_name = fixation["image_id"]
        img_path = join(self.feature_dir, img_name.replace('jpg', 'pth'))
        image_ftrs = torch.load(img_path).unsqueeze(0)

        # image = Image.open(img_path).convert('RGB')
        # if self.transform is not None:
        #     image = self.transform(image)

        images = []
        fix_vectors = []
        subjects = []
        qids = []
        performances = []
        task_embeddings = []
        for ids in self.qid_to_sub[qid]:
            fixation = self.fixations[ids]

            origin_size_y, origin_size_x = fixation["height"], fixation["width"]
            resizescale_x = origin_size_x / self.resize[1]
            resizescale_y = origin_size_y / self.resize[0]

            x_start = np.array(fixation["X"]).astype(np.float32) / resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / resizescale_y
            duration = (np.array(fixation["T_end"]).astype(np.float32)
                        - np.array(fixation["T_start"]).astype(np.float32)) / 1000.0

            length = fixation["length"]

            performance = fixation["subject_answer"] == fixation["answer"] and fixation["subject_answer"] != "faild"
            performances.append(performance)

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)

            task_embedding = self.embedding_dict[qid]
            subjects.append(fixation["subject_idx"])
            images.append(image_ftrs)
            qids.append(qid)
            task_embeddings.append(task_embedding)

        images = torch.cat(images)
        subjects = np.array(subjects)
        task_embeddings = np.array(task_embeddings)
        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "img_name": img_name,
            "subject": subjects,
            "qid": qids,
            "task_embedding": task_embeddings,
            "performance": performances
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        subject_batch = []
        img_name_batch = []
        qid_batch = []
        task_embedding_batch = []
        performance_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_subject, tmp_img_name, tmp_qid, tmp_task_embedding, tmp_performance = \
                sample["image"], sample["fix_vectors"], sample["subject"], sample["img_name"], sample["qid"], sample[
                    "task_embedding"], sample["performance"]
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            qid_batch.append(tmp_qid)
            task_embedding_batch.append(tmp_task_embedding)
            performance_batch.append(tmp_performance)

        data = {}
        data["images"] = torch.cat(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["qids"] = qid_batch
        data["task_embeddings"] = np.concatenate(task_embedding_batch)
        data["performances"] = np.concatenate(performance_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}

        return data

class AiR_evaluation(Dataset):
    """
    get AiR data for evaluation
    """

    def __init__(self,
                 stimuli_dir,
                 feature_dir,
                 fixations_dir,
                 action_map=(30, 40),
                 origin_size=(600, 800),
                 resize=(240, 320),
                 type="validation",
                 transform=None):
        self.stimuli_dir = stimuli_dir
        self.feature_dir = feature_dir
        self.fixations_dir = fixations_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.type = type
        self.transform = transform

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        # target embeddings
        self.embedding_dict = np.load(open(join("/".join(stimuli_dir.split("/")[:-1]), 'embeddings.npy'), mode='rb'),
                                      allow_pickle=True).item()

        self.fixations_file = join(self.fixations_dir, "AiR_fixations_{}.json".format(type))
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        self.fixations = fixations

        self.qid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.qid_to_sub.setdefault(fixation['question_id'], []).append(index)
        self.qid = list(self.qid_to_sub.keys())

    def __len__(self):
        # return len(self.imgid) * 15
        # return len(self.fixations)
        return len(self.qid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        qid = self.qid[idx]
        fixation = self.fixations[self.qid_to_sub[qid][0]]
        img_name = fixation["image_id"]
        img_path = join(self.feature_dir, img_name.replace('jpg', 'pth'))
        image_ftrs = torch.load(img_path).unsqueeze(0)

        # image = Image.open(img_path).convert('RGB')
        # if self.transform is not None:
        #     image = self.transform(image)

        images = []
        fix_vectors = []
        subjects = []
        qids = []
        performances = []
        task_embeddings = []
        for ids in self.qid_to_sub[qid]:
            fixation = self.fixations[ids]

            origin_size_y, origin_size_x = fixation["height"], fixation["width"]
            resizescale_x = origin_size_x / self.resize[1]
            resizescale_y = origin_size_y / self.resize[0]

            x_start = np.array(fixation["X"]).astype(np.float32) / resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / resizescale_y
            duration = (np.array(fixation["T_end"]).astype(np.float32)
                        - np.array(fixation["T_start"]).astype(np.float32)) / 1000.0

            length = fixation["length"]

            performance = fixation["subject_answer"] == fixation["answer"] and fixation["subject_answer"] != "faild"
            performances.append(performance)

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)

            task_embedding = self.embedding_dict[qid]
            subjects.append(fixation["subject_idx"])
            images.append(image_ftrs)
            qids.append(qid)
            task_embeddings.append(task_embedding)

        images = torch.cat(images)
        subjects = np.array(subjects)
        task_embeddings = np.array(task_embeddings)
        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "img_name": img_name,
            "subject": subjects,
            "qid": qids,
            "task_embedding": task_embeddings,
            "performance": performances
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        subject_batch = []
        img_name_batch = []
        qid_batch = []
        task_embedding_batch = []
        performance_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_subject, tmp_img_name, tmp_qid, tmp_task_embedding, tmp_performance = \
                sample["image"], sample["fix_vectors"], sample["subject"], sample["img_name"], sample["qid"], sample[
                    "task_embedding"], sample["performance"]
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            qid_batch.append(tmp_qid)
            task_embedding_batch.append(tmp_task_embedding)
            performance_batch.append(tmp_performance)

        data = {}
        data["images"] = torch.cat(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["qids"] = qid_batch
        data["task_embeddings"] = np.concatenate(task_embedding_batch)
        data["performances"] = np.concatenate(performance_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scanpath prediction for images")
    parser.add_argument("--img_dir", type=str, default="/home/COCOSearch18/TP/images", help="Directory to the image data (stimuli)")
    parser.add_argument("--feat_dir", type=str, default="/home/COCOSearch18/TP/image_features",
                        help="Directory to the image feature data (stimuli)")
    parser.add_argument("--fix_dir", type=str, default="/home/COCOSearch18/TP/processed", help="Directory to the raw fixation file")
    parser.add_argument("--origin_width", type=int, default=1680, help="original Width of input data")
    parser.add_argument("--origin_height", type=int, default=1050, help="original Height of input data")
    parser.add_argument("--width", type=int, default=512, help="Width of input data")
    parser.add_argument("--height", type=int, default=320, help="Height of input data")
    parser.add_argument('--im_h', default=20, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
    parser.add_argument("--blur_sigma", type=float, default=None, help="Standard deviation for Gaussian kernel")
    parser.add_argument("--batch", type=int, default=20, help="Batch size")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = COCOSearch_by_subject(args.img_dir, args.feat_dir, args.fix_dir, action_map=(args.im_h, args.im_w),
                         resize=(args.height, args.width), origin_size=(args.origin_height, args.origin_width),
                         blur_sigma=args.blur_sigma, type="train", transform=transform)
    a = train_dataset[5]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_func
    )

    # for batch in train_loader:
    #     a = 1

    test_dataset = COCOSearch_evaluation(args.img_dir, args.feat_dir, args.fix_dir, action_map=(args.im_h, args.im_w),
                         resize=(args.height, args.width), origin_size=(args.origin_height, args.origin_width),
                         type="test", transform=transform)
    a = test_dataset[5]

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        collate_fn=test_dataset.collate_func
    )

    for batch in test_loader:
        a = 1