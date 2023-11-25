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

class COCOSearch_by_subject(Dataset):
    """
    get OSIE data
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

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        # target embeddings
        self.embedding_dict = np.load(open(join(stimuli_dir, "..", 'embeddings.npy'), mode='rb'), allow_pickle=True).item()

        self.fixations_file = join(self.fixations_dir, "fixations.json")
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        fixations = [_ for _ in fixations if _["split"] == type]
        self.fixations = fixations

    def __len__(self):
        return len(self.fixations)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        fixation = self.fixations[idx]
        task = fixation["task"]
        img_name = fixation["name"]
        img_path = join(self.feature_dir, task, img_name.replace('jpg', 'pth'))
        image_ftrs = torch.load(img_path).unsqueeze(0)

        # image = Image.open(img_path).convert('RGB')
        # original_im_w = image.size[0]
        # original_im_h = image.size[1]
        # if self.transform is not None:
        #     image = self.transform(image)

        # normalize
        # pos_x = np.array(fixation["X"]).astype(np.float32) / original_im_w * self.resize[1]
        # pos_y = np.array(fixation["Y"]).astype(np.float32) / original_im_h * self.resize[0]
        # duration_raw = np.array(fixation["T"]).astype(np.float32)

        # if use cvpr paper, we do not need to normalize
        pos_x = np.array(fixation["X"]).astype(np.float32)
        pos_y = np.array(fixation["Y"]).astype(np.float32)
        duration_raw = np.array(fixation["T"]).astype(np.float32)

        if pos_y.shape[0] < self.max_length:
            padding = np.tile(np.array(self.PAD[0]), [self.max_length - pos_y.shape[0]])
            pos_y = np.concatenate([pos_y, padding], axis=0)
        else:
            pos_y = pos_y[:self.max_length]

        if pos_x.shape[0] < self.max_length:
            padding = np.tile(np.array(self.PAD[1]), [self.max_length - pos_x.shape[0]])
            pos_x = np.concatenate([pos_x, padding], axis=0)
        else:
            pos_x = pos_x[:self.max_length]

        if duration_raw.shape[0] < self.max_length:
            padding = np.tile(np.array(self.PAD[2]), [self.max_length - duration_raw.shape[0]])
            duration_raw = np.concatenate([duration_raw, padding], axis=0)
        else:
            duration_raw = duration_raw[:self.max_length]

        target = np.ascontiguousarray(np.array([pos_y, pos_x, duration_raw], dtype=np.int64).T)

        target_padding_mask = target[:, 0] == self.PAD[0]
        # in the middle of the image
        firstfix = np.array([self.resize[0] / 2, self.resize[1] / 2], np.int64)

        subject = fixation["subject"] - 1
        task_embedding = self.embedding_dict[task]

        # self.show_image(image/255)
        # self.show_image(image_resized/255)

        return {
            "image": image_ftrs,
            "subject": subject,
            "target": target,
            "target_padding_mask": target_padding_mask,
            "firstfix": firstfix,
            "img_name": img_name,
            "task": task,
            "task_embedding": task_embedding,
        }

    def collate_func(self, batch):

        img_batch = []
        target_batch = []
        target_padding_mask_batch = []
        firstfix_batch = []
        subject_batch = []
        img_name_batch = []
        task_batch = []
        task_embedding_batch = []

        for sample in batch:
            tmp_img, tmp_target, tmp_target_padding_mask,\
            tmp_firstfix, tmp_subject, tmp_img_name, tmp_task, tmp_task_embedding =\
                sample["image"], sample["target"], sample["target_padding_mask"],\
                sample["firstfix"], sample["subject"], sample["img_name"], sample["task"], sample["task_embedding"]
            img_batch.append(tmp_img)
            target_batch.append(tmp_target)
            target_padding_mask_batch.append(tmp_target_padding_mask)
            firstfix_batch.append(tmp_firstfix)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            task_batch.append(tmp_task)
            task_embedding_batch.append(tmp_task_embedding)

        data = dict()
        data["images"] = torch.cat(img_batch)
        data["targets"] = np.stack(target_batch)
        data["target_padding_masks"] = np.stack(target_padding_mask_batch)
        data["firstfixs"] = np.stack(firstfix_batch)
        data["subjects"] = np.stack(subject_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = task_batch
        data["task_embeddings"] = np.stack(task_embedding_batch)

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor
        data = {k:v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}

        return data


class COCOSearch(Dataset):
    """
    get OSIE data
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

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        self.downscale_x = resize[1] / action_map[1]
        self.downscale_y = resize[0] / action_map[0]

        # target embeddings
        self.embedding_dict = np.load(open(join(stimuli_dir, "..", 'embeddings.npy'), mode='rb'),
                                      allow_pickle=True).item()

        self.fixations_file = join(self.fixations_dir, "fixations.json")
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        fixations = [_ for _ in fixations if _["split"] == type]
        self.fixations = fixations

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault("{}/{}".format(fixation['task'], fixation['name']), []).append(index)
        self.imgid = list(self.imgid_to_sub.keys())


    def __len__(self):
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_name = self.imgid[idx]
        img_path = join(self.feature_dir, img_name.replace('jpg', 'pth'))
        image_ftrs = torch.load(img_path).unsqueeze(0)

        images = []
        subjects = []
        tasks = []
        task_embeddings = []
        target_scanpaths = []
        durations = []
        action_masks = []
        duration_masks = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]

            scanpath = np.zeros((self.max_length, self.action_map[0], self.action_map[1]), dtype=np.float32)
            # the first element denotes the termination action
            target_scanpath = np.zeros((self.max_length, self.action_map[0] * self.action_map[1] + 1), dtype=np.float32)
            duration = np.zeros(self.max_length, dtype=np.float32)
            action_mask = np.zeros(self.max_length, dtype=np.float32)
            duration_mask = np.zeros(self.max_length, dtype=np.float32)
            task = fixation["task"]

            # if use cvpr paper, we do not need to normalize
            pos_x = np.array(fixation["X"]).astype(np.float32)
            pos_y = np.array(fixation["Y"]).astype(np.float32)
            duration_raw = np.array(fixation["T"]).astype(np.float32)

            pos_x_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
            pos_y_discrete = np.zeros(self.max_length, dtype=np.int32) - 1
            for index in range(len(pos_x)):
                # only preserve the max_length ground-truth
                if index == self.max_length:
                    break
                # since pixel is start from 1 ~ max based on matlab
                pos_x_discrete[index] = ((pos_x[index] - 1) / self.downscale_x).astype(np.int32)
                pos_y_discrete[index] = ((pos_y[index] - 1) / self.downscale_y).astype(np.int32)
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

            images.append(image_ftrs)
            durations.append(duration)
            action_masks.append(action_mask)
            duration_masks.append(duration_mask)
            subjects.append(fixation["subject"] - 1)
            tasks.append(task)
            task_embedding = self.embedding_dict[task]
            task_embeddings.append(task_embedding)
            target_scanpaths.append(target_scanpath)


        images = torch.cat(images)
        subjects = np.array(subjects)
        task_embeddings = np.array(task_embeddings)
        target_scanpaths = np.array(target_scanpaths)
        durations = np.array(durations)
        action_masks = np.array(action_masks)
        duration_masks = np.array(duration_masks)

        # self.show_image(image/255)
        # self.show_image(image_resized/255)

        return {
            "image": images,
            "subject": subjects,
            "img_name": img_name,
            "duration": durations,
            "action_mask": action_masks,
            "duration_mask": duration_masks,
            "task": tasks,
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
        task_batch = []
        task_embedding_batch = []
        target_scanpath_batch = []

        for sample in batch:
            tmp_img, tmp_subject, tmp_img_name, tmp_duration, tmp_action_mask, tmp_duration_mask, \
                tmp_task, tmp_task_embedding, tmp_target_scanpath =\
                sample["image"], sample["subject"], sample["img_name"],\
                sample["duration"], sample["action_mask"], sample["duration_mask"], sample["task"], sample["task_embedding"], sample["target_scanpath"]
            img_batch.append(tmp_img)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            task_batch.append(tmp_task)
            task_embedding_batch.append(tmp_task_embedding)
            target_scanpath_batch.append(tmp_target_scanpath)

        data = dict()
        data["images"] = torch.cat(img_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["durations"] = np.concatenate(duration_batch)
        data["action_masks"] = np.concatenate(action_mask_batch)
        data["duration_masks"] = np.concatenate(duration_mask_batch)
        data["tasks"] = task_batch
        data["task_embeddings"] = np.concatenate(task_embedding_batch)
        data["target_scanpaths"] = np.concatenate(target_scanpath_batch)

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}
        return data


class COCOSearch_rl(Dataset):
    """
    get OSIE data for evaluation
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
        self.embedding_dict = np.load(open(join(stimuli_dir, "..", 'embeddings.npy'), mode='rb'),
                                      allow_pickle=True).item()

        self.fixations_file = join(self.fixations_dir, "fixations.json")
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        fixations = [_ for _ in fixations if _["split"] == type]
        self.fixations = fixations

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault("{}/{}".format(fixation['task'], fixation['name']), []).append(index)
        self.imgid = list(self.imgid_to_sub.keys())

    def __len__(self):
        # return len(self.imgid) * 15
        # return len(self.fixations)
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_name = self.imgid[idx]
        img_path = join(self.feature_dir, img_name.replace('jpg', 'pth'))
        image_ftrs = torch.load(img_path).unsqueeze(0)

        # image = Image.open(img_path).convert('RGB')
        # if self.transform is not None:
        #     image = self.transform(image)

        images = []
        fix_vectors = []
        subjects = []
        firstfixs = []
        tasks = []
        task_embeddings = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]
            task = fixation["task"]

            # normalize
            # x_start = np.array(fixation["X"]).astype(np.float32) / self.resizescale_x
            # y_start = np.array(fixation["Y"]).astype(np.float32) / self.resizescale_y
            # duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            # if use cvpr paper, we do not need to normalize
            x_start = np.array(fixation["X"]).astype(np.float32)
            y_start = np.array(fixation["Y"]).astype(np.float32)
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            # in the middle of the image
            firstfix = np.array([self.resize[0] / 2, self.resize[1] / 2], np.int64)

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})

            task_embedding = self.embedding_dict[task]

            fix_vectors.append(fix_vector)
            subjects.append(fixation["subject"] - 1)
            firstfixs.append(firstfix)
            images.append(image_ftrs)
            tasks.append(task)
            task_embeddings.append(task_embedding)

        images = torch.cat(images)
        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "firstfix": firstfixs,
            "img_name": img_name,
            "subject": subjects,
            "task": tasks,
            "task_embedding": task_embeddings
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        firstfix_batch = []
        subject_batch = []
        img_name_batch = []
        task_batch = []
        task_embedding_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_firstfix, tmp_subject, tmp_img_name, tmp_task, tmp_task_embedding = \
                sample["image"], sample["fix_vectors"], sample["firstfix"], sample["subject"], sample["img_name"], sample["task"], sample[
                    "task_embedding"]
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            firstfix_batch.append(tmp_firstfix)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            task_batch.append(tmp_task)
            task_embedding_batch.append(tmp_task_embedding)

        data = {}
        data["images"] = torch.cat(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["firstfixs"] = np.concatenate(firstfix_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = task_batch
        data["task_embeddings"] = np.concatenate(task_embedding_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}

        return data

class COCOSearch_evaluation(Dataset):
    """
    get OSIE data for evaluation
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
        self.embedding_dict = np.load(open(join(stimuli_dir, "..", 'embeddings.npy'), mode='rb'),
                                      allow_pickle=True).item()

        self.fixations_file = join(self.fixations_dir, "fixations.json")
        with open(self.fixations_file) as json_file:
            fixations = json.load(json_file)
        fixations = [_ for _ in fixations if _["split"] == type]
        self.fixations = fixations

        self.imgid_to_sub = {}
        for index, fixation in enumerate(self.fixations):
            self.imgid_to_sub.setdefault("{}/{}".format(fixation['task'], fixation['name']), []).append(index)
        self.imgid = list(self.imgid_to_sub.keys())

        objects = set([_.split("/")[0] for _ in self.imgid])

    def __len__(self):
        # return len(self.imgid) * 15
        # return len(self.fixations)
        return len(self.imgid)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        img_name = self.imgid[idx]
        img_path = join(self.feature_dir, img_name.replace('jpg', 'pth'))
        image_ftrs = torch.load(img_path).unsqueeze(0)

        # image = Image.open(img_path).convert('RGB')
        # if self.transform is not None:
        #     image = self.transform(image)

        images = []
        fix_vectors = []
        subjects = []
        firstfixs = []
        tasks = []
        task_embeddings = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]
            task = fixation["task"]

            # normalize
            # x_start = np.array(fixation["X"]).astype(np.float32) / self.resizescale_x
            # y_start = np.array(fixation["Y"]).astype(np.float32) / self.resizescale_y
            # duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            # if use cvpr paper, we do not need to normalize
            x_start = np.array(fixation["X"]).astype(np.float32)
            y_start = np.array(fixation["Y"]).astype(np.float32)
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            # in the middle of the image
            firstfix = np.array([self.resize[0] / 2, self.resize[1] / 2], np.int64)

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})

            task_embedding = self.embedding_dict[task]

            fix_vectors.append(fix_vector)
            subjects.append(fixation["subject"] - 1)
            firstfixs.append(firstfix)
            images.append(image_ftrs)
            tasks.append(task)
            task_embeddings.append(task_embedding)

        images = torch.cat(images)
        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "firstfix": firstfixs,
            "img_name": img_name,
            "subject": subjects,
            "task": tasks,
            "task_embedding": task_embeddings
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        firstfix_batch = []
        subject_batch = []
        img_name_batch = []
        task_batch = []
        task_embedding_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_firstfix, tmp_subject, tmp_img_name, tmp_task, tmp_task_embedding = \
                sample["image"], sample["fix_vectors"], sample["firstfix"], sample["subject"], sample["img_name"], sample["task"], sample[
                    "task_embedding"]
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            firstfix_batch.append(tmp_firstfix)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            task_batch.append(tmp_task)
            task_embedding_batch.append(tmp_task_embedding)

        data = {}
        data["images"] = torch.stack(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["firstfixs"] = np.stack(firstfix_batch)
        data["subjects"] = np.array(subject_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = task_batch
        data["task_embeddings"] = np.array(task_embedding_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor

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