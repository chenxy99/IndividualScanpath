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
                 AiR_stimuli_dir,
                 AiR_fixations_dir,
                 AiR_attention_bbox_dir,
                 action_map=(30, 40),
                 origin_size=(600, 800),
                 resize=(240, 320),
                 max_length=16,
                 blur_sigma=1,
                 type="train",
                 transform=None):
        self.AiR_stimuli_dir = AiR_stimuli_dir
        self.AiR_fixations_dir = AiR_fixations_dir
        self.AiR_attention_bbox_dir = AiR_attention_bbox_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.max_length = max_length
        self.blur_sigma = blur_sigma
        self.type = type
        self.transform = transform

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        self.fixations_file = join(self.AiR_fixations_dir, "AiR_fixations_{}.json".format(type))
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
        img_path = join(self.AiR_stimuli_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        attention_bbox = np.load(join(self.AiR_attention_bbox_dir, qid + ".npy")).astype(np.float32)
        attention_map = resize(attention_bbox, self.action_map)
        attention_map /= attention_map.max()
        attention_map = np.expand_dims(attention_map, axis=0)

        images = []
        subjects = []
        performances = []
        target_scanpaths = []
        durations = []
        action_masks = []
        duration_masks = []
        attention_maps = []
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

            images.append(image)
            durations.append(duration)
            action_masks.append(action_mask)
            duration_masks.append(duration_mask)
            subjects.append(fixation["subject_idx"])
            performances.append(performance)
            target_scanpaths.append(target_scanpath)
            attention_maps.append(attention_map)

        images = torch.stack(images)
        target_scanpaths = np.stack(target_scanpaths)
        durations = np.stack(durations)
        action_masks = np.stack(action_masks)
        duration_masks = np.stack(duration_masks)
        subjects = np.array(subjects)
        performances = np.array(performances)
        attention_maps = np.stack(attention_maps)

        # self.show_image(image/255)
        # self.show_image(image_resized/255)

        return {
            "image": images,
            "target_scanpath": target_scanpaths,
            "duration": durations,
            "action_mask": action_masks,
            "duration_mask": duration_masks,
            "subject": subjects,
            "qid": qid,
            "img_name": img_name,
            "performance": performances,
            "attention_map": attention_maps,
        }

    def collate_func(self, batch):

        img_batch = []
        scanpath_batch = []
        duration_batch = []
        action_mask_batch = []
        duration_mask_batch = []
        subject_batch = []
        img_name_batch = []
        qid_batch = []
        performance_batch = []
        attention_map_batch = []

        for sample in batch:
            tmp_img, tmp_scanpath, tmp_duration,\
            tmp_action_mask, tmp_duration_mask, tmp_subject, tmp_img_name, tmp_qid, tmp_performance, tmp_attention_map =\
                sample["image"], sample["target_scanpath"], sample["duration"],\
                sample["action_mask"], sample["duration_mask"], sample["subject"], sample["img_name"], \
                sample["qid"], sample["performance"], sample["attention_map"]
            img_batch.append(tmp_img)
            scanpath_batch.append(tmp_scanpath)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            qid_batch.append(tmp_qid)
            performance_batch.append(tmp_performance)
            attention_map_batch.append(tmp_attention_map)

        data = dict()
        data["images"] = torch.cat(img_batch)
        data["scanpaths"] = np.concatenate(scanpath_batch)
        data["durations"] = np.concatenate(duration_batch)
        data["action_masks"] = np.concatenate(action_mask_batch)
        data["duration_masks"] = np.concatenate(duration_mask_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["qids"] = qid_batch
        data["performances"] = np.concatenate(performance_batch)
        data["attention_maps"] = np.concatenate(attention_map_batch)

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}
        # if self.type == "train":


        return data



class AiR_evaluation(Dataset):
    """
    get AiR data for evaluation
    """

    def __init__(self,
                 AiR_stimuli_dir,
                 AiR_fixations_dir,
                 AiR_attention_bbox_dir,
                 action_map=(30, 40),
                 origin_size=(600, 800),
                 resize=(240, 320),
                 type="validation",
                 transform=None):
        self.AiR_stimuli_dir = AiR_stimuli_dir
        self.AiR_fixations_dir = AiR_fixations_dir
        self.AiR_attention_bbox_dir = AiR_attention_bbox_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.type = type
        self.transform = transform

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        self.fixations_file = join(self.AiR_fixations_dir, "AiR_fixations_{}.json".format(type))
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
        img_path = join(self.AiR_stimuli_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        attention_bbox = np.load(join(self.AiR_attention_bbox_dir, qid + ".npy")).astype(np.float32)
        attention_map = resize(attention_bbox, self.action_map)
        attention_map /= attention_map.max()
        attention_map = np.expand_dims(attention_map, axis=0)

        fix_vectors = []
        images = []
        subjects = []
        performances = []
        attention_maps = []

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

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)
            images.append(image)
            subjects.append(fixation["subject_idx"])
            performances.append(performance)
            attention_maps.append(attention_map)

        images = torch.stack(images)
        subjects = np.array(subjects)
        performances = np.array(performances)
        attention_maps = np.stack(attention_maps)

        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "img_name": img_name,
            "subject": subjects,
            "qid": qid,
            "performance": performances,
            "attention_map": attention_maps
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        subject_batch = []
        img_name_batch = []
        qid_batch = []
        performance_batch = []
        attention_map_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_subject, tmp_img_name, tmp_qid, tmp_performance, tmp_attention_map = \
                sample["image"], sample["fix_vectors"], sample["subject"], sample["img_name"], sample["qid"], sample[
                    "performance"], sample["attention_map"]
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            qid_batch.append(tmp_qid)
            performance_batch.append(tmp_performance)
            attention_map_batch.append(tmp_attention_map)

        data = {}
        data["images"] = torch.cat(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["qids"] = qid_batch
        data["performances"] = np.concatenate(performance_batch)
        data["attention_maps"] = np.concatenate(attention_map_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}

        return data


class AiR_rl(Dataset):
    """
    get AiR data for reinforcement learning
    """

    def __init__(self,
                 AiR_stimuli_dir,
                 AiR_fixations_dir,
                 AiR_attention_bbox_dir,
                 action_map=(30, 40),
                 origin_size=(600, 800),
                 resize=(240, 320),
                 type="validation",
                 transform=None):
        self.AiR_stimuli_dir = AiR_stimuli_dir
        self.AiR_fixations_dir = AiR_fixations_dir
        self.AiR_attention_bbox_dir = AiR_attention_bbox_dir
        self.action_map = action_map
        self.origin_size = origin_size
        self.resize = resize
        self.type = type
        self.transform = transform

        self.downscale_x = origin_size[1] / action_map[1]
        self.downscale_y = origin_size[0] / action_map[0]

        self.resizescale_x = origin_size[1] / resize[1]
        self.resizescale_y = origin_size[0] / resize[0]

        self.fixations_file = join(self.AiR_fixations_dir, "AiR_fixations_{}.json".format(type))
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
        img_path = join(self.AiR_stimuli_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        attention_bbox = np.load(join(self.AiR_attention_bbox_dir, qid + ".npy")).astype(np.float32)
        attention_map = resize(attention_bbox, self.action_map)
        attention_map /= attention_map.max()
        attention_map = np.expand_dims(attention_map, axis=0)

        fix_vectors = []
        images = []
        subjects = []
        performances = []
        attention_maps = []

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

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)
            images.append(image)
            subjects.append(fixation["subject_idx"])
            performances.append(performance)
            attention_maps.append(attention_map)

        images = torch.stack(images)
        subjects = np.array(subjects)
        performances = np.array(performances)
        attention_maps = np.stack(attention_maps)

        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "img_name": img_name,
            "subject": subjects,
            "qid": qid,
            "performance": performances,
            "attention_map": attention_maps
        }

    def collate_func(self, batch):

        img_batch = []
        fix_vectors_batch = []
        subject_batch = []
        img_name_batch = []
        qid_batch = []
        performance_batch = []
        attention_map_batch = []

        for sample in batch:
            tmp_img, tmp_fix_vectors, tmp_subject, tmp_img_name, tmp_qid, tmp_performance, tmp_attention_map = \
                sample["image"], sample["fix_vectors"], sample["subject"], sample["img_name"], sample["qid"], sample["performance"], sample["attention_map"]
            img_batch.append(tmp_img)
            fix_vectors_batch.append(tmp_fix_vectors)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            qid_batch.append(tmp_qid)
            performance_batch.append(tmp_performance)
            attention_map_batch.append(tmp_attention_map)

        data = {}
        data["images"] = torch.cat(img_batch)
        data["fix_vectors"] = fix_vectors_batch
        data["subjects"] =  np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["qids"] = qid_batch
        data["performances"] = np.concatenate(performance_batch)
        data["attention_maps"] = np.concatenate(attention_map_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}
        return data


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_dataset = AiR("/home/AiR_autism/stimuli", "/home/AiR_autism/processed", type="train", transform=transform)

    for idx in range(560):
        a = train_dataset[idx]
