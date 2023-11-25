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
import seaborn as sns
import scipy.ndimage as filters
from tqdm import tqdm
from scipy.io import loadmat
from torchvision import transforms


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

epsilon = 1e-7


class COCO_Search18(Dataset):
    """
    get COCO Search18 data
    """

    def __init__(self,
                 COCO_Search18_stimuli_dir,
                 COCO_Search18_fixations_dir,
                 COCO_Search18_detector_dir,
                 action_map=(30, 40),
                 resize=(240, 320),
                 max_length=16,
                 blur_sigma=1,
                 type="train",
                 split="split1",
                 transform=None,
                 saliency_map_blur_sigma=25,
                 detector_threshold=0.6):
        self.COCO_Search18_stimuli_dir = COCO_Search18_stimuli_dir
        self.COCO_Search18_fixations_dir = COCO_Search18_fixations_dir
        self.COCO_Search18_detector_dir = COCO_Search18_detector_dir
        self.action_map = action_map
        self.resize = resize
        self.max_length = max_length
        self.blur_sigma = blur_sigma
        self.type = type
        self.split = split
        self.transform = transform
        self.saliency_map_blur_sigma = saliency_map_blur_sigma
        self.detector_threshold = detector_threshold
        self.COCO_Search18_detector_file = join(self.COCO_Search18_detector_dir, "coco_search18_detector.json")
        self.object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
                            "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
                            "toilet", "tv"]
        self.name2int = dict()
        for index in range(len(self.object_name)):
            self.name2int[self.object_name[index]] = index

        with open(self.COCO_Search18_detector_file) as json_file:
            self.detector = json.load(json_file)

        self.imgs_2_det = dict()
        for index in range(len(self.detector)):
            if self.detector[index]["category"] in self.object_name and self.detector[index]["score"] >= self.detector_threshold:
                self.imgs_2_det.setdefault(self.detector[index]["image_id"], []).append(self.detector[index])

        self.fixations_file = join(self.COCO_Search18_fixations_dir, "fixations.json")
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

    def show_image_and_fixation(self, img, x, y):
        plt.figure()
        plt.imshow(img)
        plt.plot(x, y, 'xb-')
        plt.show()

    def extract_scanpath_info(self, fixation_sample):
        scanpath = np.zeros((self.max_length, self.action_map[0], self.action_map[1]), dtype=np.float32)
        # the first element denotes the termination action
        target_scanpath = np.zeros((self.max_length, self.action_map[0] * self.action_map[1] + 1), dtype=np.float32)
        duration = np.zeros(self.max_length, dtype=np.float32)
        action_mask = np.zeros(self.max_length, dtype=np.float32)
        duration_mask = np.zeros(self.max_length, dtype=np.float32)

        pos_x = np.array(fixation_sample["X"]).astype(np.float32)
        pos_x[pos_x >= self.action_map[1] * self.downscale_x] = self.action_map[1] * self.downscale_x - 1
        pos_y = np.array(fixation_sample["Y"]).astype(np.float32)
        pos_y[pos_y >= self.action_map[0] * self.downscale_y] = self.action_map[0] * self.downscale_y - 1
        duration_raw = np.array(fixation_sample["T"]).astype(np.float32)

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

        return target_scanpath, duration, action_mask, duration_mask


    def __getitem__(self, idx):
        img_name = self.imgid[idx]

        image_id = img_name.split(".")[0]
        img_path = join(self.COCO_Search18_stimuli_dir, image_id + ".jpg")
        image = Image.open(img_path).convert('RGB')
        det_size_y, det_size_x = image.height, image.width
        origin_size_y, origin_size_x = 320, 512
        if self.transform is not None:
            image = self.transform(image)
        self.downscale_x = origin_size_x / self.action_map[1]
        self.downscale_y = origin_size_y / self.action_map[0]

        attention_map = np.zeros((det_size_y, det_size_x), dtype=np.float32)
        for det in self.imgs_2_det.get(image_id.split("/")[-1], []):
            if det["category"] == self.fixations[self.imgid_to_sub[img_name][0]]["task"]:
                x_min = int(det["bbox"][0])
                x_max = int(det["bbox"][2])
                y_min = int(det["bbox"][1])
                y_max = int(det["bbox"][3])
                attention_map[y_min:y_max, x_min:x_max] = 1
        attention_map = resize(attention_map, self.action_map)
        attention_map /= (attention_map.max() + epsilon)
        attention_map = np.expand_dims(attention_map, axis=0)

        images = []
        subjects = []
        tasks = []
        target_scanpaths = []
        durations = []
        action_masks = []
        duration_masks = []
        attention_maps = []
        taskints = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]
            scanpath, duration, action_mask, duration_mask = self.extract_scanpath_info(fixation)

            task = fixation["task"]
            taskint = self.name2int[task]

            images.append(image)
            subjects.append(fixation["subject"] - 1)
            tasks.append(task)
            target_scanpaths.append(scanpath)
            durations.append(duration)
            action_masks.append(action_mask)
            duration_masks.append(duration_mask)
            attention_maps.append(attention_map)
            taskints.append(taskint)

        images = torch.stack(images)
        subjects = np.array(subjects)
        target_scanpaths = np.array(target_scanpaths)
        durations = np.array(durations)
        action_masks = np.array(action_masks)
        duration_masks = np.array(duration_masks)
        attention_maps = np.array(attention_maps)
        taskints = np.array(taskints)

        return {
            "image": images,
            "subject": subjects,
            "img_name": img_name,
            "duration": durations,
            "action_mask": action_masks,
            "duration_mask": duration_masks,
            "task": tasks,
            "target_scanpath": target_scanpaths,
            "taskint": taskints,
            "attention_map": attention_maps
        }

    def collate_func(self, batch):

        img_batch = []
        subject_batch = []
        img_name_batch = []
        duration_batch = []
        action_mask_batch = []
        duration_mask_batch = []
        task_batch = []
        target_scanpath_batch = []
        taskint_batch = []
        attention_map_batch = []

        for sample in batch:
            tmp_img, tmp_subject, tmp_img_name, tmp_duration, tmp_action_mask, tmp_duration_mask, \
                tmp_task, tmp_target_scanpath, tmp_taskint, tmp_attention_map = \
                (sample["image"], sample["subject"], sample["img_name"], \
                    sample["duration"], sample["action_mask"], sample["duration_mask"], sample["task"],
                 sample["target_scanpath"], sample["taskint"], sample["attention_map"])
            img_batch.append(tmp_img)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            task_batch.append(tmp_task)
            target_scanpath_batch.append(tmp_target_scanpath)
            taskint_batch.append(tmp_taskint)
            attention_map_batch.append(tmp_attention_map)

        data = dict()
        data["images"] = torch.cat(img_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["durations"] = np.concatenate(duration_batch)
        data["action_masks"] = np.concatenate(action_mask_batch)
        data["duration_masks"] = np.concatenate(duration_mask_batch)
        data["tasks"] = task_batch
        data["scanpaths"] = np.concatenate(target_scanpath_batch)
        data["taskints"] = np.concatenate(taskint_batch)
        data["attention_maps"] = np.concatenate(attention_map_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}
        return data


class COCO_Search18_evaluation(Dataset):
    """
    get COCO_Search18 data for evaluation
    """

    def __init__(self,
                 COCO_Search18_stimuli_dir,
                 COCO_Search18_fixations_dir,
                 COCO_Search18_detector_dir,
                 action_map=(30, 40),
                 resize=(240, 320),
                 type="validation",
                 split="split1",
                 transform=None,
                 saliency_map_blur_sigma=25,
                 detector_threshold=0.6):

        self.COCO_Search18_stimuli_dir = COCO_Search18_stimuli_dir
        self.COCO_Search18_fixations_dir = COCO_Search18_fixations_dir
        self.COCO_Search18_detector_dir = COCO_Search18_detector_dir
        self.action_map = action_map
        self.resize = resize
        self.type = type
        self.split = split
        self.transform = transform
        self.detector_threshold = detector_threshold
        self.COCO_Search18_detector_file = join(self.COCO_Search18_detector_dir, "coco_search18_detector.json")
        self.object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
                            "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
                            "toilet", "tv"]
        self.name2int = dict()
        for index in range(len(self.object_name)):
            self.name2int[self.object_name[index]] = index

        with open(self.COCO_Search18_detector_file) as json_file:
            self.detector = json.load(json_file)

        self.imgs_2_det = dict()
        for index in range(len(self.detector)):
            if self.detector[index]["category"] in self.object_name and self.detector[index][
                "score"] >= self.detector_threshold:
                self.imgs_2_det.setdefault(self.detector[index]["image_id"], []).append(self.detector[index])

        self.fixations_file = join(self.COCO_Search18_fixations_dir, "fixations.json")
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

        image_id = img_name.split(".")[0]
        img_path = join(self.COCO_Search18_stimuli_dir, image_id + ".jpg")
        image = Image.open(img_path).convert('RGB')
        det_size_y, det_size_x = image.height, image.width
        origin_size_y, origin_size_x = 320, 512
        if self.transform is not None:
            image = self.transform(image)
        self.downscale_x = origin_size_x / self.action_map[1]
        self.downscale_y = origin_size_y / self.action_map[0]
        resizescale_x = origin_size_x / self.resize[1]
        resizescale_y = origin_size_y / self.resize[0]

        attention_map = np.zeros((det_size_y, det_size_x), dtype=np.float32)
        for det in self.imgs_2_det.get(image_id.split("/")[-1], []):
            if det["category"] == self.fixations[self.imgid_to_sub[img_name][0]]["task"]:
                x_min = int(det["bbox"][0])
                x_max = int(det["bbox"][2])
                y_min = int(det["bbox"][1])
                y_max = int(det["bbox"][3])
                attention_map[y_min:y_max, x_min:x_max] = 1
        attention_map = resize(attention_map, self.action_map)
        attention_map /= (attention_map.max() + epsilon)
        attention_map = np.expand_dims(attention_map, axis=0)

        images = []
        fix_vectors = []
        subjects = []
        tasks = []
        taskints = []
        attention_maps = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]
            task = fixation["task"]
            taskint = self.name2int[task]

            # if use cvpr paper, we do not need to normalize
            x_start = np.array(fixation["X"]).astype(np.float32) / resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / resizescale_y
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)
            images.append(image)
            subjects.append(fixation["subject"] - 1)
            tasks.append(task)
            taskints.append(taskint)
            attention_maps.append(attention_map)

        images = torch.stack(images)
        subjects = np.array(subjects)
        attention_maps = np.array(attention_maps)
        taskints = np.array(taskints)

        return {
            "image": images,
            "fix_vectors": fix_vectors,
            "subject": subjects,
            "attention_map": attention_maps,
            "img_name": img_name,
            "task": tasks,
            "taskint": taskints,
        }

    def collate_func(self, batch):

        img_batch = []
        subject_batch = []
        img_name_batch = []
        fix_vectors_batch = []
        task_batch = []
        taskint_batch = []
        attention_map_batch = []

        for sample in batch:
            tmp_img, tmp_subject, tmp_img_name, tmp_fix_vectors, \
                tmp_task, tmp_taskint, tmp_attention_map = \
                (sample["image"], sample["subject"], sample["img_name"], \
                 sample["fix_vectors"], sample["task"],
                 sample["taskint"], sample["attention_map"])
            img_batch.append(tmp_img)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            fix_vectors_batch.append(tmp_fix_vectors)
            task_batch.append(tmp_task)
            taskint_batch.append(tmp_taskint)
            attention_map_batch.append(tmp_attention_map)

        data = dict()
        data["images"] = torch.cat(img_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = task_batch
        data["fix_vectors"] = fix_vectors_batch
        data["taskints"] = np.concatenate(taskint_batch)
        data["attention_maps"] = np.concatenate(attention_map_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}
        return data


class COCO_Search18_rl(Dataset):
    """
    get COCO_Search18 data for reinforcement learning
    """

    def __init__(self,
                 COCO_Search18_stimuli_dir,
                 COCO_Search18_fixations_dir,
                 COCO_Search18_detector_dir,
                 action_map=(30, 40),
                 resize=(240, 320),
                 type="train",
                 split="split1",
                 transform=None,
                 saliency_map_blur_sigma=25,
                 detector_threshold=0.6):

        self.COCO_Search18_stimuli_dir = COCO_Search18_stimuli_dir
        self.COCO_Search18_fixations_dir = COCO_Search18_fixations_dir
        self.COCO_Search18_detector_dir = COCO_Search18_detector_dir
        self.action_map = action_map
        self.resize = resize
        self.type = type
        self.split = split
        self.transform = transform
        self.detector_threshold = detector_threshold
        self.COCO_Search18_detector_file = join(self.COCO_Search18_detector_dir, "coco_search18_detector.json")
        self.object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
                            "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
                            "toilet", "tv"]
        self.name2int = dict()
        for index in range(len(self.object_name)):
            self.name2int[self.object_name[index]] = index

        with open(self.COCO_Search18_detector_file) as json_file:
            self.detector = json.load(json_file)

        self.imgs_2_det = dict()
        for index in range(len(self.detector)):
            if self.detector[index]["category"] in self.object_name and self.detector[index][
                "score"] >= self.detector_threshold:
                self.imgs_2_det.setdefault(self.detector[index]["image_id"], []).append(self.detector[index])

        self.fixations_file = join(self.COCO_Search18_fixations_dir, "fixations.json")
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

        image_id = img_name.split(".")[0]
        img_path = join(self.COCO_Search18_stimuli_dir, image_id + ".jpg")
        image = Image.open(img_path).convert('RGB')
        det_size_y, det_size_x = image.height, image.width
        origin_size_y, origin_size_x = 320, 512
        if self.transform is not None:
            image = self.transform(image)
        self.downscale_x = origin_size_x / self.action_map[1]
        self.downscale_y = origin_size_y / self.action_map[0]
        resizescale_x = origin_size_x / self.resize[1]
        resizescale_y = origin_size_y / self.resize[0]

        attention_map = np.zeros((det_size_y, det_size_x), dtype=np.float32)
        for det in self.imgs_2_det.get(image_id.split("/")[-1], []):
            if det["category"] == self.fixations[self.imgid_to_sub[img_name][0]]["task"]:
                x_min = int(det["bbox"][0])
                x_max = int(det["bbox"][2])
                y_min = int(det["bbox"][1])
                y_max = int(det["bbox"][3])
                attention_map[y_min:y_max, x_min:x_max] = 1
        attention_map = resize(attention_map, self.action_map)
        attention_map /= (attention_map.max() + epsilon)
        attention_map = np.expand_dims(attention_map, axis=0)

        images = []
        fix_vectors = []
        subjects = []
        tasks = []
        taskints = []
        attention_maps = []
        for ids in self.imgid_to_sub[img_name]:
            fixation = self.fixations[ids]
            task = fixation["task"]
            taskint = self.name2int[task]

            # if use cvpr paper, we do not need to normalize
            x_start = np.array(fixation["X"]).astype(np.float32) / resizescale_x
            y_start = np.array(fixation["Y"]).astype(np.float32) / resizescale_y
            duration = np.array(fixation["T"]).astype(np.float32) / 1000.0

            length = fixation["length"]

            fix_vector = []
            for order in range(length):
                fix_vector.append((x_start[order], y_start[order], duration[order]))
            fix_vector = np.array(fix_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                     'formats': ('f8', 'f8', 'f8')})
            fix_vectors.append(fix_vector)
            images.append(image)
            subjects.append(fixation["subject"] - 1)
            tasks.append(task)
            taskints.append(taskint)
            attention_maps.append(attention_map)

        images = torch.stack(images)
        subjects = np.array(subjects)
        attention_maps = np.array(attention_maps)
        taskints = np.array(taskints)

        return {
                "image": images,
                "fix_vectors": fix_vectors,
                "subject": subjects,
                "attention_map": attention_maps,
                "img_name": img_name,
                "task": tasks,
                "taskint": taskints,
            }

    def collate_func(self, batch):

        img_batch = []
        subject_batch = []
        img_name_batch = []
        fix_vectors_batch = []
        task_batch = []
        taskint_batch = []
        attention_map_batch = []

        for sample in batch:
            tmp_img, tmp_subject, tmp_img_name, tmp_fix_vectors, \
                tmp_task, tmp_taskint, tmp_attention_map = \
                (sample["image"], sample["subject"], sample["img_name"], \
                 sample["fix_vectors"], sample["task"],
                 sample["taskint"], sample["attention_map"])
            img_batch.append(tmp_img)
            subject_batch.append(tmp_subject)
            img_name_batch.append(tmp_img_name)
            fix_vectors_batch.append(tmp_fix_vectors)
            task_batch.append(tmp_task)
            taskint_batch.append(tmp_taskint)
            attention_map_batch.append(tmp_attention_map)

        data = dict()
        data["images"] = torch.cat(img_batch)
        data["subjects"] = np.concatenate(subject_batch)
        data["img_names"] = img_name_batch
        data["tasks"] = task_batch
        data["fix_vectors"] = fix_vectors_batch
        data["taskints"] = np.concatenate(taskint_batch)
        data["attention_maps"] = np.concatenate(attention_map_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor
        data = {k: v.unsqueeze(0) if type(v) is torch.Tensor else v for k, v in data.items()}
        return data
