import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import json
from os.path import join
from skimage import io
from PIL import Image
import os
from tqdm import tqdm
from skimage.transform import rescale, resize, downscale_local_mean

# only observe the distribution of the training set
with open("/home/COCOSearch18/TP/fixations/coco_search18_fixations_TP_validation.json") as json_file:
    fixations = json.load(json_file)

img_save_root_dir = "../../scanpath_vis/validation"

imgs_root = "/home/COCOSearch18/TP/images"

object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
               "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
               "toilet", "tv"]

print(fixations[0])


fixations_dict_all = {}
for element in fixations:
    fixations_dict_all.setdefault(element["task"], dict()).setdefault(element["name"], []).append(element)

selected_fixation_dict_all = {}
for key, values in fixations_dict_all.items():
    for img_id, value in values.items():
        for val in value:
            if val["fixOnTarget"] == False:
                selected_fixation_dict_all.setdefault(key, dict()).setdefault(img_id, []).extend(value)
            break

fixations_dict_all = selected_fixation_dict_all


def scanpath_visualization(img, scanpath_info, bbox, scores=None, save_img_name=None):
    """
    input:
        img: A (H, W, 3) np.array
        scanpath_info: A (3, L) np.array, where each columns represents the x-axis, y-axis and duration
    """
    plt.figure()
    plt.imshow(img)
    plt.plot(scanpath_info[0], scanpath_info[1], 'g-', linewidth=2)
    for index in range(scanpath_info.shape[1]):
        markersize = scanpath_info[2][index] / 300 * 30
        if index == 0:
            plt.plot(scanpath_info[0][index], scanpath_info[1][index],
                 'b', marker='o', linewidth=2, markersize=markersize, markeredgecolor='k', markeredgewidth=1, alpha=0.75)
        elif index == scanpath_info.shape[1] - 1:
            plt.plot(scanpath_info[0][index], scanpath_info[1][index],
                 'r', marker='o', linewidth=2, markersize=markersize, markeredgecolor='k', markeredgewidth=1, alpha=0.75)
        else:
            plt.plot(scanpath_info[0][index], scanpath_info[1][index],
                 'w', marker='o', linewidth=2, markersize=markersize, markeredgecolor='k', markeredgewidth=1, alpha=0.75)
        plt.text(scanpath_info[0][index], scanpath_info[1][index], index+1,
                 fontsize=10, verticalalignment='center', horizontalalignment='center')
    plt.axis("off")

    # Get the current reference
    ax = plt.gca()
    # Create a Rectangle patch
    # rect = patches.Rectangle((bbox[0] / 3.28125, bbox[1] / 3.28125), bbox[2] / 3.28125, bbox[3] / 3.28125, linewidth=1, edgecolor='y', facecolor='none')
    rect = patches.Rectangle((bbox[0] , bbox[1] ), bbox[2] , bbox[3] , linewidth=1,
                             edgecolor='y', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

    if scores:
        title_str = ""
        # for score in scores:
        #     title_str += "{:.4f}  ".format(score)
        for index in range(5, 9):
            score = scores[index]
            title_str += "{:.4f}  ".format(score)
        plt.title(title_str)

    if save_img_name:
        plt.savefig(save_img_name, bbox_inches = 'tight', pad_inches = 0, dpi=300)
        plt.close()
    else:
        plt.show()
    a=1

for object_value in tqdm(list(fixations_dict_all.keys())):
    fixations_dict = fixations_dict_all[object_value]
    for img_name in list(fixations_dict.keys()):
        gt_dir = join(img_save_root_dir, "ground_truth")
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)
        img_dir = join(join(gt_dir, object_value), img_name[:-4])
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        img_path = join(join(imgs_root, object_value), img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        image = resize(image, (320, 512), anti_aliasing=True)

        for index in range(len(fixations_dict[img_name])):
        # for index in range(1):
            selected_fixation = fixations_dict[img_name][index]
            length_val = len(selected_fixation["X"])
            selected_fixation_X = [_ * 1 for _ in selected_fixation["X"][:length_val]]
            selected_fixation_Y = [_ * 1 for _ in selected_fixation["Y"][:length_val]]
            selected_fixation_T = [_ * 1 for _ in selected_fixation["T"][:length_val]]
            scanpath_info = np.asarray(
                [selected_fixation_X, selected_fixation_Y, selected_fixation_T])
            save_img_name = join(img_dir, "subject_{}_success_{}".format(str(selected_fixation["subject"]), selected_fixation["fixOnTarget"]))
            bbox = selected_fixation["bbox"]
            scanpath_visualization(image, scanpath_info, bbox, None, save_img_name)

        # predict_dir = join(img_dir, "predictions")
        # if not os.path.exists(predict_dir):
        #     os.makedirs(predict_dir)
        # for index in range(len(predict_fixations_dict[object_value][img_name])):
        # # for index in range(0):
        #     selected_predict_fixation = predict_fixations_dict[object_value][img_name][index]
        #     selected_predict_fixation_X = [_ * 512 / 320 for _ in selected_predict_fixation["X"]]
        #     selected_predict_fixation_Y = [_ * 320 / 240 for _ in selected_predict_fixation["Y"]]
        #     selected_predict_fixation_T = [_ * 1 for _ in selected_predict_fixation["T"]]
        #     scanpath_info = np.asarray(
        #         [selected_predict_fixation_X, selected_predict_fixation_Y, selected_predict_fixation_T])
        #     save_img_name = join(predict_dir,
        #                          str("score_{:.4f}".format(selected_predict_fixation["score"][0][6]) +
        #                              "repeat_id" + str(selected_predict_fixation["subject"]) + ".jpg"))
        #     # if selected_predict_fixation["scores"][5] >= selected_predict_fixation["gt_scores"][5] and\
        #     #     selected_predict_fixation["scores"][6] >= selected_predict_fixation["gt_scores"][6]:
        #     #     scanpath_visualization(image, scanpath_info, selected_predict_fixation["scores"], save_img_name)
        #     bbox = selected_fixation["bbox"]
        #     scanpath_visualization(image, scanpath_info, bbox, None, save_img_name)


a=1