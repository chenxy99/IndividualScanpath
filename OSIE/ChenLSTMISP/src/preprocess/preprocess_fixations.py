import pandas as pd
import scipy.io as sio
import numpy as np
import json
import os
import cv2

# We use the same test set as paper
# "Visual Scanpath Prediction using IOR-ROI Recurrent Mixture Density Network" provided
test_image_names_list = ["1009.jpg", "1017.jpg", "1049.jpg", "1056.jpg", "1062.jpg", "1086.jpg", "1087.jpg",
                         "1099.jpg", "1108.jpg", "1114.jpg", "1116.jpg", "1117.jpg", "1127.jpg", "1130.jpg",
                         "1131.jpg", "1136.jpg", "1140.jpg", "1152.jpg", "1192.jpg", "1220.jpg", "1225.jpg",
                         "1226.jpg", "1252.jpg", "1255.jpg", "1269.jpg", "1295.jpg", "1307.jpg", "1360.jpg",
                         "1369.jpg", "1372.jpg", "1394.jpg", "1397.jpg", "1405.jpg", "1420.jpg", "1423.jpg",
                         "1433.jpg", "1441.jpg", "1478.jpg", "1480.jpg", "1481.jpg", "1489.jpg", "1490.jpg",
                         "1493.jpg", "1502.jpg", "1509.jpg", "1523.jpg", "1528.jpg", "1530.jpg", "1549.jpg",
                         "1555.jpg", "1558.jpg", "1567.jpg", "1576.jpg", "1581.jpg", "1595.jpg", "1596.jpg",
                         "1605.jpg", "1609.jpg", "1615.jpg", "1616.jpg", "1618.jpg", "1622.jpg", "1628.jpg",
                         "1637.jpg", "1640.jpg", "1657.jpg", "1663.jpg", "1677.jpg", "1682.jpg", "1699.jpg", ]

val_image_names_list = ['1011.jpg', '1031.jpg', '1045.jpg', '1051.jpg', '1058.jpg', '1063.jpg', '1076.jpg',
                        '1078.jpg', '1090.jpg', '1092.jpg', '1095.jpg', '1100.jpg', '1104.jpg', '1109.jpg',
                        '1129.jpg', '1135.jpg', '1146.jpg', '1149.jpg', '1162.jpg', '1166.jpg', '1170.jpg',
                        '1188.jpg', '1194.jpg', '1197.jpg', '1200.jpg', '1203.jpg', '1212.jpg', '1224.jpg',
                        '1250.jpg', '1270.jpg', '1283.jpg', '1291.jpg', '1300.jpg', '1304.jpg', '1316.jpg',
                        '1319.jpg', '1333.jpg', '1342.jpg', '1349.jpg', '1352.jpg', '1364.jpg', '1390.jpg',
                        '1407.jpg', '1424.jpg', '1432.jpg', '1435.jpg', '1461.jpg', '1468.jpg', '1469.jpg',
                        '1483.jpg', '1491.jpg', '1504.jpg', '1518.jpg', '1535.jpg', '1537.jpg', '1562.jpg',
                        '1592.jpg', '1600.jpg', '1601.jpg', '1608.jpg', '1610.jpg', '1621.jpg', '1623.jpg',
                        '1629.jpg', '1661.jpg', '1667.jpg', '1668.jpg', '1669.jpg', '1678.jpg', '1691.jpg']

val_image_names_list = sorted(val_image_names_list)
test_image_names_list = sorted(test_image_names_list)


train_image_names_list = []
image_names_list = ["{}.jpg".format(str(1001 + _)) for _ in range(700)]
for value in image_names_list:
    if value not in val_image_names_list and value not in test_image_names_list:
        train_image_names_list.append(value)


data_root = '/home/OSIE_autism'
mat_file = os.path.join(data_root, 'eye/fixations.mat')
data = sio.loadmat(mat_file)
points = data['points']
durations = data['durations']
subjectId = data['subjectId']
groupId = data['groupId']
rawPoints = data['rawPoints']
rawDurations = data['rawDurations']
pupils = data['pupils']

np.random.seed(0)
length_scanpath = []
duration_scanpath = []
x_scanpath = []
y_scanpath = []

subjectId2idx = {subjectId[0, i] - 1: i for i in range(39)}

train_list = list()
for image_name in train_image_names_list:
    image_idx = int(image_name.split(".")[0]) - 1001
    for subj_idx in range(39):
        example_dict = dict()
        idx = subjectId2idx[subj_idx]
        point = points[image_idx][idx]
        duration = durations[image_idx][idx]
        cur_subjectId = int(subjectId[0, idx] - 1)
        cur_groupId = int(groupId[0, idx] - 1)

        example_dict['name'] = image_name
        example_dict['subject'] = cur_subjectId
        example_dict['group'] = cur_groupId
        example_dict['X'] = point[0].tolist()
        example_dict['Y'] = point[1].tolist()
        example_dict['T'] = ((duration[1] - duration[0]) * 1000).tolist()
        example_dict['length'] = len(example_dict['X'])
        example_dict['split'] = 'train'

        train_list.append(example_dict)
        length_scanpath.append(len(example_dict['T']))
        duration_scanpath.extend(example_dict['T'])
        x_scanpath.extend(example_dict['X'])
        y_scanpath.extend(example_dict['Y'])

val_list = list()
for image_name in val_image_names_list:
    image_idx = int(image_name.split(".")[0]) - 1001
    for subj_idx in range(39):
        example_dict = dict()
        idx = subjectId2idx[subj_idx]
        point = points[image_idx][idx]
        duration = durations[image_idx][idx]
        cur_subjectId = int(subjectId[0, idx] - 1)
        cur_groupId = int(groupId[0, idx] - 1)

        example_dict['name'] = image_name
        example_dict['subject'] = cur_subjectId
        example_dict['group'] = cur_groupId
        example_dict['X'] = point[0].tolist()
        example_dict['Y'] = point[1].tolist()
        example_dict['T'] = ((duration[1] - duration[0]) * 1000).tolist()
        example_dict['length'] = len(example_dict['X'])
        example_dict['split'] = 'validation'

        val_list.append(example_dict)
        length_scanpath.append(len(example_dict['T']))
        duration_scanpath.extend(example_dict['T'])
        x_scanpath.extend(example_dict['X'])
        y_scanpath.extend(example_dict['Y'])

test_list = list()
for image_name in test_image_names_list:
    image_idx = int(image_name.split(".")[0]) - 1001
    for subj_idx in range(39):
        example_dict = dict()
        idx = subjectId2idx[subj_idx]
        point = points[image_idx][idx]
        duration = durations[image_idx][idx]
        cur_subjectId = int(subjectId[0, idx] - 1)
        cur_groupId = int(groupId[0, idx] - 1)

        example_dict['name'] = image_name
        example_dict['subject'] = cur_subjectId
        example_dict['group'] = cur_groupId
        example_dict['X'] = point[0].tolist()
        example_dict['Y'] = point[1].tolist()
        example_dict['T'] = ((duration[1] - duration[0]) * 1000).tolist()
        example_dict['length'] = len(example_dict['X'])
        example_dict['split'] = 'test'

        test_list.append(example_dict)
        length_scanpath.append(len(example_dict['T']))
        duration_scanpath.extend(example_dict['T'])
        x_scanpath.extend(example_dict['X'])
        y_scanpath.extend(example_dict['Y'])



data = train_list + val_list + test_list

save_json_file = '/home/OSIE_autism/processed/fixations.json'
with open(save_json_file, 'w') as f:
    json.dump(data, f, indent=2)
