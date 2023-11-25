import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import sys
import numpy as np

from models.resnet import resnet50, resnet18
epsilon = 1e-7


class ConvLSTM(nn.Module):
    def __init__(self, embed_size=512):
        super(ConvLSTM, self).__init__()
        #LSTM gates
        self.input_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.memory_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.input_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.memory_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)

        self.input = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)

        self.init_weights()

    def forward(self, x, state, spatial, semantic):
        batch, channel, col, row = x.size()

        spatial_semantic = spatial.unsqueeze(1) * semantic.unsqueeze(-1).unsqueeze(-1)

        h, c = state[0], state[1]
        i = torch.sigmoid(self.input_x(x) + self.input_h(h) + self.input(spatial_semantic))
        f = torch.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget(spatial_semantic))
        o = torch.sigmoid(self.output_x(x) + self.output_h(h) + self.output(spatial_semantic))
        g = torch.tanh(self.memory_x(x) + self.memory_h(h))

        next_c = f * c + i * g
        h = o * next_c
        state = (h, next_c)

        return h, state

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)


class semantic_att(nn.Module):
    def __init__(self, embed_size=512):
        super(semantic_att, self).__init__()
        self.semantic_lists = nn.Linear(embed_size, embed_size, bias=True)
        self.semantic_cur = nn.Linear(embed_size, embed_size, bias=True)
        self.semantic_attention = nn.Linear(embed_size, 1, bias=True)

        self.init_weights()

    def forward(self, visual_lists, visual_cur):
        '''
        visual_lists [N, T, E]
        visual_cur [N, E]
        '''
        semantic_visual_lists = self.semantic_lists(visual_lists)
        semantic_visual_cur = self.semantic_cur(visual_cur)
        semantic_attention = F.softmax(
            self.semantic_attention(torch.tanh(semantic_visual_lists + semantic_visual_cur.unsqueeze(1))), 1)
        semantic = (visual_lists * semantic_attention).sum(1)

        return semantic

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)


class spatial_att(nn.Module):
    def __init__(self, map_width=40, map_height=30):
        super(spatial_att, self).__init__()
        self.spatial_lists = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True)
        self.spatial_cur = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True)
        self.spatial_attention = nn.Conv2d(1, 1, kernel_size=(30, 40), padding=0, stride=1, bias=True)
        self.map_width = map_width
        self.map_height = map_height

        self.init_weights()

    def forward(self, visual_lists, visual_cur):
        '''
        visual_lists [N, C, H, W]
        visual_cur [N, 1, H, W]
        '''
        batch, T, height, width = visual_lists.shape
        spatial_visual_lists = self.spatial_lists(visual_lists.view(-1, 1, height, width))
        spatial_visual_cur = self.spatial_cur(visual_cur)
        semantic_attention = F.softmax(
            self.spatial_attention(torch.tanh(spatial_visual_lists.view(batch, T, height, width) + spatial_visual_cur)
                                   .view(-1, 1, height, width)).view(batch, T, 1, 1), 1)
        semantic = (visual_lists * semantic_attention).sum(1)

        return semantic

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)


class predict_head(nn.Module):
    def __init__(self, convLSTM_length, action_map_num, embedding_dim):
        super(predict_head, self).__init__()
        self.convLSTM_length = convLSTM_length
        self.action_map_num = action_map_num
        self.embedding_dim = embedding_dim
        self.sal_layer_2 = nn.Conv2d(512, action_map_num, kernel_size=1, padding=0, stride=1, bias=True)
        self.sal_layer_3 = nn.Conv2d(512, action_map_num, kernel_size=1, padding=0, stride=1, bias=True)
        self.global_avg = nn.AvgPool2d(kernel_size=(30, 40))

        self.drt_layer_1 = nn.Conv2d(512, 1, kernel_size=7, padding=2, stride=5, bias=True)
        self.drt_layer_2 = nn.Conv2d(1, 2, kernel_size=(6, 8), padding=0, stride=1, bias=True)

        self.attention_module = attention_module(512, embedding_dim, embedding_dim * 2)

        self.init_weights()

    def forward(self, encoder_feature, features, subject_embedding):
        batch = features.shape[0]
        x = features
        y = self.sal_layer_2(x)
        y = self.global_avg(y)
        t = F.relu(self.drt_layer_1(x))
        t = self.drt_layer_2(t)
        log_normal_mu = t[:, 0].view(batch, -1)
        log_normal_sigma2 = torch.exp(t[:, 1]).view(batch, -1)
        x = F.relu(self.sal_layer_3(x))
        z = torch.cat([y.view(batch, self.action_map_num, -1), x.view(batch, self.action_map_num, -1)], dim=-1)

        # get the attention weight
        attention_weights = self.attention_module(encoder_feature, x, subject_embedding)

        # attended weights
        aggr_x = (x * attention_weights.unsqueeze(-1).unsqueeze(-1)).sum(1, keepdim=True)
        aggr_z = (z * attention_weights.unsqueeze(-1)).sum(1, keepdim=True)

        if self.training == False:
            aggr_z = F.softmax(aggr_z, -1)

        predicts = {}
        # [N, T, A] A = H * W + 1
        predicts['actions'] = aggr_z
        # [N, T]
        predicts['log_normal_mu'] = log_normal_mu
        # [N, T]
        predicts['log_normal_sigma2'] = log_normal_sigma2
        # [N, T, H, W]
        predicts["action_map"] = aggr_x
        # [N, Subj_num, H, W]
        predicts["all_subject_action_map"] = x

        return predicts

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)


class attention_module(nn.Module):
    def __init__(self, feature_size, subject_embedding_size, project_size):
        super(attention_module, self).__init__()
        self.feature_size = feature_size
        self.project_size = project_size

        self.proj_feature = nn.Linear(feature_size, project_size, bias=True)
        self.proj_subject = nn.Linear(subject_embedding_size, project_size, bias=True)
        self.attention = nn.Linear(project_size, 1, bias=True)

        self.init_weights()

    def forward(self, features, attention_map, subject_embedding):
        aggr_feature = features.unsqueeze(1) * attention_map.unsqueeze(2)
        aggr_feature = aggr_feature.view(*aggr_feature.shape[:-2], -1).mean(-1)

        proj_aggr_feature = self.proj_feature(aggr_feature)
        proj_subject_embedding = self.proj_subject(subject_embedding)
        attention_weights = F.softmax(
            self.attention(torch.tanh(proj_aggr_feature + proj_subject_embedding.unsqueeze(1))), 1)
        attention_weights = attention_weights.squeeze(-1)
        return attention_weights

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

class subject_attention_module(nn.Module):
    def __init__(self, feature_size, subject_embedding_size, project_size):
        super(subject_attention_module, self).__init__()
        self.feature_size = feature_size
        self.project_size = project_size

        self.proj_feature = nn.Linear(feature_size, project_size, bias=True)
        self.proj_subject = nn.Linear(subject_embedding_size, project_size, bias=True)
        self.attention = nn.Linear(project_size, 1, bias=True)

        self.init_weights()

    def forward(self, features, subject_embedding):
        H, W = features.shape[-2], features.shape[-1]
        features = features.view(features.shape[0],  features.shape[1], -1)
        proj_aggr_feature = self.proj_feature(features.permute(0, 2, 1))
        proj_subject_embedding = self.proj_subject(subject_embedding)
        attention_weights = F.softmax(
            self.attention(torch.tanh(proj_aggr_feature + proj_subject_embedding.unsqueeze(1))), 1)
        attention_weights = attention_weights.permute(0, 2, 1).view(attention_weights.shape[0], 1, H, W)
        # attention_weights = attention_weights.squeeze(-1)
        return attention_weights

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

class baseline(nn.Module):
    def __init__(self, embed_size=512, convLSTM_length=16, min_length=1, ratio=4,
                  map_width=40, map_height=30, dropout=0.2, subject_num=15, embedding_dim=64, action_map_num=4):
        super(baseline, self).__init__()
        self.embed_size = embed_size
        self.ratio = ratio
        self.convLSTM_length = convLSTM_length
        self.min_length = min_length
        self.downsampling_rate = 8
        self.map_width = map_width
        self.map_height = map_height
        self.dropout = dropout
        self.subject_num = subject_num
        self.embedding_dim = embedding_dim
        self.action_map_num = action_map_num

        self.resnet = resnet50(pretrained=True)
        # self.resnet = resnet18(pretrained=True)
        self.dilate_resnet(self.resnet)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.sal_conv = nn.Conv2d(2048, 512, kernel_size=3, padding=1, stride=1, bias=True)
        self.lstm = ConvLSTM(self.embed_size)

        self.semantic_embed = nn.Linear(512* 2, embed_size)
        self.spatial_embed = nn.Linear(1200, 1200, bias=True)
        self.semantic_att = semantic_att(embed_size=512)
        self.spatial_att = spatial_att(map_width, map_height)

        self.object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
                            "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
                            "toilet", "tv"]
        self.int2object = {i: self.object_name[i] for i in range(len(self.object_name))}

        self.object_sal_layer = nn.ModuleDict(
            {self.object_name[i]: nn.Conv2d(512, 512, kernel_size=5, padding=2, stride=1, bias=True) for
             i in range(18)})
        self.object_head = predict_head(convLSTM_length, action_map_num, embedding_dim)

        self.subject_attention = subject_attention_module(512, embedding_dim, embedding_dim * 2)

        self.semantic_subj_embed = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 512 * 2),
        )
        self.spatial_subj_embed = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, map_width * map_height),
        )

        self.subject_embedding = nn.Embedding(num_embeddings=self.subject_num, embedding_dim=self.embedding_dim)

        self.init_weights()

    def init_hidden(self, x): #initializing hidden state as all zero
        h = torch.zeros_like(x)
        c = torch.zeros_like(x)
        return (h, c)

    def dilate_resnet(self, resnet):    #modifying resnet as in SAM paper
        resnet.layer2[0].conv1.stride = 1
        resnet.layer2[0].downsample[0].stride = 1
        resnet.layer4[0].conv1.stride = 1
        resnet.layer4[0].downsample[0].stride = 1

        for block in resnet.layer3:
            block.conv2.dilation = 2
            block.conv2.padding = 2

        for block in resnet.layer4:
            block.conv2.dilation = 4
            block.conv2.padding = 4

    def get_spatial_semantic(self, action_map, subject_related_maps, visual_feature):
        semantic_feature = action_map.expand_as(visual_feature) * visual_feature
        subject_feature = subject_related_maps.expand_as(visual_feature) * visual_feature
        feature = torch.cat([semantic_feature, subject_feature], dim=1)
        spatial_semantic_feature = feature.mean(1, keepdims=True)

        return spatial_semantic_feature

    def get_channel_semantic(self, action_map, subject_related_maps, visual_feature):
        semantic_feature = action_map.expand_as(visual_feature) * visual_feature
        subject_feature = subject_related_maps.expand_as(visual_feature) * visual_feature
        feature = torch.cat([semantic_feature, subject_feature], dim=1)
        channel_semantic_feature = feature.view(feature.shape[0], feature.shape[1], -1).mean(-1)

        return channel_semantic_feature


    def forward(self, images, subject, attention_maps, tasks):
        # scanpath is used for the extract embedding feature to the ConvLSTM modules  (We do not use it at this model)
        # durations is used in the ConvLSTM modules (We do not use it at this model)
        # active_scanpath_temporal_masks is used for training the saliency map and obtained from duration_masks

        if self.training:
            predicts = self.training_process(images, subject, attention_maps, tasks)
        else:
            predicts = self.inference(images, subject, attention_maps, tasks)

        return predicts

    def training_process(self, images, subject, attention_maps, tasks):
        # img = img.unsqueeze(0)
        batch, _, height, width = images.size()# build a one-hot performance embedding

        subj_embedding = self.subject_embedding(subject)

        spatial_subj_embed = self.spatial_subj_embed(subj_embedding).view(batch, 1, self.map_height, self.map_width)
        semantic_subj_embed = self.semantic_subj_embed(subj_embedding)

        x = self.resnet(images)
        visual_feature = F.relu(self.sal_conv(x)) #change filter size

        spatial_lists = list()
        semantic_lists = list()

        subject_related_maps = self.subject_attention(visual_feature, subj_embedding)

        spatial_feature = F.relu(self.get_spatial_semantic(attention_maps, subject_related_maps, visual_feature)) + spatial_subj_embed
        spatial_feature = self.spatial_embed(spatial_feature.view(batch, 1, -1)).view(batch, 1, 30, 40)
        spatial_lists.append(spatial_feature)
        semantic_feature = F.relu(self.get_channel_semantic(attention_maps, subject_related_maps, visual_feature)) + semantic_subj_embed
        semantic_feature = self.semantic_embed(semantic_feature)
        semantic_lists.append(semantic_feature)

        spatial_mem = self.spatial_att(torch.cat([_ for _ in spatial_lists], 1), spatial_feature)
        semantic_mem = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists], 1), semantic_feature)

        state = self.init_hidden(visual_feature)  # initialize hidden state as zeros

        #sequential model
        predict_alls = list()
        for i in range(self.convLSTM_length):
            output, state = self.lstm(visual_feature, state, spatial_mem, semantic_mem)

            features = list()
            for index in range(batch):
                features.append(self.object_sal_layer[self.int2object[int(tasks[index])]](output[index].unsqueeze(0)))
            features = torch.cat(features, axis=0)

            predict_head_rlts = self.object_head(visual_feature, features, subj_embedding)

            predict_alls.append(predict_head_rlts)

            predict_action_map = predict_head_rlts["action_map"]

            spatial_feature = F.relu(self.get_spatial_semantic(predict_action_map, subject_related_maps, visual_feature)) + spatial_subj_embed
            spatial_feature = self.spatial_embed(spatial_feature.view(batch, 1, -1)).view(batch, 1, 30, 40)
            spatial_lists.append(spatial_feature)
            semantic_feature = F.relu(self.get_channel_semantic(predict_action_map, subject_related_maps, visual_feature)) + semantic_subj_embed
            semantic_feature = self.semantic_embed(semantic_feature)
            semantic_lists.append(semantic_feature)

            spatial_mem = self.spatial_att(torch.cat([_ for _ in spatial_lists], 1), spatial_feature)
            semantic_mem = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists], 1), semantic_feature)


        predict = dict()
        for predicts, save_predict in zip([predict_alls], [predict]):
            actions_pools = list()
            log_normal_mu_pools = list()
            log_normal_sigma2_pools = list()
            action_map_pools = list()
            for i in range(self.convLSTM_length):
                actions_pools.append(predicts[i]["actions"])
                log_normal_mu_pools.append(predicts[i]["log_normal_mu"])
                log_normal_sigma2_pools.append(predicts[i]["log_normal_sigma2"])
                action_map_pools.append(predicts[i]["action_map"])
            save_predict["actions"] = torch.cat(actions_pools, axis=1)
            save_predict["log_normal_mu"] = torch.cat(log_normal_mu_pools, axis=1)
            save_predict["log_normal_sigma2"] = torch.cat(log_normal_sigma2_pools, axis=1)
            save_predict["action_map"] = torch.cat(action_map_pools, axis=1)

        predicts = {}
        # [N, T, A] A = H * W + 1
        predicts['actions'] = predict["actions"]
        # [N, T]
        predicts['log_normal_mu'] = predict["log_normal_mu"]
        # [N, T]
        predicts['log_normal_sigma2'] = predict["log_normal_sigma2"]
        return predicts

    def inference(self, images, subject, attention_maps, tasks):
        # img = img.unsqueeze(0)
        batch, _, height, width = images.size()  # build a one-hot performance embedding

        subj_embedding = self.subject_embedding(subject)

        spatial_subj_embed = self.spatial_subj_embed(subj_embedding).view(batch, 1, self.map_height, self.map_width)
        semantic_subj_embed = self.semantic_subj_embed(subj_embedding)

        x = self.resnet(images)
        visual_feature = F.relu(self.sal_conv(x))  # change filter size

        spatial_lists = list()
        semantic_lists = list()

        subject_related_maps = self.subject_attention(visual_feature, subj_embedding)

        spatial_feature = F.relu(self.get_spatial_semantic(attention_maps, subject_related_maps, visual_feature)) + spatial_subj_embed
        spatial_feature = self.spatial_embed(spatial_feature.view(batch, 1, -1)).view(batch, 1, 30, 40)
        spatial_lists.append(spatial_feature)
        semantic_feature = F.relu(self.get_channel_semantic(attention_maps, subject_related_maps, visual_feature)) + semantic_subj_embed
        semantic_feature = self.semantic_embed(semantic_feature)
        semantic_lists.append(semantic_feature)

        spatial_mem = self.spatial_att(torch.cat([_ for _ in spatial_lists], 1), spatial_feature)
        semantic_mem = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists], 1), semantic_feature)

        state = self.init_hidden(visual_feature)  # initialize hidden state as zeros

        # sequential model
        predict_alls = list()
        for i in range(self.convLSTM_length):
            output, state = self.lstm(visual_feature, state, spatial_mem, semantic_mem)

            features = list()
            for index in range(batch):
                features.append(self.object_sal_layer[self.int2object[int(tasks[index])]](output[index].unsqueeze(0)))
            features = torch.cat(features, axis=0)

            predict_head_rlts = self.object_head(visual_feature, features, subj_embedding)

            predict_alls.append(predict_head_rlts)

            predict_action_map = predict_head_rlts["action_map"]

            spatial_feature = F.relu(self.get_spatial_semantic(predict_action_map, subject_related_maps, visual_feature)) + spatial_subj_embed
            spatial_feature = self.spatial_embed(spatial_feature.view(batch, 1, -1)).view(batch, 1, 30, 40)
            spatial_lists.append(spatial_feature)
            semantic_feature = F.relu(self.get_channel_semantic(predict_action_map, subject_related_maps, visual_feature)) + semantic_subj_embed
            semantic_feature = self.semantic_embed(semantic_feature)

            semantic_lists.append(semantic_feature)

            spatial_mem = self.spatial_att(torch.cat([_ for _ in spatial_lists], 1), spatial_feature)
            semantic_mem = self.semantic_att(torch.cat([_.unsqueeze(1) for _ in semantic_lists], 1), semantic_feature)

        predict = dict()
        for predicts, save_predict in zip([predict_alls], [predict]):
            actions_pools = list()
            log_normal_mu_pools = list()
            log_normal_sigma2_pools = list()
            action_map_pools = list()
            all_subject_action_map_pools = list()
            for i in range(self.convLSTM_length):
                actions_pools.append(predicts[i]["actions"])
                log_normal_mu_pools.append(predicts[i]["log_normal_mu"])
                log_normal_sigma2_pools.append(predicts[i]["log_normal_sigma2"])
                action_map_pools.append(predicts[i]["action_map"])
                all_subject_action_map_pools.append(predicts[i]["all_subject_action_map"])
            save_predict["actions"] = torch.cat(actions_pools, axis=1)
            save_predict["log_normal_mu"] = torch.cat(log_normal_mu_pools, axis=1)
            save_predict["log_normal_sigma2"] = torch.cat(log_normal_sigma2_pools, axis=1)
            save_predict["action_map"] = torch.cat(action_map_pools, axis=1)
            predict["all_subject_action_map"] = torch.stack(all_subject_action_map_pools, axis=1)

        predicts = {}
        # [N, T, A] A = H * W + 1
        predicts["all_actions_prob"] = predict["actions"]
        # [N, T]
        predicts["log_normal_mu"] = predict["log_normal_mu"]
        # [N, T]
        predicts["log_normal_sigma2"] = predict["log_normal_sigma2"]
        # [N, T, H, W]
        predicts["action_map"] = predict["action_map"]
        # [N, T, K]
        predicts['all_subject_action_map'] = predict["all_subject_action_map"]

        return predicts

    def init_weights(self):
        for modules in [self.sal_conv.modules(),  self.object_sal_layer.modules(),
                        self.semantic_embed.modules(), self.spatial_embed.modules(),
                        self.semantic_subj_embed.modules(), self.semantic_subj_embed.modules(), self.subject_embedding.modules()]:
            for m in modules:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.001)

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
        else:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)