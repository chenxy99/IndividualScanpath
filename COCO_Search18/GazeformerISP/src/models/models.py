from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

#ResNet-50 backbone
class ResNetCOCO(nn.Module):
    def __init__(self, device = "cuda:0"):
        super(ResNetCOCO, self).__init__()
        self.resnet = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone.body.to(device)
        self.device = device
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        # x = x.to(self.device)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        bs, ch, _, _ = x.size()
        x = x.view(bs, ch, -1).permute(0, 2, 1)

        return x


class SubjectAttentionModule(nn.Module):
    def __init__(self, feature_size, subject_embedding_size, project_size):
        super(SubjectAttentionModule, self).__init__()
        self.feature_size = feature_size
        self.project_size = project_size

        self.proj_feature = nn.Linear(feature_size, project_size, bias=True)
        self.proj_subject = nn.Linear(subject_embedding_size, project_size, bias=True)
        self.attention = nn.Linear(project_size, 1, bias=True)

        self.init_weights()

    def forward(self, features, subject_embedding):
        proj_aggr_feature = self.proj_feature(features)
        proj_subject_embedding = self.proj_subject(subject_embedding)
        attention_weights = F.softmax(
            self.attention(torch.tanh(proj_aggr_feature + proj_subject_embedding.unsqueeze(0))), 0)
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

class AttentionModule(nn.Module):
    def __init__(self, feature_size, task_embedding_size, project_size):
        super(AttentionModule, self).__init__()
        self.feature_size = feature_size
        self.project_size = project_size

        self.proj_feature = nn.Linear(feature_size, project_size, bias=True)
        self.proj_task = nn.Linear(task_embedding_size, project_size, bias=True)
        self.attention = nn.Linear(project_size, 1, bias=True)

        self.init_weights()

    def forward(self, features, task_embedding):
        proj_aggr_feature = self.proj_feature(features)
        proj_task_embedding = self.proj_task(task_embedding)
        attention_weights = F.softmax(
            self.attention(torch.tanh(proj_aggr_feature + proj_task_embedding.unsqueeze(1))), 1)
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


class IntegrationModule(nn.Module):
    def __init__(self, channel_size, im_h, im_w, subject_embedding_size, dropout):
        super(IntegrationModule, self).__init__()
        self.channel_size = channel_size
        self.im_h = im_h
        self.im_w = im_w
        self.subject_embedding_size = subject_embedding_size
        self.dropout = dropout

        self.subj_spatial_feature = nn.Sequential(
            nn.Linear(subject_embedding_size, channel_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(channel_size // 2, im_h * im_w),
        )

        self.subj_semantic_feature = nn.Sequential(
            nn.Linear(subject_embedding_size, channel_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(channel_size // 2, channel_size),
        )


        self.vision_spatial_feature = nn.Linear(im_h * im_w, im_h * im_w, bias=True)
        self.vision_semantic_feature = nn.Linear(channel_size * 2, channel_size, bias=True)

        self.linear_layer = nn.Linear(2 * channel_size, channel_size, bias=True)

        self.init_weights()

    def forward(self, features, visual_att, subject_att, subjects):
        visual_feature = features.permute(1, 0, 2)
        visual_att = visual_att.unsqueeze(-1)
        x = visual_feature * visual_att

        subject_att = subject_att.unsqueeze(-1)
        u = visual_feature * subject_att

        xu = torch.cat([x, u], dim=-1)

        u_s = (F.relu(self.vision_spatial_feature(xu.sum(-1))) + self.subj_spatial_feature(subjects)).unsqueeze(-1)
        u_c = (F.relu(self.vision_semantic_feature(xu.sum(-2))) + self.subj_semantic_feature(subjects)).unsqueeze(-2)
        R = u_s * u_c

        R = R.permute(1, 0, 2)

        final_feature = self.linear_layer(torch.cat([features, R], dim=-1))

        return final_feature

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
                nn.init.normal_(m.weight, std=0.1)
                nn.init.zeros_(m.bias)
            
class Transformer(nn.Module):

    def __init__(self, d_model=512, img_hidden_dim = 2048, subject_feature_dim = 512, lm_dmodel = 768, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=512, encoder_dropout=0.1, decoder_dropout = 0.2, 
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, device = "cuda:0", args = None):
        super().__init__()
        self.device = device
        self.args = args
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                encoder_dropout, activation, normalize_before).to(device)
        encoder_norm = nn.LayerNorm(d_model)
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm).to(device)
        input_proj = nn.Linear(img_hidden_dim, d_model).to(device)
        img_transform = nn.Linear(d_model, d_model).to(device)

        text_transform = nn.Linear(lm_dmodel, d_model).to(device)
        attention_module = AttentionModule(d_model, d_model, d_model // 2)
        self.encoder =  TransformerEncoderWrapper(encoder, input_proj, img_transform, text_transform, attention_module, encoder_dropout, device).to(device)


        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                decoder_dropout, activation, normalize_before).to(device)
        decoder_norm = nn.LayerNorm(d_model)
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec).to(device)
        
        dropout = nn.Dropout(decoder_dropout)
        self.decoder = TransformerDecoderWrapper(d_model, activation, decoder, dropout, device)

        self.subject_attention_module = (
            SubjectAttentionModule(d_model, subject_embedding_size=args.subject_feature_dim, project_size=2 * args.subject_feature_dim))
        self.integration_module = IntegrationModule(args.hidden_dim, args.im_h, args.im_w, args.subject_feature_dim, args.encoder_dropout)
        
        self.d_model = d_model
        self.subject_feature_dim = subject_feature_dim
        self.lm_dmodel = lm_dmodel
        self.nhead = nhead


    def forward(self, src: Tensor, tgt: Tensor, subjects:Tensor, task:Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, 
                querypos_embed: Optional[Tensor] = None, patchpos_embed: Optional[Tensor] = None):
        memory, visual_att = self.encoder(src, mask=src_mask, subjects = subjects, task = task, src_key_padding_mask=src_key_padding_mask, patchpos_embed=patchpos_embed)


        subject_att = self.subject_attention_module(memory, subjects).permute(1, 0)
        memory = self.integration_module(features=memory, visual_att=visual_att, subject_att=subject_att, subjects=subjects)
        
        memory_task, output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              querypos_embed = querypos_embed, patchpos_embed = patchpos_embed)
        return memory_task, output
        
        
class TransformerEncoderWrapper(nn.Module):

    def __init__(self, encoder, input_proj, img_transform, text_transform, attention_module, dropout, device):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.input_proj = input_proj.to(device)
        self.img_transform = img_transform.to(device)
        
        self.text_transform = text_transform.to(device)
        self.attention_module = attention_module.to(device)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters(self.encoder)
        self._reset_parameters(self.input_proj)
        self._reset_parameters(self.img_transform)
        
        self._reset_parameters(self.text_transform)
        self._reset_parameters(self.attention_module)
            
    def _reset_parameters(self, mod):
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, subjects, task,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        src_proj = self.input_proj(src).permute(1,0,2)#input projection from 2048 -> d

        output = self.encoder(src_proj, mask=mask, src_key_padding_mask=src_key_padding_mask, patchpos_embed=patchpos_embed)#transformer encoder
        
        memory = self.img_transform(output)#project image features to multimodal space
        task_emb = self.text_transform(task)  # project task features to multimodal space

        img_feature = memory.permute(1,0,2)
        visual_att = self.attention_module(img_feature, task_emb)
        

        return memory, visual_att


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, d_model, activation, decoder, dropout, device):
        super().__init__()
        self.device = device
        self.activation = _get_activation_fn(activation)
        self.decoder = decoder.to(device)
        self.dropout = dropout
        
        self.linear = nn.Linear(d_model * 3, d_model)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos(tensor)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        

        #decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              querypos_embed = querypos_embed, patchpos_embed = patchpos_embed)
        return memory, output
        

        
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=patchpos_embed, src_mask = mask, src_key_padding_mask = src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           querypos_embed = querypos_embed, 
                           patchpos_embed = patchpos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos(tensor)

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     patchpos_embed: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, patchpos_embed)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        
        q = k = v = self.with_pos_embed(tgt, querypos_embed)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, querypos_embed),
                                   key=patchpos_embed(memory),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = v = self.with_pos_embed(tgt2, querypos_embed)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, querypos_embed),
                                   key=patchpos_embed(memory),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, 
                           querypos_embed = querypos_embed, 
                           patchpos_embed = patchpos_embed)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, 
                           querypos_embed = querypos_embed, 
                           patchpos_embed = patchpos_embed)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


