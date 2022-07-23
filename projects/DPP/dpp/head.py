"""
DPP Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
#from nms import nms
from itertools import compress
import pdb

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_coeff = 0

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        #pdb.set_trace()
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = (attn_policy + (1.0 - attn_policy) * eye)
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #pdb.set_trace()
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PredictorLGAttnST(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384, margin=0.0):
        super().__init__()
        self.margin = margin
        self.self_attn = Attention(embed_dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
        self.out_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
        )
        self.linear_dy = nn.Linear(embed_dim // 4, 2)
        self.linear_st = nn.Linear(embed_dim // 4, 2)
        self.logsf = nn.LogSoftmax(dim=-1)

    def forward(self, x, policy):
        #pdb.set_trace()
        x = x + self.self_attn(x, policy)
        B, N, C = x.size()
        x = self.out_conv(x)
        x_dy = self.logsf(self.linear_dy(x))
        x_st = self.logsf(self.linear_st(x))

        return x_dy, x_st

class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        
        # Build heads.
        num_classes = cfg.MODEL.DPP.NUM_CLASSES
        d_model = cfg.MODEL.DPP.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.DPP.DIM_FEEDFORWARD
        nhead = cfg.MODEL.DPP.NHEADS
        dropout = cfg.MODEL.DPP.DROPOUT
        activation = cfg.MODEL.DPP.ACTIVATION
        num_heads = cfg.MODEL.DPP.NUM_HEADS
        self.prune_stage = cfg.MODEL.DPP.PRUNE_STAGE#{1,3,5}

        rcnn_head = []
        for i in range(6):
            #pdb.set_trace()
            if i in self.prune_stage:
                rcnn_head.append(RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation, prune=True))
            else:
                rcnn_head.append(RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation, prune=False))
        self.head_series = nn.ModuleList(rcnn_head)

        self.return_intermediate = cfg.MODEL.DPP.DEEP_SUPERVISION
        
        # Init parameters.
        self.use_focal = cfg.MODEL.DPP.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.DPP.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, init_features):
        #pdb.set_trace()
        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes
        if init_features == None:
            proposal_features = None
        else:
            init_features = init_features.weight[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()

        policy = torch.ones(bs, proposal_features.size(1)//bs, 1, dtype=proposal_features.dtype, device=proposal_features.device)
        policy_st = torch.zeros(bs, proposal_features.size(1)//bs, 1, dtype=proposal_features.dtype, device=proposal_features.device)
        out_pred_policy = []
        out_pred_policy_st = []
        out_pred_score = []
        class_logits = torch.zeros((bboxes.size(0), bboxes.size(1), 80), dtype=bboxes.dtype, device=bboxes.device)
        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features, policy, pred_score, policy_st = rcnn_head(features, class_logits, bboxes, proposal_features, self.box_pooler, policy, policy_st)
            out_pred_policy.append(policy)
            out_pred_policy_st.append(policy_st)
            out_pred_score.append(pred_score)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()
        #pdb.set_trace()
        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), out_pred_policy, out_pred_score, out_pred_policy_st

        return class_logits[None], pred_bboxes[None], out_pred_policy, out_pred_score, out_pred_policy_st

class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0), inst_interact=None, prune=False):
        super().__init__()

        self.d_model = d_model
        self.margin = cfg.MODEL.DPP.PRUNE_MARGIN
        # dynamic.
        #pdb.set_trace()
        self.self_attn = Attention(d_model, num_heads=nhead, qkv_bias=True, qk_scale=None, attn_drop=dropout, proj_drop=dropout)
            
        self.inst_interact = DynamicConv(cfg)
        
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

        # cls.
        num_cls = cfg.MODEL.DPP.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.DPP.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = cfg.MODEL.DPP.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights
        self.prune = prune
        if prune:
            self.score_predictor = PredictorLGAttnST(d_model, self.margin)
        self.attn_coeff = 0

    def forward(self, features, logits, bboxes, pro_features, pooler, policy=None, policy_st=None):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """
        #pdb.set_trace()
        pro_features_last = pro_features
        N, nr_boxes = bboxes.shape[:2]
        
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)

        if pro_features == None:
            pro_features = roi_features.mean(0)[None]

        # self_att.
        #pdb.set_trace()
        pro_features = pro_features.view(N, nr_boxes, self.d_model)
        attn_policy_pred = policy + policy_st
        if self.prune:
            pred_score, pred_score_st = self.score_predictor(pro_features, None)
            pred_score, pred_score_st = pred_score.reshape(N, -1, 2), pred_score_st.reshape(N, -1, 2)
            if self.training:
                policy = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * attn_policy_pred
                policy_st = F.gumbel_softmax(pred_score_st, hard=True)[:, :, 0:1] * (1 - policy.detach()) * attn_policy_pred
            else:
                policy = (pred_score[:,:,0]>pred_score[:,:,1]).unsqueeze(-1) * attn_policy_pred
                policy_st = (pred_score_st[:,:,0]>pred_score_st[:,:,1]).unsqueeze(-1) * (1 - policy.detach()) * attn_policy_pred
        else:
            pred_score = None
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
        #pdb.set_trace()
        attn_policy = (policy + policy_st).detach()
        pro_features = pro_features + self.dropout1(self.self_attn(pro_features, None))

        self.attn_coeff = self.self_attn.attn_coeff

        pro_features = pro_features.permute(1,0,2)

        pro_features = self.norm1(pro_features)
        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)*policy.flatten(0,1)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features) 

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        class_logits = logits * (1-attn_policy) + class_logits.view(N, nr_boxes, -1) * attn_policy

        pred_bboxes = bboxes * (1-attn_policy) + pred_bboxes.view(N, nr_boxes, -1) * attn_policy
        
        obj_features = pro_features_last * (1-attn_policy.view(1,N*nr_boxes,-1)) + obj_features * attn_policy.view(1,N*nr_boxes,-1)

        return class_logits, pred_bboxes, obj_features, policy, pred_score, policy_st
    

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.layernorm(x)
        x = x.permute(0,3,1,2)
        return x

class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.DPP.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DPP.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.DPP.NUM_DYNAMIC
        self.num_prop = cfg.MODEL.DPP.NUM_PROPOSALS
        self.num_params = self.hidden_dim * self.dim_dynamic

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.activation = nn.ReLU(inplace=True)

        self.pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * self.pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        #pdb.set_trace()
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim) 

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)


        return features


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
    if activation == "na":
        return nn.Sequential()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
