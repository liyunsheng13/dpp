import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import os
import scipy.io as sio
from itertools import compress
import pdb

__all__ = ["DPP"]


@META_ARCH_REGISTRY.register()
class DPP(nn.Module):
    """
    Implement DPP
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DPP.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DPP.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DPP.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DPP.NUM_HEADS
        self.init_proposal_bbox = cfg.MODEL.DPP.INIT_PROPOSAL_BBOX

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        try:
            self.size_divisibility = self.backbone.size_divisibility
        except:
            self.size_divisibility = 32
        
        # Build Proposals.
        if os.path.exists(self.init_proposal_bbox):
            self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
            self.init_proposal_features = None
            self.init_proposal_boxes.weight.data=torch.Tensor(sio.loadmat(self.init_proposal_bbox)['cluster_center'])
            print ('prior bboxes are initialized from kmeans cluster')
        else:
            self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
            self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
            bbox_init = torch.rand(self.num_proposals, 4)
            bbox_xyxy = torch.zeros(self.num_proposals, 4)
            bbox_xyxy[:,0] = torch.min(bbox_init[:,0],bbox_init[:,2])
            bbox_xyxy[:,2] = torch.max(bbox_init[:,0],bbox_init[:,2])
            bbox_xyxy[:,1] = torch.min(bbox_init[:,1],bbox_init[:,3])
            bbox_xyxy[:,3] = torch.max(bbox_init[:,1],bbox_init[:,3])
            bbox_xyhw = torch.zeros(self.num_proposals, 4)
            bbox_xyhw[:,0] = (bbox_xyxy[:,0] + bbox_xyxy[:,2])/2
            bbox_xyhw[:,2] = bbox_xyxy[:,2] - bbox_xyxy[:,0]
            bbox_xyhw[:,1] = (bbox_xyxy[:,1] + bbox_xyxy[:,3])/2
            bbox_xyhw[:,3] = bbox_xyxy[:,3] - bbox_xyxy[:,1]
            self.init_proposal_boxes.weight.data = bbox_xyhw
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.DPP.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DPP.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DPP.L1_WEIGHT
        prune_weight = cfg.MODEL.DPP.PRUNE_WEIGHT
        prune_weight_st = cfg.MODEL.DPP.PRUNE_WEIGHT_ST
        prune_interIOU_weight = cfg.MODEL.DPP.PRUNE_INTERIOU_WEIGHT
        prune_interIOU_st_weight = cfg.MODEL.DPP.PRUNE_INTERIOU_ST_WEIGHT
        no_object_weight = cfg.MODEL.DPP.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DPP.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DPP.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight, "loss_prune": prune_weight, "loss_prune_st": prune_weight_st, \
                "loss_prune_inter_iou": prune_interIOU_weight, "loss_prune_inter_iou_score": prune_interIOU_st_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord, policy, out_pred_score, policy_st = self.head(features, proposal_boxes, self.init_proposal_features)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'policy': policy, 'out_pred_score': out_pred_score, 'policy_st': policy_st}
        output['pro_boxes'] = proposal_boxes
        #pdb.set_trace()

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            iteration = 5
            output = {'pred_logits': outputs_class[iteration], 'pred_boxes': outputs_coord[iteration], 'policy': policy[:iteration+1], 'out_pred_score': out_pred_score[iteration], 'policy_st': policy_st[:iteration+1]}

            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes, policy[0])

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
                processed_results[-1]['ratio'] = [pl.sum() for pl in policy]
                processed_results[-1]['ratio_st'] = [pl.sum() for pl in policy_st]
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            image_size_xyxy_tgt_match = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes) if len(gt_boxes) < self.num_proposals else self.num_proposals, 1)
            target["image_size_xyxy_tgt_match"] = image_size_xyxy_tgt_match.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes, policy):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        #pdb.set_trace()
        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device).\
                     unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, policy_per_image, image_size) in enumerate(zip(
                    scores, box_pred, policy, image_sizes
            )):
                tmp=list(compress(range(policy_per_image.size(0)), policy_per_image[:,0].long()))
                scores_per_image=scores_per_image[tmp,:]
                box_pred_per_image = box_pred_per_image[tmp,:]
                labels = torch.arange(self.num_classes, device=self.device).\
                        unsqueeze(0).repeat(len(tmp), 1).flatten(0, 1)

                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(len(tmp), sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        #pdb.set_trace()
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
