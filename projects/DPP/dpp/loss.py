"""
DPP model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit

from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

from scipy.optimize import linear_sum_assignment
from itertools import compress
import pdb

class SetCriterion(nn.Module):
    """ This class computes the loss for DPP.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DPP.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DPP.GAMMA
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)
        self.policy_stage = self.cfg.MODEL.DPP.PRUNE_STAGE#[1,3,5]
        self.num_proposal = self.cfg.MODEL.DPP.NUM_PROPOSALS
        self.prune_lowerbound = self.cfg.MODEL.DPP.PRUNE_LOWERBOUND
        self.prune_multiplier = self.cfg.MODEL.DPP.PRUNE_MULTIPLIER
        self.prune_lowerbound_st = self.cfg.MODEL.DPP.PRUNE_LOWERBOUND_ST
        self.prune_multiplier_st = self.cfg.MODEL.DPP.PRUNE_MULTIPLIER_ST
        self.prune_interIOU_alpha = self.cfg.MODEL.DPP.PRUNE_INTERIOU_ALPHA
        self.kldiv = torch.nn.KLDivLoss(reduction='none')

    def loss_labels(self, outputs, targets, indices, num_boxes, policy, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.use_focal:
            policy = policy.flatten(0, 1)
            tmp=list(compress(range(policy.size(0)), policy[:,0].long()))

            src_logits = src_logits.flatten(0, 1)[tmp,:]
            # prepare one_hot target.
            target_classes = target_classes.flatten(0, 1)[tmp]

            pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
            labels = torch.zeros_like(src_logits)
            labels[pos_inds, target_classes[pos_inds]] = 1
            # comp focal loss.
            class_loss = sigmoid_focal_loss_jit(
                src_logits,
                labels,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / num_boxes
            losses = {'loss_ce': class_loss}
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes, policy):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / max(len(src_boxes),1) #num_boxes

        image_size = torch.cat([v["image_size_xyxy_tgt_match"][:len(indices[i][0])] for i,v in enumerate(targets)])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / max(len(src_boxes),1) #num_boxes

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, policy, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, policy, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'policy'}
        policy = outputs['policy']
        policy_st = outputs['policy_st']
        if self.training:
            indices, iou = self.matcher(outputs_without_aux, targets, policy[0])
        else:
            indices, iou = self.matcher(outputs_without_aux, targets, policy[-1])

        pred_interIOU_loss = 0
        pred_interIOU_loss_st = 0
        pred_interIOUscore_loss = 0
        pred_intraIOU_loss = 0
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, policy[0]))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            
            array_indices, array_iou = [], []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, iou = self.matcher(aux_outputs, targets, policy[0])
                array_indices.append(indices)
                array_iou.append(iou)
                if i == 0:
                    accum_indices = []
                    for _ in range(len(indices)):
                        accum_indices.append(set())
                for j, indice in enumerate(indices):
                    accum_indices[j].update(set(indice[0].numpy()))

            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, iou = array_indices[i], array_iou[i]
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, policy[0], **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if i+1 in self.policy_stage:
                    mask = policy[i+1][:,:,0].detach()

                    pred_boxes = aux_outputs['pred_boxes'].flatten(0,1).detach()
                    pred_ious,_ = box_iou(pred_boxes,pred_boxes)
                    eye=torch.eye(len(pred_ious)).cuda(pred_ious.device)
                    pred_ious = (pred_ious-eye).view(aux_outputs['pred_boxes'].size(0),self.num_proposal,-1)
                    pred_ious = torch.stack([piou[:,self.num_proposal*i:self.num_proposal*(i+1)] for i, piou in enumerate(pred_ious)])
                    ##########inter IOU loss computation#################
                    coeff1 = (1-iou)**self.prune_interIOU_alpha
                    coeff2 = 1 - coeff1
                    for j in range(len(coeff1)):
                        spec_coeff = list(accum_indices[j])
                        coeff1[j,spec_coeff]=0
                        coeff2[j,spec_coeff]=5
                    if mask.sum() != 0:
                        pred_l = (coeff1*policy[i+1][:,:,0]+coeff2*(1-policy[i+1][:,:,0]))*mask/mask.sum()
                    else:
                        pred_l = (coeff1*policy[i+1][:,:,0]+coeff2*(1-policy[i+1][:,:,0]))*mask
                    pred_l_st = (coeff1*policy_st[i+1][:,:,0]+coeff2*(1-policy_st[i+1][:,:,0]))*(1-mask)
                    if (1-mask).sum() != 0:
                        pred_l_st = pred_l_st/(1-mask).sum()
                    pred_interIOU_loss += pred_l.sum()
                    pred_interIOU_loss_st += pred_l_st.sum()

        pred_loss = 0.0
        pred_loss_st = 0.0
        N = len(targets)
        if len(self.policy_stage) > 0:
            for i in range(N):
                num_pro_prune = min(max(self.prune_multiplier*len(targets[i]['labels']), self.prune_lowerbound), self.num_proposal)
                rho = (num_pro_prune/self.num_proposal)**(1/len(self.policy_stage))
                num_pro_prune_st = min(max(self.prune_multiplier_st*len(targets[i]['labels']), self.prune_lowerbound_st), self.num_proposal)
                rho_st = (num_pro_prune_st/self.num_proposal)**(1/len(self.policy_stage))
                ratio = 1
                ratio_st = 1
                for j in range(len(self.policy_stage)):
                    if self.policy_stage[j] >= len(policy):
                        continue
                    ratio = ratio*rho
                    ratio_st = ratio_st*rho_st
                    pred_loss += (policy[self.policy_stage[j]][i].mean()-ratio)**2/N
                    pred_loss_st += (policy_st[self.policy_stage[j]][i].mean()-min(ratio_st, 1-ratio))**2/N
            losses['loss_prune'] = pred_loss
            losses['loss_prune_st'] = pred_loss_st
            losses['loss_prune_inter_iou'] = pred_interIOU_loss
            losses['loss_prune_inter_st_iou'] = pred_interIOU_loss_st
        return losses



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, use_focal: bool = False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DPP.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DPP.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.iou = 0
        self.pred = 0
        self.tgt = 0
        self.policy = 0
        self.prune_score = 0

    @torch.no_grad()
    def forward(self, outputs, targets, policy):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal:
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
        image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        self.policy = policy.cpu().numpy()
        iou, _ = box_iou(out_bbox, tgt_bbox) #YsL added for analysis
        conf = 0.5*(1-cost_giou)/2 + 0.5*out_prob[:,tgt_ids]
        self.iou = iou.cpu().numpy()
        self.pred, self.tgt = out_bbox.cpu().numpy(), tgt_bbox.cpu().numpy()
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        res = [list(compress(range(policy.size(1)), policy[i,:,0].long())) for i in range(len(policy))]
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i, res[i]]) for i, c in enumerate(C.split(sizes, -1))]
        conf = conf.view(bs, num_queries, -1)
        tmp = torch.zeros(bs, num_queries, device=conf.device)
        for i, area in enumerate(conf.split(sizes, -1)):
            if area.size(-1) != 0:
                tmp[i] = area[i].max(dim=-1)[0]
        return [(torch.as_tensor(res[ii], dtype=torch.int64)[i], torch.as_tensor(j, dtype=torch.int64)) for ii, (i, j) in enumerate(indices)], tmp
