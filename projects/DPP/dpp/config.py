#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_dpp_config(cfg):
    """
    Add config for DPP.
    """
    cfg.MODEL.DPP = CN()
    cfg.MODEL.DPP.NUM_CLASSES = 80
    cfg.MODEL.DPP.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DPP.NHEADS = 8
    cfg.MODEL.DPP.DROPOUT = 0.0
    cfg.MODEL.DPP.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DPP.ACTIVATION = 'relu'
    cfg.MODEL.DPP.HIDDEN_DIM = 256
    cfg.MODEL.DPP.NUM_CLS = 1
    cfg.MODEL.DPP.NUM_REG = 3
    cfg.MODEL.DPP.NUM_HEADS = 6
    cfg.MODEL.DPP.INIT_PROPOSAL_BBOX = ""
    cfg.MODEL.DPP.ATTN = ""
    cfg.MODEL.DPP.TAU = 1.0

    # Dynamic Conv.
    cfg.MODEL.DPP.NUM_DYNAMIC = 2
    cfg.MODEL.DPP.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DPP.CLASS_WEIGHT = 2.0
    cfg.MODEL.DPP.GIOU_WEIGHT = 2.0
    cfg.MODEL.DPP.L1_WEIGHT = 5.0
    cfg.MODEL.DPP.PRUNE_WEIGHT = 1.0
    cfg.MODEL.DPP.PRUNE_WEIGHT_ST = 1.0
    cfg.MODEL.DPP.PRUNE_INTERIOU_WEIGHT = 1.0
    cfg.MODEL.DPP.PRUNE_INTERIOU_ST_WEIGHT = 1.0
    cfg.MODEL.DPP.PRUNE_INTERIOU_ALPHA = 2.0
    cfg.MODEL.DPP.DEEP_SUPERVISION = True
    cfg.MODEL.DPP.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.DPP.PRUNE_LOWERBOUND = 10.0
    cfg.MODEL.DPP.PRUNE_MULTIPLIER = 2.0
    cfg.MODEL.DPP.PRUNE_LOWERBOUND_ST = 10.0
    cfg.MODEL.DPP.PRUNE_MULTIPLIER_ST = 2.0

    # Focal Loss.
    cfg.MODEL.DPP.USE_FOCAL = True
    cfg.MODEL.DPP.ALPHA = 0.25
    cfg.MODEL.DPP.GAMMA = 2.0
    cfg.MODEL.DPP.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    #Prune
    cfg.MODEL.DPP.PRUNE_MARGIN = 0.0
    cfg.MODEL.DPP.PRUNE_STAGE = []
