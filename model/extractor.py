import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from model.pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow
from model.pointconv_util import index_points_gather as index_points, index_points_group, Conv1d


class PointConvEncoderLight(nn.Module):
    def __init__(self, weightnet=8):
        super(PointConvEncoderLight, self).__init__()
        feat_nei = 32
        self.upsample = UpsampleFlow()

        self.level0_lift = Conv1d(3, 32)
        self.level0 = PointConv(feat_nei, 32 + 3, 32, weightnet = weightnet) # out
        self.level0_1 = Conv1d(32, 32)
        
        self.level1 = PointConvD(4096, feat_nei, 32 + 3, 32, weightnet = weightnet)
        self.level1_0 = Conv1d(32, 32)# out
        self.level1_1 = Conv1d(32, 64)

        self.level2 = PointConvD(2048, feat_nei, 64 + 3, 128, weightnet = weightnet)# out
        

    def forward(self, xyz, color):
        feat_l0 = self.level0_lift(color)
        feat_l0 = self.level0(xyz, feat_l0)
        feat_l0_1 = self.level0_1(feat_l0)

        #l1
        pc_l1, feat_l1, fps_l1 = self.level1(xyz, feat_l0_1)
        feat_l1 = self.level1_0(feat_l1)
        feat_l1_2 = self.level1_1(feat_l1)

        #l2
        pc_l2, feat_l2, fps_l2 = self.level2(pc_l1, feat_l1_2)
        

        return [xyz, pc_l1, pc_l2], \
                [feat_l0, feat_l1, feat_l2], \
                [fps_l1, fps_l2]