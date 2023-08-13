import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np

from model import *


class IndependentTemporalCTX(nn.Module):

    def __init__(self, input_len, output_len, act_on_last_layer=True):
        super().__init__()
        self.act_on_last_layer = act_on_last_layer

        self.output_len = output_len
        self.tfcjoin1 = nn.Linear(input_len*1, input_len*10)#2, 5
        self.tfcjoin2 = nn.Linear(input_len*10, input_len*1)#5, 2

        self.arin = Rearrange('b n d -> b d n')
        self.arout = Rearrange('b d n -> b n d')

        self.temporal_norm1 = LN(input_len*10)#
        self.temporal_norm2 = LN(input_len*1)
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)

        self.activation1 = afnc()
        self.activation2 = afnc()

        self.reset_parameters()

    def reset_parameters(self):
        print("reset IndependentTemporalCTX")
        init_weights(self.tfcjoin1.weight)
        init_weights(self.tfcjoin2.weight)

        nn.init.constant_(self.tfcjoin1.bias, 0)
        nn.init.constant_(self.tfcjoin2.bias, 0)

    def do_forward(self, joined):

        joined = self.arin(joined)

        x_orig = joined.clone()

        joined = self.tfcjoin1(joined)
        joined = self.activation1(joined)
        joined = self.temporal_norm1(joined)
        joined = self.d1(joined)

        joined = self.tfcjoin2(joined)
        if self.act_on_last_layer:
            joined = self.activation2(joined)
            joined = self.temporal_norm2(joined)
            joined = self.d2(joined)

            joined = joined + x_orig

        joined = self.arout(joined)

        return joined

    def forward(self, x, y):
        return self.do_forward(x), self.do_forward(y)


class IndependentSpatialCTX(nn.Module):

    def __init__(self, kps_dim=39, act_on_last_layer=True):
        super().__init__()
        self.act_on_last_layer = act_on_last_layer

        self.kps_dim = kps_dim
        self.sfcjoin1 = nn.Linear(kps_dim*1, kps_dim*10) #2, 5
        self.sfcjoin2 = nn.Linear(kps_dim*10, kps_dim*1) #5, 2

        self.spatial_norm1 = LN(kps_dim*10)#5
        self.spatial_norm2 = LN(kps_dim*1)
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)

        self.activation1 = afnc()
        self.activation2 = afnc()

        self.reset_parameters()

    def reset_parameters(self):
        print("reset IndependentSpatialCTX")
        init_weights(self.sfcjoin1.weight)
        init_weights(self.sfcjoin2.weight)

        nn.init.constant_(self.sfcjoin1.bias, 0)
        nn.init.constant_(self.sfcjoin2.bias, 0)

    def do_forward(self, joined):
        x_orig = joined.clone()

        joined = self.sfcjoin1(joined)
        joined = self.activation1(joined)
        joined = self.spatial_norm1(joined)
        joined = self.d1(joined)

        joined = self.sfcjoin2(joined)
        if self.act_on_last_layer:
            joined = self.activation2(joined)
            joined = self.spatial_norm2(joined)
            joined = self.d2(joined)

            joined = joined + x_orig

        return joined

    def forward(self, x, y):
        return self.do_forward(x), self.do_forward(y)


class IndependentCtxFCL(CtxFCL):

    def __init__(self, config):
        super().__init__(config)
        self.temporalctx = IndependentTemporalCTX(self.config.dct_n, self.config.dct_n)
        self.spatialctx = IndependentSpatialCTX()


class IndependentMPF_2P(MPFSCTX_2P):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = IndependentCtxFCL(config).to(config.device)


def create_model(config):
    return IndependentMPF_2P(config=config).to(config.device)
