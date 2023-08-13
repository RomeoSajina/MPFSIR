import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


def afnc():
    return nn.PReLU()


# https://pytorch.org/docs/stable/nn.init.html
def init_weights(w):
    nn.init.xavier_uniform_(w, nn.init.calculate_gain('relu'))


class TemporalFC(nn.Module):

    def __init__(self, input_len, output_len, act_on_last_layer=True):
        super().__init__()
        self.act_on_last_layer = act_on_last_layer

        input_dim = input_len
        hidden_dim = 512
        self.output_len = output_len
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, input_dim)

        self.arin = Rearrange('b n d -> b d n')
        self.arout = Rearrange('b d n -> b n d')

        self.activation1 = afnc()
        self.activation2 = afnc()
        self.activation3 = afnc()

        self.temporal_norm0 = LN(hidden_dim)
        self.temporal_norm1 = LN(input_len)
        self.temporal_norm2 = LN(input_len)

        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)
        self.d3 = nn.Dropout(p=0.1)
        self.reset_parameters()

    def reset_parameters(self):
        print("reset TemporalFC")
        init_weights(self.fc1.weight)
        init_weights(self.fc2.weight)
        init_weights(self.fc3.weight)

        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = self.arin(x)
        x_orig = x.clone()

        x = self.fc1(x)
        x = self.activation1(x)
        x = self.temporal_norm0(x)
        x = self.d1(x)

        x = self.fc2(x)
        x = self.activation2(x)
        x = self.temporal_norm1(x)
        x = self.d2(x)
        x = x + x_orig
        x_orig = x.clone()

        x = self.fc3(x)
        if self.act_on_last_layer:
            x = self.activation3(x)
            x = self.temporal_norm2(x)
            x = self.d3(x)

            x = x + x_orig # no skip at last layer

        output = self.arout(x)

        return output[:, :self.output_len]


class TemporalCTX(nn.Module):

    def __init__(self, input_len, output_len):
        super().__init__()

        self.output_len = output_len
        self.tfcjoin1 = nn.Linear(input_len*2, input_len*5)
        self.tfcjoin2 = nn.Linear(input_len*5, input_len*2)

        self.arin = Rearrange('b n d -> b d n')
        self.arout = Rearrange('b d n -> b n d')

        self.temporal_norm1 = LN(input_len*5)
        self.temporal_norm2 = LN(input_len*2)
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)

        self.activation1 = afnc()
        self.activation2 = afnc()

        self.reset_parameters()

    def reset_parameters(self):
        print("reset TemporalCTX")
        init_weights(self.tfcjoin1.weight)
        init_weights(self.tfcjoin2.weight)

        nn.init.constant_(self.tfcjoin1.bias, 0)
        nn.init.constant_(self.tfcjoin2.bias, 0)

    def forward(self, x, y):

        joined = torch.cat((x, y), dim=1)

        joined = self.arin(joined)

        x_orig = joined.clone()

        joined = self.tfcjoin1(joined)
        joined = self.activation1(joined)
        joined = self.temporal_norm1(joined)
        joined = self.d1(joined)

        joined = self.tfcjoin2(joined)
        joined = self.activation2(joined)
        joined = self.temporal_norm2(joined)
        joined = self.d2(joined)

        joined = joined + x_orig

        joined = self.arout(joined)
        ox, oy = joined[:, :self.output_len], joined[:, self.output_len:]

        return ox, oy


class SpatialCTX(nn.Module):

    def __init__(self, kps_dim=39):
        super().__init__()

        self.kps_dim = kps_dim
        self.sfcjoin1 = nn.Linear(kps_dim*2, kps_dim*5)
        self.sfcjoin2 = nn.Linear(kps_dim*5, kps_dim*2)

        self.spatial_norm1 = LN(kps_dim*5)
        self.spatial_norm2 = LN(kps_dim*2)
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)

        self.activation1 = afnc()
        self.activation2 = afnc()

        self.reset_parameters()

    def reset_parameters(self):
        print("reset SpatialCTX")
        init_weights(self.sfcjoin1.weight)
        init_weights(self.sfcjoin2.weight)

        nn.init.constant_(self.sfcjoin1.bias, 0)
        nn.init.constant_(self.sfcjoin2.bias, 0)

    def forward(self, x, y):
        joined = torch.cat((x, y), dim=-1)
        x_orig = joined.clone()

        joined = self.sfcjoin1(joined)
        joined = self.activation1(joined)
        joined = self.spatial_norm1(joined)
        joined = self.d1(joined)

        joined = self.sfcjoin2(joined)
        joined = self.activation2(joined)
        joined = self.spatial_norm2(joined)
        joined = self.d2(joined)

        joined = joined + x_orig

        ox, oy = joined[..., :self.kps_dim], joined[..., self.kps_dim:]

        return ox, oy


class CtxFCL(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.temporal_in = TemporalFC(self.config.dct_n, self.config.dct_n)
        self.temporal_out = TemporalFC(self.config.dct_n, self.config.dct_n, False)

        self.temporalctx = TemporalCTX(self.config.dct_n, self.config.dct_n)

        self.spatialctx = SpatialCTX(kps_dim=config.num_kps*3)

    def forward(self, x, y):

        x = self.temporal_in(x)
        y = self.temporal_in(y)

        ox, oy = self.temporalctx(x, y)

        ox, oy = self.spatialctx(ox, oy)

        _ox = self.temporal_out(ox)
        _oy = self.temporal_out(oy)
        
        return _ox, _oy, ox, oy


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


class MPFSCTX_2P(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = CtxFCL(config).to(config.device)
        self.scint = nn.Sequential(
                                    nn.Linear(self.config.dct_n*config.num_kps*3, self.config.dct_n),
                                    nn.ReLU(True),
                                    LN(self.config.dct_n),
                                    nn.Dropout(p=0.1),
                                    nn.Flatten(),
                                    nn.Linear(self.config.dct_n*2, 3)
        ).to(config.device)
        self.sftmx = nn.Softmax(dim=1).to(config.device)
        
        seq_len = self.config.input_len + self.config.output_len
        dct_size = self.config.dct_n if self.config.dct_n > seq_len else seq_len
        dct_m,idct_m = get_dct_matrix(dct_size);print("using DCT {0} matrix".format(dct_size))

        self.dct_m = torch.tensor(dct_m).float().to(config.device)
        self.idct_m = torch.tensor(idct_m).float().to(config.device)
        
        print("using dct:", self.config.use_dct)

    def _prepare_in(self, x):
                            
        INPUT_LEN, OUTPUT_LEN, dct_n = self.config.input_len, self.config.output_len, self.config.dct_n
        SL = INPUT_LEN + OUTPUT_LEN

        i_idx = np.append(np.arange(0, INPUT_LEN), np.repeat([INPUT_LEN - 1], OUTPUT_LEN))
        x = x.clone().reshape(-1, x.shape[1], x.shape[2]*x.shape[3]).float()

        x = x.transpose(0, 1).reshape(x.shape[1], -1)
        
        if self.config.use_dct:
            x = torch.matmul(self.dct_m[0:dct_n, :SL], x[i_idx, :])
        else:
            x = x[i_idx, :]
           
        x = x.transpose(0, 1).reshape(-1, self.config.num_kps*3, dct_n)

        x = x.transpose(1, 2)

        return x

    def _prepare_out(self, x):
        INPUT_LEN, OUTPUT_LEN, dct_n = self.config.input_len, self.config.output_len, self.config.dct_n
        SL = INPUT_LEN + OUTPUT_LEN

        y = x.transpose(1, 2)

        y = y.view(-1, dct_n).transpose(0, 1)

        if self.config.use_dct:
            y = torch.matmul(self.idct_m[:SL, :dct_n], y)
        
        y = y.transpose(0, 1).contiguous().view(-1, self.config.num_kps*3, SL).transpose(1, 2)            
        
        return y

    def forward(self, data, train=True):

        kp0 = self._prepare_in(data['keypoints0'])
        kp1 = self._prepare_in(data['keypoints1'])

        feat_f0, feat_f1, msg0, msg1 = self.backbone(kp0, kp1)
        _sctx = self.scint( torch.cat((msg0.reshape(msg0.shape[0], 1, -1), msg1.reshape(msg1.shape[0], 1, -1)), dim=1) )

        data.update({
            'z0': self._prepare_out(feat_f0)[:, self.config.input_len:],
            'z1': self._prepare_out(feat_f1)[:, self.config.input_len:],
            's0': self.sftmx( _sctx ),
        })

        return data


class MPFSCTX_3P(MPFSCTX_2P):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, data, train=True):

        def do_for_kps(x0, x1, x2):

            kp0 = self._prepare_in(x0)
            kp1 = self._prepare_in(x1)
            kp2 = self._prepare_in(x2)

            feat_f0_1, feat_f1_0, msg0, msg1 = self.backbone(kp0, kp1)
            _sctx01 = self.scint( torch.cat((msg0.reshape(msg0.shape[0], 1, -1), msg1.reshape(msg1.shape[0], 1, -1)), dim=1) )

            feat_f1_2, feat_f2_1, msg12, msg21 = self.backbone(kp1, kp2)
            _sctx12 = self.scint( torch.cat((msg12.reshape(msg12.shape[0], 1, -1), msg21.reshape(msg21.shape[0], 1, -1)), dim=1) )

            feat_f0_2, feat_f2_0, msg02, msg20 = self.backbone(kp0, kp2)
            _sctx02 = self.scint( torch.cat((msg02.reshape(msg02.shape[0], 1, -1), msg20.reshape(msg20.shape[0], 1, -1)), dim=1) )

            return self._prepare_out((feat_f0_1+feat_f0_2)/2.), self._prepare_out((feat_f1_0+feat_f1_2)/2.), self._prepare_out((feat_f2_1+feat_f2_0)/2.), _sctx01, _sctx12, _sctx02

        z0, z1, z2, _sctx01, _sctx12, _sctx02 = do_for_kps(data['keypoints0'], data['keypoints1'], data['keypoints2'])

        data.update({
            'z0': z0[:, self.config.input_len:],
            'z1': z1[:, self.config.input_len:],
            'z2': z2[:, self.config.input_len:],

            's01': self.sftmx( _sctx01 ),
            's12': self.sftmx( _sctx12 ),
            's02': self.sftmx( _sctx02 ),
        })

        return data

    
def create_model(config, persons="2"):
    if persons == "3":
        return MPFSCTX_3P(config=config).to(config.device)
    
    return MPFSCTX_2P(config=config).to(config.device)
