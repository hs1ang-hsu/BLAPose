import os
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath

from utils.model_length import BoneLengthModel
from utils.model_conv import TemporalModelOptimized1f
from utils.bone_utils import *


class MixModel(nn.Module):
    def __init__(self, num_joints, in_features, embed_dim=256, hidden_size=512,
            num_layers=1, filter_widths=[3,3,3,3], dropout=0.25, bidirectional=True,
            channels=1024):
        super().__init__()
        
        # general elements
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.length_model = BoneLengthModel(num_joints, in_features, embed_dim=embed_dim, hidden_size=hidden_size,
                num_layers=num_layers, dropout=0, bidirectional=bidirectional)
        
        # videopose
        self.direction_model = TemporalModelOptimized1f(
            num_joints, in_features, num_joints, filter_widths=filter_widths, dropout=dropout
        )

    def set_bn_momentum(self, momentum):
        self.length_model.set_bn_momentum(momentum)
        self.direction_model.set_bn_momentum(momentum)
        
    def init_hidden(self, batch_size):
        return self.length_model.init_hidden(batch_size)
    
    def receptive_field(self):
        return self.direction_model.receptive_field()

    def eval_data_prepare(self, inputs_2d):
        receptive_field = self.receptive_field()
        inputs_2d_p = torch.squeeze(inputs_2d)
        out_num = inputs_2d_p.shape[0] - receptive_field + 1
        eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2]).to(inputs_2d.device)
        for i in range(out_num):
            eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
        return eval_input_2d

    def forward(self, x, x_chunk=None, h_0=None, L=None, eval=False):
        if L is None:
            L = self.length_model(x if x_chunk is None else x_chunk, h_0) # L.shape = (B, 16)
            for p in bone_symmetry:
                L[:,p] = L[:,p].mean(dim=1)[:,None]
        
        if not eval:
            x = self.direction_model(x)
            D, _ = poses2bone_torch(x)
            D = D.unsqueeze(1)
        else:
            x = self.eval_data_prepare(x)
            x = self.direction_model(x)
            D, _ = poses2bone_torch(x)
            D = D.unsqueeze(0)

        s = bones2poses_torch(D, L)
        if not eval:
            return s, D.squeeze()
        else:
            return s, L, D
    
    
    def _init_weights(self, length_model='', direction_model=''):
        if os.path.isfile(length_model) and os.path.isfile(direction_model):
            self.length_model._init_weights(length_model)
            self.direction_model._init_weights(direction_model)
            pass
        else:
            print('Check if specified checkpoints exist.')
            print('Length model:', length_model)
            print('Direction model:', direction_model)
            exit()
