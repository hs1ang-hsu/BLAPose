import torch
import torch.nn as nn
import numpy as np

class BoneLengthModel(torch.nn.Module):
    def __init__(self, num_joints_in, in_features, embed_dim=256, hidden_size=512,
            num_layers=2, dropout=0.25, bidirectional=True):
        super().__init__()
        
        # arch
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # model
        self.expand_conv = nn.Linear(num_joints_in*2, self.embed_dim)
        if self.bidirectional:
            self.num_gru = 2
            self.gru = nn.GRU(self.embed_dim, self.hidden_size, self.num_layers, dropout=dropout if num_layers>1 else 0,
                            bidirectional=True, batch_first=True)
        else:
            self.num_gru = 2
            gru = []
            for i in range(self.num_gru):
                gru.append(nn.GRU(self.embed_dim, self.hidden_size, self.num_layers, dropout=dropout if num_layers>1 else 0,
                            bidirectional=False, batch_first=True))
            self.gru = nn.ModuleList(gru)
        
        self.fc = nn.Linear(self.encoder_output_size, num_joints_in-1, bias=False)
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def set_bn_momentum(self, momentum):
        pass
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.h0_num_layers, batch_size, self.hidden_size).zero_()
        return hidden

    @property
    def encoder_output_size(self):
        return self.num_gru * self.hidden_size
    
    @property
    def h0_num_layers(self):
        return self.num_layers * 2 if self.bidirectional else self.num_layers

    def forward(self, x, h_0=None, causal=False):
        b, f, j, c = x.shape
        x = x.contiguous().view(b, f, -1) # (b, f, j*c)
        x = self.drop(self.relu(self.expand_conv(x)))
        
        if self.bidirectional:
            self.gru.flatten_parameters()
            if h_0 != None:
                x, h_n = self.gru(x, h_0)
                # c_0 = h_0.clone()
                # x, (h_n, c_n) = self.gru(x, (h_0, c_0))
            else:
                x, h_n = self.gru(x)
                # x, (h_n, c_n) = self.gru(x)
        
            ### h_n.shape = (B, D*H_out)
            h = torch.cat((h_n[-1], h_n[-2]), axis=-1) if self.bidirectional else h_n[-1] # h.shape = (B, H_out)
            h = self.sigmoid(h)
            h = self.fc(h)
        else:
            for i in range(self.num_gru):
                self.gru[i].flatten_parameters()
            
            h_n = []
            if h_0 != None:
                for i in range(self.num_gru):
                    x_, h_n_ = self.gru[i](x, h_0[i] if causal else h_0)
                    h_n.append(h_n_[-1])
            else:
                for i in range(self.num_gru):
                    x_, h_n_ = self.gru[i](x)
                    h_n.append(h_n_[-1])
            h = torch.cat(h_n, axis=-1) # h.shape = (B, H_out)
            h = self.sigmoid(h)
            h = self.fc(h)
        
        if causal:
            return h, h_n
        else:
            return h
    
    def _init_weights(self, pretrained=''):
        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        length_chk = torch.load(pretrained, map_location=lambda storage, loc: storage)
        need_init_state_dict = {}
        for name, m in length_chk['model_pos'].items():
            if name in parameters_names or name in buffers_names:
                need_init_state_dict[name] = m
        self.load_state_dict(need_init_state_dict, strict=True)

if __name__ == '__main__':
    model = BoneLengthModel(17, 2)
    
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('Trainable parameter count:', model_params)
    
    A = torch.zeros(64,81,17,2)
    B = model(A)
    print(B.shape)