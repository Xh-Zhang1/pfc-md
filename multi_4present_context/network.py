# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import rnn_ei1
from rnn_ei1 import EIRNN
import pdb
class Network(nn.Module):
    def __init__(self, hp, is_cuda=True):
        super(Network, self).__init__()

        self.hp = hp
          # rnn
        self.input_size = hp['n_input']
        self.hidden_size = hp['n_rnn']+hp['n_md']
        self.alpha = hp['alpha']
        self.sigma_rec = hp['sigma_rec']

        # Input weights
        self.RNN_layer = EIRNN(hp=self.hp,is_cuda=is_cuda)
        # self.softmax = torch.nn.Softmax(dim=-1)
        # self.act_fcn = lambda x: nn.functional.relu(x)

        # print('*** network ***',self)
        # for name, param in self.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()}\n")
        self.dropout_layer = nn.Dropout(hp['dropout'])
        #print(self.dropout_layer)
        #pdb.set_trace()



    def forward(self, inputs,initial_state):

        state_collector = self.RNN_layer.forward_rnn(inputs=inputs, init_state=initial_state)
        return state_collector


    def save(self,model_dir):
        save_path = os.path.join(model_dir, 'most_recent_model.pth')
        torch.save(self.state_dict(), save_path)

    def load(self, model_dir):
        if model_dir is not None:
            save_path = os.path.join(model_dir, 'most_recent_model.pth')
            if os.path.isfile(save_path):
                self.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage), strict=False)

















