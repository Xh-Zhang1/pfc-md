import sys
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.init as init
import pdb

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
from scipy.stats import ortho_group
import tools

class EIRecLinear(nn.Module):
    r"""Recurrent E-I Linear transformation.

    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """

    def __init__(self, hp,is_cuda=True):
        super().__init__()

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        hidden_size = hp['n_rnn']+hp['n_md']

        self.hp = hp
        self.hidden_size = hidden_size
        self.n_md = int(hp['n_md'])
        self.n_rnn= int(hp['n_rnn'])

        self.i_size_one = int((hp['n_rnn']*(1-self.hp['e_prop']))/3)#hidden_size - self.e_size
        self.i_size = 3*self.i_size_one
        self.pc = hp['n_rnn']-self.i_size
        self.e_size = self.pc + self.n_md

        # print('n_rnn',hp['n_rnn'])
        # print('i_size:',self.i_size)
        # print('pc',self.pc)
        # print('self.e_size :',self.e_size )
        # print('md:',self.n_md)

        mask = np.tile([1]*self.e_size+[-1]*self.i_size, (hidden_size, 1))
        if hp['mask_type'] == 'type2':
            mask = tools.mask_no_md_train_type2(hp=self.hp, md=self.n_md,pc=self.pc,i_size_one=self.i_size_one)
        if hp['mask_type'] == 'type3':
            mask = tools.mask_md_pfc_train_type3(hp=self.hp, md=self.n_md,pc=self.pc,i_size_one=self.i_size_one)
        if hp['mask_test'] == 'type2':
            #print('perform type2 ......')
            mask = tools.mask_no_md_train_type2(hp=self.hp, md=self.n_md,pc=self.pc,i_size_one=self.i_size_one)
        if hp['mask_test'] == 'type3':
            #print('perform type3 ......')
            mask = tools.mask_md_pfc_train_type3(hp=self.hp, md=self.n_md,pc=self.pc,i_size_one=self.i_size_one)
        if hp['mask_test'] == 'type3_test':
            mask = tools.type3_test(hp=self.hp, md=self.n_md,pc=self.pc,i_size_one=self.i_size_one)
        if hp['mask_test'] == 'type3_test_new':
            mask = tools.type3_test_new(hp=self.hp, md=self.n_md,pc=self.pc,i_size_one=self.i_size_one)


        np.fill_diagonal(mask, 0)
        self.mask = torch.tensor(mask, device=self.device,dtype=torch.float32)
        ############## hidden initialization
        if hp['init_hh']=='init1':
            hh_mask = torch.ones(hidden_size, hidden_size) - torch.eye(hidden_size)
            non_diag = torch.empty(hidden_size, hidden_size).normal_(0, hp['initial_std']/math.sqrt(hidden_size))
            weight = torch.eye(hidden_size)*0.999 + hh_mask * non_diag
            self.weight = nn.Parameter(weight)
        elif hp['init_hh']=='init2':# use for training
            self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        if hp['use_reset']=='yes':
            self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(6))
        # Scale E weight by E-I ratio

        self.weight.data[:, :self.pc] /= ((self.pc+self.n_md)/self.i_size)
        self.weight.data[:, self.n_rnn:self.n_rnn+self.n_md] /= (self.pc/self.i_size)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def effective_weight(self):

        if self.hp['is_EI'] == 'EI':
            effective_weights = torch.abs(self.weight) * self.mask
        elif self.hp['is_EI'] == 'EIno':
            effective_weights = self.weight

        return effective_weights

    def forward(self, input):
        # weight is non-negative

        return F.linear(input, self.effective_weight(), self.bias)







class EIRNN(nn.Module):
    """E-I RNN.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch, hidden_size)
    """

    def __init__(self, hp, is_cuda=True):
        super().__init__()

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        act_fcn = hp['activation']
        if act_fcn == 'relu':
            self.act_fcn = lambda x: nn.functional.relu(x)
        elif act_fcn == 'softplus':
            self.act_fcn = lambda x: nn.functional.softplus(x)
        self.hp = hp

        input_size =  hp['n_input']
        hidden_size = hp['n_rnn']+hp['n_md']
        output_size = hp['n_output']

        rnn_exc_inh = hp['n_rnn']

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.n_md = int(hp['n_md'])
        self.n_rnn= int(hp['n_rnn'])

        # self.e_size = int(hidden_size * hp['e_prop'])
        # self.i_size = hidden_size - self.e_size
        alpha = hp['dt'] / hp['tau']

        # self.dropout = nn.Dropout(p=0.1)


        #================= change the input =================
        #================= change the input =================
        weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(2.), 1./math.sqrt(2.))
        weight_ih[:, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.))
        self.weight_ih = nn.Parameter(weight_ih)

        # self.input2h = F.linear(input, self.weight_ih, self.bias)


        self.h2h = EIRecLinear(self.hp,is_cuda=is_cuda)

        self.weight_out = nn.Parameter(torch.empty(rnn_exc_inh, output_size).normal_(0., 0.4/math.sqrt(hidden_size)))
        self.bias_out = nn.Parameter(torch.zeros(output_size,))

        self.alpha = torch.tensor(alpha, device=self.device)
        self.sigma_rec = torch.tensor(math.sqrt(2./alpha) * hp['sigma_rec'], device=self.device)

        self._0 = torch.tensor(0., device=self.device)
        self._1 = torch.tensor(1., device=self.device)

        mask_input = tools.mask_input(input_size=self.input_size,n_rnn=self.n_rnn,n_md=self.n_md)
        self.mask_input = torch.tensor(mask_input, device=self.device,dtype=torch.float32)

    def input_effective_weights(self):
        '''
        effective_weights torch.Size([456, 456])
        weight_ih torch.Size([72, 456])
        '''
        # weight is non-negative

        if self.hp['input_mask']=='input_mask':

            effective_weights = self.weight_ih * self.mask_input
        else:
            effective_weights = self.weight_ih

        #pdb.set_trace()
        return effective_weights


    def forward_rnn(self, inputs, init_state):
        """
        Propogate input through the network.
        inputs: torch.Size([time, batch_size, dim])
        ***inputs, init_state torch.Size([102, 100, 72]) torch.Size([100, 456])


        """
        #print('(rnn_ei1) inputs, init_state',inputs.shape, init_state.shape)
        #pdb.set_trace()
        self.add_rec = torch.tensor(5, device=self.device)


        state = init_state
        state_collector = [state]

        #print('inputs',inputs.shape)
        i=0
        for input_per_step in inputs:

            # sys.exit(0)

            input_layer = torch.matmul(input_per_step, self.input_effective_weights())
            noise = torch.randn_like(state)*self.sigma_rec
            extra =  torch.ones_like(state)*0


            if 0<i<42 and self.hp['stim_epoch']=='cueing_epoch':


                if self.hp['stim_cell']=='MD1':
                    extra[:,256:356]=self.hp['stim_value']
                    # if self.hp['stim_task']=='RDM_task':
                    #extra[:,0:256]=self.hp['stim_value']
                if self.hp['stim_cell']=='MD2':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,356:456]=self.hp['stim_value']
                if self.hp['stim_cell']=='MD':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,256:456]=self.hp['stim_value']

                if self.hp['stim_cell']=='VIP':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,205:205+17]=self.hp['stim_value']

                if self.hp['stim_cell']=='SOM':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,205+17:205+17*2]=self.hp['stim_value']
                #print('extra',extra[0,:])
                if self.hp['stim_cell']=='PC':
                    extra[:,0:205]=self.hp['stim_value']
                    # if self.hp['stim_task']=='RDM_task':
                    #extra[:,0:256]=self.hp['stim_value']


            if 50<i<82 and self.hp['stim_epoch']=='delay_epoch':
                if self.hp['stim_cell']=='MD1':
                    #print('stim_cell MD1')
                    extra[:,256:356]=self.hp['stim_value']
                    # if self.hp['stim_task']=='RDM_task':
                    #     extra[:,0:256]=self.hp['stim_value']

                if self.hp['stim_cell']=='MD2':
                    #print('stim_cell MD2')
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,356:456]=self.hp['stim_value']

                if self.hp['stim_cell']=='MD':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,256:456]=self.hp['stim_value']
                if self.hp['stim_cell']=='VIP':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,205:205+17]=self.hp['stim_value']
                if self.hp['stim_cell']=='SOM':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,205+17:205+17*2]=self.hp['stim_value']
                if self.hp['stim_cell']=='PV':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,205+17*2:205+17*3]=self.hp['stim_value']

                if self.hp['stim_cell']=='PC':
                    #print(self.hp['stim_cell'],self.hp['stim_value'])
                    extra[:,0:205]=self.hp['stim_value']
                    # if self.hp['stim_task']=='RDM_task':
                    #extra[:,0:256]=self.hp['stim_value']


            current_layer = self.h2h(self.act_fcn(state))
            #print(extra)


            state_new = input_layer+current_layer+noise+ extra
            state = (self._1 - self.alpha) * state + self.alpha * state_new
            state_collector.append(state)
            i+=1


        return state_collector

    def forward_fixed_point(self, inputs, init_state):
        """
        Propogate input through the network.
        inputs: torch.Size([time, batch_size, dim])
        ***inputs, init_state torch.Size([102, 100, 72]) torch.Size([100, 456])


        """
        #print('(rnn_ei1) inputs, init_state',inputs.shape, init_state.shape)
        #pdb.set_trace()
        self.add_rec = torch.tensor(5, device=self.device)


        state = init_state
        state_collector = [state]

        #print('inputs',inputs.shape)
        i=0
        for input_per_step in inputs:

            #print(i,'input_per_step',input_per_step.shape,state.shape)

            input_layer = torch.matmul(input_per_step, self.input_effective_weights())
            noise = torch.randn_like(state)*self.sigma_rec
            extra =  torch.ones_like(state)*0


            if 0<i<42 and self.hp['stim_epoch']=='cueing_epoch':
                if self.hp['stim_cell']=='MD1':
                    extra[:,256:356]=self.hp['stim_value']
                    # if self.hp['stim_task']=='RDM_task':
                    #extra[:,0:256]=self.hp['stim_value']
                if self.hp['stim_cell']=='MD2':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,356:456]=self.hp['stim_value']
                if self.hp['stim_cell']=='MD':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,256:456]=self.hp['stim_value']

                if self.hp['stim_cell']=='VIP':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,205:205+17]=self.hp['stim_value']

                if self.hp['stim_cell']=='SOM':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,205+17:205+17*2]=self.hp['stim_value']
                #print('extra',extra[0,:])


            if 42<i<82 and self.hp['stim_epoch']=='delay_epoch':
                if self.hp['stim_cell']=='MD1':
                    extra[:,256:356]=self.hp['stim_value']
                    # if self.hp['stim_task']=='RDM_task':
                    #     extra[:,0:256]=self.hp['stim_value']

                if self.hp['stim_cell']=='MD2':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,356:456]=self.hp['stim_value']

                if self.hp['stim_cell']=='MD':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,256:456]=self.hp['stim_value']
                if self.hp['stim_cell']=='VIP':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,205:205+17]=self.hp['stim_value']
                if self.hp['stim_cell']=='SOM':
                    #extra[:,0:256]=self.hp['stim_value']
                    extra[:,205+17:205+17*2]=self.hp['stim_value']

            current_layer = self.h2h(self.act_fcn(state))


            state_new = input_layer+current_layer+noise+ extra
            state = (self._1 - self.alpha) * state + self.alpha * state_new
            state_collector.append(state)
            i+=1


        return state_collector

    def out_weight_clipper(self):
        self.weight_out.data.clamp_(0.)

    def self_weight_clipper(self):
        diag_element = self.h2h.weight.diag().data.clamp_(0., 1.)
        self.h2h.weight.data[range(self.hidden_size), range(self.hidden_size)] = diag_element


