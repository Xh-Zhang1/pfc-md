"""Utility functions."""

import os
import errno
import json
import pickle
import numpy as np
import torch

import default
import network

def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')

    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            hp = json.load(f)
    else:
        hp = default.get_default_hp(rule_name='both_RDM_HL_task')
        #print('hp.json:',fname)

    hp['seed'] = np.random.randint(0, 1000000)
    hp['rng'] = np.random.RandomState(hp['seed'])
    return hp


def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)


def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_log(log, log_name='log.json'):
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, log_name)
    with open(fname, 'w') as f:
        json.dump(log, f)


def load_log(model_dir, log_name='log.json'):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, log_name)
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log


def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data


def sequence_mask(lens):
    '''
    Input: lens: numpy array of integer

    Return sequence mask
    Example: if lens = [3, 5, 4]
    Then the return value will be
    tensor([[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]], dtype=torch.uint8)
    :param lens:
    :return:
    '''
    max_len = max(lens)
    # return torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
    return torch.t(torch.arange(max_len).expand(len(lens), max_len) < torch.tensor(np.expand_dims(lens, 1), dtype=torch.float32))


def elapsed_time(totaltime):
    hrs  = int(totaltime//3600)
    mins = int(totaltime%3600)//60
    secs = int(totaltime%60)
    return '{}h {}m {}s elapsed'.format(hrs, mins, secs)
def mask_no_md_train_type2(hp,md,pc,i_size_one):
    '''
    connection MD to PC
    '''
    np.random.seed(hp['seed'])
    n_md1=int(md/2)
    n_md2=int(md/2)

    n_VIP=i_size_one
    n_SOM=i_size_one
    n_PV=i_size_one
    n_PC=pc

    PC_PC  = np.tile([1]*n_PC, (n_PC, 1))
    PC_VIP = np.tile([1]*n_PC, (n_VIP,1))
    PC_SOM = np.tile([1]*n_PC, (n_SOM,1))
    PC_PV  = np.tile([1]*n_PC, (n_PV, 1))
    PC_md1 = np.tile([0]*n_PC, (n_md1,1))
    PC_md2 = np.tile([0]*n_PC, (n_md2,1))

    VIP_PC  = np.tile([0]*n_VIP, (n_PC, 1))
    VIP_VIP = np.tile([0]*n_VIP, (n_VIP,1))
    VIP_SOM = np.tile([-1]*n_VIP, (n_SOM,1))
    VIP_PV  = np.tile([0]*n_VIP, (n_PV, 1))
    VIP_md1 = np.tile([0]*n_VIP, (n_md1,1))
    VIP_md2 = np.tile([0]*n_VIP, (n_md2,1))


    SOM_PC  = np.tile([-1]*n_SOM, (n_PC, 1))
    SOM_VIP = np.tile([-1]*n_SOM, (n_VIP,1))
    SOM_SOM = np.tile([0]*n_SOM, (n_SOM,1))
    SOM_PV  = np.tile([-1]*n_SOM, (n_PV, 1))
    SOM_md1 = np.tile([0]*n_SOM, (n_md1,1))
    SOM_md2 = np.tile([0]*n_SOM, (n_md2,1))


    PV_PC  = np.tile([-1]*n_PV, (n_PC, 1))
    PV_VIP = np.tile([0]*n_PV,  (n_VIP,1))
    PV_SOM = np.tile([0]*n_PV,  (n_SOM,1))
    PV_PV  = np.tile([-1]*n_PV, (n_PV, 1))
    PV_md1 = np.tile([0]*n_PV,  (n_md1,1))
    PV_md2 = np.tile([0]*n_PV,  (n_md2,1))


    md1_PC  = np.tile([0]*n_md1, (n_PC, 1))
    md1_VIP = np.tile([0]*n_md1,  (n_VIP,1))
    md1_SOM = np.tile([0]*n_md1,  (n_SOM,1))
    md1_PV  = np.tile([0]*n_md1, (n_PV, 1))
    md1_md1 = np.tile([0]*n_md1,  (n_md1,1))
    md1_md2 = np.tile([0]*n_md1,  (n_md2,1))

    md2_PC  = np.tile([0]*n_md2, (n_PC, 1))
    md2_VIP = np.tile([0]*n_md2,  (n_VIP,1))
    md2_SOM = np.tile([0]*n_md2,  (n_SOM,1))
    md2_PV  = np.tile([0]*n_md2, (n_PV, 1))
    md2_md1 = np.tile([0]*n_md2,  (n_md1,1))
    md2_md2 = np.tile([0]*n_md2,  (n_md2,1))

    if hp['add_mask']=='add_PC_to_PC':
        PC_PC = np.tile([hp['scale_PC_PC']*1]*n_PC, (n_PC,1))


    if hp['remove_mask']=='rm_PC_VIP':
        PC_VIP_zero = int(n_PC*hp['PC_VIP_zero'])
        PC_VIP_one  = n_PC - int(n_PC*hp['PC_VIP_zero'])

        PC_VIP_1 = np.tile([0]*PC_VIP_zero, (n_VIP,1))
        PC_VIP_2 = np.tile([1]*PC_VIP_one,  (n_VIP,1))
        PC_VIP = np.concatenate((PC_VIP_1, PC_VIP_2), axis=1)


    elif hp['remove_mask']=='rm_PC_PV':
        PC_PV_zero = int(n_PC*hp['PC_PV_zero'])
        PC_PV_one  = n_PC - int(n_PC*hp['PC_PV_zero'])

        PC_PV_1 = np.tile([0]*PC_PV_zero, (n_PV,1))
        PC_PV_2 = np.tile([1]*PC_PV_one,  (n_PV,1))
        PC_PV = np.concatenate((PC_PV_1, PC_PV_2), axis=1)

    elif hp['remove_mask']=='rm_VIP_PC':
        VIP_PC_zero = int(n_VIP*hp['VIP_PC_zero'])
        VIP_PC_one  = n_VIP - int(n_VIP*hp['VIP_PC_zero'])

        VIP_PC_1 = np.tile([0]*VIP_PC_zero, (n_PC,1))
        VIP_PC_2 = np.tile([-1]*VIP_PC_one,  (n_PC,1))
        VIP_PC = np.concatenate((VIP_PC_1, VIP_PC_2), axis=1)

    elif hp['remove_mask']=='rm_PV_PC':
        PV_PC_zero = int(n_PV*hp['PV_PC_zero'])
        PV_PC_one  = n_PV - int(n_PV*hp['PV_PC_zero'])

        PV_PC_1 = np.tile([0]*PV_PC_zero, (n_PC,1))
        PV_PC_2 = np.tile([-1]*PV_PC_one,  (n_PC,1))
        PV_PC = np.concatenate((PV_PC_1, PV_PC_2), axis=1)

    elif hp['remove_mask']=='rm_SOM_PC':
        SOM_PC_zero = int(n_SOM*hp['SOM_PC_zero'])
        SOM_PC_one  = n_SOM - int(n_SOM*hp['SOM_PC_zero'])

        SOM_PC_1 = np.tile([0]*SOM_PC_zero, (n_PC,1))
        SOM_PC_2 = np.tile([-1]*SOM_PC_one,  (n_PC,1))
        SOM_PC = np.concatenate((SOM_PC_1, SOM_PC_2), axis=1)



    mask_col_PC  = np.concatenate((PC_PC, PC_VIP, PC_SOM, PC_PV, PC_md1, PC_md2),  axis=0)
    mask_col_VIP = np.concatenate((VIP_PC,VIP_VIP,VIP_SOM,VIP_PV,VIP_md1,VIP_md2), axis=0)
    mask_col_SOM = np.concatenate((SOM_PC,SOM_VIP,SOM_SOM,SOM_PV,SOM_md1,SOM_md2), axis=0)
    mask_col_PV  = np.concatenate((PV_PC, PV_VIP,PV_SOM,PV_PV,PV_md1,PV_md2),      axis=0)
    mask_col_md1 = np.concatenate((md1_PC,md1_VIP,md1_SOM,md1_PV,md1_md1,md1_md2), axis=0)
    mask_col_md2 = np.concatenate((md2_PC,md2_VIP,md2_SOM,md2_PV,md2_md1,md2_md2), axis=0)


    mask = np.concatenate((mask_col_PC, mask_col_VIP, mask_col_SOM, mask_col_PV,mask_col_md1,mask_col_md2), axis=1)
    #print('mask','\n',mask)


    return mask



def mask_md_pfc_train_type8(hp,md,pc,i_size_one):
    '''
    connection MD to PC
    '''
    p_md1=hp['p_md1']#0.6
    np.random.seed(hp['seed'])
    n_md1=int(md*p_md1)
    n_md2=int(md)-n_md1
    #print(n_md1,n_md2)

    n_VIP=i_size_one
    n_SOM=i_size_one
    n_PV=i_size_one
    #print('888888888888888888 n_PV',n_PV)
    n_PC=pc
    sp = hp['sparsity_pc_md']

    PC_PC  = np.tile([1]*n_PC, (n_PC, 1))
    PC_PV  = np.tile([1]*n_PC, (n_PV, 1))
    PC_SOM = np.tile([1]*n_PC, (n_SOM,1))
    PC_VIP = np.tile([1]*n_PC, (n_VIP,1))

    PC_md1 = np.tile([1]*n_PC, (n_md1,1))
    PC_md2 = np.tile([1]*n_PC, (n_md2,1))

    PV_PC  = np.tile([-1]*n_PV, (n_PC, 1))
    PV_PV  = np.tile([-1]*n_PV, (n_PV, 1))
    PV_SOM = np.tile([0]*n_PV,  (n_SOM,1))
    PV_VIP = np.tile([0]*n_PV,  (n_VIP,1))
    PV_md1 = np.tile([0]*n_PV,  (n_md1,1))
    PV_md2 = np.tile([0]*n_PV,  (n_md2,1))

    SOM_PC  = np.tile([-1]*n_SOM, (n_PC, 1))
    SOM_PV  = np.tile([-1]*n_SOM, (n_PV, 1))
    SOM_SOM = np.tile([0]*n_SOM, (n_SOM,1))
    SOM_VIP = np.tile([-1]*n_SOM, (n_VIP,1))
    SOM_md1 = np.tile([0]*n_SOM, (n_md1,1))
    SOM_md2 = np.tile([0]*n_SOM, (n_md2,1))

    VIP_PC  = np.tile([0]*n_VIP, (n_PC, 1))
    VIP_VIP = np.tile([0]*n_VIP, (n_VIP,1))
    VIP_SOM = np.tile([-1]*n_VIP, (n_SOM,1))
    VIP_PV  = np.tile([0]*n_VIP, (n_PV, 1))
    VIP_md1 = np.tile([0]*n_VIP, (n_md1,1))
    VIP_md2 = np.tile([0]*n_VIP, (n_md2,1))



    md1_PC  = np.tile([1]*n_md1, (n_PC, 1))
    md1_PV  = np.tile([1]*n_md1, (n_PV, 1))
    md1_VIP = np.tile([0]*n_md1,  (n_VIP,1))
    md1_SOM = np.tile([0]*n_md1,  (n_SOM,1))
    md1_md1 = np.tile([0]*n_md1,  (n_md1,1))
    md1_md2 = np.tile([0]*n_md1,  (n_md2,1))

    md2_PC  = np.tile([1]*n_md2, (n_PC, 1))
    md2_PV  = np.tile([0]*n_md2, (n_PV, 1))
    md2_VIP = np.tile([1]*n_md2,  (n_VIP,1))
    md2_SOM = np.tile([0]*n_md2,  (n_SOM,1))
    md2_md1 = np.tile([0]*n_md2,  (n_md1,1))
    md2_md2 = np.tile([0]*n_md2,  (n_md2,1))

    ##sparsity pc to md1
    PC_md2_one = int(n_PC*sp)
    PC_md2_zero  = n_PC - PC_md2_one

    PC_md2_1 = np.tile([0]*PC_md2_zero, (n_md2,1))
    PC_md2_2 = np.tile([1]*PC_md2_one,  (n_md2,1))
    PC_md2 = np.concatenate((PC_md2_1, PC_md2_2), axis=1)
    #print(sp,'PC_md2',PC_md2)



    if hp['add_mask']=='add_PC_to_md1':
        PC_md1 = np.tile([hp['scale_PC_md1']*1]*n_PC, (n_md1,1))

    if hp['add_mask']=='add_PC_to_md2':
        PC_md2 = np.tile([hp['scale_PC_md']*1]*n_PC, (n_md2,1))




    if hp['add_mask']=='add_PC_to_PC':
        PC_PC = np.tile([hp['scale_PC_PC']*1]*n_PC, (n_PC,1))

    if hp['add_mask']=='add_SOM_to_PC':
        SOM_PC = np.tile([hp['scale_SOM_PC']*(-1)]*n_SOM, (n_PC,1))
    if hp['add_mask']=='add_PV_to_PC':
        PV_PC = np.tile([hp['scale_PV_PC']*(-1)]*n_PV, (n_PC,1))


    if hp['add_mask']=='add_md1_to_PC':
        md1_PC = np.tile([hp['scale_md1_PC']*1]*n_md1, (n_PC,1))
        md2_PC = np.tile([hp['scale_md2_PC'] * 1] * n_md2, (n_PC, 1))

    if hp['add_mask']=='add_md1_to_PV':
        md1_PV = np.tile([hp['scale_md1_PV']*1]*n_md1, (n_PV,1))




    mask_col_PC  = np.concatenate((PC_PC, PC_VIP, PC_SOM, PC_PV, PC_md1, PC_md2),  axis=0)
    mask_col_VIP = np.concatenate((VIP_PC,VIP_VIP,VIP_SOM,VIP_PV,VIP_md1,VIP_md2), axis=0)
    mask_col_SOM = np.concatenate((SOM_PC,SOM_VIP,SOM_SOM,SOM_PV,SOM_md1,SOM_md2), axis=0)
    mask_col_PV  = np.concatenate((PV_PC, PV_VIP,PV_SOM,PV_PV,PV_md1,PV_md2),      axis=0)
    mask_col_md1 = np.concatenate((md1_PC,md1_VIP,md1_SOM,md1_PV,md1_md1,md1_md2), axis=0)
    mask_col_md2 = np.concatenate((md2_PC,md2_VIP,md2_SOM,md2_PV,md2_md1,md2_md2), axis=0)


    mask = np.concatenate((mask_col_PC, mask_col_VIP, mask_col_SOM, mask_col_PV,mask_col_md1,mask_col_md2), axis=1)
    #print('mask','\n',mask)
    return mask




def mask_input0(input_size,n_rnn,n_md):
    '''
    connection MD to PC
    '''

    n_pfc = n_rnn

    n_context=2

    n_cue_sensory = input_size-n_context
    CueSensory_pfc = np.tile([1]*n_cue_sensory, (n_pfc, 1))
    CueSensory_md = np.tile([1]*n_cue_sensory, (n_md, 1))

    context_pfc = np.tile([1]*n_context, (n_pfc, 1))
    context_md = np.tile([1]*n_context, (n_md, 1))

    mask_col_CueSensory  = np.concatenate((CueSensory_pfc, CueSensory_md),  axis=0)
    mask_col_context  = np.concatenate((context_pfc, context_md),  axis=0)

    mask = np.concatenate((mask_col_CueSensory,mask_col_context), axis=1)

    return mask.T

def mask_input1(input_size,n_rnn,n_md):
    '''
    connection MD to PC
    '''
    n_pfc = n_rnn

    n_context=0

    n_cue_sensory = input_size-n_context
    CueSensory_pfc = np.tile([1]*n_cue_sensory, (n_pfc, 1))
    CueSensory_md = np.tile([0]*n_cue_sensory, (n_md, 1))

    context_pfc = np.tile([0]*n_context, (n_pfc, 1))
    context_md = np.tile([1]*n_context, (n_md, 1))

    mask_col_CueSensory  = np.concatenate((CueSensory_pfc, CueSensory_md),  axis=0)
    mask_col_context  = np.concatenate((context_pfc, context_md),  axis=0)

    mask = np.concatenate((mask_col_CueSensory,mask_col_context), axis=1)


    return mask.T

def mask_input2(input_size,n_rnn,n_md):
    '''
    connection MD to PC
    '''
    n_pfc = n_rnn

    n_context=2

    n_cue_sensory = input_size-n_context
    CueSensory_pfc = np.tile([1]*n_cue_sensory, (n_pfc, 1))
    CueSensory_md = np.tile([0]*n_cue_sensory, (n_md, 1))

    context_pfc = np.tile([0]*n_context, (n_pfc, 1))
    context_md = np.tile([1]*n_context, (n_md, 1))

    mask_col_CueSensory  = np.concatenate((CueSensory_pfc, CueSensory_md),  axis=0)
    mask_col_context  = np.concatenate((context_pfc, context_md),  axis=0)

    mask = np.concatenate((mask_col_CueSensory,mask_col_context), axis=1)


    return mask.T




def weight_mask_A(hp, is_cuda):
    #print('=================== weight_mask_A')
    '''
        W = A*H + C
    '''
    if hp['mod_type'] == 'training':
        model_dir = hp['model_dir_current']
    else :
        model_dir = hp['model_dir_A_hh']
    #print('****** model_dir',model_dir)

    n_rnn = hp['n_rnn']
    n_md = hp['n_md']
    A_hh_weight = np.load(model_dir+'/model_A_hh.npy')
    #print('A_hh_weight',A_hh_weight[0:5,0])

    # generate H and C
    H = np.ones((n_rnn+n_md, n_rnn+n_md))
    H_1 = np.zeros((n_rnn, n_rnn))
    H[0:n_rnn,0:n_rnn] = H_1

    C = np.zeros((n_rnn+n_md, n_rnn+n_md))
    C[0:n_rnn,0:n_rnn] = A_hh_weight[0:n_rnn,0:n_rnn]

    return H,C




def generate_seq_training(c_cue,config,cue_duration,p_coh,batch_size):
    num_zero = int(cue_duration * config['sparsity_HL'])
    num_nonzero = cue_duration - num_zero

    HL_sequences = []

    for i in range(batch_size):
        HL_sequence_zero = 0.0 * np.ones(num_zero)
        HL_sequence_h = c_cue[i] * np.ones(int(p_coh * num_nonzero))
        # print('HL_sequence_h',cue_duration,HL_sequence_h)
        HL_sequence_l = -c_cue[i] * np.ones(num_nonzero - int(p_coh * num_nonzero))
        # print('HL_sequence_l',HL_sequence_l)
        HL_sequence = np.concatenate((HL_sequence_h, HL_sequence_l, HL_sequence_zero), axis=0)
        np.random.shuffle(HL_sequence)
        HL_sequences.append(HL_sequence)
    HL_sequences = np.array(HL_sequences)
    #print('HL_sequences',HL_sequences.shape,HL_sequences)


    return HL_sequences




def generate_seq_cue_plus_type1(c_cue, config, cue_duration, p_coh, batch_size):
    num_zero = int(cue_duration * config['sparsity_HL'])
    num_nonzero = cue_duration - num_zero

    HL_sequences = []
    num_front = config['num_front']

    HL_sequence_zero = 0.0 * np.ones(num_zero)
    HL_sequence_front = 1 * np.ones(num_front)


    print(c_cue[0])
    for i in range(batch_size):
        if c_cue[i] == 1:
            HL_sequence_h = 1 * np.ones(int(p_coh * num_nonzero)-num_front)
            HL_sequence_l = -1 * np.ones(num_nonzero - int(p_coh * num_nonzero))
        elif c_cue[i] == -1:
            HL_sequence_h = -1 * np.ones(int(p_coh * num_nonzero))
            HL_sequence_l = 1 * np.ones(num_nonzero - int(p_coh * num_nonzero) - num_front)
        # print('HL_sequence_l',HL_sequence_l)
        HL_sequence_back = np.concatenate((HL_sequence_h, HL_sequence_l, HL_sequence_zero), axis=0)
        np.random.shuffle(HL_sequence_back)

        HL_sequence = np.concatenate((HL_sequence_front,HL_sequence_back), axis=0)
        # np.random.shuffle(HL_sequence)
        HL_sequences.append(HL_sequence)
    HL_sequences = np.array(HL_sequences)
    print('*** HL_sequences',HL_sequences.shape,HL_sequences[0,:])


    return HL_sequences

def generate_seq_cue_plus_type2(c_cue, config, cue_duration, p_coh, batch_size):
    num_zero = int(cue_duration * config['sparsity_HL'])
    num_nonzero = cue_duration - num_zero
    p_coh = config['p_coh']

    num_front = config['num_front']

    HL_sequence_zero = 0.0 * np.ones(num_zero)

    HL_sequences = []
    #print(c_cue[0])
    for i in range(batch_size):
        if c_cue[i] == 1:
            seq_plus = int(p_coh * num_nonzero)
            seq_minus = num_nonzero - seq_plus
            seq_type1_1 =  1 * np.ones(int(num_front))
            seq_type1_2 = -1 * np.ones(int(num_front))
            seq_type1_3 =  1 * np.ones(int(num_front))
            seq_type1_4 = -1 * np.ones(seq_minus - int(num_front))
            seq_type1_5 =  1 * np.ones(seq_plus - 2*int(num_front))
            HL_sequence = np.concatenate((seq_type1_1, seq_type1_2, seq_type1_3, seq_type1_4, seq_type1_5), axis=0)
        if c_cue[i] == -1:
            seq_minus = int(p_coh * num_nonzero)
            seq_plus= num_nonzero - seq_minus

            seq_type1_1 = 1 * np.ones(int(num_front))
            seq_type1_2 = -1 * np.ones(int(num_front))
            if seq_plus-int(num_front) >0:
                seq_type1_3 = 1 * np.ones(seq_plus-int(num_front))
            else:
                seq_type1_3 = np.array([])

            seq_type1_4 = -1 * np.ones(seq_minus - int(num_front))
            HL_sequence = np.concatenate((seq_type1_1, seq_type1_2, seq_type1_3, seq_type1_4), axis=0)
        HL_sequences.append(HL_sequence)
    HL_sequences = np.array(HL_sequences)
    #np.random.shuffle(HL_sequences)

    print('HL_sequences',HL_sequences.shape,HL_sequences[0,:])

    return HL_sequences

def generate_seq_cue_plus_type3(c_cue, config, cue_duration, p_coh, batch_size):
    num_zero = int(cue_duration * config['sparsity_HL'])
    num_nonzero = cue_duration - num_zero
    p_coh = config['p_coh']


    #print(c_cue[0])
    HL_sequences = []
    for i in range(batch_size):
        if c_cue[0] == 1:
            seq_plus = int(p_coh * num_nonzero)
            seq_minus = num_nonzero - seq_plus
            seq_1 = []
            for j in range(seq_minus):
                seq_1.append([1, -1,1])
            seq_1 = np.array(seq_1).flatten()
            np.random.shuffle(seq_1)
            seq_2 =  1 * np.ones(seq_plus-2*int(seq_minus))
            #print(seq_1,seq_2)
            HL_sequence = np.concatenate((seq_1,seq_2), axis=0)

        if c_cue[0] == -1:

            seq_minus = int(p_coh * num_nonzero)
            seq_plus = num_nonzero - seq_minus
            seq_1 = []
            for j in range(seq_plus):
                seq_1.append([1, -1])
            seq_1 = np.array(seq_1).flatten()
            np.random.shuffle(seq_1)
            seq_2 = -1 * np.ones(seq_minus - int(seq_plus))
            HL_sequence = np.concatenate((seq_1, seq_2), axis=0)
        HL_sequences.append(HL_sequence)
    HL_sequences = np.array(HL_sequences)
    #np.random.shuffle(HL_sequences)

    print('HL_sequences',HL_sequences.shape,HL_sequences[0,:])

    return HL_sequences

def generate_seq_cue_minus_type1(c_cue, config, cue_duration, p_coh, batch_size):
    num_zero = int(cue_duration * config['sparsity_HL'])
    num_nonzero = cue_duration - num_zero

    HL_sequences = []
    num_front = config['num_front']

    HL_sequence_zero = 0.0 * np.ones(num_zero)
    HL_sequence_front = -1 * np.ones(num_front)

    for i in range(batch_size):
        if c_cue[i] == -1:
            HL_sequence_h = -1 * np.ones(int(p_coh * num_nonzero)-num_front)
            HL_sequence_l = 1 * np.ones(num_nonzero - int(p_coh * num_nonzero))
        elif c_cue[i] == 1:
            HL_sequence_h = 1 * np.ones(int(p_coh * num_nonzero))
            HL_sequence_l = -1 * np.ones(num_nonzero - int(p_coh * num_nonzero) - num_front)
        # print('HL_sequence_l',HL_sequence_l)
        HL_sequence_back = np.concatenate((HL_sequence_h, HL_sequence_l, HL_sequence_zero), axis=0)
        np.random.shuffle(HL_sequence_back)

        HL_sequence = np.concatenate((HL_sequence_front,HL_sequence_back), axis=0)
        # np.random.shuffle(HL_sequence)
        HL_sequences.append(HL_sequence)
    HL_sequences = np.array(HL_sequences)
    print('*** HL_sequences',HL_sequences.shape,HL_sequences[0,:])

    return HL_sequences

def generate_seq_cue_1_minus(c_cue, config, cue_duration, p_coh, batch_size):
    num_zero = int(cue_duration * config['sparsity_HL'])
    num_nonzero = cue_duration - num_zero
    p_coh =config['p_coh']

    HL_sequences = []
    num_front = config['num_front']

    HL_sequence_zero = 0.0 * np.ones(num_zero)
    HL_sequence_front = -1 * np.ones(num_front)


    print(c_cue[0])
    for i in range(batch_size):
        HL_sequence_h = c_cue[i] * np.ones(int(p_coh * num_nonzero)-num_front)
        HL_sequence_l = -c_cue[i] * np.ones(num_nonzero - int(p_coh * num_nonzero))
        if c_cue[0] == 1:
            HL_sequence_h = 1 * np.ones(int(p_coh * num_nonzero))
            HL_sequence_l = -1 * np.ones(num_nonzero - int(p_coh * num_nonzero) - num_front)

        if c_cue[0] == -1:
            HL_sequence_h = -1 * np.ones(int(p_coh * num_nonzero) - num_front)
            HL_sequence_l = 1 * np.ones(num_nonzero - int(p_coh * num_nonzero))
        # print('HL_sequence_l',HL_sequence_l)
        HL_sequence_back = np.concatenate((HL_sequence_h, HL_sequence_l, HL_sequence_zero), axis=0)
        np.random.shuffle(HL_sequence_back)

        HL_sequence = np.concatenate((HL_sequence_front,HL_sequence_back), axis=0)
        # np.random.shuffle(HL_sequence)
        HL_sequences.append(HL_sequence)
    HL_sequences = np.array(HL_sequences)

    return HL_sequences



















