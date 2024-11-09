"""Utility functions."""

import os
import errno
import json
import pickle
import numpy as np
import torch

import default


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


def mask_type1_train(hp,md,pc,i_size_one):
    np.random.seed(hp['seed'])

    n_md1=md
    n_md2=md

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
    md1_VIP = np.tile([1]*n_md1,  (n_VIP,1))
    md1_SOM = np.tile([0]*n_md1,  (n_SOM,1))
    md1_PV  = np.tile([0]*n_md1, (n_PV, 1))
    md1_md1 = np.tile([0]*n_md1,  (n_md1,1))
    md1_md2 = np.tile([0]*n_md1,  (n_md2,1))

    md2_PC  = np.tile([0]*n_md2, (n_PC, 1))
    md2_VIP = np.tile([0]*n_md2,  (n_VIP,1))
    md2_SOM = np.tile([0]*n_md2,  (n_SOM,1))
    md2_PV  = np.tile([1]*n_md2, (n_PV, 1))
    md2_md1 = np.tile([0]*n_md2,  (n_md1,1))
    md2_md2 = np.tile([0]*n_md2,  (n_md2,1))

    if hp['remove_mask']=='rm_md1_VIP':
        md1_VIP = np.tile([0]*n_md1,  (n_VIP,1))
    elif hp['remove_mask']=='rm_md2_PV':
        md2_PV  = np.tile([0]*n_md2, (n_PV, 1))
    elif hp['remove_mask']=='remove_PC_VIP':
        PC_VIP = np.tile([0]*n_PC, (n_VIP,1))
    elif hp['remove_mask']=='remove_PC_PV':
        PC_PV  = np.tile([0]*n_PC, (n_PV, 1))



    mask_col_PC  = np.concatenate((PC_PC, PC_VIP, PC_SOM, PC_PV, PC_md1, PC_md2),  axis=0)
    mask_col_VIP = np.concatenate((VIP_PC,VIP_VIP,VIP_SOM,VIP_PV,VIP_md1,VIP_md2), axis=0)
    mask_col_SOM = np.concatenate((SOM_PC,SOM_VIP,SOM_SOM,SOM_PV,SOM_md1,SOM_md2), axis=0)
    mask_col_PV  = np.concatenate((PV_PC, PV_VIP,PV_SOM,PV_PV,PV_md1,PV_md2),      axis=0)
    mask_col_md1 = np.concatenate((md1_PC,md1_VIP,md1_SOM,md1_PV,md1_md1,md1_md2), axis=0)
    mask_col_md2 = np.concatenate((md2_PC,md2_VIP,md2_SOM,md2_PV,md2_md1,md2_md2), axis=0)


    mask = np.concatenate((mask_col_PC, mask_col_VIP, mask_col_SOM, mask_col_PV,mask_col_md1,mask_col_md2), axis=1)
    #print('mask','\n',mask)


    return mask


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


def mask_md_pfc_train_type3(hp,md,pc,i_size_one):
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
    PC_md1 = np.tile([1]*n_PC, (n_md1,1))
    PC_md2 = np.tile([1]*n_PC, (n_md2,1))

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


    md1_PC  = np.tile([0.5]*n_md1, (n_PC, 1))
    md1_VIP = np.tile([1]*n_md1,  (n_VIP,1))
    md1_SOM = np.tile([0]*n_md1,  (n_SOM,1))
    md1_PV  = np.tile([0]*n_md1, (n_PV, 1))
    md1_md1 = np.tile([0]*n_md1,  (n_md1,1))
    md1_md2 = np.tile([0]*n_md1,  (n_md2,1))

    md2_PC  = np.tile([0.5]*n_md2, (n_PC, 1))
    md2_VIP = np.tile([0]*n_md2,  (n_VIP,1))
    md2_SOM = np.tile([0]*n_md2,  (n_SOM,1))
    md2_PV  = np.tile([1]*n_md2, (n_PV, 1))
    md2_md1 = np.tile([0]*n_md2,  (n_md1,1))
    md2_md2 = np.tile([0]*n_md2,  (n_md2,1))



    mask_col_PC  = np.concatenate((PC_PC, PC_VIP, PC_SOM, PC_PV, PC_md1, PC_md2),  axis=0)
    mask_col_VIP = np.concatenate((VIP_PC,VIP_VIP,VIP_SOM,VIP_PV,VIP_md1,VIP_md2), axis=0)
    mask_col_SOM = np.concatenate((SOM_PC,SOM_VIP,SOM_SOM,SOM_PV,SOM_md1,SOM_md2), axis=0)
    mask_col_PV  = np.concatenate((PV_PC, PV_VIP,PV_SOM,PV_PV,PV_md1,PV_md2),      axis=0)
    mask_col_md1 = np.concatenate((md1_PC,md1_VIP,md1_SOM,md1_PV,md1_md1,md1_md2), axis=0)
    mask_col_md2 = np.concatenate((md2_PC,md2_VIP,md2_SOM,md2_PV,md2_md1,md2_md2), axis=0)


    mask = np.concatenate((mask_col_PC, mask_col_VIP, mask_col_SOM, mask_col_PV,mask_col_md1,mask_col_md2), axis=1)
    #print('mask','\n',mask)
    return mask

def mask_pfc_only(hp,md,pc,i_size_one):
    '''
    connection MD to PC
    '''
    p_md1=hp['p_md1']#0.6
    np.random.seed(hp['seed'])
    n_md1=int(md*p_md1)

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

    PV_PC  = np.tile([-1]*n_PV, (n_PC, 1))
    PV_PV  = np.tile([-1]*n_PV, (n_PV, 1))
    PV_SOM = np.tile([0]*n_PV,  (n_SOM,1))
    PV_VIP = np.tile([0]*n_PV,  (n_VIP,1))

    SOM_PC  = np.tile([-1]*n_SOM, (n_PC, 1))
    SOM_PV  = np.tile([-1]*n_SOM, (n_PV, 1))
    SOM_SOM = np.tile([0]*n_SOM, (n_SOM,1))
    SOM_VIP = np.tile([-1]*n_SOM, (n_VIP,1))

    VIP_PC  = np.tile([0]*n_VIP, (n_PC, 1))
    VIP_PV = np.tile([0] * n_VIP, (n_PV, 1))
    VIP_SOM = np.tile([-1]*n_VIP, (n_SOM,1))
    VIP_VIP = np.tile([0] * n_VIP, (n_VIP, 1))

    mask_col_PC  = np.concatenate((PC_PC, PC_PV, PC_SOM, PC_VIP),  axis=0)
    mask_col_PV  = np.concatenate((PV_PC, PV_PV,PV_SOM,PV_VIP),      axis=0)
    mask_col_SOM = np.concatenate((SOM_PC,SOM_PV,SOM_SOM,SOM_VIP), axis=0)
    mask_col_VIP = np.concatenate((VIP_PC,VIP_PV,VIP_SOM,VIP_VIP), axis=0)

    mask = np.concatenate((mask_col_PC, mask_col_PV, mask_col_SOM, mask_col_VIP), axis=1)
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
    imbalance_strength = hp['imbalance_strength']

    PC_PC  = np.tile([1]*n_PC, (n_PC, 1))
    PC_PV  = np.tile([1]*n_PC, (n_PV, 1))
    PC_SOM = np.tile([1]*n_PC, (n_SOM,1))
    PC_VIP = np.tile([1]*n_PC, (n_VIP,1))
    PC_md1 = np.tile([1]*n_PC, (n_md1,1))
    PC_md2 = np.tile([1]*n_PC, (n_md2,1))

    PV_PC  = np.tile([-1*imbalance_strength]*n_PV, (n_PC, 1))
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
    VIP_PV = np.tile([0] * n_VIP, (n_PV, 1))
    VIP_SOM = np.tile([-1]*n_VIP, (n_SOM,1))
    VIP_VIP = np.tile([0] * n_VIP, (n_VIP, 1))
    VIP_md1 = np.tile([0]*n_VIP, (n_md1,1))
    VIP_md2 = np.tile([0]*n_VIP, (n_md2,1))



    md1_PC  = np.tile([1]*n_md1, (n_PC, 1))
    md1_PV  = np.tile([1]*n_md1, (n_PV, 1))
    md1_SOM = np.tile([0] * n_md1, (n_SOM, 1))
    md1_VIP = np.tile([0]*n_md1,  (n_VIP,1))
    md1_md1 = np.tile([0]*n_md1,  (n_md1,1))
    md1_md2 = np.tile([0]*n_md1,  (n_md2,1))

    md2_PC  = np.tile([1]*n_md2, (n_PC, 1))
    md2_PV  = np.tile([0]*n_md2, (n_PV, 1))
    md2_SOM = np.tile([0]*n_md2,  (n_SOM,1))
    md2_VIP = np.tile([1] * n_md2, (n_VIP, 1))
    md2_md1 = np.tile([0]*n_md2,  (n_md1,1))
    md2_md2 = np.tile([0]*n_md2,  (n_md2,1))

    ##sparsity pc to md1
    PC_md2_one = int(n_PC*sp)
    PC_md2_zero  = n_PC - PC_md2_one

    PC_md2_1 = np.tile([0]*PC_md2_zero, (n_md2,1))
    PC_md2_2 = np.tile([1]*PC_md2_one,  (n_md2,1))
    PC_md2 = np.concatenate((PC_md2_1, PC_md2_2), axis=1)
    #print(sp,'PC_md2',PC_md2)





    mask_col_PC  = np.concatenate((PC_PC, PC_PV, PC_SOM, PC_VIP, PC_md1, PC_md2),  axis=0)
    mask_col_PV  = np.concatenate((PV_PC, PV_PV,PV_SOM,PV_VIP,PV_md1,PV_md2),      axis=0)
    mask_col_SOM = np.concatenate((SOM_PC,SOM_PV,SOM_SOM,SOM_VIP,SOM_md1,SOM_md2), axis=0)

    mask_col_VIP = np.concatenate((VIP_PC,VIP_PV,VIP_SOM,VIP_VIP,VIP_md1,VIP_md2), axis=0)
    mask_col_md1 = np.concatenate((md1_PC,md1_PV,md1_SOM,md1_VIP,md1_md1,md1_md2), axis=0)
    mask_col_md2 = np.concatenate((md2_PC,md2_PV,md2_SOM,md2_VIP,md2_md1,md2_md2), axis=0)


    mask = np.concatenate((mask_col_PC, mask_col_PV, mask_col_SOM, mask_col_VIP,mask_col_md1,mask_col_md2), axis=1)
    #print('mask','\n',mask)
    return mask
def mask_md_pfc_train_type9(hp,md,pc,i_size_one):
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
    PC_md1 = np.tile([0.1]*n_PC, (n_md1,1))
    PC_md2 = np.tile([0.1]*n_PC, (n_md2,1))

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
    VIP_PV = np.tile([0] * n_VIP, (n_PV, 1))
    VIP_SOM = np.tile([-1]*n_VIP, (n_SOM,1))
    VIP_VIP = np.tile([0] * n_VIP, (n_VIP, 1))
    VIP_md1 = np.tile([0]*n_VIP, (n_md1,1))
    VIP_md2 = np.tile([0]*n_VIP, (n_md2,1))


    md1_PC  = np.tile([0.1]*n_md1, (n_PC, 1))
    md1_PV  = np.tile([1]*n_md1, (n_PV, 1))
    md1_SOM = np.tile([0] * n_md1, (n_SOM, 1))
    md1_VIP = np.tile([0]*n_md1,  (n_VIP,1))
    md1_md1 = np.tile([0]*n_md1,  (n_md1,1))
    md1_md2 = np.tile([0]*n_md1,  (n_md2,1))

    md2_PC  = np.tile([0.1]*n_md2, (n_PC, 1))
    md2_PV  = np.tile([0]*n_md2, (n_PV, 1))
    md2_SOM = np.tile([0]*n_md2,  (n_SOM,1))
    md2_VIP = np.tile([1] * n_md2, (n_VIP, 1))
    md2_md1 = np.tile([0]*n_md2,  (n_md1,1))
    md2_md2 = np.tile([0]*n_md2,  (n_md2,1))

    ##sparsity pc to md1
    PC_md2_one = int(n_PC*sp)
    PC_md2_zero  = n_PC - PC_md2_one

    PC_md2_1 = np.tile([0]*PC_md2_zero, (n_md2,1))
    PC_md2_2 = np.tile([1]*PC_md2_one,  (n_md2,1))
    PC_md2 = 0.1*np.concatenate((PC_md2_1, PC_md2_2), axis=1)
    #print(sp,'PC_md2',PC_md2)





    mask_col_PC  = np.concatenate((PC_PC, PC_PV, PC_SOM, PC_VIP, PC_md1, PC_md2),  axis=0)
    mask_col_PV  = np.concatenate((PV_PC, PV_PV,PV_SOM,PV_VIP,PV_md1,PV_md2),      axis=0)
    mask_col_SOM = np.concatenate((SOM_PC,SOM_PV,SOM_SOM,SOM_VIP,SOM_md1,SOM_md2), axis=0)

    mask_col_VIP = np.concatenate((VIP_PC,VIP_PV,VIP_SOM,VIP_VIP,VIP_md1,VIP_md2), axis=0)
    mask_col_md1 = np.concatenate((md1_PC,md1_PV,md1_SOM,md1_VIP,md1_md1,md1_md2), axis=0)
    mask_col_md2 = np.concatenate((md2_PC,md2_PV,md2_SOM,md2_VIP,md2_md1,md2_md2), axis=0)


    mask = np.concatenate((mask_col_PC, mask_col_PV, mask_col_SOM, mask_col_VIP,mask_col_md1,mask_col_md2), axis=1)
    #print('mask','\n',mask)
    return mask


def type3_test_new(hp,md,pc,i_size_one):
    '''
    change dm1 to MD2
    change dm2 to MD1

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
    PC_md1 = np.tile([1]*n_PC, (n_md1,1))
    PC_md2 = np.tile([1]*n_PC, (n_md2,1))

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

    md1_PC  = np.tile([0.5]*n_md1, (n_PC, 1))
    md1_VIP = np.tile([1]*n_md1,  (n_VIP,1))
    md1_SOM = np.tile([0]*n_md1,  (n_SOM,1))
    md1_PV  = np.tile([0]*n_md1, (n_PV, 1))
    md1_md1 = np.tile([0]*n_md1,  (n_md1,1))
    md1_md2 = np.tile([0]*n_md1,  (n_md2,1))

    md2_PC  = np.tile([0.5]*n_md2, (n_PC, 1))
    md2_VIP = np.tile([0]*n_md2,  (n_VIP,1))
    md2_SOM = np.tile([0]*n_md2,  (n_SOM,1))
    md2_PV  = np.tile([1]*n_md2, (n_PV, 1))
    md2_md1 = np.tile([0]*n_md2,  (n_md1,1))
    md2_md2 = np.tile([0]*n_md2,  (n_md2,1))

    if hp['remove_mask']=='rm_all_md':
        #print('rm_all_md')
        PC_md1 = np.tile([0]*n_PC, (n_md1,1))
        PC_md2 = np.tile([0]*n_PC, (n_md2,1))
        md1_PC  = np.tile([0]*n_md1, (n_PC, 1))
        md2_PC  = np.tile([0]*n_md2, (n_PC, 1))
        md1_VIP = np.tile([0]*n_md1,  (n_VIP,1))
        md2_PV  = np.tile([0]*n_md2, (n_PV, 1))


    if hp['add_mask']=='add_PC_to_md1':
        PC_md1 = np.tile([hp['scale_PC_md']*1]*n_PC, (n_md1,1))


    if hp['add_mask']=='add_PC_to_md2':
        PC_md2 = np.tile([hp['scale_PC_md']*1]*n_PC, (n_md2,1))

    if hp['add_mask']=='add_md_to_PC':
        md1_PC = np.tile([hp['scale_md_PC']*0.5]*n_md1, (n_PC,1))
        md2_PC = np.tile([hp['scale_md_PC']*0.5]*n_md2, (n_PC,1))

    if hp['add_mask']=='add_md1_to_PC':
        md2_PC = np.tile([hp['scale_md_PC']*0.5]*n_md2, (n_PC,1))

    if hp['add_mask']=='add_PC_to_PC':
        PC_PC = np.tile([hp['scale_PC_PC']*1]*n_PC, (n_PC,1))

    if hp['add_mask']=='add_SOM_to_PC':
        SOM_PC = np.tile([hp['scale_SOM_PC']*(-1)]*n_SOM, (n_PC,1))
    if hp['add_mask']=='add_PV_to_PC':
        PV_PC = np.tile([hp['scale_PV_PC']*(-1)]*n_PV, (n_PC,1))




    if hp['add_mask']=='add_md1_to_VIP':
        md1_VIP = np.tile([hp['scale_md1_VIP']*1]*n_md1, (n_VIP,1))
    if hp['add_mask']=='add_md2_to_PV':
        md2_PV = np.tile([hp['scale_md2_PV']*1]*n_md2, (n_PV,1))





    elif hp['remove_mask']=='rm_md1_VIP':
        n_md1_zero = int(n_md2*hp['percent_zero'])
        #print('n_md2_zero',n_md2_zero)
        md1_VIP_1 = np.tile([0]*n_md1_zero, (n_VIP,1))
        md1_VIP_2 = np.tile([1]*(n_md1-n_md1_zero),  (n_VIP,1))
        md1_VIP = np.concatenate((md1_VIP_1,md1_VIP_2), axis=1)

    elif hp['remove_mask']=='rm_md2_PV':
        md2_PV_zero = int(n_md2*hp['percent_zero'])
        md2_PV_1 = np.tile([0]*md2_PV_zero, (n_PV,1))
        md2_PV_2 = np.tile([1]*(n_md2-md2_PV_zero),  (n_PV,1))
        md2_PV = np.concatenate((md2_PV_1,md2_PV_2), axis=1)

    elif hp['remove_mask']=='rm_md_PC':
        n_md1_zero = int(n_md1*hp['percent_zero'])
        #print('n_md2_zero',n_md2_zero)
        md1_PC_1 = np.tile([0]*n_md1_zero, (n_PC,1))
        md1_PC_2 = np.tile([0.5]*(n_md1-n_md1_zero),  (n_PC,1))
        md1_PC = np.concatenate((md1_PC_1,md1_PC_2), axis=1)

        n_md2_zero = int(n_md2*hp['percent_zero'])
        #print('n_md2_zero',n_md2_zero)
        md2_PC_1 = np.tile([0]*n_md2_zero, (n_PC,1))
        md2_PC_2 = np.tile([0.5]*(n_md2-n_md2_zero),  (n_PC,1))
        md2_PC = np.concatenate((md2_PC_1,md2_PC_2), axis=1)

    elif hp['remove_mask']=='rm_PC_md1':
        PC_md1_zero = int(n_PC*hp['percent_zero'])
        PC_md1_one  = n_PC - int(n_PC*hp['percent_zero'])

        PC_md1_1 = np.tile([0]*PC_md1_zero, (n_md1,1))
        PC_md1_2 = np.tile([1]*PC_md1_one,  (n_md1,1))
        PC_md1 = np.concatenate((PC_md1_1, PC_md1_2), axis=1)
    elif hp['remove_mask']=='rm_PC_md2':
        PC_md2_zero = int(n_PC*hp['percent_zero'])
        PC_md2_one  = n_PC - int(n_PC*hp['percent_zero'])

        PC_md2_1 = np.tile([0]*PC_md2_zero, (n_md2,1))
        PC_md2_2 = np.tile([1]*PC_md2_one,  (n_md2,1))
        PC_md2 = np.concatenate((PC_md2_1, PC_md2_2), axis=1)

    elif hp['remove_mask']=='rm_PC_md':
        PC_md1_zero = int(n_PC*hp['percent_zero'])
        PC_md1_one  = n_PC - int(n_PC*hp['percent_zero'])

        PC_md1_1 = np.tile([0]*PC_md1_zero, (n_md1,1))
        PC_md1_2 = np.tile([1]*PC_md1_one,  (n_md1,1))
        PC_md1 = np.concatenate((PC_md1_1, PC_md1_2), axis=1)


        PC_md2_zero = int(n_PC*hp['percent_zero'])
        PC_md2_one  = n_PC - int(n_PC*hp['percent_zero'])

        PC_md2_1 = np.tile([0]*PC_md2_zero, (n_md2,1))
        PC_md2_2 = np.tile([1]*PC_md2_one,  (n_md2,1))
        PC_md2 = np.concatenate((PC_md2_1, PC_md2_2), axis=1)





    elif hp['remove_mask']=='rm_md1_PC':
        n_md1_zero = int(n_md1*hp['percent_zero'])
        #print('n_md2_zero',n_md2_zero)
        md1_PC_1 = np.tile([0]*n_md1_zero, (n_PC,1))
        md1_PC_2 = np.tile([0.5]*(n_md1-n_md1_zero),  (n_PC,1))
        md1_PC = np.concatenate((md1_PC_1,md1_PC_2), axis=1)

    elif hp['remove_mask']=='rm_md2_PC':
        n_md2_zero = int(n_md2*hp['percent_zero'])
        #print('n_md2_zero',n_md2_zero)
        md2_PC_1 = np.tile([0]*n_md2_zero, (n_PC,1))
        md2_PC_2 = np.tile([0.5]*(n_md2-n_md2_zero),  (n_PC,1))
        md2_PC = np.concatenate((md2_PC_1,md2_PC_2), axis=1)


    ####################### scale inhibitory neuron #######################
    if hp['remove_mask']=='scale_SOM_to_VIP':
        SOM_VIP = np.tile([hp['scale_SOM_VIP']*(-1)]*n_SOM, (n_VIP,1))


    if hp['remove_mask']=='scale_VIP_to_SOM':
        VIP_SOM = np.tile([hp['scale_VIP_SOM']*(-1)]*n_VIP, (n_SOM,1))

    if hp['remove_mask']=='scale_both_VIP_and_SOM':
        VIP_SOM = np.tile([hp['scale_VIP_SOM']*(-1)]*n_VIP, (n_SOM,1))
        SOM_VIP = np.tile([hp['scale_SOM_VIP']*(-1)]*n_SOM, (n_VIP,1))

    if hp['remove_mask']=='scale_SOM_to_PV':
        SOM_PV  = np.tile([hp['scale_SOM_PV']*(-1)]*n_SOM, (n_PV,1))




    if hp['remove_mask']=='scale_SOM_to_PC':
        SOM_PC = np.tile([hp['scale_SOM_PC']*(-1)]*n_SOM, (n_PC,1))

    if hp['remove_mask']=='scale_PV_to_PC':
        PV_PC = np.tile([hp['scale_PV_PC']*(-1)]*n_PV, (n_PC,1))






    ####################### remove inhibitory neuron #######################

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

    elif hp['remove_mask']=='rm_SOM_VIP':
        SOM_VIP_zero = int(n_SOM*hp['SOM_VIP_zero'])
        SOM_VIP_one  = n_SOM - int(n_SOM*hp['SOM_VIP_zero'])

        SOM_VIP_1 = np.tile([0]*SOM_VIP_zero, (n_VIP,1))
        SOM_VIP_2 = np.tile([-1]*SOM_VIP_one,  (n_VIP,1))
        SOM_VIP = np.concatenate((SOM_VIP_1, SOM_VIP_2), axis=1)



    elif hp['remove_mask']=='rm_SOM_PV':
        SOM_PV_zero = int(n_SOM*hp['SOM_PV_zero'])
        SOM_PV_one  = n_SOM - int(n_SOM*hp['SOM_PV_zero'])

        SOM_PV_1 = np.tile([0]*SOM_PV_zero, (n_PV,1))
        SOM_PV_2 = np.tile([-1]*SOM_PV_one,  (n_PV,1))
        SOM_PV = np.concatenate((SOM_PV_1, SOM_PV_2), axis=1)










    elif hp['remove_mask']=='rm_PC_VIP':
        PC_VIP_zero = int(n_PC*hp['PC_VIP_zero'])
        PC_VIP_one  = n_PC - int(n_PC*hp['PC_VIP_zero'])

        PC_VIP_1 = np.tile([0]*PC_VIP_zero, (n_VIP,1))
        PC_VIP_2 = np.tile([1]*PC_VIP_one,  (n_VIP,1))
        PC_VIP = np.concatenate((PC_VIP_1, PC_VIP_2), axis=1)

    elif hp['remove_mask']=='rm_PC_SOM':
        PC_SOM_zero = int(n_PC*hp['PC_SOM_zero'])
        PC_SOM_one  = n_PC - int(n_PC*hp['PC_SOM_zero'])

        PC_SOM_1 = np.tile([0]*PC_SOM_zero, (n_SOM,1))
        PC_SOM_2 = np.tile([1]*PC_SOM_one,  (n_SOM,1))
        PC_SOM = np.concatenate((PC_SOM_1, PC_SOM_2), axis=1)


    elif hp['remove_mask']=='rm_PC_PV':
        PC_PV_zero = int(n_PC*hp['PC_PV_zero'])
        PC_PV_one  = n_PC - int(n_PC*hp['PC_PV_zero'])

        PC_PV_1 = np.tile([0]*PC_PV_zero, (n_PV,1))
        PC_PV_2 = np.tile([1]*PC_PV_one,  (n_PV,1))
        PC_PV = np.concatenate((PC_PV_1, PC_PV_2), axis=1)





    mask_col_PC  = np.concatenate((PC_PC, PC_VIP, PC_SOM, PC_PV, PC_md1, PC_md2),  axis=0)
    mask_col_VIP = np.concatenate((VIP_PC,VIP_VIP,VIP_SOM,VIP_PV,VIP_md1,VIP_md2), axis=0)
    mask_col_SOM = np.concatenate((SOM_PC,SOM_VIP,SOM_SOM,SOM_PV,SOM_md1,SOM_md2), axis=0)
    mask_col_PV  = np.concatenate((PV_PC, PV_VIP,PV_SOM,PV_PV,PV_md1,PV_md2),      axis=0)
    mask_col_md1 = np.concatenate((md1_PC,md1_VIP,md1_SOM,md1_PV,md1_md1,md1_md2), axis=0)
    mask_col_md2 = np.concatenate((md2_PC,md2_VIP,md2_SOM,md2_PV,md2_md1,md2_md2), axis=0)


    mask = np.concatenate((mask_col_PC, mask_col_VIP, mask_col_SOM, mask_col_PV,mask_col_md1,mask_col_md2), axis=1)

    #print('mask','\n',mask)
    return mask

def type3_test(hp,md,pc,i_size_one):
    '''
    change dm1 to md2
    change dm2 to MD1

    '''
    np.random.seed(hp['seed'])
    n_MD2=int(md/2)
    n_MD1=int(md/2)

    n_VIP=i_size_one
    n_SOM=i_size_one
    n_PV=i_size_one
    n_PC=pc

    PC_PC  = np.tile([1]*n_PC, (n_PC, 1))
    PC_VIP = np.tile([1]*n_PC, (n_VIP,1))
    PC_SOM = np.tile([1]*n_PC, (n_SOM,1))
    PC_PV  = np.tile([1]*n_PC, (n_PV, 1))
    PC_MD2 = np.tile([1]*n_PC, (n_MD2,1))
    PC_MD1 = np.tile([1]*n_PC, (n_MD1,1))

    VIP_PC  = np.tile([0]*n_VIP, (n_PC, 1))
    VIP_VIP = np.tile([0]*n_VIP, (n_VIP,1))
    VIP_SOM = np.tile([-1]*n_VIP, (n_SOM,1))
    VIP_PV  = np.tile([0]*n_VIP, (n_PV, 1))
    VIP_MD2 = np.tile([0]*n_VIP, (n_MD2,1))
    VIP_MD1 = np.tile([0]*n_VIP, (n_MD1,1))


    SOM_PC  = np.tile([-1]*n_SOM, (n_PC, 1))
    SOM_VIP = np.tile([-1]*n_SOM, (n_VIP,1))
    SOM_SOM = np.tile([0]*n_SOM, (n_SOM,1))
    SOM_PV  = np.tile([-1]*n_SOM, (n_PV, 1))
    SOM_MD2 = np.tile([0]*n_SOM, (n_MD2,1))
    SOM_MD1 = np.tile([0]*n_SOM, (n_MD1,1))


    PV_PC  = np.tile([-1]*n_PV, (n_PC, 1))
    PV_VIP = np.tile([0]*n_PV,  (n_VIP,1))
    PV_SOM = np.tile([0]*n_PV,  (n_SOM,1))
    PV_PV  = np.tile([-1]*n_PV, (n_PV, 1))
    PV_MD2 = np.tile([0]*n_PV,  (n_MD2,1))
    PV_MD1 = np.tile([0]*n_PV,  (n_MD1,1))


    MD2_PC  = np.tile([0.5]*n_MD2, (n_PC, 1))
    MD2_VIP = np.tile([1]*n_MD2,  (n_VIP,1))
    MD2_SOM = np.tile([0]*n_MD2,  (n_SOM,1))
    MD2_PV  = np.tile([0]*n_MD2, (n_PV, 1))
    MD2_MD2 = np.tile([0]*n_MD2,  (n_MD2,1))
    MD2_MD1 = np.tile([0]*n_MD2,  (n_MD1,1))

    MD1_PC  = np.tile([0.5]*n_MD1, (n_PC, 1))
    MD1_VIP = np.tile([0]*n_MD1,  (n_VIP,1))
    MD1_SOM = np.tile([0]*n_MD1,  (n_SOM,1))
    MD1_PV  = np.tile([1]*n_MD1, (n_PV, 1))
    MD1_MD2 = np.tile([0]*n_MD1,  (n_MD2,1))
    MD1_MD1 = np.tile([0]*n_MD1,  (n_MD1,1))


    mask_col_PC  = np.concatenate((PC_PC, PC_VIP, PC_SOM, PC_PV, PC_MD2, PC_MD1),  axis=0)
    mask_col_VIP = np.concatenate((VIP_PC,VIP_VIP,VIP_SOM,VIP_PV,VIP_MD2,VIP_MD1), axis=0)
    mask_col_SOM = np.concatenate((SOM_PC,SOM_VIP,SOM_SOM,SOM_PV,SOM_MD2,SOM_MD1), axis=0)
    mask_col_PV  = np.concatenate((PV_PC, PV_VIP,PV_SOM,PV_PV,PV_MD2,PV_MD1),      axis=0)
    mask_col_MD2 = np.concatenate((MD2_PC,MD2_VIP,MD2_SOM,MD2_PV,MD2_MD2,MD2_MD1), axis=0)
    mask_col_MD1 = np.concatenate((MD1_PC,MD1_VIP,MD1_SOM,MD1_PV,MD1_MD2,MD1_MD1), axis=0)


    mask = np.concatenate((mask_col_PC, mask_col_VIP, mask_col_SOM, mask_col_PV,mask_col_MD2,mask_col_MD1), axis=1)
    #print('mask','\n',mask)
    return mask


def mask_input(input_size,n_rnn,n_md):
    '''
    connection MD to PC
    '''

    n_pfc = n_rnn

    n_context=2

    n_cue_sensory = input_size-n_context
    CueSensory_pfc = np.tile([1]*n_cue_sensory, (n_pfc, 1))
    CueSensory_md = np.tile([1]*n_cue_sensory, (n_md, 1))

    context_pfc = np.tile([0]*n_context, (n_pfc, 1))
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

    n_context=2

    n_cue_sensory = input_size-n_context
    CueSensory_pfc = np.tile([1]*n_cue_sensory, (n_pfc, 1))
    CueSensory_md = np.tile([1]*n_cue_sensory, (n_md, 1))

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


def mask_input3(input_size,n_rnn,n_md):
    '''
    connection MD to PC
    '''
    n_pfc = n_rnn

    n_context=2

    n_cue_sensory = input_size-n_context
    CueSensory_pfc = np.tile([1]*n_cue_sensory, (n_pfc, 1))
    context_pfc = np.tile([1] * n_context, (n_pfc, 1))

    CueSensory_md = np.tile([0]*n_cue_sensory, (n_md, 1))
    context_md = np.tile([0]*n_context, (n_md, 1))

    mask_col_CueSensory  = np.concatenate((CueSensory_pfc, CueSensory_md),  axis=0)
    mask_col_context  = np.concatenate((context_pfc, context_md),  axis=0)

    mask = np.concatenate((mask_col_CueSensory,mask_col_context), axis=1)

    return mask.T