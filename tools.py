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




def mask_pfc_par_local(hp, n_rnn):
    '''
    connection MD to PC
    '''


    n_pfc = int(n_rnn/2)
    n_parietal = n_rnn-n_pfc

    n_pfcE = int(n_pfc * hp['e_prop'])  # hp['n_rnn']-self.i_size
    n_pfcI = n_pfc - n_pfcE

    n_parietalE = int(n_parietal * hp['e_prop'])  # hp['n_rnn']-self.i_size
    n_parietalI = n_parietal - n_parietalE

    sparsity = hp['sparsity']


    pfcE_pfcE       = np.tile([1]*n_pfcE, (n_pfcE, 1))
    pfcE_pfcI       = np.tile([1] * n_pfcE, (n_pfcI, 1))
    pfcE_parietalE  = np.tile([1]*int(n_pfcE*sparsity)+[0]*(n_pfcE-int(sparsity*n_pfcE)), (n_parietalE,1))
    pfcE_parietalI  = np.tile([0] * n_pfcE, (n_parietalI, 1))

    pfcI_pfcE      = np.tile([-1] * n_pfcI, (n_pfcE, 1))
    pfcI_pfcI      = np.tile([-1] * n_pfcI, (n_pfcI, 1))
    pfcI_parietalE = np.tile([0] * n_pfcI, (n_parietalE, 1))
    pfcI_parietalI = np.tile([0] * n_pfcI, (n_parietalI, 1))


    parietalE_pfcE      = np.tile([1]*int(n_parietalE*sparsity)+[0]*(n_parietalE-int(sparsity*n_parietalE)), (n_pfcE, 1))
    parietalE_pfcI      = np.tile([0] * n_parietalE, (n_pfcI, 1))
    parietalE_parietalE = np.tile([1] * n_parietalE, (n_parietalE, 1))
    parietalE_parietalI = np.tile([1] * n_parietalE, (n_parietalI, 1))


    parietalI_pfcE      = np.tile([0] * n_parietalI, (n_pfcE, 1))
    parietalI_pfcI      = np.tile([0] * n_parietalI, (n_pfcI, 1))
    parietalI_parietalE = np.tile([-1]  * n_parietalI, (n_parietalE, 1))
    parietalI_parietalI = np.tile([-1]  * n_parietalI, (n_parietalI, 1))


    mask_col_pfcE       = np.concatenate((pfcE_pfcE, pfcE_pfcI, pfcE_parietalE, pfcE_parietalI), axis=0)
    mask_col_pfcI       = np.concatenate((pfcI_pfcE, pfcI_pfcI, pfcI_parietalE, pfcI_parietalI), axis=0)
    mask_col_parietalE  = np.concatenate((parietalE_pfcE, parietalE_pfcI, parietalE_parietalE, parietalE_parietalI), axis=0)
    mask_col_parietalI  = np.concatenate((parietalI_pfcE, parietalI_pfcI, parietalI_parietalE, parietalI_parietalI), axis=0)

    mask = np.concatenate((mask_col_pfcE, mask_col_pfcI, mask_col_parietalE,  mask_col_parietalI), axis=1)

    return mask


def mask_pfc_par_1cross(hp, n_rnn):
    '''
    connection MD to PC
    '''


    n_pfc = int(n_rnn/2)
    n_parietal = n_rnn-n_pfc

    n_pfcE = int(n_pfc * hp['e_prop'])  # hp['n_rnn']-self.i_size
    n_pfcI = n_pfc - n_pfcE

    n_parietalE = int(n_parietal * hp['e_prop'])  # hp['n_rnn']-self.i_size
    n_parietalI = n_parietal - n_parietalE

    sparsity = hp['sparsity']


    pfcE_pfcE       = np.tile([1]*n_pfcE, (n_pfcE, 1))
    pfcE_pfcI       = np.tile([1] * n_pfcE, (n_pfcI, 1))
    pfcE_parietalE  = np.tile([1]*int(n_pfcE*sparsity)+[0]*(n_pfcE-int(sparsity*n_pfcE)), (n_parietalE,1))
    pfcE_parietalI  = np.tile([1] * int(n_pfcE*sparsity)+[0]*(n_pfcE-int(sparsity*n_pfcE)), (n_parietalI, 1))

    pfcI_pfcE      = np.tile([-1] * n_pfcI, (n_pfcE, 1))
    pfcI_pfcI      = np.tile([-1] * n_pfcI, (n_pfcI, 1))
    pfcI_parietalE = np.tile([0] * n_pfcI, (n_parietalE, 1))
    pfcI_parietalI = np.tile([0] * n_pfcI, (n_parietalI, 1))


    parietalE_pfcE      = np.tile([1]*int(n_parietalE*sparsity)+[0]*(n_parietalE-int(sparsity*n_parietalE)), (n_pfcE, 1))
    parietalE_pfcI      = np.tile([1] * int(n_parietalE*sparsity)+[0]*(n_parietalE-int(sparsity*n_parietalE)), (n_pfcI, 1))
    parietalE_parietalE = np.tile([1] * n_parietalE, (n_parietalE, 1))
    parietalE_parietalI = np.tile([1] * n_parietalE, (n_parietalI, 1))


    parietalI_pfcE      = np.tile([0] * n_parietalI, (n_pfcE, 1))
    parietalI_pfcI      = np.tile([0] * n_parietalI, (n_pfcI, 1))
    parietalI_parietalE = np.tile([-1]  * n_parietalI, (n_parietalE, 1))
    parietalI_parietalI = np.tile([-1]  * n_parietalI, (n_parietalI, 1))


    mask_col_pfcE  = np.concatenate((pfcE_pfcE, pfcE_pfcI, pfcE_parietalE, pfcE_parietalI), axis=0)
    mask_col_pfcI  = np.concatenate((pfcI_pfcE, pfcI_pfcI, pfcI_parietalE, pfcI_parietalI), axis=0)
    mask_col_parietalE  = np.concatenate((parietalE_pfcE, parietalE_pfcI, parietalE_parietalE, parietalE_parietalI), axis=0)
    mask_col_parietalI  = np.concatenate((parietalI_pfcE, parietalI_pfcI, parietalI_parietalE, parietalI_parietalI), axis=0)

    mask = np.concatenate((mask_col_pfcE, mask_col_pfcI, mask_col_parietalE,  mask_col_parietalI), axis=1)

    return mask

def mask_pfc_par_2cross(hp, n_rnn):
    '''
    connection MD to PC
    '''


    n_pfc = int(n_rnn/2)
    n_parietal = n_rnn-n_pfc

    n_pfcE = int(n_pfc * hp['e_prop'])  # hp['n_rnn']-self.i_size
    n_pfcI = n_pfc - n_pfcE

    n_parietalE = int(n_parietal * hp['e_prop'])  # hp['n_rnn']-self.i_size
    n_parietalI = n_parietal - n_parietalE

    sparsity = hp['sparsity']


    pfcE_pfcE       = np.tile([1] * n_pfcE, (n_pfcE, 1))
    pfcE_pfcI       = np.tile([1] * n_pfcE, (n_pfcI, 1))
    pfcE_parietalE  = np.tile([1] * int(n_pfcE*sparsity)+[0]*(n_pfcE-int(sparsity*n_pfcE)), (n_parietalE,1))
    pfcE_parietalI  = np.tile([1] * int(n_pfcE*sparsity)+[0]*(n_pfcE-int(sparsity*n_pfcE)), (n_parietalI, 1))

    pfcI_pfcE      = np.tile([-1] * n_pfcI, (n_pfcE, 1))
    pfcI_pfcI      = np.tile([-1] * n_pfcI, (n_pfcI, 1))
    pfcI_parietalE = np.tile([-1] * int(n_pfcI*sparsity)+[0]*(n_pfcI-int(sparsity*n_pfcI)), (n_parietalE, 1))
    pfcI_parietalI = np.tile([-1] * int(n_pfcI*sparsity)+[0]*(n_pfcI-int(sparsity*n_pfcI)), (n_parietalI, 1))


    parietalE_pfcE      = np.tile([1] * int(n_parietalE*sparsity)+[0]*(n_parietalE-int(sparsity*n_parietalE)), (n_pfcE, 1))
    parietalE_pfcI      = np.tile([1] * int(n_parietalE*sparsity)+[0]*(n_parietalE-int(sparsity*n_parietalE)), (n_pfcI, 1))
    parietalE_parietalE = np.tile([1] * n_parietalE, (n_parietalE, 1))
    parietalE_parietalI = np.tile([1] * n_parietalE, (n_parietalI, 1))


    parietalI_pfcE      = np.tile([-1] * int(n_parietalI*sparsity)+[0]*(n_parietalI-int(sparsity*n_parietalI)), (n_pfcE, 1))
    parietalI_pfcI      = np.tile([-1] * int(n_parietalI*sparsity)+[0]*(n_parietalI-int(sparsity*n_parietalI)), (n_pfcI, 1))
    parietalI_parietalE = np.tile([-1]  * n_parietalI, (n_parietalE, 1))
    parietalI_parietalI = np.tile([-1]  * n_parietalI, (n_parietalI, 1))


    mask_col_pfcE       = np.concatenate((pfcE_pfcE, pfcE_pfcI, pfcE_parietalE, pfcE_parietalI), axis=0)
    mask_col_pfcI       = np.concatenate((pfcI_pfcE, pfcI_pfcI, pfcI_parietalE, pfcI_parietalI), axis=0)
    mask_col_parietalE  = np.concatenate((parietalE_pfcE, parietalE_pfcI, parietalE_parietalE, parietalE_parietalI), axis=0)
    mask_col_parietalI  = np.concatenate((parietalI_pfcE, parietalI_pfcI, parietalI_parietalE, parietalI_parietalI), axis=0)

    mask = np.concatenate((mask_col_pfcE, mask_col_pfcI, mask_col_parietalE,  mask_col_parietalI), axis=1)

    return mask






if __name__ == '__main__':


    hp = {}
    hp['e_prop'] = 0.75
    hp['sparsity']=1
    n_rnn = 16

    mask = mask_pfc_par_local(hp, n_rnn)
    print('mask','\n',mask)