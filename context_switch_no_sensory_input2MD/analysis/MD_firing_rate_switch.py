import os
import sys

import numpy as np
import json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as colors
import seaborn as sns

import default
import tools


import Sequence_lib



rule_name = 'HL_task'  # sys.argv[1]#'HL_task'
hp = default.get_default_hp(rule_name=rule_name)

hp['activation'] = 'softplus'
# hp['input_mask']='no'
hp['dropout_model'] = 0.0
hp['dropout'] = 0.0

hp['mask_test'] = 'type8'
hp['mask_type'] = 'type8'
hp['input_mask'] = 'mask3'
hp['sparsity_pc_md'] = 0.25

hp['stim_std'] = 0.1
hp['stim_std_test'] = 0.1
hp['mask_test'] = 'type8'
hp['start_switch'] = True
hp['mod_type'] = 'testing'
hp['model_dir_current'] = 'no'
n_rnn = hp['n_rnn']

# =========================  plot =============================
fig_path = hp['root_path'] + '/Figures/'
figure_path = os.path.join(fig_path, 'EI_balance/')
# figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'_idx_'+str(idx)+'/')
tools.mkdir_p(figure_path)


data_root = hp['root_path'] + '/Datas/'
data_path = os.path.join(data_root, 'EI_balance/')
# figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'_idx_'+str(idx)+'/')
tools.mkdir_p(data_path)

fs=10
# =========================  plot =============================

hp['lsq1'] = 0.999
hp['lsq2'] = 0.0
hp['load_model_coh'] = 1
hp['n_rnn'] = 256
hp['n_md'] = 30
hp['cue_duration'] = 800


epoch = Sequence_lib.get_epoch(hp=hp)
cue_on = epoch['cue_on']
cue_off = epoch['cue_off']
stim_on = epoch['stim_on']
response_on = epoch['response_on']

def get_model_dir_diff_context(model_idx, context):
    hp['model_idx'] = model_idx

    print(hp['switch_context'])
    model_dir = 'no'
    model_name = 'no'

    if context == 'con_A':
        # model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
        #              + str(hp['model_idx'])

        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) \
                     + '_' + str(hp['lsq1']) + '_' + str(600) + '_model_' + str(hp['model_idx'])

        # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)
        local_folder_name = os.path.join('/' + 'model_A_' + str(hp['load_model_coh']), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'

    elif context == 'con_A2B':
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(hp['loadA_idx'])

        local_folder_name = os.path.join('/' + 'model_A2B_' + str(hp['load_model_coh']), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'
        hp['model_dir_A_hh'] = model_dir
    elif context == 'con_B2A':
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(hp['loadA_idx']) + '_B' + str(hp['loadB_idx'])
        # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)

        local_folder_name = os.path.join('/' + 'model_B2A_' + str(hp['load_model_coh']), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'
        hp['model_dir_A_B_hh'] = model_dir
    print('model_dir_current',hp['model_dir_current'])
    return model_dir, model_name



def MD_switch_fr_A_Bfail_B_transient(model_idx,i_dx,idx_switch):
    fig_path_0 = os.path.join(fig_path, 'MD_switch_fr_A_Bfail_B_transient/')
    tools.mkdir_p(fig_path_0)

    data_path_0 = os.path.join(data_path, 'MD_switch_fr_A_Bfail_B_transient/')
    tools.mkdir_p(data_path_0)

    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx

    model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A += str(12)

    model_dir_A2B_fail, model_name_A2B_fail = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B_fail += str(2)

    model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B += str(12)


    Sequence_lib.MD_switch_fr_A_B_B(fig_path_0, data_path_0,hp, model_dir_A, model_dir_A2B_fail, model_dir_A2B,model_name_A2B)
    Sequence_lib.MD_switch_fr_A_Bfail_Bsucc_cell_example(fig_path_0, data_path_0,hp, model_dir_A,model_dir_A2B_fail,model_dir_A2B,model_name_A2B)

MD_switch_fr_A_Bfail_B_transient(model_idx=1,i_dx=12,idx_switch=12)





def MD_switch_fr_A_B_AA(model_idx,i_dx):
    fig_path_0 = os.path.join(fig_path, 'MD_switch_fr_A_B_AA/')
    tools.mkdir_p(fig_path_0)

    data_path_0 = os.path.join(data_path, 'MD_switch_fr_A_B_AA/')
    tools.mkdir_p(data_path_0)

    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx


    model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A += str(idx)

    model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B += str(idx)

    model_dir_B2A, model_name_B2A = get_model_dir_diff_context(model_idx, context='con_B2A')
    model_dir_B2A += str(idx)

    Sequence_lib.MD_switch_fr_A_B_AA_cell_example(fig_path_0,data_path_0, hp, model_dir_A, model_dir_A2B, model_dir_B2A,model_name_A2B)


MD_switch_fr_A_B_AA(model_idx=1,i_dx=12)











