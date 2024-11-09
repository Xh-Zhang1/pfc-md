import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import default
import tools

import SNR_lib

batch_size = 1

rule_name = 'HL_task'  # sys.argv[1]#'HL_task'
hp = default.get_default_hp(rule_name=rule_name)

hp['dropout_model'] = 0.0
hp['dropout'] = 0.0

hp['mask_test'] = 'type8'
hp['mask_type'] = 'type8'
hp['sparsity_pc_md'] = 0.25

hp['n_rnn'] = 256
hp['n_md'] = 30
hp['exc_output'] = 150

hp['stim_std'] = 0.1
hp['stim_std_test'] = 0.1
hp['mask_test'] = 'type8'
hp['start_switch'] = True
hp['mod_type'] = 'testing'
hp['model_dir_current'] = 'no'
hp['input_mask']='mask3'
batch_size = 100
# =========================  plot =============================
fig_path = hp['root_path'] + '/Figures/'


data_root = hp['root_path'] + '/Datas/'
data_path = os.path.join(data_root, 'plot_scatter/')

# =========================  plot =============================
epoch = SNR_lib.get_epoch(hp=hp)
load_model_coh = 1

def get_model_dir_diff_context(model_idx, context):
    hp['model_idx'] = model_idx
    hp['switch_context'] = context
    print(hp['switch_context'])
    model_dir = 'no'
    model_name = 'no'

    if context == 'con_A':
        # model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
        #              + str(hp['model_idx'])

        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) \
                     + '_' + str(hp['lsq1']) + '_' + str(hp['cue_delay']) + '_model_' + str(hp['model_idx'])

        # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)
        local_folder_name = os.path.join('/' + 'model_A_' + str(load_model_coh), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'

    elif context == 'con_A2B':
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(hp['loadA_idx'])

        local_folder_name = os.path.join('/' + 'model_A2B_' + str(load_model_coh), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'
        hp['model_dir_A_hh'] = model_dir
        print('**  model_dir_A_hh', hp['model_dir_A_hh'])

    elif context == 'con_B2A':
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(hp['loadA_idx']) + '_B' + str(hp['loadB_idx'])
        # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)

        local_folder_name = os.path.join('/' + 'model_B2A_' + str(load_model_coh), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'
        hp['model_dir_A_B_hh'] = model_dir
        print('**  model_dir_A_B_hh',hp['model_dir_A_B_hh'] )



    return model_dir, model_name



def plot_scatter_md_diff_context(model_idx,idx):
    fig_path_0 = os.path.join(fig_path, 'plot_scatter_md_diff_context/')
    tools.mkdir_p(fig_path_0)

    data_path_0 = os.path.join(data_path, 'plot_scatter_md_diff_context/')
    tools.mkdir_p(data_path_0)

    hp['lsq1'] = 0.999
    hp['lsq2'] = 0.001
    hp['p_coh'] = 1
    hp['mod_type'] = 'test'
    model_idx = model_idx
    idx = idx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx



    model_dir_A, model_name = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A2B, _ = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_B2A, _ = get_model_dir_diff_context(model_idx, context='con_B2A')
    hp['model_dir_current'] = model_dir_A2B

    model_dir_A = model_dir_A + str(idx) + '/'
    model_dir_A2B = model_dir_A2B + str(100) + '/'
    model_dir_B2A = model_dir_B2A + str(idx) + '/'


    SNR_lib.scatters_sns_MD_different_context(fig_path_0, data_path_0, hp, epoch, idx, model_name,
                                               period='cue',
                                               model_dir_A=model_dir_A, model_dir_A2B=model_dir_A2B,
                                               model_dir_B2A=model_dir_B2A)


for model_idx in np.array([1]):
    for idx in np.array([12]):
        plot_scatter_md_diff_context(model_idx,idx)
#
