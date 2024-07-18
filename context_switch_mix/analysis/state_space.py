import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as colors

import default
import tools
import run

import state_space_lib,Sequence_lib



rule_name = 'HL_task'  # sys.argv[1]#'HL_task'
hp = default.get_default_hp(rule_name=rule_name)

hp['activation'] = 'softplus'
# hp['input_mask']='no'
hp['dropout_model'] = 0.0
hp['dropout'] = 0.0

hp['mask_test'] = 'type8'
hp['mask_type'] = 'type8'
hp['input_mask'] = 'mask0'
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
figure_path = os.path.join(fig_path, 'state_space/')
# figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'_idx_'+str(idx)+'/')
tools.mkdir_p(figure_path)


data_root = hp['root_path'] + '/Datas/'
data_path = os.path.join(data_root, 'state_space/')
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
hp['cue_delay'] = 600

epoch = Sequence_lib.get_epoch(hp=hp)
cue_on = epoch['cue_on']
cue_off = epoch['cue_off']
stim_on = epoch['stim_on']
response_on = epoch['response_on']

def get_model_dir_diff_context(model_idx, context):
    hp['model_idx'] = model_idx
    model_dir = 'no'
    model_name = 'no'

    if context == 'con_A':
        # model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
        #              + str(hp['model_idx'])

        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) \
                     + '_' + str(hp['lsq1']) + '_' + str(hp['cue_delay']) + '_model_' + str(hp['model_idx'])

        # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)
        local_folder_name = os.path.join('/' + 'model_A_' + str(hp['load_model_coh']), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'

    elif context == 'con_A2B':
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(hp['loadA_idx'])

        local_folder_name = os.path.join('/' + 'model_A2B_' + str(hp['load_model_coh']), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'
        hp['model_dir_A_hh'] = model_dir
        print('model_dir_current', hp['model_dir_A_hh'])
    elif context == 'con_B2A':
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(hp['loadA_idx']) + '_B' + str(hp['loadB_idx'])
        # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)

        local_folder_name = os.path.join('/' + 'model_B2A_' + str(hp['load_model_coh']), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'
        hp['model_dir_A_hh'] = model_dir
        print('model_dir_current', hp['model_dir_A_hh'])
    #print('model_dir_current',hp['model_dir_A_hh'])
    return model_dir, model_name

def PCA_conA():
    data_path_0 = os.path.join(data_path, 'PCA_conA/')
    tools.mkdir_p(data_path_0)

    coh_HL=0.92

    # model_idx = 2
    # idx = 12
    # hp['loadA_idx'] = idx
    # hp['loadB_idx'] = idx
    # ######rule1
    # model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    # model_dir_A += str(idx)



    state_space_lib.PCA_plot_2D_cue_color(data_path_0,data_path,model_dir=None,model_name=None,idx=None,context='con_A',
                                    hp=hp,
                                    task_name='HL_task',
                                    start_time=cue_on-1,
                                    end_time=stim_on,
                                    p_coh=0.92)



PCA_conA()




def PCA_conA_switch_contB_mean():
    data_path_0 = os.path.join(data_path, 'PCA_conA_switch_contB_mean/')
    tools.mkdir_p(data_path_0)

    # hp['loadA_idx'] = i_dx
    # hp['loadB_idx'] = i_dx
    #
    # idx = i_dx
    # model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    # model_dir_A += str(i_dx)
    #
    #
    # model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    # model_dir_A2B += str(i_dx)
    # print('model_dir_A2B:', model_dir_A2B)
    # model_dir_A2B_1, model_name_A2B_1 = get_model_dir_diff_context(model_idx, context='con_A2B')
    # model_dir_A2B_1 += str(3)

    state_space_lib.switch_2D_A_B_mean(data_path_0, data_path, model_dir_A=None, model_dir_A2B=None,
                                     hp=hp,
                                     start_time=cue_on-1,
                                     end_time=cue_off - 1,
                                     )

    state_space_lib.switch_2D_B_B1_mean(data_path_0, data_path, model_dir_1=None, model_dir_2=None,
                                       hp=hp,
                                       start_time=cue_on - 1,
                                       end_time=cue_off - 1,
                                       )
    #
    state_space_lib.switch_2D_A_B_B1_mean(data_path_0, data_path, model_dir_1=None, model_dir_2=None,
                                          model_dir_3=None,
                                        hp=hp,
                                        start_time=cue_on - 1,
                                        end_time=cue_off-1,
                                        )

PCA_conA_switch_contB_mean()

