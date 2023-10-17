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

import default_switch
import tools_switch


import Sequence_lib, EI_balance_lib



rule_name = 'HL_task'  # sys.argv[1]#'HL_task'
hp = default_switch.get_default_hp(rule_name=rule_name)

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
figure_path = os.path.join(fig_path, 'fig7_EI_balance/')
# figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'_idx_'+str(idx)+'/')
tools_switch.mkdir_p(figure_path)


data_root = hp['root_path'] + '/Datas_switch/'
data_path = os.path.join(data_root, 'fig7_EI_balance/')
# figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'_idx_'+str(idx)+'/')
tools_switch.mkdir_p(data_path)

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

    print(hp['switch_context'])
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
    elif context == 'con_B2A':
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(hp['loadA_idx']) + '_B' + str(hp['loadB_idx'])
        # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)

        local_folder_name = os.path.join('/' + 'model_B2A_' + str(hp['load_model_coh']), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'
        hp['model_dir_A_B_hh'] = model_dir
    print('model_dir_current',hp['model_dir_current'])
    return model_dir, model_name


def plot_figure(data_0,cell):
    # normalize
    #print("data,data.max",data)
    #'''
    for i in range(0, data_0.shape[1]):
        if np.max(data_0[:, i])<0:
            data_0[:, i] = 0
        else:
            data_0[:, i] = (data_0[:, i] / np.max(data_0[:, i]))
            #print("np.max(data[:, i])",np.max(data[:, i]))
    #'''
    X_0, Y_0 = np.mgrid[0:data_0.shape[0]*20:20, 0:data_0.shape[1]]

    fig = plt.figure(figsize=(3.0, 2.6))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])

    plt.gca().set_xlabel('Time (ms)', fontsize=fs+1)
    plt.gca().set_ylabel('Neuron (Sorted)', fontsize=fs+1)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title(cell,fontsize=8)

    # Make the plot
    #cmap = plt.get_cmap('viridis')#viridis_r
    plt.pcolormesh(X_0, Y_0, data_0)

    m = cm.ScalarMappable(cmap=mpl.rcParams["image.cmap"])#cmap=mpl.rcParams["image.cmap"]
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)

    cbar = fig.colorbar(m, aspect=15)

    cbar.set_ticks([0,  1])
    cbar.ax.tick_params(labelsize=fs+1)
    #cbar.ax.set_title('Normalized\n activity', fontsize=fs+1)
    fig.savefig(figure_path+cell+'.eps', format='eps', dpi=1000)
    plt.show()

def filter_data(data_0):

    for i in range(0, data_0.shape[1]):
        if np.max(data_0[:, i]) < 0:
            data_0[:, i] = 0
        else:
            data_0[:, i] = (data_0[:, i] / np.max(data_0[:, i]))
            # print("np.max(data[:, i])",np.max(data[:, i]))
    # '''
    X_0, Y_0 = np.mgrid[0:data_0.shape[0] * 20:20, 0:data_0.shape[1]]
    return X_0, Y_0

def get_data_all_context(pfc_HL_vis_A, pfc_HL_vis_A2B, pfc_HL_vis_B2A):
    start_time = epoch['cue_off'] + 1
    end_time = epoch['stim_on'] - 2

    # '''
    func_activity_threshold_0 = 0.2

    data_vis_A = pfc_HL_vis_A[start_time:end_time, 0:205]

    max_firing_rate_vis_A = np.max(data_vis_A, axis=0)
    pick_idx_vis_A = np.argwhere(max_firing_rate_vis_A > func_activity_threshold_0).squeeze()

    data_vis_A = data_vis_A[:, pick_idx_vis_A]
    peak_time_vis_A = np.argmax(data_vis_A, axis=0)

    peak_order_vis_A = np.argsort(peak_time_vis_A, axis=0)
    print('peak_order_vis_A', peak_order_vis_A)
    data_vis_A = data_vis_A[:, peak_order_vis_A]

    X_vis_A, Y_vis_A = filter_data(data_vis_A)

    ######### context_A2B #################
    data_vis_A2B = pfc_HL_vis_A2B[start_time:end_time, 0:205]
    data_vis_A2B = data_vis_A2B[:, pick_idx_vis_A]
    data_vis_A2B = data_vis_A2B[:, peak_order_vis_A]

    X_vis_A2B, Y_vis_A2B = filter_data(data_vis_A2B)

    ######### context_B2A #################
    data_vis_B2A = pfc_HL_vis_B2A[start_time:end_time, 0:205]
    data_vis_B2A = data_vis_B2A[:, pick_idx_vis_A]
    data_vis_B2A = data_vis_B2A[:, peak_order_vis_A]

    X_vis_B2A, Y_vis_B2A = filter_data(data_vis_B2A)

    return X_vis_A, Y_vis_A,data_vis_A,X_vis_A2B, Y_vis_A2B,data_vis_A2B, X_vis_B2A, Y_vis_B2A,data_vis_B2A


def test_model_peak_order_three_context(model_idx,i_dx,idx_switch):
    figure_path = os.path.join(fig_path, 'test_model_peak_order_three/')
    tools_switch.mkdir_p(figure_path)

    data_path_1 = os.path.join(data_path, 'test_model_peak_order_three/')
    tools_switch.mkdir_p(data_path_1)

    coh_HL=0.92

    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx


    batch_size = 100


    #######rule1
    start_time = epoch['cue_off']-1
    end_time = epoch['stim_on']-2

    model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A += str(idx)

    model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B += str(idx_switch)

    model_dir_B2A, model_name_B2A = get_model_dir_diff_context(model_idx, context='con_B2A')
    model_dir_B2A += str(idx_switch)
    print('model_dir_A:',model_dir_A)
    # print('model_dir_A2B:',model_dir_A2B)
    # print('model_dir_B2A:', model_dir_B2A)


    # pfc_HL_vis_A, md_HL_vis_A = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',model_dir=model_dir_A,
    #                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,context='con_A',model_dir=model_dir_A,
    #                                                 cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,context='con_A2B',model_dir=model_dir_A2B,
    #                                                 cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,context='con_A2B',model_dir=model_dir_A2B,
    #                                                 cue=1, p_cohs=coh_HL,batch_size=batch_size)
    # #
    # pfc_HL_vis_B2A, md_HL_vis_B2A = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,context='con_B2A',model_dir=model_dir_B2A,
    #                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_B2A, md_HL_aud_B2A = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,context='con_B2A',model_dir=model_dir_B2A,
    #                                                 cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    #
    # np.save(data_path+'pfc_HL_vis_A.npy',pfc_HL_vis_A)
    # np.save(data_path + 'pfc_HL_vis_A2B.npy', pfc_HL_vis_A2B)
    # np.save(data_path + 'pfc_HL_vis_B2A.npy', pfc_HL_vis_B2A)
    # np.save(data_path + 'pfc_HL_aud_A.npy', pfc_HL_aud_A)
    # np.save(data_path + 'pfc_HL_aud_A2B.npy', pfc_HL_aud_A2B)
    # np.save(data_path + 'pfc_HL_aud_B2A.npy', pfc_HL_aud_B2A)

    pfc_HL_vis_A   = np.load(data_path + 'pfc_HL_vis_A.npy')
    pfc_HL_vis_A2B = np.load(data_path + 'pfc_HL_vis_A2B.npy')
    pfc_HL_vis_B2A = np.load(data_path + 'pfc_HL_vis_B2A.npy')
    pfc_HL_aud_A   = np.load(data_path + 'pfc_HL_aud_A.npy')
    pfc_HL_aud_A2B = np.load(data_path + 'pfc_HL_aud_A2B.npy')
    pfc_HL_aud_B2A = np.load(data_path + 'pfc_HL_aud_B2A.npy')


    X_vis_A, Y_vis_A,data_vis_A,X_vis_A2B, Y_vis_A2B,data_vis_A2B, \
        X_vis_B2A, Y_vis_B2A,data_vis_B2A = get_data_all_context(pfc_HL_vis_A, pfc_HL_vis_A2B, pfc_HL_vis_B2A)



    ##### plot activity #################################

    start = 0
    end = response_on - 2
    max_idx_exc = np.argsort(np.mean(pfc_HL_vis_A[cue_on:response_on, 0:205], axis=0))
    #print('max_idx_exc',max_idx_exc)
    cell_idx_exc = range(0,20,1)#max_idx_exc[0:205]#range(0,20,1)
    cell_idx_md1 = range(0,18,1)  # max_idx_md1[0:20]
    cell_idx_inh = range(205+17*2, 256, 1)  # max_idx_md1[0:20]


    # y_md2 = 1.2 * np.max(md_HL_vis_A[cue_on:stim_on - 2, 18:12])

    fig, axs = plt.subplots(3, 3, figsize=(9,9))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)

    axs[0,0].pcolormesh(X_vis_A, Y_vis_A, data_vis_A)
    axs[1, 0].pcolormesh(X_vis_A2B, Y_vis_A2B, data_vis_A2B)
    axs[2, 0].pcolormesh(X_vis_B2A, Y_vis_B2A, data_vis_B2A)

    for i in np.array(cell_idx_exc):
        axs[0,1].plot(pfc_HL_vis_A[cue_on + start:end, i], label=str(i))
        axs[1,1].plot(pfc_HL_vis_A2B[cue_on + start:end, i], label=str(i))
        axs[2,1].plot(pfc_HL_vis_B2A[cue_on + start:end, i], label=str(i))
        #axs[0,1].legend(fontsize=5)

    for i in np.array(cell_idx_exc):
        axs[0,2].plot(pfc_HL_aud_A[cue_on + start:end, i], label=str(i))
        axs[1,2].plot(pfc_HL_aud_A2B[cue_on + start:end, i], label=str(i))
        axs[2,2].plot(pfc_HL_aud_B2A[cue_on + start:end, i], label=str(i))
        #axs[0,1].legend(fontsize=5)

    # for i in np.array(cell_idx_inh):
    #     axs[0,2].plot(pfc_HL_vis_A[cue_on + start:end, i], label=str(i))
    #     axs[1,2].plot(pfc_HL_vis_A2B[cue_on + start:end, i], label=str(i))
    #     axs[2,2].plot(pfc_HL_vis_B2A[cue_on + start:end, i], label=str(i))
    #     #axs[0,1].legend(fontsize=5)


    # for i in range(3):
    #     for j in range(1,4):
    #         axs[i,1].set_ylim([y_exc_min, y_exc])
    #         axs[i,2].set_ylim([y_exc_min, y_exc])
    #         axs[i,3].set_ylim([y_md1_min, y_md1])
    #         axs[i,j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
    #         axs[i,j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')
    #

    plt.savefig(figure_path+model_name_B2A+'_'+str(idx_switch)+'.png')
    plt.show()

test_model_peak_order_three_context(model_idx=5,i_dx=15,idx_switch=15)



def ratio_EI_vis_plot(model_idx,i_dx):
    fig_path_0 = os.path.join(fig_path, 'ratio_EI_vis_plot/')
    tools_switch.mkdir_p(fig_path_0)

    data_path_1 = os.path.join(data_path, 'ratio_EI_vis_plot/')
    tools_switch.mkdir_p(data_path_1)


    model_idx = model_idx
    hp['loadA_idx'] = i_dx
    hp['loadB_idx'] = i_dx

    #######
    model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A += str(i_dx)

    model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B += str(i_dx)

    model_dir_A2B_fail, model_name_A2B_fail = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B_fail += str(1)

    EI_balance_lib.ratio_EI_vis(data_path_1, hp,model_dir_A, model_dir_A2B,model_dir_A2B_fail)
ratio_EI_vis_plot(model_idx=5,i_dx=15)


def MD_switch_fr_A_B_B(model_idx,i_dx,idx_switch):
    fig_path_0 = os.path.join(fig_path, 'MD_switch_fr_A_B_B/')
    tools_switch.mkdir_p(fig_path_0)

    data_path_1 = os.path.join(data_path, 'MD_switch_fr_A_B_B/')
    tools_switch.mkdir_p(data_path_1)

    coh_HL=0.92

    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx
    batch_size = 100



    model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A += str(40)

    model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B += str(15)

    model_dir_B2A, model_name_B2A = get_model_dir_diff_context(model_idx, context='con_B2A')
    model_dir_B2A += str(20)

    model_dir_A2B_fail, model_name_A2B_fail = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B_fail += str(1)



    Sequence_lib.MD_switch_fr_A_B_B(fig_path_0, data_path_1,hp, model_dir_A, model_dir_A2B_fail, model_dir_A2B)
    Sequence_lib.MD_switch_fr_A_B_B_cell_example(fig_path_0,data_path_1, hp, model_dir_A, model_dir_A2B_fail, model_dir_A2B)
MD_switch_fr_A_B_B(model_idx=5,i_dx=15,idx_switch=15)

def plot_scatter_md_diff_context(model_idx,i_dx):
    fig_path_0 = os.path.join(figure_path, 'plot_scatter_md_diff_context/')
    tools_switch.mkdir_p(fig_path_0)

    data_path_1 = os.path.join(data_path, 'plot_scatter_md_diff_context/')
    tools_switch.mkdir_p(data_path_1)


    hp['seed'] = 23


    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx


    model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A += str(40)

    model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B += str(10)

    model_dir_B2A, model_name_B2A = get_model_dir_diff_context(model_idx, context='con_B2A')
    model_dir_B2A += str(20)



    Sequence_lib.scatters_sns_MD_different_context_example(data_path_1, hp, epoch, idx, model_name_A,period='cue',
                            model_dir_A=model_dir_A, model_dir_A2B=model_dir_A2B,model_dir_B2A=model_dir_B2A)



plot_scatter_md_diff_context(model_idx=5,i_dx=15)

