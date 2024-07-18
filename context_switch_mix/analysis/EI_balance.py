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
import run

import Sequence_lib,EI_balance_lib



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




def exc_received_from_md1(model_idx,i_dx):
    figure_path = os.path.join(fig_path, 'weight_plot_sorted_md1/')
    tools.mkdir_p(figure_path)

    coh_HL = 0.92

    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx

    batch_size = 100

    #######rule1
    model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A += str(idx)

    model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B += str(i_dx)
    model_dir_A2B_fail, model_name_A2B_fail = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B_fail += str(1)

    model_dir_B2A, model_name_B2A = get_model_dir_diff_context(model_idx, context='con_B2A')
    model_dir_B2A += str(idx)
    #print('model_dir_A:', model_dir_A)
    effective_weights_A = Sequence_lib.get_weight_A(model_name_A, hp, model_dir_A)


    effective_weights_A2B =Sequence_lib.get_weight_A2B(model_name_A2B, hp, model_dir_A2B)
    effective_weights_A2B_fail =Sequence_lib.get_weight_A2B(model_name_A2B_fail, hp, model_dir_A2B_fail)
    effective_weights_B2A =Sequence_lib.get_weight_B2A(model_name_B2A, hp, model_dir_B2A)

    # pfc_HL_vis_A, md_HL_vis_A = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',model_dir=model_dir_A,
    #                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,context='con_A', model_dir=model_dir_A,
    #                                 cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',model_dir=model_dir_A2B,
    #                                 cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,context='con_A2B',model_dir=model_dir_A2B,
    #                                 cue=1, p_cohs=coh_HL,batch_size=batch_size)
    #
    # pfc_HL_vis_A2B_fail, md_HL_vis_A2B_fail = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,context='con_A2B',model_dir=model_dir_A2B_fail,
    #                                 cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    # pfc_HL_aud_A2B_fail, md_HL_aud_A2B_fail = Sequence_lib.get_neurons_activity_mode_test1(context_name='HL_task',hp=hp, context='con_A2B',model_dir=model_dir_A2B_fail,
    #                                 cue=1, p_cohs=coh_HL,batch_size=batch_size)
    #
    # np.save(figure_path+'pfc_HL_vis_A.npy',pfc_HL_vis_A)
    # np.save(figure_path+'pfc_HL_aud_A.npy',pfc_HL_aud_A)
    # np.save(figure_path + 'pfc_HL_vis_A2B.npy', pfc_HL_vis_A2B)
    # np.save(figure_path + 'pfc_HL_aud_A2B.npy', pfc_HL_aud_A2B)
    # np.save(figure_path + 'pfc_HL_vis_A2B_fail.npy', pfc_HL_vis_A2B_fail)
    # np.save(figure_path + 'pfc_HL_aud_A2B_fail.npy', pfc_HL_aud_A2B_fail)
    #
    # np.save(figure_path + 'md_HL_vis_A.npy', md_HL_vis_A)
    # np.save(figure_path + 'md_HL_aud_A.npy', md_HL_aud_A)
    # np.save(figure_path + 'md_HL_vis_A2B.npy', md_HL_vis_A2B)
    # np.save(figure_path + 'md_HL_aud_A2B.npy', md_HL_aud_A2B)
    # np.save(figure_path + 'md_HL_vis_A2B_fail.npy', md_HL_vis_A2B_fail)
    # np.save(figure_path + 'md_HL_aud_A2B_fail.npy', md_HL_aud_A2B_fail)
    #
    pfc_HL_vis_A = np.load(figure_path+'pfc_HL_vis_A.npy')
    pfc_HL_aud_A = np.load(figure_path + 'pfc_HL_aud_A.npy')
    pfc_HL_vis_A2B = np.load(figure_path + 'pfc_HL_vis_A2B.npy')
    pfc_HL_aud_A2B = np.load(figure_path + 'pfc_HL_aud_A2B.npy')
    pfc_HL_vis_A2B_fail = np.load(figure_path + 'pfc_HL_vis_A2B_fail.npy')
    pfc_HL_aud_A2B_fail = np.load(figure_path + 'pfc_HL_aud_A2B_fail.npy')

    md_HL_vis_A = np.load(figure_path + 'md_HL_vis_A.npy')
    md_HL_aud_A = np.load(figure_path + 'md_HL_aud_A.npy')
    md_HL_vis_A2B = np.load(figure_path + 'md_HL_vis_A2B.npy')
    md_HL_aud_A2B = np.load(figure_path + 'md_HL_aud_A2B.npy')
    md_HL_vis_A2B_fail = np.load(figure_path + 'md_HL_vis_A2B_fail.npy')
    md_HL_aud_A2B_fail = np.load(figure_path + 'md_HL_aud_A2B_fail.npy')


    Sum_exc_As=[]
    Sum_inh_As = []
    Sum_exc_A2Bs = []
    Sum_inh_A2Bs = []
    Sum_exc_A2Bs_fail = []
    Sum_inh_A2Bs_fail = []

    start=cue_off
    end=stim_on
    # cell_aud = [12,38,145,195,179,131]
    cell_vis = [87,128]#[30,101,151]
    cell = cell_vis
    cell = range(205)

    for i_select in np.array(cell):

        ##### from md1
        mat_md1_exc_A = effective_weights_A[i_select, 256:256 + 18]
        fr_md1_vis_A_mean = np.mean(md_HL_vis_A[start:end,0:18],axis=0)
        input_from_md1 = np.mean(mat_md1_exc_A*fr_md1_vis_A_mean)
        #input_from_md1 = np.mean(mat_md1_exc_A)


        ##### from md2
        mat_md2_exc_A = effective_weights_A[i_select, 256 + 18:]
        fr_md2_vis_A_mean = np.mean(md_HL_vis_A[start:end, 18:], axis=0)
        input_from_md2 = np.mean(mat_md2_exc_A * fr_md2_vis_A_mean)
        #input_from_md2 = np.mean(mat_md2_exc_A ,axis=0)

        ##### from inh
        mat_Inh_Exc_A = effective_weights_A[i_select, 205:256]
        fr_Inh_vis_A_mean = np.mean(pfc_HL_vis_A[start:end,205:256],axis=0)
        input_from_inh = np.mean(mat_Inh_Exc_A * fr_Inh_vis_A_mean)
        #input_from_inh = np.mean(mat_Inh_Exc_A,axis=0)
        ##### from exc
        mat_Exc_Exc_A = effective_weights_A[i_select, 0:205]
        fr_Exc_vis_A_mean = np.mean(pfc_HL_vis_A[start:end, 0:205], axis=0)
        input_from_Exc = np.mean(mat_Exc_Exc_A * fr_Exc_vis_A_mean)
        #input_from_Exc = np.mean(mat_Exc_Exc_A,axis=0)


        Sum_A_exc = input_from_md1 + input_from_md2 + input_from_Exc


        Sum_exc_As.append(Sum_A_exc)
        Sum_inh_As.append(np.abs(input_from_inh))

    for i_select in np.array(cell):
        ##### from md1
        mat_md1_exc_A2B = effective_weights_A2B[i_select, 256:256 + 18]
        fr_md1_vis_A2B_mean = np.mean(md_HL_vis_A2B[start:end,0:18],axis=0)
        input_from_md1 = np.mean(mat_md1_exc_A2B*fr_md1_vis_A2B_mean)
        #input_from_md1 = np.mean(mat_md1_exc_A2B,axis=0)


        ##### from md2
        mat_md2_exc_A2B = effective_weights_A2B[i_select, 256 + 18:]
        fr_md2_vis_A2B_mean = np.mean(md_HL_vis_A2B[start:end, 18:], axis=0)
        input_from_md2 = np.mean(mat_md2_exc_A2B * fr_md2_vis_A2B_mean)
        #input_from_md2 = np.mean(mat_md2_exc_A2B,axis=0)

        ##### from inh
        mat_Inh_Exc_A2B = effective_weights_A2B[i_select, 205:256]
        fr_Inh_vis_A2B_mean = np.mean(pfc_HL_vis_A2B[start:end,205:256],axis=0)
        input_from_inh = np.mean(mat_Inh_Exc_A2B * fr_Inh_vis_A2B_mean)
        #input_from_inh = np.mean(mat_Inh_Exc_A2B,axis= 0)


        ##### from exc
        mat_Exc_Exc_A2B = effective_weights_A2B[i_select, 0:205]
        fr_Exc_vis_A2B_mean = np.mean(pfc_HL_vis_A2B[start:end, 0:205], axis=0)
        input_from_Exc = np.mean(mat_Exc_Exc_A2B * fr_Exc_vis_A2B_mean)
        #input_from_Exc = np.mean(mat_Exc_Exc_A2B,axis=0)

        Sum_A2B_exc = input_from_md1 + input_from_md2 + input_from_Exc

        Sum_exc_A2Bs.append(Sum_A2B_exc)
        Sum_inh_A2Bs.append(np.abs(input_from_inh))

    for i_select in np.array(cell):
        ##### from md1
        mat_md1_exc_A2B_fail = effective_weights_A2B_fail[i_select, 256:256 + 18]
        fr_md1_vis_A2B_mean_fail = np.mean(md_HL_vis_A2B_fail[start:end,0:18],axis=0)
        input_from_md1_fail = np.mean(mat_md1_exc_A2B_fail*fr_md1_vis_A2B_mean_fail)
        #input_from_md1_fail = np.mean(mat_md1_exc_A2B_fail,axis=0)


        ##### from md2
        mat_md2_exc_A2B_fail = effective_weights_A2B_fail[i_select, 256 + 18:]
        fr_md2_vis_A2B_mean_fail = np.mean(md_HL_vis_A2B_fail[start:end, 18:], axis=0)
        input_from_md2_fail = np.mean(mat_md2_exc_A2B_fail * fr_md2_vis_A2B_mean_fail)
        #input_from_md2_fail = np.mean(mat_md2_exc_A2B_fail,axis=0)


        ##### from inh
        mat_Inh_Exc_A2B_fail = effective_weights_A2B_fail[i_select, 205:256]
        fr_Inh_vis_A2B_mean_fail = np.mean(pfc_HL_vis_A2B_fail[start:end,205:256],axis=0)
        input_from_inh_fail = np.mean(mat_Inh_Exc_A2B_fail * fr_Inh_vis_A2B_mean_fail)
        #input_from_inh_fail = np.mean(mat_Inh_Exc_A2B_fail,axis=0)


        ##### from exc
        mat_Exc_Exc_A2B_fail = effective_weights_A2B_fail[i_select, 0:205]
        fr_Exc_vis_A2B_mean_fail = np.mean(pfc_HL_vis_A2B_fail[start:end, 0:205], axis=0)
        input_from_Exc_fail = np.mean(mat_Exc_Exc_A2B_fail * fr_Exc_vis_A2B_mean_fail)
        #input_from_Exc_fail = np.mean(mat_Exc_Exc_A2B_fail)

        Sum_A2B_exc_fail = input_from_md1_fail + input_from_md2_fail + input_from_Exc_fail

        Sum_exc_A2Bs_fail.append(Sum_A2B_exc_fail)
        Sum_inh_A2Bs_fail.append(np.abs(input_from_inh_fail))

    ratio_A = np.array(Sum_exc_As)/np.array(Sum_inh_As)
    ratio_A2B = np.array(Sum_exc_A2Bs) / np.array(Sum_inh_A2Bs)
    ratio_A2Bs_fail = np.array(Sum_exc_A2Bs_fail) / np.array(Sum_inh_A2Bs_fail)


    #find cell
    select_ratio_A=[]
    select_ratio_A2B = []
    for i in range(205):
        if 1.3>ratio_A[i]>0.6 and ratio_A[i]-ratio_A2B[i]>0.15:
            select_ratio_A.append(i)
        if 1.3>ratio_A2B[i]>0.6 and ratio_A2B[i]-ratio_A[i]>0.15:
            select_ratio_A2B.append(i)
    print('select_ratio_A2B',select_ratio_A2B)

    #select_ratio_A = [87,128]
    fig1 = plt.figure(figsize=(4, 4))
    ax1 = fig1.add_axes([0.2, 0.2, 0.7, 0.7])
    plt.scatter(ratio_A, ratio_A2B, color='grey')


    # select_ratio_A = [12,38,59,83,87,113,128,131,128,145,174,191,195]
    # select_ratio_A2B = [13, 21, 26, 37, 40, 43, 44, 45, 58, 71, 74, 77, 105, 106, 109, 122, 129, 138, 143, 146, 154, 170, 173, 176, 202]

    for i in np.array(select_ratio_A):
         plt.scatter(ratio_A[i], ratio_A2B[i], label=str(i),color='b')

    for i in np.array(select_ratio_A2B):
        plt.scatter(ratio_A[i], ratio_A2B[i], label=str(i),color='r')

    plt.scatter(ratio_A[128], ratio_A2B[128],color='yellow',marker='*')
    #plt.legend(fontsize=5)

    max = 1.5
    plt.plot([0, max], [0, max], color='grey')
    plt.xlim([-0.001, max])
    plt.ylim([-0.001, max])
    plt.xlabel('context A')
    plt.ylabel('context B')
    plt.title(str(start)+'_'+str(end))

    plt.savefig(figure_path + 'scatter.png')
    plt.show()

    # fig2 = plt.figure(figsize=(4, 4))
    # ax2 = fig2.add_axes([0.2, 0.2, 0.7, 0.7])
    # plt.scatter(ratio_A, ratio_A2Bs_fail, color='orange')
    # max = 1.7  # np.max(fr_rnn_1)*( np.max(fr_rnn_1)>np.max(fr_rnn_2))+np.max(fr_rnn_2)*( np.max(fr_rnn_2)>np.max(fr_rnn_1))
    # plt.plot([0, max], [0, max], color='grey')
    #
    # # plt.xlim([-0.001, 1.7])
    # # plt.ylim([-0.001, 1.7])
    # plt.xlabel('A')
    # plt.ylabel('B_fail')
    # plt.show()


    ######## plot activity ##################################
    #'''

    y_exc=1.2
    start_time = 0
    end_time = response_on - 2
    #end = cue_off + 12

    fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.97, left=0.1, hspace=0.2, wspace=0.2)
    fig.suptitle('blue:'+str(start_time)+'_'+str(end_time), fontsize=10)
    cell_idx_exc = select_ratio_A#range(205)#select_ratio_A2B

    j = -1
    for i in np.array(cell_idx_exc):
        j += 1
        axs[0, 0].plot(pfc_HL_vis_A[cue_on + start_time:end_time, i], label=str(i))
        axs[1, 0].plot(pfc_HL_vis_A2B[cue_on + start_time:end_time, i], label=str(i))
        axs[2, 0].plot(pfc_HL_vis_A2B_fail[cue_on + start_time:end_time, i], label=str(i))
        #axs[0, 0].legend(fontsize=8)
        axs[0, 0].set_title('vis')

    for i in np.array(cell_idx_exc):
        axs[0, 1].plot(pfc_HL_aud_A[cue_on + start_time:end_time, i], label=str(i))
        axs[1, 1].plot(pfc_HL_aud_A2B[cue_on + start_time:end_time, i], label=str(i))
        axs[2, 1].plot(pfc_HL_aud_A2B_fail[cue_on + start_time:end_time, i], label=str(i))
        axs[0, 1].legend(fontsize=8)
        axs[0, 1].set_title('aud')
    for i in range(3):
        for j in range(2):
            axs[i,0].set_ylim([0, y_exc])
            axs[i,1].set_ylim([0, y_exc])
            axs[i,1].set_yticks([])
            axs[i,j].set_xticks([])
            axs[i,j].axvspan(cue_off - 2 - start_time, cue_off - 2 - start_time, color='grey', label='cue_off')
            axs[i,j].axvspan(stim_on - 2 - start_time, stim_on - 2 - start_time, color='grey', label='stim_on')
    plt.savefig(figure_path+model_name_B2A+'.png')
    plt.show()

    fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.97, left=0.1, hspace=0.2, wspace=0.2)
    fig.suptitle('red:'+str(start_time)+'_'+str(end_time), fontsize=10)
    cell_idx_exc = select_ratio_A2B  # range(205)#select_ratio_A2B

    j = -1
    for i in np.array(cell_idx_exc):
        j += 1
        axs[0, 0].plot(pfc_HL_vis_A[cue_on + start_time:end_time, i], label=str(i))
        axs[1, 0].plot(pfc_HL_vis_A2B[cue_on + start_time:end_time, i], label=str(i))
        axs[2, 0].plot(pfc_HL_vis_A2B_fail[cue_on + start_time:end_time, i], label=str(i))
        # axs[0, 0].legend(fontsize=8)
        axs[0, 0].set_title('vis')

    for i in np.array(cell_idx_exc):
        axs[0, 1].plot(pfc_HL_aud_A[cue_on + start_time:end_time, i], label=str(i))
        axs[1, 1].plot(pfc_HL_aud_A2B[cue_on + start_time:end_time, i], label=str(i))
        axs[2, 1].plot(pfc_HL_aud_A2B_fail[cue_on + start_time:end_time, i], label=str(i))
        axs[0, 1].legend(fontsize=10)
        axs[0, 1].set_title('aud')
    for i in range(3):
        for j in range(2):
            axs[i, 0].set_ylim([0, y_exc])
            axs[i, 1].set_ylim([0, y_exc])
            axs[i, 1].set_yticks([])
            axs[i, j].set_xticks([])
            axs[i, j].axvspan(cue_off - 2 - start_time, cue_off - 2 - start_time, color='grey', label='cue_off')
            axs[i, j].axvspan(stim_on - 2 - start_time, stim_on - 2 - start_time, color='grey', label='stim_on')
    plt.savefig(figure_path + model_name_B2A + '.png')
    plt.show()
    #'''


def ratio_EI_vis_plot():
    data_path_0 = os.path.join(data_path, 'ratio_EI_vis_plot/')
    tools.mkdir_p(data_path_0)

    coh_HL = 0.92
    batch_size = 100


    #######
    # model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    # model_dir_A += str(i_dx)
    #
    # model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    # model_dir_A2B += str(i_dx)
    #
    # model_dir_A2B_fail, model_name_A2B_fail = get_model_dir_diff_context(model_idx, context='con_A2B')
    # model_dir_A2B_fail += str(1)

    EI_balance_lib.ratio_EI_vis(data_path_0, hp,model_dir_A=None, model_dir_A2B=None,model_dir_A2B_fail=None)

    data_path_0 = os.path.join(data_path, 'ratio_EI_aud_plot/')
    tools.mkdir_p(data_path_0)
    EI_balance_lib.ratio_EI_aud(data_path_0, hp, model_dir_A=None, model_dir_A2B=None,model_dir_A2B_fail=None)
ratio_EI_vis_plot()



def plot_speed_train():

    idx = 15
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx
    trials_A=[]
    trials_A2B = []
    trials_B2A = []
    #######rule1
    #[31, 32, 33, 34, 37, 38, 39, 40, 41, 43, 44, 45, 47]
    file = [51,53,54,58,59,
            60,61,62,64,65,
            67,71,72,75,77,
            78,79,81,82,83,
            84,85,86,87]

    for model_idx in np.array(file[:20]):
        print('***************************',model_idx)
        model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
        model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
        model_dir_B2A, model_name_B2A = get_model_dir_diff_context(model_idx, context='con_B2A')


        print('model_dir_A',model_dir_A)
        print('model_dir_A2B', model_dir_A2B)
        print('model_dir_B2A', model_dir_B2A)

        with open(model_dir_A+'log.json') as user_file_A:
            parsed_json_A = json.load(user_file_A)
        success_trial_A = parsed_json_A['success_trial']
        trials_A.append(success_trial_A[0])

        with open(model_dir_A2B+'log.json') as user_file_A2B:
            parsed_json_A2B = json.load(user_file_A2B)
        success_trial_A2B = parsed_json_A2B['success_trial']
        trials_A2B.append(success_trial_A2B[0])

        with open(model_dir_B2A+'log.json') as user_file_B2A:
            parsed_json_B2A = json.load(user_file_B2A)
        success_trial_B2A = parsed_json_B2A['success_trial']
        trials_B2A.append(success_trial_B2A[0])

    trials_A_normal = np.array(trials_A)/np.array(trials_A)
    trials_A2B_normal = np.array(trials_A2B)/np.array(trials_A)
    trials_B2A_normal = np.array(trials_B2A)/np.array(trials_A)


    IT_1 = np.mean(trials_A_normal)
    IT_2 = np.mean(trials_A2B_normal)
    IT_3 = np.mean(trials_B2A_normal)
    IT_std_1 = np.std(trials_A_normal)/np.sqrt(trials_A_normal.shape[0])
    IT_std_2 = np.std(trials_A2B_normal)/np.sqrt(trials_A_normal.shape[0])
    IT_std_3 = np.std(trials_B2A_normal)/np.sqrt(trials_A_normal.shape[0])

    IT_mean = [IT_1,IT_2,IT_3]
    IT_std = [IT_std_1,IT_std_2,IT_std_3]

    name_context = ['A','B','AA']

    fig = plt.figure(figsize=(2.5, 2))
    ax = fig.add_axes([0.3, 0.15, 0.6, 0.7])

    ax.bar(name_context, IT_mean, yerr =IT_std,width=0.3, color='tab:orange')

    plt.ylabel('ratio', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('train_speed',fontsize=8)
    plt.yticks([0,0.5,1],fontsize=11)

    #plt.legend(fontsize=8)
    fig.savefig(figure_path + 'train_speed.eps', format='eps', dpi=1000)
    plt.show()
#plot_speed_train()



def MD_switch_fr_A_B_B():
    coh_HL=0.92


    #Sequence_lib.MD_switch_fr_A_B_B(figure_path, data_path,hp, model_dir_A, model_dir_A2B_fail, model_dir_A2B)
    Sequence_lib.MD_switch_fr_A_B_B_cell_example(figure_path,data_path, hp)
MD_switch_fr_A_B_B()

def plot_scatter_md_diff_context(model_idx,i_dx):
    data_path_0 = os.path.join(data_path, 'plot_scatter_md_diff_context/')
    tools.mkdir_p(data_path_0)
    hp['seed'] = 23


    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx


    # Sequence_lib.scatters_sns_MD_different_context_example(fig_path_0, hp, epoch, idx, model_name_A,period='cue',
    #                         model_dir_A=model_dir_A, model_dir_A2B=model_dir_A2B,model_dir_B2A=model_dir_B2A)

    Sequence_lib.sns_PFC_shift_different_context(data_path_0, hp)



plot_scatter_md_diff_context(model_idx=5,i_dx=15)

