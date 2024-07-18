import os
import sys

import numpy as np

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
import matplotlib.colors as colors
import Sequence_lib
clrs = sns.color_palette("Set2")#sns.color_palette("muted")#muted


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
figure_path = os.path.join(fig_path, 'connnect_change_plot/')
# figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'_idx_'+str(idx)+'/')
tools.mkdir_p(figure_path)


data_root = hp['root_path'] + '/Datas/'
data_path = os.path.join(data_root, 'connnect_change_plot/')
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

def get_data_one_context(pfc_HL_vis_A):
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


    return X_vis_A, Y_vis_A,data_vis_A


def weight_plot(model_idx,i_dx):
    figure_path = os.path.join(fig_path, 'test_model_peak_order_three/')
    tools.mkdir_p(figure_path)

    coh_HL = 0.92

    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx

    batch_size = 80

    #######rule1
    model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A += str(idx)

    model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B += str(idx)

    model_dir_B2A, model_name_B2A = get_model_dir_diff_context(model_idx, context='con_B2A')
    model_dir_B2A += str(idx)
    #print('model_dir_A:', model_dir_A)

    Sequence_lib.all_weight_hotmap(figure_path, hp, model_dir_A, model_dir_A2B, model_dir_B2A)


#weight_plot(model_idx=5,i_dx=15)



def weight_plot_sorted_all(model_idx,i_dx):
    figure_path = os.path.join(fig_path, 'test_model_peak_order_three/')
    tools.mkdir_p(figure_path)

    coh_HL = 0.92

    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx

    batch_size = 80

    # #######rule1

    effective_weights_A = np.load(data_path + 'effective_weights_A.npy')
    effective_weights_A2B = np.load(data_path + 'effective_weights_A2B.npy')
    effective_weights_B2A = np.load(data_path + 'effective_weights_B2A.npy')



    mat_md1_pc_A = effective_weights_A[0:205, 256:256 + 18]
    mat_md1_pv_A = effective_weights_A[205 + 17 * 2:256, 256:256 + 18]
    mat_pc_md1_A = effective_weights_A[256:256 + 18, 0:205]

    mat_md1_pc_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_md1_pv_A2B = effective_weights_A2B[205 + 17 * 2:256, 256:]
    mat_pc_md1_A2B = effective_weights_A2B[256:, 0:205]

    mat_md1_pc_B2A = effective_weights_B2A[0:205, 256:256 + 18]
    mat_md1_pv_B2A = effective_weights_B2A[205 + 17 * 2:256, 256:256 + 18]
    mat_pc_md1_B2A = effective_weights_B2A[256:256 + 18, 0:205]

    diff_conA_conA2B_index_md1_pc = np.argsort(np.mean(mat_md1_pc_A2B-mat_md1_pc_A,axis=1))
    print('****** diff_conA_conA2B_index_md1_pc', diff_conA_conA2B_index_md1_pc)
    mat_md1_pc_A = effective_weights_A[diff_conA_conA2B_index_md1_pc, 256:256 + 18]
    mat_md1_pc_A2B = effective_weights_A2B[diff_conA_conA2B_index_md1_pc, 256:256 + 18]
    mat_md1_pc_B2A = effective_weights_B2A[diff_conA_conA2B_index_md1_pc, 256:256 + 18]



    mat_md2_pc_A = effective_weights_A[0:205, 256 + 18:]
    mat_md2_vip_A = effective_weights_A[205 + 17 * 0:256 + 17 * 1, 256 + 18:]
    mat_pc_md2_A = effective_weights_A[256 + 18:, 0:205]

    mat_md2_pc_A2B = effective_weights_A2B[0:205, 256 + 18:256 + 2 * 18]
    mat_md2_vip_A2B = effective_weights_A2B[205 + 17 * 0:256 + 17 * 1, 256 + 18:256 + 2 * 18]
    mat_pc_md2_A2B = effective_weights_A2B[256 + 18:256 + 2 * 18, 0:205]

    mat_md2_pc_B2A = effective_weights_B2A[0:205, 256 + 18:256 + 2 * 18]
    mat_md2_vip_B2A = effective_weights_B2A[205 + 17 * 0:256 + 17 * 1, 256 + 18:256 + 2 * 18]
    mat_pc_md2_B2A = effective_weights_B2A[256 + 18:256 + 2 * 18, 0:205]

    diff_conA_conA2B_index_md2_pc = np.argsort(np.mean(mat_md2_pc_A2B - mat_md2_pc_A, axis=1))
    print('****** diff_conA_conA2B_index_md2_pc', diff_conA_conA2B_index_md2_pc)
    mat_md2_pc_A   = effective_weights_A[diff_conA_conA2B_index_md2_pc, 256:256 + 18]
    mat_md2_pc_A2B = effective_weights_A2B[diff_conA_conA2B_index_md2_pc, 256:256 + 18]
    mat_md2_pc_B2A = effective_weights_B2A[diff_conA_conA2B_index_md2_pc, 256:256 + 18]






    start_exc_1 = 0
    end_exc_1 = 205



    data = mat_md1_pc_A2B[start_exc_1:end_exc_1,:].T
    median_val = np.median(data)
    max_val = np.max(data)
    print('median_val,max_val:',median_val,max_val)

    cmap = plt.cm.get_cmap('summer')
    norm = colors.Normalize(vmin=-median_val, vmax=median_val)
    norm.autoscale_None(data )


    start_exc = 150
    end_exc = 206
    idx_select = diff_conA_conA2B_index_md1_pc[start_exc:end_exc]
    print('idx_select',idx_select)
    print(idx_select[4],idx_select[25],idx_select[37],idx_select[39])

    print(mat_md1_pc_A[150+4, :])
    print(mat_md1_pc_A2B[150+4,:])
    print(mat_md1_pc_B2A[150+4,:])


    fig, axs = plt.subplots(3, 1, figsize=(10, 5))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.05, hspace=0.3, wspace=0.3)

    #fig.suptitle('md1_PC', fontsize=14)

    axs[0].imshow(mat_md1_pc_A[start_exc:end_exc,:].T, aspect='auto', cmap=cmap, norm=norm,alpha=1)
    axs[1].imshow(mat_md1_pc_A2B[start_exc:end_exc,:].T, aspect='auto', cmap=cmap, norm=norm,alpha=1)
    axs[2].imshow(mat_md1_pc_B2A[start_exc:end_exc,:].T, aspect='auto', cmap=cmap, norm=norm,alpha=1)
    #plt.colorbar()
    axs[0].set_title('A')
    axs[1].set_title('A2B')
    axs[2].set_title('B2A')
    plt.show()


    # fig_md2, axs_md2 = plt.subplots(3, 1, figsize=(10, 5))
    # plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.05, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name_B2A + ';' + 'md2_PC', fontsize=14)
    # # fig.suptitle('md1_PC', fontsize=14)
    #
    # axs_md2[0].imshow(mat_md2_pc_A[start_exc:end_exc, :].T, aspect='auto',cmap=cmap_name)
    # axs_md2[1].imshow(mat_md2_pc_A2B[start_exc:end_exc, :].T, aspect='auto',cmap=cmap_name)
    # axs_md2[2].imshow(mat_md2_pc_B2A[start_exc:end_exc, :].T, aspect='auto',cmap=cmap_name)
    #
    # axs_md2[0].set_title('A')
    # axs_md2[1].set_title('A2B')
    # axs_md2[2].set_title('B2A')
    # plt.show()

    # fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    # plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.05, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name_B2A + ';' + 'md1_PC', fontsize=14)
    # # fig.suptitle('md1_PC', fontsize=14)
    #
    # axs[0].imshow(mat_md1_pc_A[start_exc:end_exc,:].T, aspect='auto')
    # axs[1].imshow(mat_md1_pc_A2B[start_exc:end_exc,:].T, aspect='auto')
    # axs[2].imshow(mat_md1_pc_B2A[start_exc:end_exc,:].T, aspect='auto')
    #
    # axs[0].set_title('A')
    # axs[1].set_title('A2B')
    # axs[2].set_title('B2A')
    # plt.show()
    #print(mat_md1_pc_A[0,0],mat_md1_pc_A2B[0,0],mat_md1_pc_B2A[0,0])

    # fig3, axs3 = plt.subplots(3, 1, figsize=(6, 8))
    # plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.05, hspace=0.3, wspace=0.3)
    # fig3.suptitle(model_name_B2A + ';' + 'md2_PC', fontsize=14)
    # # fig.suptitle('md1_PC', fontsize=14)
    #
    # axs3[0].imshow(mat_md2_pc_A[start_exc:end_exc,:].T, aspect='auto')
    # axs3[1].imshow(mat_md2_pc_A2B[start_exc:end_exc,:].T, aspect='auto')
    # axs3[2].imshow(mat_md2_pc_B2A[start_exc:end_exc,:].T, aspect='auto')
    #
    # axs3[0].set_title('A')
    # axs3[1].set_title('A2B')
    # axs3[2].set_title('B2A')
    # plt.show()





    # fig1, axs1 = plt.subplots(3, 1, figsize=(4, 10))
    # plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # #fig1.suptitle(model_name_B2A+';'+'md1_pv', fontsize=12)
    # fig1.suptitle('md1_pv', fontsize=14)
    #
    # axs1[0].imshow(mat_md1_pv_A.T, aspect='auto')
    # axs1[1].imshow(mat_md1_pv_A2B.T, aspect='auto')
    # axs1[2].imshow(mat_md1_pv_B2A.T, aspect='auto')
    #
    # axs1[0].set_title('A')
    # axs1[1].set_title('A2B')
    # axs1[2].set_title('B2A')
    # plt.show()
    #




    # fig2, axs2 = plt.subplots(1, 3, figsize=(6, 10))
    # plt.subplots_adjust(top=0.9, bottom=0.05, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # # fig1.suptitle(model_name_B2A+';'+'pc_md1', fontsize=12)
    # fig2.suptitle('pc_md1', fontsize=14)
    #
    # axs2[0].imshow(mat_pc_md1_A.T, aspect='auto')
    # axs2[1].imshow(mat_pc_md1_A2B.T, aspect='auto')
    # axs2[2].imshow(mat_pc_md1_B2A.T, aspect='auto')
    #
    # axs2[0].set_title('A')
    # axs2[1].set_title('A2B')
    # axs2[2].set_title('B2A')
    # plt.show()

weight_plot_sorted_all(model_idx=5,i_dx=15)



def histgram_weight_plot_all(model_idx,i_dx):
    data_path_0 = os.path.join(data_path, 'histgram_weight_plot_all/')
    tools.mkdir_p(data_path_0)


    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx

    batch_size = 80

    # #######rule1
    # model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    # model_dir_A += str(idx)
    #
    # model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    # model_dir_A2B += str(15)
    #
    # model_dir_B2A, model_name_B2A = get_model_dir_diff_context(model_idx, context='con_B2A')
    # model_dir_B2A += str(idx)


    Sequence_lib.histgram_delta_weight_all(data_path_0,hp)


#histgram_weight_plot_all(model_idx=5,i_dx=15)




