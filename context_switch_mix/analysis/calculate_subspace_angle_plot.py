

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
import default
import tools
import run

import calculate_subspace_angle_lib,Sequence_lib



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
figure_path = os.path.join(fig_path, 'calculate_angle/')
# figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'_idx_'+str(idx)+'/')
tools.mkdir_p(figure_path)
print('figure_path:',figure_path)


data_root = hp['root_path'] + '/Datas/'
data_path = os.path.join(data_root, 'calculate_angle/')
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
        #print('model_dir_current', hp['model_dir_A_hh'])
    elif context == 'con_B2A':
        model_name = str(hp['mask_type']) + '_' + str(hp['n_rnn']) + '_' + str(hp['n_md']) + '_model_' \
                     + str(hp['model_idx']) + '_load_A' + str(hp['loadA_idx']) + '_B' + str(hp['loadB_idx'])
        # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)

        local_folder_name = os.path.join('/' + 'model_B2A_' + str(hp['load_model_coh']), model_name)
        model_dir = hp['root_path'] + local_folder_name + '/'
        hp['model_dir_A_hh'] = model_dir
        #print('model_dir_current', hp['model_dir_A_hh'])
    #print('model_dir_current',hp['model_dir_A_hh'])
    return model_dir, model_name


########################## calculate angle between md pfc in context A #######################################


def calculate_angle_between_context(neuron,period,model_idx,loadA_idx,loadB_idx,cue,p_coh):

    ###########################################
    hp['loadA_idx'] = loadA_idx
    hp['loadB_idx'] = loadB_idx

    model_dir_A, model_name_A = get_model_dir_diff_context(model_idx, context='con_A')
    model_dir_A += str(loadA_idx)
    print('model_dir_A:', model_dir_A)

    model_dir_A2B, model_name_A2B = get_model_dir_diff_context(model_idx, context='con_A2B')
    model_dir_A2B += str(loadB_idx)
    print('model_dir_A2B:', model_dir_A2B)
    model_dir_B2A, model_name_B2A = get_model_dir_diff_context(model_idx, context='con_B2A')
    model_dir_B2A += str(0)
    print('model_dir_B2A:', model_dir_B2A)


    if period=='cue':
        start_time = cue_on +1
        end_time = cue_off - 1
    elif period=='cuedelay':
        start_time = cue_on +1
        end_time = stim_on - 1

    elif period=='delay':
        start_time = cue_off +1
        end_time = stim_on - 1



    degree_0,degree_1,degree_2 = calculate_subspace_angle_lib.angle_between_different_context(figure_path, data_path,
                                        model_dir_A=model_dir_A, model_dir_A2B=model_dir_A2B,
                                        model_dir_B2A=model_dir_B2A,
                                        hp=hp,
                                        start_time=start_time,
                                        end_time=end_time,
                                        neuron=neuron,
                                        cue = cue,
                                        p_coh=p_coh,

                                        )
    return degree_0,degree_1,degree_2











def plot_violinplot_10model_diff_context(neuron,cue,p_coh):
    figure_path_0 = os.path.join(figure_path, 'plot_violinplot_10model_diff_context/')
    tools.mkdir_p(figure_path_0)
    idx=0

    degree_0_cue_list = np.load(data_path + 'degree_0_cue_list_' + neuron + '.npy').tolist()
    degree_1_cue_list = np.load(data_path + 'degree_1_cue_list_' + neuron + '.npy').tolist()
    degree_2_cue_list = np.load(data_path + 'degree_2_cue_list_' + neuron + '.npy').tolist()

    degree_0_delay_list = np.load(data_path + 'degree_0_delay_list_' + neuron + '.npy').tolist()
    degree_1_delay_list = np.load(data_path + 'degree_1_delay_list_' + neuron + '.npy').tolist()
    degree_2_delay_list = np.load(data_path + 'degree_2_delay_list_' + neuron + '.npy').tolist()


    print(neuron+' degree_0_cue:',degree_0_cue_list,degree_1_cue_list,degree_2_cue_list)
    print(neuron+' degree_0_delay:',degree_0_delay_list, degree_1_delay_list, degree_2_delay_list)

    data_0_cue = np.array([degree_0_cue_list,degree_1_cue_list,degree_2_cue_list]).T
    data_0_delay = np.array([degree_0_delay_list, degree_1_delay_list, degree_2_delay_list]).T
    print('data_0_delay',data_0_delay.shape)

    # Create a boxplot
    colors_list_pfc = sns.light_palette("red",n_colors=8)
    colors_list_md = sns.color_palette("Blues",n_colors=8)  # sns.light_palette("blue",n_colors=6)#sns.color_palette("Paired")
    if neuron == 'pfc':
        colors = [colors_list_pfc[1], colors_list_pfc[3], colors_list_pfc[5]]
    elif neuron == 'md':
        colors = [colors_list_md[1], colors_list_md[3], colors_list_md[5]]

    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.17, 0.15, 0.77, 0.75])
    # plt.boxplot(data)
    # colors_list = sns.color_palette("hls", 8)
    sns.violinplot(data=data_0_cue, palette=colors, linecolor="k", inner='point', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.xticks([1, 2, 3,4,5], [str(cohs[0]),str(cohs[1]),str(cohs[2]),str(cohs[3]),str(cohs[4])],fontsize=13)
    plt.xlabel('', fontsize=10)
    # plt.ylabel('angle (deg.)', fontsize=15)
    plt.title(neuron + ';'+'cue='+str(cue)+'; cueing', fontsize=10)
    plt.ylim([0, 110])
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=11)
    # fig.savefig(figure_path_0 + task_name + '_angle.png')
    fig.savefig(figure_path_0 + neuron + '_cue'+str(cue) + '; cueing.pdf')
    plt.show()

    ################# delay ###########################
    ################# delay ###########################
    ################# delay ###########################

    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.17, 0.15, 0.77, 0.75])

    sns.violinplot(data=data_0_delay, palette=colors, linecolor="k", inner='point', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.xticks([1, 2, 3,4,5], [str(cohs[0]),str(cohs[1]),str(cohs[2]),str(cohs[3]),str(cohs[4])],fontsize=13)
    plt.xlabel('', fontsize=10)
    # plt.ylabel('angle (deg.)', fontsize=15)
    plt.title(neuron + ';' + 'cue=' + str(cue)+'; delay', fontsize=10)
    plt.ylim([0, 110])
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=11)
    # fig.savefig(figure_path_0 + task_name + '_angle.png')
    fig.savefig(figure_path_0 + neuron + '_cue' + str(cue) + 'delay.pdf')
    plt.show()


def plot_angle_paper():
    p_coh=0.92
    plot_violinplot_10model_diff_context(neuron='pfc',cue=1,p_coh=p_coh)
    # plot_violinplot_10model_diff_context(neuron='pfc',cue=-1,p_coh=p_coh)
    #plot_violinplot_10model_diff_context(neuron='md',cue=1,p_coh=p_coh)
    plot_violinplot_10model_diff_context(neuron='md',cue=-1,p_coh=p_coh)

plot_angle_paper()














































































