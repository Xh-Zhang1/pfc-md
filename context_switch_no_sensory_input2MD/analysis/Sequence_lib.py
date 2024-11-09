import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import torch
import seaborn as sns
import matplotlib.colors as colors
import default
import tools
# import run

clrs = sns.color_palette("muted")#muted
clrs_fill = sns.color_palette('pastel')
def get_epoch(hp):
    # print('************ perform HL_task')
    dt = hp['dt']
    batch_size = 1
    rng = hp['rng']

    # cue stim
    cue_time = int(hp['cue_duration'])
    cue_delay_time = int(hp['cue_delay'])
    cue_on = (rng.uniform(40, 40, batch_size) / dt).astype(int)
    cue_duration = int(cue_time / dt)  # 60
    cue_off = cue_on + cue_duration
    cue_delay = (rng.uniform(cue_delay_time, cue_delay_time, batch_size) / dt).astype(int)

    # stim epoch
    stim_time = int(hp['stim'])
    stim1_on = cue_off + cue_delay
    stim1_during = (rng.uniform(stim_time, stim_time, batch_size) / dt).astype(int)
    stim1_off = stim1_on + stim1_during
    response_on = stim1_off

    # response end time
    response_duration = int(hp['response_time'] / dt)
    response_off = response_on + response_duration


    cue_on = cue_on[0];cue_off=cue_off[0];stim_on=stim1_on[0];stim_off=stim1_off[0];response_on=response_on[0];response_off=response_off[0]
    print('cue_on,cue_off,stim_on,stim_off,response_on',cue_on,cue_off,stim_on,stim_off,response_on)

    epoch = {'cue_on':cue_on,
             'cue_off':cue_off,
             'stim_on':stim_on,
             'stim_off':stim_off,
             'response_on':response_on,
             'response_off':response_off}

    return epoch




def MD_switch_fr_A_B_B(figure_path, data_path,hp, model_dir_1, model_dir_2,
                                     model_dir_3,model_name_A2B):
    data_path = os.path.join(data_path, 'MD_switch_fr_A_B_B/')
    tools.mkdir_p(data_path)
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.99
    batch_size = 100


    # pfc_HL_both_A, md_HL_both_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',
    #                                                             model_dir=model_dir_1,
    #                                                             cue=None, p_cohs=coh_HL, batch_size=batch_size)
    #
    #
    # pfc_HL_both_A2B, md_HL_both_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
    #                                                                 model_dir=model_dir_2,
    #                                                                 cue=None, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_both_B2A, md_HL_both_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',
    #                                                                 model_dir=model_dir_3,
    #                                                                 cue=None, p_cohs=coh_HL, batch_size=batch_size)
    #
    #
    # np.save(data_path+'md_HL_both_A.npy',md_HL_both_A)
    # np.save(data_path + 'md_HL_both_A2B.npy', md_HL_both_A2B)
    # np.save(data_path + 'md_HL_both_B2A.npy', md_HL_both_B2A)






    md_HL_both_A = np.load(data_path+'md_HL_both_A.npy')
    md_HL_both_A2B = np.load(data_path + 'md_HL_both_A2B.npy')
    md_HL_both_B2A = np.load(data_path + 'md_HL_both_B2A.npy')

    md_HL_both_A_cell = md_HL_both_A
    md_HL_both_A2B_cell = md_HL_both_A2B
    md_HL_both_B2A_cell = md_HL_both_B2A



    #cell_idx_md1 = [3,5,8,9,10,16]
    cell_idx_md1 = range(16)
    start = 0
    end = response_on - 2
    md_HL_both_A = np.mean(md_HL_both_A[cue_on + start:end, cell_idx_md1],axis=0)
    md_HL_both_A2B = np.mean(md_HL_both_A2B[cue_on + start:end, cell_idx_md1], axis=0)
    md_HL_both_B2A = np.mean(md_HL_both_B2A[cue_on + start:end, cell_idx_md1], axis=0)

    md_HL_both_A_mean = np.mean(md_HL_both_A)
    md_HL_both_A2B_mean = np.mean(md_HL_both_A2B)
    md_HL_both_B2A_mean = np.mean(md_HL_both_B2A)
    print( md_HL_both_A_mean)

    md_HL_both_A_std = np.std(md_HL_both_A)/md_HL_both_A.shape[0]
    md_HL_both_A2B_std = np.std(md_HL_both_A2B)/md_HL_both_A.shape[0]
    md_HL_both_B2A_std = np.std(md_HL_both_B2A)/md_HL_both_A.shape[0]

    #'''
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.25, 0.2, 0.7, 0.7])

    x= np.arange(0,3,1)
    y = [md_HL_both_A_mean,md_HL_both_A2B_mean,md_HL_both_B2A_mean]
    print(x,y)
    plt.xticks([0,1,2], fontsize=11)

    yerr = [md_HL_both_A_std,md_HL_both_A2B_std,md_HL_both_B2A_std]


    plt.errorbar(x, y, yerr=yerr,fmt='-o',c ='tab:orange')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel('Faring rate', fontsize=12)
    plt.title(model_name_A2B, fontsize=6)


    plt.savefig(figure_path + model_name_A2B+'MD_switch_fr_A_B_B.png')
    plt.savefig(figure_path + model_name_A2B + 'MD_switch_fr_A_B_B.pdf')

    plt.show()
    #'''

def MD_switch_fr_A_B_B_cell_example(figure_path, data_path,hp, model_dir_1, model_dir_2,model_dir_3):
    data_path = os.path.join(data_path, 'MD_switch_fr_A_B_B_cell_example/')
    tools.mkdir_p(data_path)
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.99
    batch_size = 100

    context1 = 'con_A'
    context2 = 'con_A2B'
    context3 = 'con_A2B'
    #


    # pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                             model_dir=model_dir_1,
    #                                                             cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                             model_dir=model_dir_1,
    #                                                             cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context2,
    #                                                                 model_dir=model_dir_2,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context2,
    #                                                                 model_dir=model_dir_2,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context3,
    #                                                                 model_dir=model_dir_3,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context3,
    #                                                                 model_dir=model_dir_3,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # np.save(data_path+'md_HL_vis_A.npy',md_HL_vis_A)
    # np.save(data_path+'md_HL_vis_A2B.npy',md_HL_vis_A2B)
    # np.save(data_path + 'md_HL_vis_B2A.npy', md_HL_vis_B2A)
    # np.save(data_path + 'md_HL_aud_A.npy', md_HL_aud_A)
    # np.save(data_path + 'md_HL_aud_A2B.npy', md_HL_aud_A2B)
    # np.save(data_path + 'md_HL_aud_B2A.npy', md_HL_aud_B2A)





    md_HL_vis_A   = np.load(data_path + 'md_HL_vis_A.npy')
    md_HL_vis_A2B = np.load(data_path + 'md_HL_vis_A2B.npy')
    md_HL_vis_B2A = np.load(data_path + 'md_HL_vis_B2A.npy')
    md_HL_aud_A   = np.load(data_path + 'md_HL_aud_A.npy')
    md_HL_aud_A2B = np.load(data_path + 'md_HL_aud_A2B.npy')
    md_HL_aud_B2A = np.load(data_path + 'md_HL_aud_B2A.npy')

    ##### plot activity #################################
    start = 0
    end = response_on - 2
    cell_idx_md1 = [3]#[3,5,8,9,10,16]

    fig, axs = plt.subplots(3, 2, figsize=(3, 4))
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    y_md1 = 2.7  # * np.max(md_HL_vis_A2B[cue_on:stim_on, 0:18])
    y_md1_min = 0.4  # 0.9 * np.min(md_HL_vis_A2B[cue_on:stim_on, 0:18])

    for i in np.array(cell_idx_md1):
        axs[0, 0].plot(md_HL_vis_A[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[1, 0].plot(md_HL_vis_A2B[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[2, 0].plot(md_HL_vis_B2A[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[0, 0].legend(fontsize=5)
        axs[0, 0].set_title('vis', fontsize=7)
    for i in np.array(cell_idx_md1):
        axs[0, 1].plot(md_HL_aud_A[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[1, 1].plot(md_HL_aud_A2B[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[2, 1].plot(md_HL_aud_B2A[cue_on + start:end, i], c='tab:blue', label=str(i))
        axs[0, 1].set_title('aud', fontsize=7)
        # axs[0, 1].legend(fontsize=5)

    for i in range(3):
        for j in range(2):
            axs[i, j].spines[['left', 'right', 'top']].set_visible(False)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].axvspan(cue_off - 2 - 1 - start, stim_on - 2 - 1 - start, facecolor='#EBEBEB')

            axs[i, j].set_ylim([y_md1_min, y_md1])
            axs[i, j].set_ylim([y_md1_min, y_md1])
            # axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            # axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    #plt.savefig(figure_path + 'MD_switch_fr_A_B_B_cell_example.eps', format='eps', dpi=1000)
    plt.savefig(figure_path + 'MD_switch_fr_A_B_B_cell_example.png')

    plt.show()


def MD_switch_fr_A_Bfail_Bsucc_cell_example(figure_path, data_path, hp, model_dir_1, model_dir_2,model_dir_3,model_name_A2B):

    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.99
    batch_size = 200

    context1 = 'con_A'
    context2 = 'con_A2B'
    context3 = 'con_B2A'

    # pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                             model_dir=model_dir_1,
    #                                                             cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                             model_dir=model_dir_1,
    #                                                             cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context2,
    #                                                                 model_dir=model_dir_2,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context2,
    #                                                                 model_dir=model_dir_2,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context3,
    #                                                                 model_dir=model_dir_3,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context3,
    #                                                                 model_dir=model_dir_3,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # np.save(data_path+'md_HL_vis_A.npy',md_HL_vis_A)
    # np.save(data_path+'md_HL_vis_A2B.npy',md_HL_vis_A2B)
    # np.save(data_path + 'md_HL_vis_B2A.npy', md_HL_vis_B2A)
    # np.save(data_path + 'md_HL_aud_A.npy', md_HL_aud_A)
    # np.save(data_path + 'md_HL_aud_A2B.npy', md_HL_aud_A2B)
    # np.save(data_path + 'md_HL_aud_B2A.npy', md_HL_aud_B2A)

    md_HL_vis_A   = np.load(data_path + 'md_HL_vis_A.npy')
    md_HL_vis_A2B = np.load(data_path + 'md_HL_vis_A2B.npy')
    md_HL_vis_B2A = np.load(data_path + 'md_HL_vis_B2A.npy')
    md_HL_aud_A   = np.load(data_path + 'md_HL_aud_A.npy')
    md_HL_aud_A2B = np.load(data_path + 'md_HL_aud_A2B.npy')
    md_HL_aud_B2A = np.load(data_path + 'md_HL_aud_B2A.npy')

    ##### plot activity #################################
    start = 0
    end = response_on - 5

    for cell_idx in np.array([15]):
        cell_idx_md1 = [cell_idx]  # [3,5,8,9,10,16]

        fig, axs = plt.subplots(3, 2, figsize=(3, 4),sharey=True)
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
        y_md1 = 2.7  # * np.max(md_HL_vis_A2B[cue_on:stim_on, 0:18])
        y_md1_min = 0.4  # 0.9 * np.min(md_HL_vis_A2B[cue_on:stim_on, 0:18])

        for i in np.array(cell_idx_md1):
            axs[0, 0].plot(md_HL_vis_A[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[1, 0].plot(md_HL_vis_A2B[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[2, 0].plot(md_HL_vis_B2A[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[0, 0].legend(fontsize=5)
            axs[0, 0].set_title('vis', fontsize=7)
        for i in np.array(cell_idx_md1):
            axs[0, 1].plot(md_HL_aud_A[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[1, 1].plot(md_HL_aud_A2B[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[2, 1].plot(md_HL_aud_B2A[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[0, 1].set_title('aud', fontsize=7)
            # axs[0, 1].legend(fontsize=5)

        for i in range(3):
            for j in range(2):
                axs[i, j].spines[['left', 'right', 'top']].set_visible(False)
                # axs[i, j].set_xticks([])
                # axs[i, j].set_yticks([])
                axs[i, j].axvspan(cue_off - 2 - 1 - start, stim_on - 2 - 1 - start, facecolor='#EBEBEB')

                # axs[i, j].set_ylim([y_md1_min, y_md1])
                # axs[i, j].set_ylim([y_md1_min, y_md1])
                # axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
                # axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

        plt.savefig(figure_path + model_name_A2B+'_cell'+str(cell_idx)+'.pdf')
        plt.savefig(figure_path + model_name_A2B+'_cell'+str(cell_idx)+'.png')

        plt.show()



def MD_switch_fr_A_B_AA_cell_example(figure_path, data_path,hp, model_dir_1, model_dir_2,model_dir_3,model_name_A2B):
    data_path = os.path.join(data_path, 'MD_switch_fr_A_B_AA_cell_example/')
    tools.mkdir_p(data_path)
    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.99
    batch_size = 200

    context1 = 'con_A'
    context2 = 'con_A2B'
    context3 = 'con_B2A'

    # pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                             model_dir=model_dir_1,
    #                                                             cue=1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context1,
    #                                                             model_dir=model_dir_1,
    #                                                             cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context2,
    #                                                                 model_dir=model_dir_2,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context2,
    #                                                                 model_dir=model_dir_2,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context3,
    #                                                                 model_dir=model_dir_3,
    #                                                                 cue=-1, p_cohs=coh_HL, batch_size=batch_size)
    # pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context=context3,
    #                                                                 model_dir=model_dir_3,
    #                                                                 cue=1, p_cohs=coh_HL, batch_size=batch_size)
    #
    # np.save(data_path+'md_HL_vis_A.npy',md_HL_vis_A)
    # np.save(data_path+'md_HL_vis_A2B.npy',md_HL_vis_A2B)
    # np.save(data_path + 'md_HL_vis_B2A.npy', md_HL_vis_B2A)
    # np.save(data_path + 'md_HL_aud_A.npy', md_HL_aud_A)
    # np.save(data_path + 'md_HL_aud_A2B.npy', md_HL_aud_A2B)
    # np.save(data_path + 'md_HL_aud_B2A.npy', md_HL_aud_B2A)

    md_HL_vis_A   = np.load(data_path + 'md_HL_vis_A.npy')
    md_HL_vis_A2B = np.load(data_path + 'md_HL_vis_A2B.npy')
    md_HL_vis_B2A = np.load(data_path + 'md_HL_vis_B2A.npy')
    md_HL_aud_A   = np.load(data_path + 'md_HL_aud_A.npy')
    md_HL_aud_A2B = np.load(data_path + 'md_HL_aud_A2B.npy')
    md_HL_aud_B2A = np.load(data_path + 'md_HL_aud_B2A.npy')

    ##### plot activity #################################
    start = 0
    end = response_on - 5

    for cell_idx in range(9,10):
        cell_idx_md1 = [cell_idx]  # [3,5,8,9,10,16]

        fig, axs = plt.subplots(3, 2, figsize=(3, 4),sharey=True)
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
        y_md1 = 2.7  # * np.max(md_HL_vis_A2B[cue_on:stim_on, 0:18])
        y_md1_min = 0.4  # 0.9 * np.min(md_HL_vis_A2B[cue_on:stim_on, 0:18])

        for i in np.array(cell_idx_md1):
            axs[0, 0].plot(md_HL_vis_A[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[1, 0].plot(md_HL_vis_A2B[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[2, 0].plot(md_HL_vis_B2A[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[0, 0].legend(fontsize=5)
            axs[0, 0].set_title('vis', fontsize=7)
        for i in np.array(cell_idx_md1):
            axs[0, 1].plot(md_HL_aud_A[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[1, 1].plot(md_HL_aud_A2B[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[2, 1].plot(md_HL_aud_B2A[cue_on + start:end, i], c='tab:blue', label=str(i))
            axs[0, 1].set_title('aud', fontsize=7)
            # axs[0, 1].legend(fontsize=5)

        for i in range(3):
            for j in range(2):
                axs[i, j].spines[['left', 'right', 'top']].set_visible(False)
                # axs[i, j].set_xticks([])
                # axs[i, j].set_yticks([])
                axs[i, j].axvspan(cue_off - 2 - 1 - start, stim_on - 2 - 1 - start, facecolor='#EBEBEB')

                # axs[i, j].set_ylim([y_md1_min, y_md1])
                # axs[i, j].set_ylim([y_md1_min, y_md1])
                # axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
                # axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

        plt.savefig(figure_path + model_name_A2B+'_cell'+str(cell_idx)+'.pdf')

        #plt.savefig(figure_path + model_name_A2B+'_cell'+str(cell_idx)+'.png')

        plt.show()








