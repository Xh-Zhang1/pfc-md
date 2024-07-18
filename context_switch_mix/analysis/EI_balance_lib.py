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
import torch

import default
import tools
import run

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

def generate_test_trial(context_name,hp,model_dir,
                        cue=None,
                        p_coh=0.9,
                        batch_size=1,
                        c_vis=None,
                        c_aud=None):

    rng  = hp['rng']

    if cue is None:
        c_cue = hp['rng'].choice([1,-1], (batch_size,))
    else:
        c_cue = hp['rng'].choice([cue], (batch_size,))
    if c_vis is None:
        c_vis = hp['gamma_noise']*hp['rng'].choice([-0.2,0.2], (batch_size,))
    else:
        c_vis = hp['rng'].choice([c_vis], (batch_size,))

    if c_aud is None:
        c_aud = hp['gamma_noise']*hp['rng'].choice([-0.2,0.2], (batch_size,))
    else:
        c_aud = hp['rng'].choice([c_aud], (batch_size,))

    runnerObj = run.Runner(rule_name=context_name, hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='test')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            p_coh=p_coh,
                                            c_cue=c_cue,
                                            c_vis=c_vis,
                                            c_aud=c_aud)


    return trial_input, run_result

def get_neurons_activity_mode_test1(context_name,hp,context,model_dir,cue,p_cohs,batch_size):

    hp['switch_context']=context

    runnerObj = run.Runner(rule_name=context_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='test1')
    trial_input, run_result = runnerObj.run(batch_size=batch_size, c_cue=cue, p_coh=p_cohs)


    #### average value over batch_sizes for hidden state
    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
    firing_rate_list = list([firing_rate[:,i,:] for i in range(batch_size)])
    firing_rate_mean = np.mean(np.array(firing_rate_list),axis=0)
    firing_rate = firing_rate_mean
    #### MD
    firing_rate_md = run_result.firing_rate_md.detach().cpu().numpy()
    fr_MD_list = list([firing_rate_md[:,i,:] for i in range(batch_size)])
    fr_MD_mean = np.mean(np.array(fr_MD_list),axis=0)
    fr_MD = fr_MD_mean

    return firing_rate_mean,fr_MD


def get_weight_A(hp,model_dir):
    n_md = int(hp['n_md'])
    n_rnn= int(hp['n_rnn'])
    i_size_one = int((hp['n_rnn']*(1-hp['e_prop']))/3)#hidden_size - self.e_size
    i_size = 3*i_size_one
    pc = hp['n_rnn']-i_size
    e_size = pc + n_md


    runnerObj = run.Runner(rule_name='HL_task', hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='test1')
    weight = runnerObj.model.RNN_layer.h2h.weight.detach().numpy()
    mask = tools.mask_md_pfc_train_type8(hp=hp, md=n_md,pc=pc,i_size_one=i_size_one)

    effective_weights = np.abs(weight) * mask

    print('effective_weights',effective_weights[0:5,0])
    print(effective_weights.shape)

    mat_md1_pc = effective_weights[0:100, 256:256 + 18]
    mat_md1_pv = effective_weights[205 + 17 * 2:256, 256:256 + 18]

    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    # plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name, fontsize=8)
    #
    # axs[0].imshow(mat_md1_pc.T,aspect='auto')
    # axs[1].imshow(mat_md1_pv.T, aspect='auto')
    # plt.show()

    return effective_weights


def get_weight_A2B(hp,model_dir):
    n_md = int(hp['n_md'])
    n_rnn= int(hp['n_rnn'])
    i_size_one = int((hp['n_rnn']*(1-hp['e_prop']))/3)#hidden_size - self.e_size
    i_size = 3*i_size_one
    pc = hp['n_rnn']-i_size
    e_size = pc + n_md

    H,C = tools.weight_mask_A(hp, is_cuda=False)

    runnerObj = run.Runner(rule_name='HL_task', hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='test1')
    weight = runnerObj.model.RNN_layer.h2h.weight.detach()
    mask = tools.mask_md_pfc_train_type8(hp=hp, md=n_md,pc=pc,i_size_one=i_size_one)

    C = np.abs(C) * mask
    base_weight = torch.abs(weight) * mask
    effective_weights = base_weight * H + C


    print('effective_weights',effective_weights[0:5,0])

    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    # plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name, fontsize=8)
    #
    # mat_md1_pc = effective_weights[0:100, 256:256 + 18]
    # mat_md1_pv = effective_weights[205 + 17 * 2:256, 256:256 + 18]
    #
    # axs[0].imshow(mat_md1_pc.T, aspect='auto')
    # axs[1].imshow(mat_md1_pv.T, aspect='auto')
    # plt.show()
    return effective_weights.numpy()



def get_weight_B2A(hp,model_dir):
    n_md = int(hp['n_md'])
    n_rnn= int(hp['n_rnn'])
    i_size_one = int((hp['n_rnn']*(1-hp['e_prop']))/3)#hidden_size - self.e_size
    i_size = 3*i_size_one
    pc = hp['n_rnn']-i_size
    e_size = pc + n_md

    H,C = tools.weight_mask_A(hp, is_cuda=False)

    runnerObj = run.Runner(rule_name='HL_task', hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='test1')
    weight = runnerObj.model.RNN_layer.h2h.weight.detach()
    mask = tools.mask_md_pfc_train_type8(hp=hp, md=n_md,pc=pc,i_size_one=i_size_one)

    C = np.abs(C) * mask
    base_weight = torch.abs(weight) * mask
    effective_weights = base_weight * H + C


    print('effective_weights',effective_weights[0:5,0])

    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    # plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, left=0.1, hspace=0.3, wspace=0.3)
    # fig.suptitle(model_name, fontsize=8)
    #
    # mat_md1_pc = effective_weights[0:100, 256:256 + 18]
    # mat_md1_pv = effective_weights[205 + 17 * 2:256, 256:256 + 18]
    #
    # axs[0].imshow(mat_md1_pc.T, aspect='auto')
    # axs[1].imshow(mat_md1_pv.T, aspect='auto')
    # plt.show()

    return effective_weights.numpy()




def activity_three_context_md1(fig_path,hp,model_idx,i_dx,model_dir_A, model_dir_A2B,
                                model_dir_B2A, model_name_B2A,
                               start_cell):
    figure_path = os.path.join(fig_path, 'activity_three_context/')
    tools.mkdir_p(figure_path)

    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']



    coh_HL=0.92

    model_idx = model_idx
    idx = i_dx
    hp['loadA_idx'] = idx
    hp['loadB_idx'] = idx
    batch_size = 50

    #######rule1

    pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',model_dir=model_dir_A,
                                                    cue=1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A',model_dir=model_dir_A,
                                                    cue=-1, p_cohs=coh_HL,batch_size=batch_size)

    pfc_HL_vis_A2B, md_HL_vis_A2B =get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',model_dir=model_dir_A2B,
                                                    cue=-1, p_cohs=coh_HL,batch_size=batch_size)
    pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_A2B',model_dir=model_dir_A2B,
                                                    cue=1, p_cohs=coh_HL,batch_size=batch_size)
    #
    pfc_HL_vis_B2A, md_HL_vis_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_B2A',model_dir=model_dir_B2A,
                                                    cue=1, p_cohs=coh_HL, batch_size=batch_size)
    pfc_HL_aud_B2A, md_HL_aud_B2A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp, context='con_B2A',model_dir=model_dir_B2A,
                                                    cue=-1, p_cohs=coh_HL,batch_size=batch_size)

    np.save(figure_path+'pfc_HL_vis_A.npy',pfc_HL_vis_A)
    np.save(figure_path + 'pfc_HL_aud_A.npy', pfc_HL_aud_A)
    np.save(figure_path + 'pfc_HL_vis_A2B.npy', pfc_HL_vis_A2B)
    np.save(figure_path + 'pfc_HL_aud_A2B.npy', pfc_HL_aud_A2B)
    np.save(figure_path + 'pfc_HL_vis_B2A.npy', pfc_HL_vis_B2A)
    np.save(figure_path + 'pfc_HL_aud_B2A.npy', pfc_HL_aud_B2A)

    pfc_HL_vis_A = np.load(figure_path+'pfc_HL_vis_A.npy')
    pfc_HL_aud_A = np.load(figure_path+'pfc_HL_aud_A.npy')
    pfc_HL_vis_A2B = np.load(figure_path + 'pfc_HL_vis_A2B.npy')
    pfc_HL_aud_A2B = np.load(figure_path + 'pfc_HL_aud_A2B.npy')
    pfc_HL_vis_B2A = np.load(figure_path + 'pfc_HL_vis_B2A.npy')
    pfc_HL_aud_B2A = np.load(figure_path + 'pfc_HL_aud_B2A.npy')

    ###### get the weight and sort the weight
    effective_weights_A = get_weight_A(hp, model_dir_A)
    effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    effective_weights_B2A = get_weight_B2A(hp, model_dir_B2A)








    mat_md1_pc_A = effective_weights_A[0:205, 256:256 + 18]
    mat_md1_pc_A2B = effective_weights_A2B[0:205, 256:256 + 18]
    mat_md1_pc_B2A = effective_weights_B2A[0:205, 256:256 + 18]

    mat_md2_pc_A = effective_weights_A[0:205, 256 + 18:]
    mat_md2_pc_A2B = effective_weights_A2B[0:205, 256 + 18:]
    mat_md2_pc_B2A = effective_weights_B2A[0:205, 256 + 18:]

    diff_conA_conA2B_index_md1_pc = np.argsort(np.mean(mat_md1_pc_A2B - mat_md1_pc_A, axis=1))

    mat_md1_pc_A = mat_md1_pc_A[diff_conA_conA2B_index_md1_pc, :]
    mat_md1_pc_A2B = mat_md1_pc_A2B[diff_conA_conA2B_index_md1_pc, :]
    mat_md1_pc_B2A = mat_md1_pc_B2A[diff_conA_conA2B_index_md1_pc, :]

    mat_md2_pc_A = mat_md2_pc_A[diff_conA_conA2B_index_md1_pc, :]
    mat_md2_pc_A2B = mat_md2_pc_A2B[diff_conA_conA2B_index_md1_pc, :]
    mat_md2_pc_B2A = mat_md2_pc_B2A[diff_conA_conA2B_index_md1_pc, :]

    print(np.mean(mat_md1_pc_A2B - mat_md1_pc_A[0, :]))
    print(np.mean(mat_md1_pc_A2B - mat_md1_pc_A[1, :]))
    print(np.mean(mat_md1_pc_A2B - mat_md1_pc_A[2, :]))

    ##### plot activity #################################
    start = 0
    end = response_on - 2

    cell = range(start_cell, start_cell + 3)  # [158,197,199,203,204]#
    # for i in np.array(cell):
    #     print('**********',i)
    #     print(mat_md1_pc_A[i, :])
    #     print(mat_md1_pc_A2B[i,:])
    #     print(mat_md1_pc_B2A[i,:])
    #
    #     fig = plt.figure(figsize=(3.0, 2.6))
    #     ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])
    #     plt.plot(mat_md1_pc_A[i, :],'o-',label='A')
    #     plt.plot(mat_md1_pc_A2B[i, :],'o-',label='B')
    #     plt.plot(mat_md1_pc_B2A[i, :],'o-',label='AA')
    #     plt.legend()
    #     plt.title(str(i))
    #     plt.show()

    cell_idx_exc = diff_conA_conA2B_index_md1_pc[
        cell]  # max_idx_exc[0:205]#np.array([148, 188, 147, 24])#range(0,20,1)#max_idx_exc[0:205]#range(0,20,1)
    print('cell_idx_exc', cell_idx_exc)
    cell_idx_md1 = range(0, 9, 1)  # np.array([1,3,9,11])#range(0,18,1)  # max_idx_md1[0:20]
    cell_idx_inh = range(205 + 17 * 2, 256, 1)  # max_idx_md1[0:20]

    y_exc = 1.2  # 1.5* np.max(pfc_HL_vis_A[cue_on:stim_on, cell_idx_exc])
    y_inh = 1.2 * np.max(pfc_HL_vis_A[cue_on:stim_on - 2, 205 + 17 * 2:256])

    fig, axs = plt.subplots(3, 4, figsize=(8, 5))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.97, left=0.05, hspace=0.2, wspace=0.2)
    fig.suptitle(model_name_B2A, fontsize=8)

    k = -1
    for i in np.array(cell):
        k += 1
        axs[k, 2].plot(mat_md1_pc_A[i, :], 'o-', label='1', c='tab:blue')
        axs[k, 2].plot(mat_md1_pc_A2B[i, :], 'o-', label='2', c='tab:orange')
        axs[k, 2].plot(mat_md1_pc_B2A[i, :], 'o-', label='11', c='tab:green', alpha=0.3)
        axs[k, 2].set_title(cell[k])
        axs[k, 2].set_ylim([0, 0.07])
        axs[0, 2].legend(fontsize=5)

        axs[k, 3].plot(mat_md2_pc_A[i, :], 'o-', label='1', c='tab:blue')
        axs[k, 3].plot(mat_md2_pc_A2B[i, :], 'o-', label='2', c='tab:orange')
        axs[k, 3].plot(mat_md2_pc_B2A[i, :], 'o-', label='11', c='tab:green', alpha=0.3)
        axs[k, 3].set_title(cell[k])
        axs[k, 3].set_ylim([0, 0.07])
        axs[0, 3].legend(fontsize=5)

    j = -1
    for i in np.array(cell_idx_exc):
        j += 1
        axs[0, 0].plot(pfc_HL_vis_A[cue_on + start:end, i], label=str(cell[j]))
        axs[1, 0].plot(pfc_HL_aud_A2B[cue_on + start:end, i], label=str(cell[j]))
        axs[2, 0].plot(pfc_HL_vis_B2A[cue_on + start:end, i], label=str(cell[j]))
        axs[0, 0].legend(fontsize=5)
        axs[0, 0].set_title('LP>HP')

    for i in np.array(cell_idx_exc):
        axs[0, 1].plot(pfc_HL_aud_A[cue_on + start:end, i], label=str(i))
        axs[1, 1].plot(pfc_HL_vis_A2B[cue_on + start:end, i], label=str(i))
        axs[2, 1].plot(pfc_HL_aud_B2A[cue_on + start:end, i], label=str(i))
        axs[0, 1].legend(fontsize=5)
        axs[0, 1].set_title('LP<HP')

    # for i in np.array(cell_idx_inh):
    #     axs[0,2].plot(pfc_HL_vis_A[cue_on + start:end, i], label=str(i))
    #     axs[1,2].plot(pfc_HL_vis_A2B[cue_on + start:end, i], label=str(i))
    #     axs[2,2].plot(pfc_HL_vis_B2A[cue_on + start:end, i], label=str(i))
    #     #axs[0,1].legend(fontsize=5)

    for i in range(3):
        for j in range(2):
            axs[i, 0].set_ylim([0, y_exc])
            axs[i, 1].set_ylim([0, y_exc])
            axs[i, 1].set_yticks([])
            axs[i, j].set_xticks([])

            axs[i, j].axvspan(cue_off - 2 - 1 - start, cue_off - 2 - 1 - start, color='grey', label='cue_off')
            axs[i, j].axvspan(stim_on - 2 - 1 - start, stim_on - 2 - 1 - start, color='grey', label='stim_on')

    plt.savefig(figure_path + model_name_B2A + '_' + str(start_cell) + '.png')
    plt.show()
    #"""



def ratio_EI_vis(data_path,hp,model_dir_A, model_dir_A2B,
                                model_dir_A2B_fail):


    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.92
    batch_size = 100

    # pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                                                                          context='con_A', model_dir=model_dir_A,
    #                                                                          cue=1, p_cohs=coh_HL,
    #                                                                          batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                                                                          context='con_A', model_dir=model_dir_A,
    #                                                                          cue=-1, p_cohs=coh_HL,
    #                                                                          batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                                                                              context='con_A2B',
    #                                                                              model_dir=model_dir_A2B,
    #                                                                              cue=-1, p_cohs=coh_HL,
    #                                                                              batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                                                                              context='con_A2B',
    #                                                                              model_dir=model_dir_A2B,
    #                                                                              cue=1, p_cohs=coh_HL,
    #                                                                              batch_size=batch_size)
    #
    # pfc_HL_vis_A2B_fail, md_HL_vis_A2B_fail = get_neurons_activity_mode_test1(context_name='HL_task',
    #                                                                                        hp=hp, context='con_A2B',
    #                                                                                        model_dir=model_dir_A2B_fail,
    #                                                                                        cue=-1, p_cohs=coh_HL,
    #                                                                                        batch_size=batch_size)
    # pfc_HL_aud_A2B_fail, md_HL_aud_A2B_fail = get_neurons_activity_mode_test1(context_name='HL_task',
    #                                                                                        hp=hp, context='con_A2B',
    #                                                                                        model_dir=model_dir_A2B_fail,
    #                                                                                        cue=1, p_cohs=coh_HL,
    #                                                                                        batch_size=batch_size)
    #
    # np.save(data_path + 'pfc_HL_vis_A.npy', pfc_HL_vis_A)
    # np.save(data_path + 'pfc_HL_aud_A.npy', pfc_HL_aud_A)
    # np.save(data_path + 'pfc_HL_vis_A2B.npy', pfc_HL_vis_A2B)
    # np.save(data_path + 'pfc_HL_aud_A2B.npy', pfc_HL_aud_A2B)
    # np.save(data_path + 'pfc_HL_vis_A2B_fail.npy', pfc_HL_vis_A2B_fail)
    # np.save(data_path + 'pfc_HL_aud_A2B_fail.npy', pfc_HL_aud_A2B_fail)
    #
    # np.save(data_path + 'md_HL_vis_A.npy', md_HL_vis_A)
    # np.save(data_path + 'md_HL_aud_A.npy', md_HL_aud_A)
    # np.save(data_path + 'md_HL_vis_A2B.npy', md_HL_vis_A2B)
    # np.save(data_path + 'md_HL_aud_A2B.npy', md_HL_aud_A2B)
    # np.save(data_path + 'md_HL_vis_A2B_fail.npy', md_HL_vis_A2B_fail)
    # np.save(data_path + 'md_HL_aud_A2B_fail.npy', md_HL_aud_A2B_fail)


    pfc_HL_vis_A = np.load(data_path + 'pfc_HL_vis_A.npy')
    pfc_HL_aud_A = np.load(data_path + 'pfc_HL_aud_A.npy')
    pfc_HL_vis_A2B = np.load(data_path + 'pfc_HL_vis_A2B.npy')
    pfc_HL_aud_A2B = np.load(data_path + 'pfc_HL_aud_A2B.npy')
    pfc_HL_vis_A2B_fail = np.load(data_path + 'pfc_HL_vis_A2B_fail.npy')
    pfc_HL_aud_A2B_fail = np.load(data_path + 'pfc_HL_aud_A2B_fail.npy')

    md_HL_vis_A = np.load(data_path + 'md_HL_vis_A.npy')
    md_HL_aud_A = np.load(data_path + 'md_HL_aud_A.npy')
    md_HL_vis_A2B = np.load(data_path + 'md_HL_vis_A2B.npy')
    md_HL_aud_A2B = np.load(data_path + 'md_HL_aud_A2B.npy')
    md_HL_vis_A2B_fail = np.load(data_path + 'md_HL_vis_A2B_fail.npy')
    md_HL_aud_A2B_fail = np.load(data_path + 'md_HL_aud_A2B_fail.npy')








    # effective_weights_A = get_weight_A(hp, model_dir_A)
    # effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    # effective_weights_A2B_fail = get_weight_A2B(hp, model_dir_A2B_fail)
    # np.save(data_path + 'effective_weights_A.npy', effective_weights_A)
    # np.save(data_path + 'effective_weights_A2B.npy', effective_weights_A2B)
    # np.save(data_path + 'effective_weights_A2B_fail.npy', effective_weights_A2B_fail)

    effective_weights_A = np.load(data_path + 'effective_weights_A.npy')
    effective_weights_A2B = np.load(data_path + 'effective_weights_A2B.npy')
    effective_weights_A2B_fail = np.load(data_path + 'effective_weights_A2B_fail.npy')







    mat_md1_pc_A = effective_weights_A[0:205, 256:256 + 18]
    mat_md1_pc_A2B = effective_weights_A2B[0:205, 256:256 + 18]

    mat_md2_pc_A = effective_weights_A[0:205, 256 + 18:]
    mat_md2_pc_A2B = effective_weights_A2B[0:205, 256 + 18:]

    diff_conA_conA2B_index_md1_pc = np.argsort(np.mean(mat_md1_pc_A2B - mat_md1_pc_A, axis=1))
    print('diff_conA_conA2B_index_md1_pc',diff_conA_conA2B_index_md1_pc.shape,diff_conA_conA2B_index_md1_pc)



    Sum_exc_As = []
    Sum_inh_As = []
    Sum_exc_A2Bs = []
    Sum_inh_A2Bs = []
    Sum_exc_A2Bs_fail = []
    Sum_inh_A2Bs_fail = []

    start = cue_off
    end = stim_on

    cell = range(205)

    for i_select in np.array(cell):
        ##### from md1
        mat_md1_exc_A = effective_weights_A[i_select, 256:256 + 18]
        fr_md1_vis_A_mean = np.mean(md_HL_vis_A[start:end, 0:18], axis=0)
        input_from_md1 = np.mean(mat_md1_exc_A * fr_md1_vis_A_mean)
        # input_from_md1 = np.mean(mat_md1_exc_A)

        ##### from md2
        mat_md2_exc_A = effective_weights_A[i_select, 256 + 18:]
        fr_md2_vis_A_mean = np.mean(md_HL_vis_A[start:end, 18:], axis=0)
        input_from_md2 = np.mean(mat_md2_exc_A * fr_md2_vis_A_mean)
        # input_from_md2 = np.mean(mat_md2_exc_A ,axis=0)

        ##### from inh
        mat_Inh_Exc_A = effective_weights_A[i_select, 205:256]
        fr_Inh_vis_A_mean = np.mean(pfc_HL_vis_A[start:end, 205:256], axis=0)
        input_from_inh = np.mean(mat_Inh_Exc_A * fr_Inh_vis_A_mean)
        # input_from_inh = np.mean(mat_Inh_Exc_A,axis=0)
        ##### from exc
        mat_Exc_Exc_A = effective_weights_A[i_select, 0:205]
        fr_Exc_vis_A_mean = np.mean(pfc_HL_vis_A[start:end, 0:205], axis=0)
        input_from_Exc = np.mean(mat_Exc_Exc_A * fr_Exc_vis_A_mean)
        # input_from_Exc = np.mean(mat_Exc_Exc_A,axis=0)

        Sum_A_exc = input_from_md1 + input_from_md2 + input_from_Exc

        Sum_exc_As.append(Sum_A_exc)
        Sum_inh_As.append(np.abs(input_from_inh))

    for i_select in np.array(cell):
        ##### from md1
        mat_md1_exc_A2B = effective_weights_A2B[i_select, 256:256 + 18]
        fr_md1_vis_A2B_mean = np.mean(md_HL_vis_A2B[start:end, 0:18], axis=0)
        input_from_md1 = np.mean(mat_md1_exc_A2B * fr_md1_vis_A2B_mean)
        # input_from_md1 = np.mean(mat_md1_exc_A2B,axis=0)

        ##### from md2
        mat_md2_exc_A2B = effective_weights_A2B[i_select, 256 + 18:]
        fr_md2_vis_A2B_mean = np.mean(md_HL_vis_A2B[start:end, 18:], axis=0)
        input_from_md2 = np.mean(mat_md2_exc_A2B * fr_md2_vis_A2B_mean)
        # input_from_md2 = np.mean(mat_md2_exc_A2B,axis=0)

        ##### from inh
        mat_Inh_Exc_A2B = effective_weights_A2B[i_select, 205:256]
        fr_Inh_vis_A2B_mean = np.mean(pfc_HL_vis_A2B[start:end, 205:256], axis=0)
        input_from_inh = np.mean(mat_Inh_Exc_A2B * fr_Inh_vis_A2B_mean)
        # input_from_inh = np.mean(mat_Inh_Exc_A2B,axis= 0)

        ##### from exc
        mat_Exc_Exc_A2B = effective_weights_A2B[i_select, 0:205]
        fr_Exc_vis_A2B_mean = np.mean(pfc_HL_vis_A2B[start:end, 0:205], axis=0)
        input_from_Exc = np.mean(mat_Exc_Exc_A2B * fr_Exc_vis_A2B_mean)
        # input_from_Exc = np.mean(mat_Exc_Exc_A2B,axis=0)

        Sum_A2B_exc = input_from_md1 + input_from_md2 + input_from_Exc

        Sum_exc_A2Bs.append(Sum_A2B_exc)
        Sum_inh_A2Bs.append(np.abs(input_from_inh))

    for i_select in np.array(cell):
        ##### from md1
        mat_md1_exc_A2B_fail = effective_weights_A2B_fail[i_select, 256:256 + 18]
        fr_md1_vis_A2B_mean_fail = np.mean(md_HL_vis_A2B_fail[start:end, 0:18], axis=0)
        input_from_md1_fail = np.mean(mat_md1_exc_A2B_fail * fr_md1_vis_A2B_mean_fail)
        # input_from_md1_fail = np.mean(mat_md1_exc_A2B_fail,axis=0)

        ##### from md2
        mat_md2_exc_A2B_fail = effective_weights_A2B_fail[i_select, 256 + 18:]
        fr_md2_vis_A2B_mean_fail = np.mean(md_HL_vis_A2B_fail[start:end, 18:], axis=0)
        input_from_md2_fail = np.mean(mat_md2_exc_A2B_fail * fr_md2_vis_A2B_mean_fail)
        # input_from_md2_fail = np.mean(mat_md2_exc_A2B_fail,axis=0)

        ##### from inh
        mat_Inh_Exc_A2B_fail = effective_weights_A2B_fail[i_select, 205:256]
        fr_Inh_vis_A2B_mean_fail = np.mean(pfc_HL_vis_A2B_fail[start:end, 205:256], axis=0)
        input_from_inh_fail = np.mean(mat_Inh_Exc_A2B_fail * fr_Inh_vis_A2B_mean_fail)
        # input_from_inh_fail = np.mean(mat_Inh_Exc_A2B_fail,axis=0)

        ##### from exc
        mat_Exc_Exc_A2B_fail = effective_weights_A2B_fail[i_select, 0:205]
        fr_Exc_vis_A2B_mean_fail = np.mean(pfc_HL_vis_A2B_fail[start:end, 0:205], axis=0)
        input_from_Exc_fail = np.mean(mat_Exc_Exc_A2B_fail * fr_Exc_vis_A2B_mean_fail)
        # input_from_Exc_fail = np.mean(mat_Exc_Exc_A2B_fail)

        Sum_A2B_exc_fail = input_from_md1_fail + input_from_md2_fail + input_from_Exc_fail

        Sum_exc_A2Bs_fail.append(Sum_A2B_exc_fail)
        Sum_inh_A2Bs_fail.append(np.abs(input_from_inh_fail))

    ratio_A = np.array(Sum_exc_As) / np.array(Sum_inh_As)
    ratio_A2B = np.array(Sum_exc_A2Bs) / np.array(Sum_inh_A2Bs)
    ratio_A2B_fail = np.array(Sum_exc_A2Bs_fail) / np.array(Sum_inh_A2Bs_fail)

    # find cell
    # select_ratio_A=[]
    # select_ratio_A2B = []
    # for i in range(205):
    #     if 1.3>ratio_A[i]>0.6 and ratio_A[i]-ratio_A2B[i]>0.15:
    #         select_ratio_A.append(i)
    #     if 1.3>ratio_A2B[i]>0.6 and ratio_A2B[i]-ratio_A[i]>0.15:
    #         select_ratio_A2B.append(i)
    # print('select_ratio_A2B',select_ratio_A2B)
    #############################
    fig1 = plt.figure(figsize=(2.7, 2.7))
    ax1 = fig1.add_axes([0.23, 0.22, 0.7, 0.7])
    s = 45
    # colors_1 = sns.color_palette("YlOrBr", 10)
    # colors_2 = sns.color_palette("light:b", 10)
    colors_0 = sns.color_palette("hls", 8)
    #plt.scatter(ratio_A, ratio_A2B, marker="o", s=s, color='grey', edgecolors='white')

    select_ratio_A = diff_conA_conA2B_index_md1_pc[0:33]#[179, 12, 131, 38, 145, 195]
    select_ratio_A2B = [13, 44, 106, 170, 143]
    front = diff_conA_conA2B_index_md1_pc[0:30]
    media = diff_conA_conA2B_index_md1_pc[30:100]
    last = diff_conA_conA2B_index_md1_pc[100:]
    for i in np.array(front):
        plt.scatter(ratio_A[i], ratio_A2B[i], marker="o", s=40, c='tab:blue', label=str(i), edgecolors='white')

    for i in np.array(last):
        plt.scatter(ratio_A[i], ratio_A2B[i], marker="o", s=40, c='tab:red', label=str(i), edgecolors='white')

    for i in np.array(media):
        plt.scatter(ratio_A[i], ratio_A2B[i], marker="o", s=40, c='tab:green', label=str(i), edgecolors='white')


    # for i in np.array(select_ratio_A):
    #     plt.scatter(ratio_A[i], ratio_A2B[i], marker="X", s=40, c='#EE6363', label=str(i))
    # for i in np.array(select_ratio_A2B):
    #     plt.scatter(ratio_A[i], ratio_A2B[i], marker="X", s=40, c='#4682B4', label=str(i))

    max = 1.5
    plt.plot([0, max], [0, max], color='grey')
    plt.xlim([-0.001, max])
    plt.ylim([-0.001, max])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('E/I (con_A)', fontsize=12)
    plt.ylabel('E/I (con_B)', fontsize=12)
    plt.title(str(start) + '_' + str(end), fontsize=5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()

    #########################
    fig1 = plt.figure(figsize=(2.7, 2.7))
    ax1 = fig1.add_axes([0.23, 0.22, 0.7, 0.7])
    s = 45
    # colors_1 = sns.color_palette("YlOrBr", 10)
    # colors_2 = sns.color_palette("light:b", 10)
    colors_0 = sns.color_palette("hls", 8)
    # plt.scatter(ratio_A, ratio_A2B, marker="o", s=s, color='grey', edgecolors='white')

    select_ratio_A = diff_conA_conA2B_index_md1_pc[0:33]  # [179, 12, 131, 38, 145, 195]
    select_ratio_A2B = [13, 44, 106, 170, 143]
    front = diff_conA_conA2B_index_md1_pc[0:30]
    media = diff_conA_conA2B_index_md1_pc[30:100]
    last = diff_conA_conA2B_index_md1_pc[100:]
    for i in np.array(front):
        plt.scatter(ratio_A[i], ratio_A2B_fail[i], marker="o", s=40, c='tab:blue', label=str(i), edgecolors='white')

    for i in np.array(last):
        plt.scatter(ratio_A[i], ratio_A2B_fail[i], marker="o", s=40, c='tab:red', label=str(i), edgecolors='white')

    for i in np.array(media):
        plt.scatter(ratio_A[i], ratio_A2B_fail[i], marker="o", s=40, c='tab:green', label=str(i), edgecolors='white')

    # for i in np.array(select_ratio_A):
    #     plt.scatter(ratio_A[i], ratio_A2B[i], marker="X", s=40, c='#EE6363', label=str(i))
    # for i in np.array(select_ratio_A2B):
    #     plt.scatter(ratio_A[i], ratio_A2B[i], marker="X", s=40, c='#4682B4', label=str(i))

    max = 1.5
    plt.plot([0, max], [0, max], color='grey')
    plt.xlim([-0.001, max])
    plt.ylim([-0.001, max])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('E/I (con_A)', fontsize=12)
    plt.ylabel('E/I (con_B_fail)', fontsize=12)
    plt.title('fail'+str(start) + '_' + str(end), fontsize=5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


def ratio_EI_aud(data_path,hp,model_dir_A, model_dir_A2B,
                                model_dir_A2B_fail):


    epoch = get_epoch(hp=hp)
    cue_on = epoch['cue_on']
    cue_off = epoch['cue_off']
    stim_on = epoch['stim_on']
    response_on = epoch['response_on']

    coh_HL = 0.92
    batch_size = 100

    # pfc_HL_vis_A, md_HL_vis_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                                                                          context='con_A', model_dir=model_dir_A,
    #                                                                          cue=1, p_cohs=coh_HL,
    #                                                                          batch_size=batch_size)
    # pfc_HL_aud_A, md_HL_aud_A = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                                                                          context='con_A', model_dir=model_dir_A,
    #                                                                          cue=-1, p_cohs=coh_HL,
    #                                                                          batch_size=batch_size)
    #
    # pfc_HL_vis_A2B, md_HL_vis_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                                                                              context='con_A2B',
    #                                                                              model_dir=model_dir_A2B,
    #                                                                              cue=-1, p_cohs=coh_HL,
    #                                                                              batch_size=batch_size)
    # pfc_HL_aud_A2B, md_HL_aud_A2B = get_neurons_activity_mode_test1(context_name='HL_task', hp=hp,
    #                                                                              context='con_A2B',
    #                                                                              model_dir=model_dir_A2B,
    #                                                                              cue=1, p_cohs=coh_HL,
    #                                                                              batch_size=batch_size)
    #
    # pfc_HL_vis_A2B_fail, md_HL_vis_A2B_fail = get_neurons_activity_mode_test1(context_name='HL_task',
    #                                                                                        hp=hp, context='con_A2B',
    #                                                                                        model_dir=model_dir_A2B_fail,
    #                                                                                        cue=-1, p_cohs=coh_HL,
    #                                                                                        batch_size=batch_size)
    # pfc_HL_aud_A2B_fail, md_HL_aud_A2B_fail = get_neurons_activity_mode_test1(context_name='HL_task',
    #                                                                                        hp=hp, context='con_A2B',
    #                                                                                        model_dir=model_dir_A2B_fail,
    #                                                                                        cue=1, p_cohs=coh_HL,
    #                                                                                        batch_size=batch_size)
    #
    # np.save(data_path + 'pfc_HL_vis_A.npy', pfc_HL_vis_A)
    # np.save(data_path + 'pfc_HL_aud_A.npy', pfc_HL_aud_A)
    # np.save(data_path + 'pfc_HL_vis_A2B.npy', pfc_HL_vis_A2B)
    # np.save(data_path + 'pfc_HL_aud_A2B.npy', pfc_HL_aud_A2B)
    # np.save(data_path + 'pfc_HL_vis_A2B_fail.npy', pfc_HL_vis_A2B_fail)
    # np.save(data_path + 'pfc_HL_aud_A2B_fail.npy', pfc_HL_aud_A2B_fail)
    #
    # np.save(data_path + 'md_HL_vis_A.npy', md_HL_vis_A)
    # np.save(data_path + 'md_HL_aud_A.npy', md_HL_aud_A)
    # np.save(data_path + 'md_HL_vis_A2B.npy', md_HL_vis_A2B)
    # np.save(data_path + 'md_HL_aud_A2B.npy', md_HL_aud_A2B)
    # np.save(data_path + 'md_HL_vis_A2B_fail.npy', md_HL_vis_A2B_fail)
    # np.save(data_path + 'md_HL_aud_A2B_fail.npy', md_HL_aud_A2B_fail)

    pfc_HL_vis_A = np.load(data_path + 'pfc_HL_vis_A.npy')
    pfc_HL_aud_A = np.load(data_path + 'pfc_HL_aud_A.npy')
    pfc_HL_vis_A2B = np.load(data_path + 'pfc_HL_vis_A2B.npy')
    pfc_HL_aud_A2B = np.load(data_path + 'pfc_HL_aud_A2B.npy')
    pfc_HL_vis_A2B_fail = np.load(data_path + 'pfc_HL_vis_A2B_fail.npy')
    pfc_HL_aud_A2B_fail = np.load(data_path + 'pfc_HL_aud_A2B_fail.npy')

    md_HL_vis_A = np.load(data_path + 'md_HL_vis_A.npy')
    md_HL_aud_A = np.load(data_path + 'md_HL_aud_A.npy')
    md_HL_vis_A2B = np.load(data_path + 'md_HL_vis_A2B.npy')
    md_HL_aud_A2B = np.load(data_path + 'md_HL_aud_A2B.npy')
    md_HL_vis_A2B_fail = np.load(data_path + 'md_HL_vis_A2B_fail.npy')
    md_HL_aud_A2B_fail = np.load(data_path + 'md_HL_aud_A2B_fail.npy')


    # effective_weights_A = get_weight_A(hp, model_dir_A)
    # effective_weights_A2B = get_weight_A2B(hp, model_dir_A2B)
    # effective_weights_A2B_fail = get_weight_A2B(hp, model_dir_A2B_fail)
    #
    # np.save(data_path + 'effective_weights_A.npy', effective_weights_A)
    # np.save(data_path + 'effective_weights_A2B.npy', effective_weights_A2B)
    # np.save(data_path + 'effective_weights_A2B_fail.npy', effective_weights_A2B_fail)

    effective_weights_A = np.load(data_path + 'effective_weights_A.npy')
    effective_weights_A2B = np.load(data_path + 'effective_weights_A2B.npy')
    effective_weights_A2B_fail = np.load(data_path + 'effective_weights_A2B_fail.npy')


    mat_md1_pc_A = effective_weights_A[0:205, 256:256 + 18]
    mat_md1_pc_A2B = effective_weights_A2B[0:205, 256:256 + 18]

    mat_md2_pc_A = effective_weights_A[0:205, 256 + 18:]
    mat_md2_pc_A2B = effective_weights_A2B[0:205, 256 + 18:]

    diff_conA_conA2B_index_md1_pc = np.argsort(np.mean(mat_md1_pc_A2B - mat_md1_pc_A, axis=1))
    print('diff_conA_conA2B_index_md1_pc',diff_conA_conA2B_index_md1_pc.shape,diff_conA_conA2B_index_md1_pc)



    Sum_exc_As = []
    Sum_inh_As = []
    Sum_exc_A2Bs = []
    Sum_inh_A2Bs = []
    Sum_exc_A2Bs_fail = []
    Sum_inh_A2Bs_fail = []

    start = cue_off
    end = stim_on

    cell = range(205)

    for i_select in np.array(cell):
        ##### from md1
        mat_md1_exc_A = effective_weights_A[i_select, 256:256 + 18]
        fr_md1_vis_A_mean = np.mean(md_HL_vis_A[start:end, 0:18], axis=0)
        input_from_md1 = np.mean(mat_md1_exc_A * fr_md1_vis_A_mean)
        # input_from_md1 = np.mean(mat_md1_exc_A)

        ##### from md2
        mat_md2_exc_A = effective_weights_A[i_select, 256 + 18:]
        fr_md2_vis_A_mean = np.mean(md_HL_vis_A[start:end, 18:], axis=0)
        input_from_md2 = np.mean(mat_md2_exc_A * fr_md2_vis_A_mean)
        # input_from_md2 = np.mean(mat_md2_exc_A ,axis=0)

        ##### from inh
        mat_Inh_Exc_A = effective_weights_A[i_select, 205:256]
        fr_Inh_vis_A_mean = np.mean(pfc_HL_vis_A[start:end, 205:256], axis=0)
        input_from_inh = np.mean(mat_Inh_Exc_A * fr_Inh_vis_A_mean)
        # input_from_inh = np.mean(mat_Inh_Exc_A,axis=0)
        ##### from exc
        mat_Exc_Exc_A = effective_weights_A[i_select, 0:205]
        fr_Exc_vis_A_mean = np.mean(pfc_HL_vis_A[start:end, 0:205], axis=0)
        input_from_Exc = np.mean(mat_Exc_Exc_A * fr_Exc_vis_A_mean)
        # input_from_Exc = np.mean(mat_Exc_Exc_A,axis=0)

        Sum_A_exc = input_from_md1 + input_from_md2 + input_from_Exc

        Sum_exc_As.append(Sum_A_exc)
        Sum_inh_As.append(np.abs(input_from_inh))

    for i_select in np.array(cell):
        ##### from md1
        mat_md1_exc_A2B = effective_weights_A2B[i_select, 256:256 + 18]
        fr_md1_vis_A2B_mean = np.mean(md_HL_vis_A2B[start:end, 0:18], axis=0)
        input_from_md1 = np.mean(mat_md1_exc_A2B * fr_md1_vis_A2B_mean)
        # input_from_md1 = np.mean(mat_md1_exc_A2B,axis=0)

        ##### from md2
        mat_md2_exc_A2B = effective_weights_A2B[i_select, 256 + 18:]
        fr_md2_vis_A2B_mean = np.mean(md_HL_vis_A2B[start:end, 18:], axis=0)
        input_from_md2 = np.mean(mat_md2_exc_A2B * fr_md2_vis_A2B_mean)
        # input_from_md2 = np.mean(mat_md2_exc_A2B,axis=0)

        ##### from inh
        mat_Inh_Exc_A2B = effective_weights_A2B[i_select, 205:256]
        fr_Inh_vis_A2B_mean = np.mean(pfc_HL_vis_A2B[start:end, 205:256], axis=0)
        input_from_inh = np.mean(mat_Inh_Exc_A2B * fr_Inh_vis_A2B_mean)
        # input_from_inh = np.mean(mat_Inh_Exc_A2B,axis= 0)

        ##### from exc
        mat_Exc_Exc_A2B = effective_weights_A2B[i_select, 0:205]
        fr_Exc_vis_A2B_mean = np.mean(pfc_HL_vis_A2B[start:end, 0:205], axis=0)
        input_from_Exc = np.mean(mat_Exc_Exc_A2B * fr_Exc_vis_A2B_mean)
        # input_from_Exc = np.mean(mat_Exc_Exc_A2B,axis=0)

        Sum_A2B_exc = input_from_md1 + input_from_md2 + input_from_Exc

        Sum_exc_A2Bs.append(Sum_A2B_exc)
        Sum_inh_A2Bs.append(np.abs(input_from_inh))

    for i_select in np.array(cell):
        ##### from md1
        mat_md1_exc_A2B_fail = effective_weights_A2B_fail[i_select, 256:256 + 18]
        fr_md1_vis_A2B_mean_fail = np.mean(md_HL_vis_A2B_fail[start:end, 0:18], axis=0)
        input_from_md1_fail = np.mean(mat_md1_exc_A2B_fail * fr_md1_vis_A2B_mean_fail)
        # input_from_md1_fail = np.mean(mat_md1_exc_A2B_fail,axis=0)

        ##### from md2
        mat_md2_exc_A2B_fail = effective_weights_A2B_fail[i_select, 256 + 18:]
        fr_md2_vis_A2B_mean_fail = np.mean(md_HL_vis_A2B_fail[start:end, 18:], axis=0)
        input_from_md2_fail = np.mean(mat_md2_exc_A2B_fail * fr_md2_vis_A2B_mean_fail)
        # input_from_md2_fail = np.mean(mat_md2_exc_A2B_fail,axis=0)

        ##### from inh
        mat_Inh_Exc_A2B_fail = effective_weights_A2B_fail[i_select, 205:256]
        fr_Inh_vis_A2B_mean_fail = np.mean(pfc_HL_vis_A2B_fail[start:end, 205:256], axis=0)
        input_from_inh_fail = np.mean(mat_Inh_Exc_A2B_fail * fr_Inh_vis_A2B_mean_fail)
        # input_from_inh_fail = np.mean(mat_Inh_Exc_A2B_fail,axis=0)

        ##### from exc
        mat_Exc_Exc_A2B_fail = effective_weights_A2B_fail[i_select, 0:205]
        fr_Exc_vis_A2B_mean_fail = np.mean(pfc_HL_vis_A2B_fail[start:end, 0:205], axis=0)
        input_from_Exc_fail = np.mean(mat_Exc_Exc_A2B_fail * fr_Exc_vis_A2B_mean_fail)
        # input_from_Exc_fail = np.mean(mat_Exc_Exc_A2B_fail)

        Sum_A2B_exc_fail = input_from_md1_fail + input_from_md2_fail + input_from_Exc_fail

        Sum_exc_A2Bs_fail.append(Sum_A2B_exc_fail)
        Sum_inh_A2Bs_fail.append(np.abs(input_from_inh_fail))

    ratio_A = np.array(Sum_exc_As) / np.array(Sum_inh_As)
    ratio_A2B = np.array(Sum_exc_A2Bs) / np.array(Sum_inh_A2Bs)
    ratio_A2B_fail = np.array(Sum_exc_A2Bs_fail) / np.array(Sum_inh_A2Bs_fail)

    # find cell
    # select_ratio_A=[]
    # select_ratio_A2B = []
    # for i in range(205):
    #     if 1.3>ratio_A[i]>0.6 and ratio_A[i]-ratio_A2B[i]>0.15:
    #         select_ratio_A.append(i)
    #     if 1.3>ratio_A2B[i]>0.6 and ratio_A2B[i]-ratio_A[i]>0.15:
    #         select_ratio_A2B.append(i)
    # print('select_ratio_A2B',select_ratio_A2B)
    #############################
    fig1 = plt.figure(figsize=(2.7, 2.7))
    ax1 = fig1.add_axes([0.23, 0.22, 0.7, 0.7])
    s = 45
    # colors_1 = sns.color_palette("YlOrBr", 10)
    # colors_2 = sns.color_palette("light:b", 10)
    colors_0 = sns.color_palette("hls", 8)
    #plt.scatter(ratio_A, ratio_A2B, marker="o", s=s, color='grey', edgecolors='white')

    select_ratio_A = diff_conA_conA2B_index_md1_pc[0:33]#[179, 12, 131, 38, 145, 195]
    select_ratio_A2B = [13, 44, 106, 170, 143]
    front = diff_conA_conA2B_index_md1_pc[0:30]
    media = diff_conA_conA2B_index_md1_pc[30:100]
    last = diff_conA_conA2B_index_md1_pc[100:]
    for i in np.array(front):
        plt.scatter(ratio_A[i], ratio_A2B[i], marker="o", s=40, c='tab:blue', label=str(i), edgecolors='white')

    for i in np.array(last):
        plt.scatter(ratio_A[i], ratio_A2B[i], marker="o", s=40, c='tab:red', label=str(i), edgecolors='white')

    for i in np.array(media):
        plt.scatter(ratio_A[i], ratio_A2B[i], marker="o", s=40, c='tab:green', label=str(i), edgecolors='white')


    # for i in np.array(select_ratio_A):
    #     plt.scatter(ratio_A[i], ratio_A2B[i], marker="X", s=40, c='#EE6363', label=str(i))
    # for i in np.array(select_ratio_A2B):
    #     plt.scatter(ratio_A[i], ratio_A2B[i], marker="X", s=40, c='#4682B4', label=str(i))

    max = 1.5
    plt.plot([0, max], [0, max], color='grey')
    plt.xlim([-0.001, max])
    plt.ylim([-0.001, max])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('E/I (con_A)', fontsize=12)
    plt.ylabel('E/I (con_B)', fontsize=12)
    plt.title('aud_'+str(start) + '_' + str(end), fontsize=8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()

    #########################
    fig1 = plt.figure(figsize=(2.7, 2.7))
    ax1 = fig1.add_axes([0.23, 0.22, 0.7, 0.7])
    s = 45
    # colors_1 = sns.color_palette("YlOrBr", 10)
    # colors_2 = sns.color_palette("light:b", 10)
    colors_0 = sns.color_palette("hls", 8)
    # plt.scatter(ratio_A, ratio_A2B, marker="o", s=s, color='grey', edgecolors='white')

    select_ratio_A = diff_conA_conA2B_index_md1_pc[0:33]  # [179, 12, 131, 38, 145, 195]
    select_ratio_A2B = [13, 44, 106, 170, 143]
    front = diff_conA_conA2B_index_md1_pc[0:30]
    media = diff_conA_conA2B_index_md1_pc[30:101]
    last = diff_conA_conA2B_index_md1_pc[101:]
    for i in np.array(front):
        plt.scatter(ratio_A[i], ratio_A2B_fail[i], marker="o", s=40, c='tab:blue', label=str(i), edgecolors='white')

    for i in np.array(last):
        plt.scatter(ratio_A[i], ratio_A2B_fail[i], marker="o", s=40, c='tab:red', label=str(i), edgecolors='white')

    for i in np.array(media):
        plt.scatter(ratio_A[i], ratio_A2B_fail[i], marker="o", s=40, c='tab:green', label=str(i), edgecolors='white')

    # for i in np.array(select_ratio_A):
    #     plt.scatter(ratio_A[i], ratio_A2B[i], marker="X", s=40, c='#EE6363', label=str(i))
    # for i in np.array(select_ratio_A2B):
    #     plt.scatter(ratio_A[i], ratio_A2B[i], marker="X", s=40, c='#4682B4', label=str(i))

    max = 1.5
    plt.plot([0, max], [0, max], color='grey')
    plt.xlim([-0.001, max])
    plt.ylim([-0.001, max])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('E/I (con_A)', fontsize=12)
    plt.ylabel('E/I (con_B_fail)', fontsize=12)
    plt.title('aud_fail_'+str(start) + '_' + str(end), fontsize=8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()
