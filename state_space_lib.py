

from __future__ import division
import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import seaborn as sns
from matplotlib.collections import LineCollection
fs=10

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



def PCA_plot_2D_cue_color(figure_path,data_path,model_dir,model_name,context,idx,hp,
                    task_name,
                    start_time=0,
                    end_time=0,
                    p_coh=0.9):


    hp['switch_context'] = context
    if context == 'con_A' or context == 'con_B2A':
        cue=1
    if context == 'con_A2B':
        cue=-1

    batch_size=50


    ########################## generate the data ##############################

    # trial_input_0, run_result_0 = generate_test_trial(model_dir=model_dir, hp=hp, task_name=task_name,
    #                                                   batch_size=batch_size,
    #                                                   cue=cue, p_coh=p_coh,
    #                                                   c_vis=None,
    #                                                   c_aud=None)
    #
    # trial_input_1, run_result_1 = generate_test_trial(model_dir=model_dir, hp=hp, task_name=task_name,
    #                                                   batch_size=batch_size,
    #                                                   cue=-cue, p_coh=p_coh,
    #                                                   c_vis=None,
    #                                                   c_aud=None)
    # firing_rate_cue_0 = run_result_0.firing_rate_binder.detach().cpu().numpy()
    # firing_rate_cue_1 = run_result_1.firing_rate_binder.detach().cpu().numpy()
    #
    # np.save(data_path + 'firing_rate_cue_0.npy', firing_rate_cue_0)
    # np.save(data_path + 'firing_rate_cue_1.npy', firing_rate_cue_1)
    # with open(data_path + 'epochs.pickle', 'wb') as f:
    #     pickle.dump(trial_input_0.epochs, f)






    firing_rate_cue_0 = np.load(data_path + 'firing_rate_cue_0.npy')
    firing_rate_cue_1= np.load(data_path + 'firing_rate_cue_1.npy')
    with open(data_path + 'epochs.pickle', 'rb') as f:
        epochs = pickle.load(f)

    ############################################################################
    start_projection=start_time
    end_projection = end_time
    start_time_0, _ = epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)

    firing_rate_list_0 = list(firing_rate_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    print("concate_firing_rate_0", concate_firing_rate_0.shape)


    ############################################################################
    start_time_1=start_time_0#, _ = trial_input_1.epochs['interval']
    end_time_1 = end_time_0#trial_input_1.epochs['interval']

    firing_rate_list_1 = list(firing_rate_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)
    print("concate_firing_rate_1", concate_firing_rate_1.shape)


    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1),axis=0)
    print("*** concate_firing_rate", concate_firing_rate.shape)

    # plot
    _alpha_list = [0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8]

    pca = PCA(n_components=3)
    pca.fit(concate_firing_rate)

    explained_variance_ratio=pca.explained_variance_ratio_
    print('explained_variance_ratio',explained_variance_ratio)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)

    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    time_size = end_time-start_time
    delim = np.cumsum(time_size)
    print('delim',delim)

    print("##concate_firing_rate_transform",concate_firing_rate_transform.shape)
    concate_transform_split = np.split(concate_firing_rate_transform, delim[:-1], axis=0)
    print('concate_transform_split',len(concate_transform_split))


    fig = plt.figure(figsize=(3.0, 3))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])


    for i in range(0, int(len(concate_transform_split)/2)):

        x = concate_transform_split[i][:, 0]
        y = concate_transform_split[i][:, 1]
        cmap = sns.color_palette("viridis", as_cmap=True)

        colors = cmap(np.linspace(0, 1, len(x)))

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments,alpha=0.5, colors=colors,zorder=0)

        ax.add_collection(lc)

        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1],linewidth=0.2,marker='o', color='red')
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',
                   color='orange')
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',
                   color='orange')
    #fig.colorbar(lc, ax=ax)

    for i in range(int(len(concate_transform_split) / 2),len(concate_transform_split)):
        x = concate_transform_split[i][:, 0]
        y = concate_transform_split[i][:, 1]
        cmap = sns.color_palette("viridis", as_cmap=True)#sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
        colors = cmap(np.linspace(0, 1, len(x)))

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, alpha=0.5,linestyles='-',colors=colors,zorder=0)

        ax.add_collection(lc)
        # if i==batch_size+2:
        #fig.colorbar(lc)

        #ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], linewidth=0.5,color=colors_1[5],zorder=0)
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], linewidth=0.2,marker='o', color='blue')

        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',color='orange')
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',color='orange')

    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top','left','bottom']].set_visible(False)


    ax.grid(True)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.title(model_name+'/'+str(idx)+':'+str(start_time[0])+'_'+str(end_time[0]),fontsize=5)
    plt.savefig(figure_path + 'PCA_conA2B_fail'+str(start_time[0])+'_'+str(end_time[0])+'.eps', format='eps', dpi=1000)
    plt.savefig(figure_path + 'PCA_conA2B_fail'+str(start_time[0])+'_'+str(end_time[0])+'.svg', format='svg', dpi=1000)


    # # # color bar
    # fig2 = plt.figure(figsize=(3, 2.1))
    # ax2 = fig2.add_axes([0.2, 0.05, 0.03, 0.75])
    # cmap = plt.get_cmap('rainbow', batch_size)
    # mpl.colorbar.ColorbarBase(ax2, cmap=cmap, spacing='proportional', ticks=[1/7*0.5, 0.5, 1-1/7*0.5])
    # ax2.set_yticklabels(['0','', '1'], size=fs)
    # ax2.set_title('cue', fontsize=fs)

    plt.show()
    return fig


def conA_switch_contB(figure_path,data_path,model_dir_A,model_dir_A2B,hp,
                    start_time=0,
                    end_time=0):
    p_coh = 0.99
    batch_size=50


    ########################## generate the data ##############################

    # trial_input_A_0, run_result_A_0 = generate_test_trial_context(model_dir=model_dir_A, context='con_A',hp=hp, task_name='HL_task',
    #                                                   batch_size=batch_size,cue=1, p_coh=p_coh)
    #
    # trial_input_A_1, run_result_A_1 = generate_test_trial_context(model_dir=model_dir_A, context='con_A',hp=hp, task_name='HL_task',
    #                                                   batch_size=batch_size,cue=-1, p_coh=p_coh)
    #
    # trial_input_B_0, run_result_B_0 = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
    #                                                       task_name='HL_task',
    #                                                       batch_size=batch_size, cue=1, p_coh=p_coh)
    #
    # trial_input_B_1, run_result_B_1 = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
    #                                                       task_name='HL_task',
    #                                                       batch_size=batch_size, cue=-1, p_coh=p_coh)
    #
    #
    #
    # firing_rate_A_0 = run_result_A_0.firing_rate_binder.detach().cpu().numpy()
    # firing_rate_A_1 = run_result_A_1.firing_rate_binder.detach().cpu().numpy()
    # firing_rate_B_0 = run_result_B_0.firing_rate_binder.detach().cpu().numpy()
    # firing_rate_B_1 = run_result_B_1.firing_rate_binder.detach().cpu().numpy()
    #
    # np.save(data_path + 'firing_rate_A_0.npy', firing_rate_A_0)
    # np.save(data_path + 'firing_rate_A_1.npy', firing_rate_A_1)
    # np.save(data_path + 'firing_rate_B_0.npy', firing_rate_B_0)
    # np.save(data_path + 'firing_rate_B_1.npy', firing_rate_B_1)
    # with open(data_path + 'epochs.pickle', 'wb') as f:
    #     pickle.dump(trial_input_A_0.epochs, f)





    with open(data_path + 'epochs.pickle', 'rb') as f:
        epochs = pickle.load(f)

    firing_rate_A_0 = np.load(data_path + 'firing_rate_A_0.npy')
    firing_rate_A_1 = np.load(data_path + 'firing_rate_A_1.npy')
    firing_rate_B_0 = np.load(data_path + 'firing_rate_B_0.npy')
    firing_rate_B_1 = np.load(data_path + 'firing_rate_B_1.npy')

    ############################################################################
    start_projection=start_time
    end_projection = end_time
    start_time_0, _ = epochs['stimulus']

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)
    print('start_time_0',start_time_0)
    print('end_time_0', end_time_0)
    print('firing_rate_A_0',firing_rate_A_0.shape)

    ############################################################################
    fr_list_A_0 = list(firing_rate_A_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A_0 = np.concatenate(fr_list_A_0, axis=0)

    ############################################################################
    fr_list_A_1 = list(firing_rate_A_1[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A_1 = np.concatenate(fr_list_A_1, axis=0)

    ############################################################################
    fr_list_B_0 = list(firing_rate_B_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B_0 = np.concatenate(fr_list_B_0, axis=0)

    ############################################################################
    fr_list_B_1 = list(firing_rate_B_1[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B_1 = np.concatenate(fr_list_B_1, axis=0)


    concate_fr_all = np.concatenate((concate_fr_A_0, concate_fr_A_1,concate_fr_B_0,concate_fr_B_1),axis=0)

    # plot
    _alpha_list = [0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8]

    pca = PCA(n_components=3)
    pca.fit(concate_fr_all)

    explained_variance_ratio=pca.explained_variance_ratio_
    print('explained_variance_ratio',explained_variance_ratio)

    concate_fr_transform = pca.transform(concate_fr_all)

    start_time = np.concatenate((start_time_0, start_time_0, start_time_0, start_time_0), axis=0)
    end_time = np.concatenate((end_time_0, end_time_0, end_time_0, end_time_0), axis=0)
    time_size = end_time-start_time
    delim = np.cumsum(time_size)

    print("##concate_firing_rate_transform",concate_fr_transform.shape)
    concate_transform_split = np.split(concate_fr_transform, delim[:-1], axis=0)
    print('concate_transform_split',len(concate_transform_split))


    fig = plt.figure(figsize=(3.0, 3))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

    for i in range(0, batch_size):
        ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], linewidth=0.5,color='r',zorder=0)
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1],linewidth=0.2,marker='o', color='red')
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',color='orange')

    for i in range(batch_size,batch_size*2):
        ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], linewidth=0.5,color='b',zorder=0)
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], linewidth=0.2,marker='o', color='blue')
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',color='orange')



    for i in range(batch_size*2,batch_size*3):
        ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], linewidth=0.5,color='pink',zorder=0)
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], linewidth=0.5,marker='o', edgecolors='black',color='pink')
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',color='orange')
    for i in range(batch_size*3,batch_size*4):
        ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], linewidth=0.5,color='tab:blue',zorder=0)
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], linewidth=0.5,marker='o', edgecolors='black',color='tab:blue')
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',color='orange')


    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top','left','bottom']].set_visible(False)


    ax.grid(True)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.title(str(start_time[0])+'_'+str(end_time[0]),fontsize=5)
    plt.savefig(figure_path + 'PCA_conA2B_fail'+str(start_time[0])+'_'+str(end_time[0])+'.eps', format='eps', dpi=1000)
    plt.savefig(figure_path + 'PCA_conA2B_fail'+str(start_time[0])+'_'+str(end_time[0])+'.svg', format='svg', dpi=1000)


    # # # color bar
    # fig2 = plt.figure(figsize=(3, 2.1))
    # ax2 = fig2.add_axes([0.2, 0.05, 0.03, 0.75])
    # cmap = plt.get_cmap('rainbow', batch_size)
    # mpl.colorbar.ColorbarBase(ax2, cmap=cmap, spacing='proportional', ticks=[1/7*0.5, 0.5, 1-1/7*0.5])
    # ax2.set_yticklabels(['0','', '1'], size=fs)
    # ax2.set_title('cue', fontsize=fs)

    plt.show()
    return fig



def switch_2D_A_B_mean(figure_path,data_path,model_dir_A,model_dir_A2B,hp,
                    start_time=0,
                    end_time=0):
    p_coh = 0.99
    batch_size=20


    ########################## generate the data ##############################

    # trial_input_A_0, run_result_A_0 = generate_test_trial_context(model_dir=model_dir_A, context='con_A',hp=hp, task_name='HL_task',
    #                                                   batch_size=batch_size,cue=1, p_coh=p_coh)
    #
    # trial_input_A_1, run_result_A_1 = generate_test_trial_context(model_dir=model_dir_A, context='con_A',hp=hp, task_name='HL_task',
    #                                                   batch_size=batch_size,cue=-1, p_coh=p_coh)
    #
    # trial_input_B_0, run_result_B_0 = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
    #                                                       task_name='HL_task',
    #                                                       batch_size=batch_size, cue=1, p_coh=p_coh)
    #
    # trial_input_B_1, run_result_B_1 = generate_test_trial_context(model_dir=model_dir_A2B, context='con_A2B', hp=hp,
    #                                                       task_name='HL_task',
    #                                                       batch_size=batch_size, cue=-1, p_coh=p_coh)
    #
    # firing_rate_A_0 = run_result_A_0.firing_rate_binder.detach().cpu().numpy()
    # firing_rate_A_1 = run_result_A_1.firing_rate_binder.detach().cpu().numpy()
    # firing_rate_B_0 = run_result_B_0.firing_rate_binder.detach().cpu().numpy()
    # firing_rate_B_1 = run_result_B_1.firing_rate_binder.detach().cpu().numpy()
    #
    # np.save(data_path + 'firing_rate_A_0.npy', firing_rate_A_0)
    # np.save(data_path + 'firing_rate_A_1.npy', firing_rate_A_1)
    # np.save(data_path + 'firing_rate_B_0.npy', firing_rate_B_0)
    # np.save(data_path + 'firing_rate_B_1.npy', firing_rate_B_1)
    # with open(data_path + 'epochs.pickle', 'wb') as f:
    #     pickle.dump(trial_input_A_0.epochs, f)
    #

    firing_rate_A_0 = np.load(data_path + 'firing_rate_A_0.npy')
    firing_rate_A_1= np.load(data_path + 'firing_rate_A_1.npy')
    firing_rate_B_0 = np.load(data_path + 'firing_rate_B_0.npy')
    firing_rate_B_1 = np.load(data_path + 'firing_rate_B_1.npy')
    with open(data_path + 'epochs.pickle', 'rb') as f:
        epochs = pickle.load(f)

    ############################################################################
    start_projection=start_time
    end_projection = end_time
    print('time',end_time-start_time)
    #start_time_0, _ = trial_input_A_0.epochs['stimulus']
    start_time_0, _ = epochs['stimulus']
    print('start_time_0',start_time_0)

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)

    ############################################################################
    fr_list_A_0 = list(firing_rate_A_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A_0 = np.concatenate(fr_list_A_0, axis=0)

    fr_list_A_1 = list(firing_rate_A_1[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_A_1 = np.concatenate(fr_list_A_1, axis=0)

    ############################################################################
    fr_list_B_0 = list(firing_rate_B_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B_0 = np.concatenate(fr_list_B_0, axis=0)

    fr_list_B_1 = list(firing_rate_B_1[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    concate_fr_B_1 = np.concatenate(fr_list_B_1, axis=0)
    print('concate_fr_B_1 ',concate_fr_B_1.shape)



    concate_fr_all = np.concatenate((concate_fr_A_0, concate_fr_A_1,concate_fr_B_0,concate_fr_B_1),axis=0)
    print('concate_fr_all',concate_fr_all.shape)

    # plot

    pca = PCA(n_components=3)
    pca.fit(concate_fr_all)

    explained_variance_ratio=pca.explained_variance_ratio_
    concate_fr_transform = pca.transform(concate_fr_all)

    start_time = np.concatenate((start_time_0, start_time_0, start_time_0, start_time_0), axis=0)
    end_time = np.concatenate((end_time_0, end_time_0, end_time_0, end_time_0), axis=0)
    time_size = end_time-start_time
    delim = np.cumsum(time_size)
    concate_transform_split = np.split(concate_fr_transform, delim[:-1], axis=0)
    concate_transform_split = np.array(concate_transform_split)

    fr_A_0 = concate_transform_split[0:1*batch_size,:,:]
    fr_A_1 = concate_transform_split[1*batch_size:2*batch_size, :,:]
    fr_B_0 = concate_transform_split[2*batch_size:3*batch_size, :,:]
    fr_B_1 = concate_transform_split[3 * batch_size: 4 * batch_size, :,:]

    print('fr_A_0',fr_A_0.shape)

    fr_A_0 = np.mean(fr_A_0,axis=0)
    fr_A_1 = np.mean(fr_A_1, axis=0)
    fr_B_0 = np.mean(fr_B_0, axis=0)
    fr_B_1 = np.mean(fr_B_1, axis=0)


    fig = plt.figure(figsize=(3.0, 3))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

    color_1 = ['tab:purple', 'grey', 'tab:green', 'hotpink']
    color_2 = ['tab:purple', 'grey', 'tab:green', 'hotpink']
    lw=2
    s = 1

    ax.plot(fr_A_0[:, 0], fr_A_0[:, 1], linewidth=lw, color=color_1[0], zorder=0)
    ax.plot(fr_A_1[:, 0], fr_A_1[:, 1], linewidth=lw, color=color_2[0], zorder=0)
    ax.plot(fr_B_0[:, 0], fr_B_0[:, 1], linewidth=lw, color=color_1[1], zorder=0)
    ax.plot(fr_B_1[:, 0], fr_B_1[:, 1], linewidth=lw, color=color_2[1], zorder=0)

    ax.scatter(fr_A_0[-1, 0], fr_A_0[-1, 1], linewidth=s, marker='o', color='red')
    ax.scatter(fr_A_1[-1, 0], fr_A_1[-1, 1], linewidth=s, marker='o', color='blue')

    ax.scatter(fr_B_0[-1, 0], fr_B_0[-1, 1], linewidth=s, marker='o', color='blue')
    ax.scatter(fr_B_1[-1, 0], fr_B_1[-1, 1], linewidth=s, marker='o', color='red')
    ax.scatter(fr_A_0[0, 0], fr_A_0[0, 1], linewidth=1, marker='*', color='orange')
    ax.scatter(fr_A_1[0, 0], fr_A_1[0, 1], linewidth=1, marker='*', color='orange')
    ax.scatter(fr_B_0[0, 0], fr_B_0[0, 1], linewidth=1, marker='*', color='orange')
    ax.scatter(fr_B_1[0, 0], fr_B_1[0, 1], linewidth=1, marker='*', color='orange')


    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top','left','bottom']].set_visible(False)


    ax.grid(True)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.title(str(start_time[0])+'_'+str(end_time[0]),fontsize=5)
    plt.savefig(figure_path + 'switch_2D_A_B_mean'+str(start_time[0])+'_'+str(end_time[0])+'.eps', format='eps', dpi=1000)
    plt.savefig(figure_path + 'switch_2D_A_B_mean'+str(start_time[0])+'_'+str(end_time[0])+'.svg', format='svg', dpi=1000)


    # # # color bar
    # fig2 = plt.figure(figsize=(3, 2.1))
    # ax2 = fig2.add_axes([0.2, 0.05, 0.03, 0.75])
    # cmap = plt.get_cmap('rainbow', batch_size)
    # mpl.colorbar.ColorbarBase(ax2, cmap=cmap, spacing='proportional', ticks=[1/7*0.5, 0.5, 1-1/7*0.5])
    # ax2.set_yticklabels(['0','', '1'], size=fs)
    # ax2.set_title('cue', fontsize=fs)

    plt.show()
    return fig

