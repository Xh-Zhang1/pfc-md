import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ranksums
from scipy.stats import wilcoxon

import default
import tools
import run
import neuron_activity_supple_lib
from scipy.io import savemat

batch_size=1



rule_name = 'RDM_task'#sys.argv[1]#'HL_task'
hp=default.get_default_hp(rule_name=rule_name)

hp['rng']=np.random.RandomState(0)
############ model
hp['sparsity_HL']=0.2
hp['sparsity_RDM']=0.0
#hp['input_mask']='no'
hp['mask_test']='type3'
hp['init_hh']='init2'
hp['get_SNR']='no'
hp['scale_random']=5.0
hp['scale_RDM']=1


hp['scale_random']=5.0
hp['stim_std']=0.1
hp['dropout_model']=0.0

hp['mask_type']='type3'
hp['stim_std_test']=0.1
hp['dropout']=0.0
hp['input_mask']='no'
hp['model_idx']=59
idx = 9#8#9,12


#=========================  plot =============================
model_name = str(hp['mask_type'])+'_'+str(hp['n_rnn'])+'_'+str(hp['n_md'])+ \
             '_'+str(hp['activation'])+'_sr'+str(hp['scale_random'])+'_'+str(hp['stim_std'])+ \
             '_drop'+str(hp['dropout_model'])+'_'+str(hp['model_idx'])

local_folder_name = os.path.join('/'+'model_'+str(hp['p_coh']),  model_name, str(idx))
model_dir = hp['root_path']+local_folder_name+'/'
if not os.path.isfile(model_dir):
    print('model_dir', model_dir)
    print('$$$$$$$$$$$$$$$ path not exist !!!!!')
    #sys.exit(0)


print('model_dir',model_dir)
epoch = neuron_activity_supple_lib.get_epoch(model_dir, hp)

fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'Figure2/',model_name+'/')
tools.mkdir_p(figure_path)

data_path = os.path.join(hp['root_path'], 'Datas','Figure2/')
tools.mkdir_p(data_path)


coh_RDM=0.9;coh_HL=0.9
def plot_scatter():
    data_path_1 = os.path.join(data_path, 'plot_scatter/')
    tools.mkdir_p(data_path_1)

    start = epoch['cue_off']+1
    end =epoch['stim_on']

    neuron_activity_supple_lib.plot_rule(figure_path=figure_path, data_path=data_path_1, model_dir=model_dir, hp=hp, start=start, end=end,
                                              cue=1, c_vis=0.2, c_aud=0.2, coh_RDM=coh_RDM, coh_HL=coh_HL)

    neuron_activity_supple_lib.plot_rule(figure_path=figure_path, data_path=data_path_1, model_dir=model_dir, hp=hp, start=start, end=end,
                                              cue=-1, c_vis=0.2, c_aud=0.2, coh_RDM=coh_RDM, coh_HL=coh_HL)

    neuron_activity_supple_lib.plot_context(figure_path=figure_path, data_path=data_path_1, model_dir=model_dir, hp=hp, start=start, end=end,
                                                 context_name='HL_task', p_cohs=0.9)
    # #
    neuron_activity_supple_lib.plot_context(figure_path=figure_path, data_path=data_path_1, model_dir=model_dir, hp=hp, start=start, end=end,
                                                 context_name='RDM_task', p_cohs=0.9)

plot_scatter()




def neuron_tuning_4panel():#
    data_path_1 = os.path.join(data_path, 'neuron_tuning_4panel/')
    tools.mkdir_p(data_path_1)

    neuron_activity_supple_lib.plot_exc_units_panel4(data_path_1,figure_path,model_name,model_dir,hp,c_vis=-0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)
    neuron_activity_supple_lib.plot_exc_units_panel4_supple(data_path_1,figure_path,model_name,model_dir,hp,c_vis=-0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)

    neuron_activity_supple_lib.plot_MD2_units_pref_panel4(data_path_1,figure_path,model_name,model_dir,idx,hp,c_vis=0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)
    neuron_activity_supple_lib.plot_MD1_units_pref_panel4(data_path_1,figure_path,model_name,model_dir,idx,hp,c_vis=0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)

    neuron_activity_supple_lib.plot_MD2_units_pref_panel4_supple(data_path_1,figure_path,model_name,model_dir,idx,hp,c_vis=0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)
    neuron_activity_supple_lib.plot_MD1_units_pref_panel4_supple(data_path_1,figure_path,model_name,model_dir,idx,hp,c_vis=0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)


neuron_tuning_4panel()




import matplotlib as mpl
import os,sys
import numpy as np
import pdb
import matplotlib.pyplot as plt

from matplotlib import cm

import network
import default
import tools
import run
import neuron_activity_supple_lib

hp=default.get_default_hp(rule_name='HL_task',random_seed=1)
hp['stim_epoch']='no'
hp['mask_test'] = 'type3'
hp['stim_value']=0
hp['scale_PC_MD']=1
hp['dropout']=0.0
hp['dropout_model']=0.0
hp['cue_delay'] = 800
hp['init_hh']='init2'

batch_size=1
fs = 10


hp['stim_std']=0.1
hp['stim_std_test']=0.1
hp['dropout_model']=0.0
hp['model_idx']=59
idx_select=9


fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'sequence/')
#figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'_idx_'+str(idx)+'/')
tools.mkdir_p(figure_path)

data_path = os.path.join(hp['root_path'], 'Datas','Fixed_point/')
tools.mkdir_p(data_path)
#=========================  plot =============================
hp['mask_type_model']='type3'
model_name = str(hp['mask_type_model'])+'_'+str(hp['n_rnn'])+'_'+str(hp['n_md'])+ \
             '_'+str(hp['activation'])+'_sr'+str(hp['scale_random'])+'_'+str(hp['stim_std'])+ \
             '_drop'+str(hp['dropout_model'])+'_'+str(hp['model_idx'])

local_folder_name = os.path.join('/'+'model_'+str(hp['p_coh']),  model_name, str(idx_select))
model_dir = hp['root_path']+local_folder_name+'/'

print('model_dir',model_dir)
epoch = neuron_activity_supple_lib.get_epoch(model_dir, hp)

model = network.Network(hp, is_cuda=False)
model.load(model_dir)


def plot_figure(data_0,cell,task_name,cue):
    # normalize
    #print("data,data.max",data)
    for i in range(0, data_0.shape[1]):
        if np.max(data_0[:, i])<0:
            data_0[:, i] = 0
        else:
            data_0[:, i] = (data_0[:, i] / np.max(data_0[:, i]))
            #print("np.max(data[:, i])",np.max(data[:, i]))
    X_0, Y_0 = np.mgrid[0:data_0.shape[0]*20:20, 0:data_0.shape[1]]

    fig = plt.figure(figsize=(3.0, 2.6))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])

    plt.gca().set_xlabel('Time (ms)', fontsize=fs+1)
    plt.gca().set_ylabel('Neuron (Sorted)', fontsize=fs+1)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title(cell+'_'+task_name+'_'+str(cue))

    # Make the plot
    #cmap = plt.get_cmap('viridis')#viridis_r
    plt.pcolormesh(X_0, Y_0, data_0)
    print('X, Y',data_0)

    m = cm.ScalarMappable(cmap=mpl.rcParams["image.cmap"])#cmap=mpl.rcParams["image.cmap"]
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)

    cbar = fig.colorbar(m, aspect=15)

    cbar.set_ticks([0,  1])
    cbar.ax.tick_params(labelsize=fs+1)
    #cbar.ax.set_title('Normalized\n activity', fontsize=fs+1)
    fig.savefig(figure_path+cell+'_'+task_name+'_'+str(cue)+'.eps', format='eps', dpi=1000)
    plt.show()

#generate firing rate for PCA
def activity_peak_order_plot_diff_rule(task_name,cue):
    data_path_1 = os.path.join(data_path, 'activity_peak_order_plot_diff_rule/')
    tools.mkdir_p(data_path_1)



    hp['use_reset']='no'
    #######rule1
    # trial_input_0, run_result_0 = generate_test_trial(model_dir=model_dir,hp=hp,task_name='RDM_task',batch_size=batch_size,
    #                                                   cue=cue,
    #                                                   p_coh=0.92,
    #                                                   c_vis=0.2,
    #                                                   c_aud=0.2)
    #
    # firing_rate_cue_0 = run_result_0.firing_rate_binder.detach().cpu().numpy()
    # np.save(data_path_1+'firing_rate_cue_0.npy', firing_rate_cue_0)
    firing_rate_cue_0 = np.load(data_path_1+'firing_rate_cue_0.npy')



    start_time = epoch['cue_off'] + 4
    end_time = epoch['stim_on'] - 2
    func_activity_threshold_0 = 0.2

    data_0 = firing_rate_cue_0[start_time:end_time,0,0:205]

    max_firing_rate_0 = np.max(data_0, axis=0)
    pick_idx_0 = np.argwhere(max_firing_rate_0 > func_activity_threshold_0).squeeze()
    print("pick_idx",pick_idx_0.shape)

    data_0 = data_0[:, pick_idx_0]
    peak_time_0 = np.argmax(data_0, axis=0)

    peak_order_0 = np.argsort(peak_time_0, axis=0)
    data_0 = data_0[:, peak_order_0]
    print(data_0.shape)



    #######rule1
    # trial_input_1, run_result_1 = generate_test_trial(model_dir=model_dir,hp=hp,task_name='RDM_task',batch_size=batch_size,
    #                                                   cue=-cue,
    #                                                   p_coh=0.92,
    #                                                   c_vis=0.2,
    #                                                   c_aud=0.2)
    #
    #
    #
    # firing_rate_cue_1 = run_result_1.firing_rate_binder.detach().cpu().numpy()
    # np.save(data_path_1 + 'firing_rate_cue_1.npy', firing_rate_cue_1)
    firing_rate_cue_1 = np.load(data_path_1 + 'firing_rate_cue_1.npy')



    start_time = epoch['cue_off'] + 2
    end_time = epoch['stim_on'] - 2



    data_1 = firing_rate_cue_1[start_time:end_time,0,0:205]
    data_1 = data_1[:, pick_idx_0]

    data_1 = data_1[:, peak_order_0]
    print(data_1.shape)

    #######task1
    # trial_input_2, run_result_2 = generate_test_trial(model_dir=model_dir,hp=hp,task_name='HL_task',batch_size=batch_size,
    #                                                   cue=cue,
    #                                                   p_coh=0.92,
    #                                                   c_vis=0.2,
    #                                                   c_aud=0.2)
    #
    #
    # firing_rate_cue_2 = run_result_2.firing_rate_binder.detach().cpu().numpy()
    # np.save(data_path_1 + 'firing_rate_cue_2.npy', firing_rate_cue_2)
    firing_rate_cue_2 = np.load(data_path_1 + 'firing_rate_cue_2.npy')








    start_time = epoch['cue_off'] + 2
    end_time = epoch['stim_on'] - 2
    data_2 = firing_rate_cue_2[start_time:end_time,0,0:205]
    data_2 = data_2[:, pick_idx_0]

    data_2 = data_2[:, peak_order_0]
    print(data_1.shape)



    plot_figure(data_0,cell='vis_sort',task_name='RDM_task',cue=cue)
    plot_figure(data_1,cell='vis_sort',task_name='RDM_task',cue=-cue)
    plot_figure(data_2,cell='vis_sort',task_name='HL_task',cue=cue)
activity_peak_order_plot_diff_rule(task_name='RDM_task',cue=1)


def activity_peak_order_plot_diff_task(cue):
    data_path_1 = os.path.join(data_path, 'activity_peak_order_plot_diff_task/')
    tools.mkdir_p(data_path_1)


    hp['use_reset']='no'
    #######rule1
    # trial_input_0, run_result_0 = generate_test_trial(model_dir=model_dir,hp=hp,task_name='RDM_task',batch_size=batch_size,
    #                                                   cue=cue,
    #                                                   p_coh=0.92,
    #                                                   c_vis=0.2,
    #                                                   c_aud=0.2)
    #
    # firing_rate_cue_0 = run_result_0.firing_rate_binder.detach().cpu().numpy()
    # np.save(data_path_1 + 'firing_rate_cue_0.npy', firing_rate_cue_0)



    firing_rate_cue_0 = np.load(data_path_1 + 'firing_rate_cue_0.npy')

    start_time = epoch['cue_off'] + 2
    end_time = epoch['stim_on'] - 2
    func_activity_threshold_0 = 0.2





    data_0 = firing_rate_cue_0[start_time:end_time,0,0:205]

    max_firing_rate_0 = np.max(data_0, axis=0)
    pick_idx_0 = np.argwhere(max_firing_rate_0 > func_activity_threshold_0).squeeze()
    print("pick_idx",pick_idx_0.shape)

    data_0 = data_0[:, pick_idx_0]
    peak_time_0 = np.argmax(data_0, axis=0)

    peak_order_0 = np.argsort(peak_time_0, axis=0)
    data_0 = data_0[:, peak_order_0]
    print(data_0.shape)

    #######rule1
    # trial_input_1, run_result_1 = generate_test_trial(model_dir=model_dir,hp=hp,task_name='HL_task',batch_size=batch_size,
    #                                                   cue=cue,
    #                                                   p_coh=0.92,
    #                                                   c_vis=0.2,
    #                                                   c_aud=0.2)
    # firing_rate_cue_1 = run_result_1.firing_rate_binder.detach().cpu().numpy()
    # np.save(data_path_1 + 'firing_rate_cue_1.npy', firing_rate_cue_1)



    firing_rate_cue_1 = np.load(data_path_1 + 'firing_rate_cue_1.npy')

    start_time = epoch['cue_off'] + 2
    end_time = epoch['stim_on'] - 2




    data_1 = firing_rate_cue_1[start_time:end_time,0,0:205]
    data_1 = data_1[:, pick_idx_0]

    data_1 = data_1[:, peak_order_0]
    print(data_1.shape)



    plot_figure(data_0,cell='exc',task_name='RDM_task',cue=cue)
    plot_figure(data_1,cell='exc',task_name='HL_task',cue=cue)
activity_peak_order_plot_diff_task(cue=1)








