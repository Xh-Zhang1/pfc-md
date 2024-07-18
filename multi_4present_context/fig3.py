import sys,os
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import default
import tools
import Regression_state_space_lib_sparsity


hp=default.get_default_hp(rule_name='RDM_task')

############ model
hp['p_coh']=0.92
hp['n_rnn'] = 256
hp['activation'] = 'softplus'

hp['mask_type']='type3'
hp['mask_test']='type3'
hp['n_md'] = 200
hp['cue_duration'] = 800
hp['cue_delay'] = 800
hp['stim'] = 200

hp['stim_std']=0.1
hp['scale_random']=5.0
hp['dropout_model']=0.0

hp['model_idx']=59
idxs = [9]

hp['dropout']=0.0
hp['stim_std_test']=0.0

fig_path = hp['root_path']+'/Figures/'
#figure_path = os.path.join(fig_path,str(model_name),str(context_name)+'/')
figure_path = os.path.join(fig_path,'Regress'+'/')
tools.mkdir_p(figure_path)


hp['figure_path'] = figure_path

hp['gamma_noise']=1.0
batch_size = 1

data_path = os.path.join(hp['root_path'], 'Datas','fig3/')
tools.mkdir_p(data_path)

def plot_regress_HL(model_idx,idx):
    data_path = os.path.join(hp['root_path'], 'Datas', 'fig3/')
    tools.mkdir_p(data_path)

    data_path_1 = os.path.join(data_path, 'plot_regress_HL/')
    tools.mkdir_p(data_path_1)
    hp['model_idx']=model_idx
    idx = idx
    model_name = str(hp['mask_type'])+'_'+str(hp['n_rnn'])+'_'+str(hp['n_md'])+ \
                 '_'+str(hp['activation'])+'_sr'+str(hp['scale_random'])+'_'+str(hp['stim_std'])+ \
                 '_drop'+str(hp['dropout_model'])+'_'+str(hp['model_idx'])
    local_folder_name = os.path.join('/'+'model_'+str(hp['p_coh']),  model_name, str(idx))
    model_dir = hp['root_path']+local_folder_name+'/'

    epoch = Regression_state_space_lib_sparsity.get_epoch(model_dir, hp)
    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];response_on = epoch['response_on'];response_off = epoch['response_off']
    stim_on = epoch['stim_on']
    #data path

    start_projection_Q = cue_on-1
    end_projection_Q = stim_on
    coh = [0.6,0.7,0.92]


    #
    Regression_state_space_lib_sparsity.Plot_statespace_cue_epoch_3coh_HL(data_path = data_path_1,model_name = model_name, model_dir=model_dir, idx=idx,
                                                           hp=hp,
                                                           Q_task_name='HL_task',
                                                           task_name='HL_task',
                                                           sparsity_HL=0.0,
                                                           start_projection_Q=start_projection_Q,
                                                           end_projection_Q=end_projection_Q,

                                                           start_projection=cue_on,
                                                           end_projection=stim_on-1,
                                                           p_cohs_plot=[0.92,0.7,0.6]
                                                           )

    Regression_state_space_lib_sparsity.Plot_statespace_diff_sparsity_HL(data_path = data_path_1,model_name = model_name, model_dir=model_dir, idx=idx,
                                                              hp=hp,
                                                              Q_task_name='HL_task',
                                                              task_name='HL_task',
                                                              start_projection_Q=start_projection_Q,
                                                              end_projection_Q=end_projection_Q,

                                                              start_projection=cue_on-1,
                                                              end_projection=cue_off-1,
                                                              p_cohs_plot=0.92,
                                                             )

    Regression_state_space_lib_sparsity.calculate_velocity_cue_HL_separate(data_path=data_path_1, model_name=model_name,
                                                                  model_dir=model_dir, idx=idx,
                                                                  hp=hp,
                                                                  Q_task_name='HL_task',
                                                                  task_name='HL_task',
                                                                  sparsity_HL=0.0,
                                                                  start_projection_Q=start_projection_Q,
                                                                  end_projection_Q=end_projection_Q,

                                                                  start_projection=cue_on - 1,
                                                                  end_projection=stim_on,
                                                                  # stim_on-1,#cue_off-1,
                                                                  p_cohs_plot=coh
                                                                  )

plot_regress_HL(model_idx=59,idx=9)


def plot_regress_RDM(model_idx,idx):
    data_path = os.path.join(hp['root_path'], 'Datas', 'fig3/')
    tools.mkdir_p(data_path)

    data_path_1 = os.path.join(data_path, 'plot_regress_RDM/')
    tools.mkdir_p(data_path_1)

    hp['model_idx'] = model_idx
    idx = idx

    model_name = str(hp['mask_type'])+'_'+str(hp['n_rnn'])+'_'+str(hp['n_md'])+ \
                 '_'+str(hp['activation'])+'_sr'+str(hp['scale_random'])+'_'+str(hp['stim_std'])+ \
                 '_drop'+str(hp['dropout_model'])+'_'+str(hp['model_idx'])
    local_folder_name = os.path.join('/'+'model_'+str(hp['p_coh']),  model_name, str(idx))
    model_dir = hp['root_path']+local_folder_name+'/'

    epoch = Regression_state_space_lib_sparsity.get_epoch(model_dir, hp)
    cue_on = epoch['cue_on'];cue_off = epoch['cue_off'];response_on = epoch['response_on'];response_off = epoch['response_off']
    stim_on = epoch['stim_on']
    #data path
    hp['data_path'] = data_path
    start_projection_Q = cue_on
    end_projection_Q = stim_on

    coh = [0.1,0.4,0.92]


    Regression_state_space_lib_sparsity.Plot_statespace_cue_epoch_3coh_RDM(data_path = data_path_1,figure_path = figure_path,model_name = model_name, model_dir=model_dir, idx=idx,
                                                                       hp=hp,
                                                                       Q_task_name='RDM_task',
                                                                       task_name='RDM_task',
                                                                       sparsity_HL=0.0,
                                                                       start_projection_Q=start_projection_Q,
                                                                       end_projection_Q=end_projection_Q,

                                                                       start_projection=cue_on-1,
                                                                       end_projection=cue_off-1,
                                                                       p_cohs_plot=[0.8,0.4,0.2]
                                                                       )
    #


    Regression_state_space_lib_sparsity.Plot_statespace_diff_sparsity_RDM(data_path = data_path_1,figure_path = figure_path,model_name = model_name, model_dir=model_dir, idx=idx,
                                                              hp=hp,
                                                              Q_task_name='RDM_task',
                                                              task_name='RDM_task',
                                                              start_projection_Q=start_projection_Q,
                                                              end_projection_Q=end_projection_Q,

                                                              start_projection=cue_on-1,
                                                              end_projection=cue_off-1,
                                                              p_cohs_plot=0.92,
                                                             )


    Regression_state_space_lib_sparsity.calculate_velocity_cue_RDM_separate(data_path = data_path_1,model_name = model_name, model_dir=model_dir, idx=idx,
                                                                  hp=hp,
                                                                  Q_task_name='RDM_task',
                                                                  task_name='RDM_task',
                                                                  sparsity_RDM=0.0,
                                                                  start_projection_Q=start_projection_Q,
                                                                  end_projection_Q=end_projection_Q,

                                                                  start_projection=cue_on-1,
                                                                  end_projection=stim_on,
                                                                  p_cohs_plot=[0.1,0.4,0.92])

plot_regress_RDM(model_idx=59,idx=9)

