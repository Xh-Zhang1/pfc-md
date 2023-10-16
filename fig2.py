import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import default
import tools
import neuron_activity_lib


rule_name = 'RDM_task'
hp=default.get_default_hp(rule_name=rule_name)


############ model
############ model
hp['sparsity_HL']=0.2
hp['sparsity_RDM']=0.0
#hp['input_mask']='no'
hp['mask_test']='type3'
hp['init_hh']='init2'
hp['get_SNR']='no'
hp['scale_random']=5.0
hp['scale_RDM']=1

hp['sparsity_HL']=0.0
hp['sparsity_RDM']=0.0
hp['mask_type']='type3'

hp['scale_random']=5.0
hp['stim_std']=0.1
hp['dropout_model']=0.0
hp['model_idx']=59
hp['stim_std_test']=0.0
hp['dropout']=0.0
idx = 9

#=========================  plot =============================
model_name = str(hp['mask_type'])+'_'+str(hp['n_rnn'])+'_'+str(hp['n_md'])+ \
             '_'+str(hp['activation'])+'_sr'+str(hp['scale_random'])+'_'+str(hp['stim_std'])+ \
             '_drop'+str(hp['dropout_model'])+'_'+str(hp['model_idx'])

local_folder_name = os.path.join('/'+'model_'+str(hp['p_coh']),  model_name, str(idx))
model_dir = hp['root_path']+local_folder_name+'/'

epoch = neuron_activity_lib.get_epoch(model_dir,hp)

fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'neuron_activity/',model_name+'/')
tools.mkdir_p(figure_path)

data_path = os.path.join(hp['root_path'], 'Datas','neuron_activity/')
tools.mkdir_p(data_path)


start = epoch['cue_off']
end =epoch['stim_on']

neuron_activity_lib.plot_rule(figure_path=figure_path,data_path=data_path,start=start,end=end,cue=1)

neuron_activity_lib.plot_rule(figure_path=figure_path,data_path=data_path,start=start,end=end,cue=-1)

neuron_activity_lib.plot_context(figure_path=figure_path,data_path=data_path,start=start,end=end,
                                 context_name='HL_task')

neuron_activity_lib.plot_context(figure_path=figure_path,data_path=data_path,start=start,end=end,
                                 context_name='RDM_task')




# neuron_activity_lib.md1_diff_uncertainty_HL(data_path,figure_path,epoch=epoch)
# neuron_activity_lib.md1_diff_uncertainty_RDM(data_path,figure_path,epoch=epoch)
#
#
# neuron_activity_lib.md2_diff_uncertainty_HL(data_path,figure_path,epoch=epoch)
# neuron_activity_lib.md2_diff_uncertainty_RDM(data_path,figure_path,epoch=epoch)
#
# neuron_activity_lib.inh_diff_uncertainty_HL(data_path,figure_path,epoch=epoch)
# neuron_activity_lib.inh_diff_uncertainty_RDM(data_path,figure_path,epoch=epoch)
# #
# neuron_activity_lib.md1_diff_sparsity_RDM(data_path,figure_path,epoch=epoch)
# neuron_activity_lib.md1_diff_sparsity_HL(data_path,figure_path,epoch=epoch)

# #


def neuron_tuning():
    coh_RDM=0.92;coh_HL=0.92
    hp['sparsity_RDM']=0.0

    #neuron_activity_lib.plot_exc_units_perfVis4(data_path,figure_path,model_name,model_dir,hp,c_vis=-0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)
    #neuron_activity_lib.plot_exc_units_perfAud4(data_path,figure_path,model_name,model_dir,hp,c_vis=-0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)
    # neuron_activity_lib.plot_inh_units_perfHL4(data_path, figure_path, model_name, model_dir, idx, hp, c_vis=-0.2, c_aud=-0.2, coh_RDM=0.8, coh_HL=0.8, epoch=epoch)
    # neuron_activity_lib.plot_inh_units_perfRDM4(data_path,figure_path,model_name,model_dir,idx,hp,c_vis=-0.2,c_aud=-0.2,coh_RDM=0.8,coh_HL=0.8,epoch=epoch)
    # #
    neuron_activity_lib.plot_MD1_units_perfRDM(data_path,figure_path,model_name,model_dir,idx,hp,c_vis=0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)
    neuron_activity_lib.plot_MD1_units_perfHL(data_path,figure_path,model_name,model_dir,idx,hp,c_vis=0.2,c_aud=0.2,coh_RDM=coh_RDM,coh_HL=coh_HL,epoch=epoch)

#neuron_tuning()
