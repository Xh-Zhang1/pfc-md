import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import default
import tools



rule_name = 'HL_task'
hp=default.get_default_hp(rule_name=rule_name,random_seed=1)

############ model
hp['p_coh']=0.92
hp['n_rnn'] = 256
hp['activation'] = 'softplus'

hp['n_md'] = 200
hp['cue_duration'] = 800
hp['cue_delay'] = 800
hp['stim'] = 200
hp['stim_std']=0.1

hp['scale_RDM']=1.0
hp['scale_HL']=1.0

hp['dropout']=0.0
hp['dropout_model']=0.0
hp['model_idx']=59


model_name_1 = str(hp['n_rnn'])+'_'+str(hp['n_md'])+ \
             '_'+str(hp['activation'])+'_'+str(hp['cue_delay'])+'_'+str(hp['stim'])+ \
             '_'+str(hp['dropout_model'])+'_'+str(hp['model_idx'])
fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'psychometric/')
tools.mkdir_p(figure_path)

data_path_type3 = os.path.join(hp['root_path'], 'Datas','psychometric/','type3_'+model_name_1+'/')
tools.mkdir_p(data_path_type3)

data_path_type2 = os.path.join(hp['root_path'], 'Datas','psychometric/','type2_'+model_name_1+'/')
tools.mkdir_p(data_path_type2)


cohs_RDM = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
cohs_HL  = np.array([0.5,0.6,0.65,0.67,0.7,0.72,0.8,0.9,1.0])


###################### plot_psychometric_HL
print('#### data_path_type3',data_path_type3)


def plot_psychometric_HL():

    choice_perf_HL_type3 = np.load(data_path_type3+'choice_perf_HL_type3.npy')
    choice_perf_HL_type2 = np.load(data_path_type2+'choice_perf_HL_type2.npy')

    mean_HL_type3 = np.mean(choice_perf_HL_type3,axis=0)
    error_HL_type3 = np.std(choice_perf_HL_type3, axis=0) / np.sqrt(choice_perf_HL_type3.shape[0])


    x_type3 = cohs_HL
    y_para_type3 = mean_HL_type3
    err_type3 = error_HL_type3

    weights_type3 = [1.0/max(_,0.01) for _ in err_type3]
    def logistic_growth_type3(x, A1, A2, x_0, p):
        return A2 + (A1-A2)/(1+(x/x_0)**p)

    x_plot_type3 = np.linspace(0.5, 1.0, 100)
    paras_type3, paras_cov_type3 = curve_fit(logistic_growth_type3, x_type3, y_para_type3,absolute_sigma=True,sigma = weights_type3)
    para_curve_type3 = logistic_growth_type3(x_plot_type3, *paras_type3)


    # ##### type2 ########
    mean_HL_type2 = np.mean(choice_perf_HL_type2,axis=0)
    error_HL_type2 = np.std(choice_perf_HL_type2, axis=0) / np.sqrt(choice_perf_HL_type2.shape[0])

    x_type2 = cohs_HL
    y_para_type2 = mean_HL_type2
    err_type2 = error_HL_type2

    weights_type2 = [1.0/max(_,0.2) for _ in err_type2]
    def logistic_growth(x, A1, A2, x_0, p):
        return A2 + (A1-A2)/(1+(x/x_0)**p)

    x_plot_type2 = np.linspace(0.5, 1.0, 100)
    paras_type2, paras_cov_type2 = curve_fit(logistic_growth, x_type2, y_para_type2,absolute_sigma=True,sigma = weights_type2)
    para_curve_type2 = logistic_growth(x_plot_type2, *paras_type2)


    ##===================== plot =====================#####
    fig0 = plt.figure(figsize=(3.,3.0))
    ax0 = fig0.add_axes([0.22,0.23,0.75,0.68])
    font_ticket = 12

    plt.errorbar(x_type3,y_para_type3, err_type3, color = 'black', fmt = 'o')
    plt.plot(x_plot_type3, para_curve_type3, color = 'black', label = "PFC-MD")

    plt.errorbar(x_type2,y_para_type2, err_type2, color = 'tab:blue', fmt = 'o')
    plt.plot(x_plot_type2, para_curve_type2, color = 'tab:blue', label = "PFC")


    plt.title('CAC subtask', fontsize=font_ticket+1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.5, 0.6, 0.7, 0.8,0.9,1.0], fontsize=font_ticket)
    plt.yticks([0.3, 0.5,0.7,0.9], fontsize=font_ticket)
    plt.xlabel('LP/HP cue congruence', fontsize=font_ticket)
    plt.ylabel('Proportion correct', fontsize=font_ticket)
    plt.legend(fontsize=9)
    fig0.savefig(figure_path + 'plot_psychometric_HL' + '.png')
    fig0.savefig(figure_path+'plot_psychometric_HL'+'.eps', format='eps', dpi=1000)

    plt.show()
plot_psychometric_HL()


#"""
#plot_psychometric_RDM
def plot_psychometric_RDM():
    choice_perf_RDM_type3 = np.load(data_path_type3+'choice_perf_RDM_type3.npy')
    mean_RDM_type3 = np.mean(choice_perf_RDM_type3,axis=0)
    error_RDM_type3 = np.std(choice_perf_RDM_type3, axis=0) / np.sqrt(choice_perf_RDM_type3.shape[0])


    x_type3 = cohs_RDM
    y_para_type3 = mean_RDM_type3
    err_type3 = error_RDM_type3

    weights_type3 = [1.0/max(_,0.01) for _ in err_type3]
    def logistic_growth_type3(x, A1, A2, x_0, p):
        return A2 + (A1-A2)/(1+(x/x_0)**p)

    x_plot_type3 = np.linspace(0, 0.8, 100)
    paras_type3, paras_cov_type3 = curve_fit(logistic_growth_type3, x_type3, y_para_type3,absolute_sigma=True,sigma = weights_type3)
    para_curve_type3 = logistic_growth_type3(x_plot_type3, *paras_type3)


    choice_perf_RDM_type2 = np.load(data_path_type2+'choice_perf_RDM_type2.npy')
    mean_RDM_type2 = np.mean(choice_perf_RDM_type2,axis=0)
    error_RDM_type2 = np.std(choice_perf_RDM_type2, axis=0) / np.sqrt(choice_perf_RDM_type2.shape[0])

    x_type2 = cohs_RDM
    y_para_type2 = mean_RDM_type2
    err_type2 = error_RDM_type2

    weights_type2 = [1.0/max(_,0.2) for _ in err_type2]
    def logistic_growth(x, A1, A2, x_0, p):
        return A2 + (A1-A2)/(1+(x/x_0)**p)

    x_plot_type2 = np.linspace(0, 0.8, 100)
    paras_type2, paras_cov_type2 = curve_fit(logistic_growth, x_type2, y_para_type2,absolute_sigma=True,sigma = weights_type2)
    para_curve_type2 = logistic_growth(x_plot_type2, *paras_type2)


    ##===================== plot =====================#####
    fig0 = plt.figure(figsize=(3., 3.0))
    ax0 = fig0.add_axes([0.22, 0.23, 0.75, 0.68])
    font_ticket = 12
    plt.errorbar(x_type3,y_para_type3, err_type3, color = 'black', fmt = 'o')
    plt.plot(x_plot_type3, para_curve_type3, color = 'black', label = "PFC-MD")

    plt.errorbar(x_type2,y_para_type2, err_type2, color = 'tab:blue', fmt = 'o')
    plt.plot(x_plot_type2, para_curve_type2, color = 'tab:blue', label = "PFC")


    plt.title('RDM subtask', fontsize=font_ticket+1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=font_ticket)
    plt.yticks([0.3, 0.5,0.7,0.9], fontsize=font_ticket)
    plt.xlabel('Motion coherence', fontsize=font_ticket)
    plt.ylabel('Proportion correct', fontsize=font_ticket)
    plt.legend(fontsize=10)
    fig0.savefig(figure_path + 'plot_psychometric_RDM' + '.png')
    fig0.savefig(figure_path + 'plot_psychometric_RDM' + '.eps', format='eps', dpi=1000)

    plt.show()

plot_psychometric_RDM()


######################################
hp=default.get_default_hp(rule_name='RDM_task',random_seed=1)
fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'psychometric/')
tools.mkdir_p(figure_path)
data_path_type3 = os.path.join(hp['root_path'], 'Datas','psychometric/','type3'+'/')
tools.mkdir_p(data_path_type3)
data_path_type2 = os.path.join(hp['root_path'], 'Datas','psychometric/','type2'+'/')
tools.mkdir_p(data_path_type2)

def fit_data_RDM(cohs_RDM,choice_perf_RDM_0):
    mean_RDM_0 = np.mean(choice_perf_RDM_0,axis=0)
    error_RDM_0 = np.std(choice_perf_RDM_0, axis=0) / np.sqrt(choice_perf_RDM_0.shape[0])

    x_0 = cohs_RDM
    y_para_0 = mean_RDM_0
    err_0 = error_RDM_0
    weights_0 = [1.0/max(_,0.01) for _ in err_0]
    def logistic_growth_0(x, A1, A2, x_0, p):
        return A2 + (A1-A2)/(1+(x/x_0)**p)

    x_plot_0 = np.linspace(0, 0.8, 100)
    paras_0, paras_cov_0 = curve_fit(logistic_growth_0, x_0, y_para_0,absolute_sigma=True,sigma = weights_0)
    para_curve_0 = logistic_growth_0(x_plot_0, *paras_0)
    return x_0,y_para_0, err_0, x_plot_0, para_curve_0
def plot_cue_sparsity_RDM():
    #plot_psychometric_RDM_type3
    cohs_RDM  = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    sparsity_1=0.0
    sparsity_2=0.2
    sparsity_3=0.5
    sparsity_4=0.6
    sparsity_5=0.7


    choice_perf_RDM_1 = np.load(data_path_type3+'choice_perf_RDM_type3_'+str(sparsity_1)+'.npy')
    choice_perf_RDM_2 = np.load(data_path_type3+'choice_perf_RDM_type3_'+str(sparsity_2)+'.npy')
    choice_perf_RDM_3 = np.load(data_path_type3+'choice_perf_RDM_type3_'+str(sparsity_3)+'.npy')
    choice_perf_RDM_4 = np.load(data_path_type3+'choice_perf_RDM_type3_'+str(sparsity_4)+'.npy')
    choice_perf_RDM_5 = np.load(data_path_type3+'choice_perf_RDM_type3_'+str(sparsity_5)+'.npy')

    ##### sparisity 1 ########
    x_1,y_para_1, err_1, x_plot_1, para_curve_1 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_1)
    x_2,y_para_2, err_2, x_plot_2, para_curve_2 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_2)
    x_3,y_para_3, err_3, x_plot_3, para_curve_3 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_3)
    x_4,y_para_4, err_4, x_plot_4, para_curve_4 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_4)
    x_5,y_para_5, err_5, x_plot_5, para_curve_5 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_5)


    ##===================== plot =====================#####
    fig0 = plt.figure(figsize=(3., 3.0))
    ax0 = fig0.add_axes([0.22, 0.23, 0.75, 0.68])
    font_ticket = 12


    plt.errorbar(x_1,y_para_1, err_1, color = 'black', alpha=1.0,fmt = 'o',label = "0")
    plt.plot(x_plot_1, para_curve_1, color = 'black',  alpha=1.0)


    plt.errorbar(x_3,y_para_3, err_3, color = 'olive',  alpha=1.0,fmt = 'o',label = "0.5")
    plt.plot(x_plot_3, para_curve_3, color = 'olive', alpha=1.0)


    plt.errorbar(x_5,y_para_5, err_5, color = 'peru',  alpha=1.0,fmt = 'o',label = "0.7")
    plt.plot(x_plot_5, para_curve_5, color = 'peru', alpha=1.0)


    #plt.title('RDM subtask', fontsize=font_ticket+1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=font_ticket)
    plt.yticks([0.3, 0.5,0.7,0.9], fontsize=font_ticket)
    plt.xlabel('Cue conherence', fontsize=font_ticket)
    plt.ylabel('Proportion correct', fontsize=font_ticket)
    plt.legend(fontsize=10)
    fig0.savefig(figure_path + 'sparsity_RDM' + '.png')
    fig0.savefig(figure_path + 'sparsity_RDM' + '.eps', format='eps', dpi=1000)
    plt.show()
plot_cue_sparsity_RDM()


def plot_psychometric_RDM_type2():
    cohs_RDM = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    sparsity_1=0.0
    sparsity_2=0.2
    sparsity_3=0.5
    sparsity_4=0.6
    sparsity_5=0.7


    choice_perf_RDM_1 = np.load(data_path_type2+'choice_perf_RDM_type2_'+str(sparsity_1)+'.npy')
    choice_perf_RDM_2 = np.load(data_path_type2+'choice_perf_RDM_type2_'+str(sparsity_2)+'.npy')
    choice_perf_RDM_3 = np.load(data_path_type2+'choice_perf_RDM_type2_'+str(sparsity_3)+'.npy')
    choice_perf_RDM_4 = np.load(data_path_type2+'choice_perf_RDM_type2_'+str(sparsity_4)+'.npy')
    choice_perf_RDM_5 = np.load(data_path_type2+'choice_perf_RDM_type2_'+str(sparsity_5)+'.npy')

    x_1,y_para_1, err_1, x_plot_1, para_curve_1 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_1)
    x_2,y_para_2, err_2, x_plot_2, para_curve_2 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_2)
    x_3,y_para_3, err_3, x_plot_3, para_curve_3 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_3)
    # x_4,y_para_4, err_4, x_plot_4, para_curve_4 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_4)
    x_5,y_para_5, err_5, x_plot_5, para_curve_5 = fit_data_RDM(cohs_RDM=cohs_RDM,choice_perf_RDM_0=choice_perf_RDM_5)


    ##===================== plot =====================#####
    fig0 = plt.figure(figsize=(3., 3.0))
    ax0 = fig0.add_axes([0.22, 0.23, 0.75, 0.68])
    font_ticket = 12


    plt.errorbar(x_1,y_para_1, err_1, color = 'tab:blue', alpha=1.0,fmt = 'o')
    plt.plot(x_plot_1, para_curve_1, color = 'tab:blue',  alpha=1.0,label = "s=0.0")



    plt.errorbar(x_3,y_para_3, err_3, color = 'olive',  alpha=1.0,fmt = 'o')
    plt.plot(x_plot_3, para_curve_3, color = 'olive', alpha=1.0,label = "s=0.5")



    plt.errorbar(x_5,y_para_5, err_5, color = 'peru',  alpha=1.0,fmt = 'o')
    plt.plot(x_plot_5, para_curve_5, color = 'peru', alpha=1.0,label = "s=0.7")


    plt.title(r'RDM_task: pfc_alone', fontsize=font_ticket+1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim([0.3,1])
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=font_ticket)
    plt.yticks([0.3, 0.5,0.7,0.9], fontsize=font_ticket)
    plt.ylabel('Proportion correct', fontsize=font_ticket)
    plt.legend(fontsize=9)
    fig0.savefig(figure_path + 'plot_psychometric_RDM_type2' + '.png')
    fig0.savefig(figure_path + 'plot_psychometric_RDM_type2' + '.eps', format='eps', dpi=1000)
    plt.show()
#plot_psychometric_RDM_type2()

def compare_two_type_under_sparsity_RDM():
    # plot_psychometric_RDM_type3
    cohs_RDM = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    sparsity_1 = 0.0
    sparsity_2 = 0.2
    sparsity_3 = 0.5
    sparsity_4 = 0.6
    sparsity_5 = 0.7

    choice_perf_RDM_type2 = np.load(data_path_type3 + 'choice_perf_RDM_type3_' + str(sparsity_2) + '.npy')


    choice_perf_RDM_type3 = np.load(data_path_type2 + 'choice_perf_RDM_type2_' + str(sparsity_2) + '.npy')

    ##### sparisity 1 ########
    x_type2, y_para_type2, err_type2, x_plot_type2, para_curve_type2 = fit_data_RDM(cohs_RDM=cohs_RDM, choice_perf_RDM_0=choice_perf_RDM_type2)
    x_type3, y_para_type3, err_type3, x_plot_type3, para_curve_type3 = fit_data_RDM(cohs_RDM=cohs_RDM, choice_perf_RDM_0=choice_perf_RDM_type3)

    ##===================== plot =====================#####
    fig0 = plt.figure(figsize=(3., 3.0))
    ax0 = fig0.add_axes([0.22, 0.23, 0.75, 0.68])
    font_ticket = 12

    plt.errorbar(x_type3, y_para_type3, err_type3, color='black', fmt='o')
    plt.plot(x_plot_type3, para_curve_type3, color='black', label="PFC-MD")

    plt.errorbar(x_type2, y_para_type2, err_type2, color='tab:blue', fmt='o')
    plt.plot(x_plot_type2, para_curve_type2, color='tab:blue', label="PFC")

    plt.title('RDM subtask', fontsize=font_ticket+1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=font_ticket)
    plt.yticks([0.3, 0.5, 0.7, 0.9], fontsize=font_ticket)
    plt.xlabel('Cue conherence', fontsize=font_ticket)
    plt.ylabel('Proportion correct', fontsize=font_ticket)
    plt.legend(fontsize=9)
    fig0.savefig(figure_path + 'compare_two_type_under_sparsity_RDM' + '.png')
    fig0.savefig(figure_path + 'compare_two_type_under_sparsity_RDM' + '.eps', format='eps', dpi=1000)
    plt.show()

#compare_two_type_under_sparsity_RDM()




#======================================================
fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'psychometric/')
tools.mkdir_p(figure_path)
data_path_type3 = os.path.join(hp['root_path'], 'Datas','psychometric/','type3'+'/')
tools.mkdir_p(data_path_type3)
data_path_type2 = os.path.join(hp['root_path'], 'Datas','psychometric/','type2'+'/')
tools.mkdir_p(data_path_type2)


def fit_data_HL(cohs_HL, choice_perf_HL_0):
    mean_HL_0 = np.mean(choice_perf_HL_0, axis=0)
    error_HL_0 = np.std(choice_perf_HL_0, axis=0) / np.sqrt(choice_perf_HL_0.shape[0])
    x_0 = cohs_HL
    y_para_0 = mean_HL_0
    err_0 = error_HL_0
    weights_0 = [1.0 / max(_, 0.01) for _ in err_0]

    def logistic_growth_0(x, A1, A2, x_0, p):
        return A2 + (A1 - A2) / (1 + (x / x_0) ** p)

    x_plot_0 = np.linspace(0.5, 1.0, 100)
    paras_0, paras_cov_0 = curve_fit(logistic_growth_0, x_0, y_para_0, absolute_sigma=True, sigma=weights_0)
    para_curve_0 = logistic_growth_0(x_plot_0, *paras_0)
    return x_0, y_para_0, err_0, x_plot_0, para_curve_0
def psychometric_HL_diff_sparsity():
    #plot_psychometric_HL_type3
    cohs_HL  = np.array([0.5,0.6,0.65,0.67,0.7,0.72,0.8,0.9,1.0])
    sparsity_1=0.0
    sparsity_2=0.2
    sparsity_3=0.5
    sparsity_4=0.6
    sparsity_5=0.75

    choice_perf_HL_1 = np.load(data_path_type3+'choice_perf_HL_type3_'+str(sparsity_1)+'.npy')
    choice_perf_HL_2 = np.load(data_path_type3+'choice_perf_HL_type3_'+str(sparsity_2)+'.npy')
    choice_perf_HL_3 = np.load(data_path_type3+'choice_perf_HL_type3_'+str(sparsity_3)+'.npy')
    choice_perf_HL_4 = np.load(data_path_type3+'choice_perf_HL_type3_'+str(sparsity_4)+'.npy')
    choice_perf_HL_5 = np.load(data_path_type3+'choice_perf_HL_type3_'+str(sparsity_5)+'.npy')

    x_1,y_para_1, err_1, x_plot_1, para_curve_1 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_1)
    x_2,y_para_2, err_2, x_plot_2, para_curve_2 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_2)
    x_3,y_para_3, err_3, x_plot_3, para_curve_3 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_3)
    x_4,y_para_4, err_4, x_plot_4, para_curve_4 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_4)
    x_5,y_para_5, err_5, x_plot_5, para_curve_5 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_5)


    ##===================== plot =====================#####
    fig0 = plt.figure(figsize=(3., 3.0))
    ax0 = fig0.add_axes([0.22, 0.23, 0.75, 0.68])
    font_ticket = 12


    plt.errorbar(x_1,y_para_1, err_1, color = 'black', alpha=1.0,fmt = 'o',label = "0")
    plt.plot(x_plot_1, para_curve_1, color = 'black',  alpha=1.0)

    plt.errorbar(x_3,y_para_3, err_3, color = 'olive',  alpha=1.0,fmt = 'o',label = "0.5")
    plt.plot(x_plot_3, para_curve_3, color = 'olive', alpha=1.0)

    plt.errorbar(x_5,y_para_5, err_5, color = 'peru',  alpha=1.0,fmt = 'o',label = "0.7")
    plt.plot(x_plot_5, para_curve_5, color = 'peru', alpha=1.0)


    #plt.title('CAC subtask', fontsize=font_ticket+1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.5, 0.6, 0.7, 0.8,0.9,1.0], fontsize=12)
    plt.yticks([0.3, 0.5,0.7,0.9], fontsize=12)
    plt.xlabel('Cue congruence', fontsize=12)
    plt.ylabel('Proportion correct', fontsize=12)
    plt.legend(fontsize=10)
    fig0.savefig(figure_path + 'sparsity_HL' + '.png')
    fig0.savefig(figure_path + 'sparsity_HL' + '.eps', format='eps', dpi=1000)
    plt.show()
psychometric_HL_diff_sparsity()

def compare_two_type_under_sparsity_HL():
    #plot_psychometric_HL_type3
    cohs_HL  = np.array([0.5,0.6,0.65,0.67,0.7,0.72,0.8,0.9,1.0])
    sparsity_1=0.0
    sparsity_2=0.2
    sparsity_3=0.5
    sparsity_4=0.6
    sparsity_5=0.7

    choice_perf_HL_type3 = np.load(data_path_type3+'choice_perf_HL_type3_'+str(sparsity_2)+'.npy')
    choice_perf_HL_type2 = np.load(data_path_type2+'choice_perf_HL_type2_'+str(sparsity_2)+'.npy')


    x_type2,y_para_type2, err_type2, x_plot_type2, para_curve_type2 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_type2)
    x_type3,y_para_type3, err_type3, x_plot_type3, para_curve_type3 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_type3)



    ##===================== plot =====================#####
    fig0 = plt.figure(figsize=(3., 3.0))
    ax0 = fig0.add_axes([0.22, 0.23, 0.75, 0.68])
    font_ticket = 12

    plt.errorbar(x_type3, y_para_type3, err_type3, color='black', fmt='o')
    plt.plot(x_plot_type3, para_curve_type3, color='black', label="PFC-MD")

    plt.errorbar(x_type2, y_para_type2, err_type2, color='tab:blue', fmt='o')
    plt.plot(x_plot_type2, para_curve_type2, color='tab:blue', label="PFC")


    plt.title('CAC subtask', fontsize=font_ticket+1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.5, 0.6, 0.7, 0.8,0.9,1.0], fontsize=12)
    plt.yticks([0.3, 0.5,0.7,0.9], fontsize=12)
    plt.xlabel('Cue congruence', fontsize=font_ticket)
    plt.ylabel('Proportion correct', fontsize=12)
    plt.legend(fontsize=10)
    fig0.savefig(figure_path + 'compare_two_type_under_sparsity_H' + '.png')
    fig0.savefig(figure_path + 'compare_two_type_under_sparsity_H' + '.eps', format='eps', dpi=1000)
    plt.show()
compare_two_type_under_sparsity_HL()


def plot_psychometric_HL_type2():

    cohs_HL  = np.array([0.5,0.6,0.65,0.67,0.7,0.72,0.8,0.9,1.0])
    sparsity_1=0.0
    sparsity_2=0.2
    sparsity_3=0.5
    sparsity_4=0.6
    sparsity_5=0.7

    choice_perf_HL_1 = np.load(data_path_type2+'choice_perf_HL_type2_'+str(sparsity_1)+'.npy')
    choice_perf_HL_2 = np.load(data_path_type2+'choice_perf_HL_type2_'+str(sparsity_2)+'.npy')
    choice_perf_HL_3 = np.load(data_path_type2+'choice_perf_HL_type2_'+str(sparsity_3)+'.npy')
    choice_perf_HL_4 = np.load(data_path_type2+'choice_perf_HL_type2_'+str(sparsity_4)+'.npy')
    choice_perf_HL_5 = np.load(data_path_type2+'choice_perf_HL_type2_'+str(sparsity_5)+'.npy')


    x_1,y_para_1, err_1, x_plot_1, para_curve_1 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_1)
    x_2,y_para_2, err_2, x_plot_2, para_curve_2 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_2)
    x_3,y_para_3, err_3, x_plot_3, para_curve_3 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_3)
    x_4,y_para_4, err_4, x_plot_4, para_curve_4 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_4)
    x_5,y_para_5, err_5, x_plot_5, para_curve_5 = fit_data_HL(cohs_HL=cohs_HL,choice_perf_HL_0=choice_perf_HL_5)


    ##===================== plot =====================#####
    fig = plt.figure(figsize=(2.7,2.5))
    ax = fig.add_axes([0.22,0.2,0.75,0.75])
    font_ticket = 11

    plt.errorbar(x_1,y_para_1, err_1, color = 'tab:blue', alpha=1.0,fmt = 'o')
    plt.plot(x_plot_1, para_curve_1, color = 'tab:blue',  alpha=1.0,label = "0.0")


    plt.errorbar(x_3,y_para_3, err_3, color = 'olive',  alpha=1.0,fmt = 'o')
    plt.plot(x_plot_3, para_curve_3, color = 'olive', alpha=1.0,label = "0.5")

    plt.errorbar(x_5,y_para_5, err_5, color = 'peru',  alpha=1.0,fmt = 'o')
    plt.plot(x_plot_5, para_curve_5, color = 'peru', alpha=1.0,label = "0.7")


    plt.title('CAC_task', fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.5, 0.6, 0.7, 0.8,0.9,1.0], fontsize=font_ticket)
    plt.yticks([0.3, 0.5,0.7,0.9], fontsize=font_ticket)
    plt.ylim([0.3,1])
    plt.ylabel('Proportion correct', fontsize=font_ticket)
    plt.legend(fontsize=10)
    plt.show()
#"""






################## histogram
#'''
fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'psychometric/')
tools.mkdir_p(figure_path)

data_path = os.path.join(hp['root_path'], 'Datas','psychometric/','error/')
tools.mkdir_p(data_path)


#plot_error_HL
def plot_error_HL():
    perf_choices_coh = np.load(data_path+'perf_choices_coh_HL.npy')
    rule_errors_coh = np.load(data_path+'rule_errors_coh_HL.npy')
    sensory_errors_coh = np.load(data_path+'sensory_errors_coh_HL.npy')
    both_errors_coh = np.load(data_path+'both_errors_coh_HL.npy')

    perf_choices_coh_mean = np.mean(perf_choices_coh,axis=1)
    perf_choices_coh_std = np.std(perf_choices_coh,axis=1)

    rule_errors_coh_mean = np.mean(rule_errors_coh,axis=1)
    rule_errors_coh_std = np.std(rule_errors_coh,axis=1)
    sensory_errors_coh_mean = np.mean(sensory_errors_coh,axis=1)
    sensory_errors_coh_std = np.std(sensory_errors_coh,axis=1)
    both_errors_coh_mean = np.mean(both_errors_coh,axis=1)
    both_errors_coh_std = np.std(both_errors_coh,axis=1)


    IT_0 = [perf_choices_coh_mean[0], rule_errors_coh_mean[0], sensory_errors_coh_mean[0], both_errors_coh_mean[0]]
    IT_std_0 = [perf_choices_coh_std[0], rule_errors_coh_std[0], sensory_errors_coh_std[0], both_errors_coh_std[0]]

    IT_1 = [perf_choices_coh_mean[1], rule_errors_coh_mean[1], sensory_errors_coh_mean[1], both_errors_coh_mean[1]]
    IT_std_1 = [perf_choices_coh_std[1], rule_errors_coh_std[1], sensory_errors_coh_std[1], both_errors_coh_std[1]]

    IT_2 = [perf_choices_coh_mean[2], rule_errors_coh_mean[2], sensory_errors_coh_mean[2], both_errors_coh_mean[2]]
    IT_std_2 = [perf_choices_coh_std[2], rule_errors_coh_std[2], sensory_errors_coh_std[2], both_errors_coh_std[2]]


    barWidth = 0.25
    fig = plt.figure(figsize =(3., 2.6))
    ax = fig.add_axes([0.22, 0.23, 0.75, 0.68])
    font_ticket = 12

    br1 = np.arange(len(IT_0))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, IT_2, yerr = IT_std_2,color ='#F7903D', alpha=0.9,width = barWidth,
            label =str(0.7))
    plt.bar(br2, IT_1, yerr = IT_std_1,color ='#4D85BD', alpha=0.9,width = barWidth,
            label =str(0.65))
    plt.bar(br3, IT_0, yerr = IT_std_0,color ='#59A95A', alpha=0.9,width = barWidth,
            label =str(0.6))

    plt.ylabel('Trial fraction', fontsize = font_ticket)
    plt.xticks([r + barWidth for r in range(len(IT_0))],
               ['correct', 'rule', 'sensory', 'both'],fontsize = font_ticket)
    ax.tick_params(axis='x', rotation=35)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('CAC subtask',fontsize=font_ticket+1)
    plt.yticks([0.0,0.3,0.6,0.9],fontsize = font_ticket)
    plt.legend()
    fig.savefig(figure_path + 'error_HL' + '.png')
    fig.savefig(figure_path + 'error_HL' + '.eps', format='eps', dpi=1000)
    plt.show()

plot_error_HL()


#plot_error_RDM
def plot_error_RDM():
    perf_choices_coh = np.load(data_path+'perf_choices_coh_RDM.npy')
    rule_errors_coh = np.load(data_path+'rule_errors_coh_RDM.npy')
    sensory_errors_coh = np.load(data_path+'sensory_errors_coh_RDM.npy')
    both_errors_coh = np.load(data_path+'both_errors_coh_RDM.npy')

    perf_choices_coh_mean = np.mean(perf_choices_coh,axis=1)
    perf_choices_coh_std = np.std(perf_choices_coh,axis=1)

    rule_errors_coh_mean = np.mean(rule_errors_coh,axis=1)
    rule_errors_coh_std = np.std(rule_errors_coh,axis=1)
    sensory_errors_coh_mean = np.mean(sensory_errors_coh,axis=1)
    sensory_errors_coh_std = np.std(sensory_errors_coh,axis=1)
    both_errors_coh_mean = np.mean(both_errors_coh,axis=1)
    both_errors_coh_std = np.std(both_errors_coh,axis=1)



    IT_0 = [perf_choices_coh_mean[0], rule_errors_coh_mean[0], sensory_errors_coh_mean[0], both_errors_coh_mean[0]]
    IT_std_0 = [perf_choices_coh_std[0], rule_errors_coh_std[0], sensory_errors_coh_std[0], both_errors_coh_std[0]]

    IT_1 = [perf_choices_coh_mean[1], rule_errors_coh_mean[1], sensory_errors_coh_mean[1], both_errors_coh_mean[1]]
    IT_std_1 = [perf_choices_coh_std[1], rule_errors_coh_std[1], sensory_errors_coh_std[1], both_errors_coh_std[1]]

    IT_2 = [perf_choices_coh_mean[2], rule_errors_coh_mean[2], sensory_errors_coh_mean[2], both_errors_coh_mean[2]]
    IT_std_2 = [perf_choices_coh_std[2], rule_errors_coh_std[2], sensory_errors_coh_std[2], both_errors_coh_std[2]]


    barWidth = 0.25
    fig = plt.figure(figsize=(3.0, 2.6))
    ax = fig.add_axes([0.22, 0.23, 0.75, 0.68])
    font_ticket = 12

    br1 = np.arange(len(IT_0))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, IT_2, yerr = IT_std_2,color ='#F7903D', alpha=0.9,width = barWidth,
            label =str(0.4))
    plt.bar(br2, IT_1, yerr = IT_std_1,color ='#4D85BD', alpha=0.9,width = barWidth,
            label =str(0.2))
    plt.bar(br3, IT_0, yerr = IT_std_0,color ='#59A95A', alpha=0.9,width = barWidth,
            label =str(0.1))


    plt.ylabel('Trial fraction', fontsize = font_ticket)
    plt.xticks([r + barWidth for r in range(len(IT_0))],['correct', 'rule', 'sensory', 'both'],fontsize = 12)
    ax.tick_params(axis='x', rotation=35)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('RDM subtask',fontsize=font_ticket+1)
    plt.yticks([0.0,0.3,0.6,0.9],fontsize = font_ticket)
    plt.legend()
    fig.savefig(figure_path + 'error_RDM' + '.png')
    fig.savefig(figure_path + 'error_RDM' + '.eps', format='eps', dpi=1000)
    plt.show()
plot_error_RDM()

#'''


