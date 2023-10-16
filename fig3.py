import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ranksums
from scipy.stats import wilcoxon
import seaborn as sns
import default
import tools
from scipy.io import savemat


hp=default.get_default_hp(rule_name='RDM_task')
hp['rng']=np.random.RandomState(0)



fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'SNR_lib/')
tools.mkdir_p(figure_path)

data_path = os.path.join(hp['root_path'], 'Datas','SNR_lib_1/')
tools.mkdir_p(data_path)


#plot_SNR_RDM_ifMD
SNR_dB_10s_cue = np.load(data_path + 'md_RDM_10s_cue.npy')
SNR_dB_20s_cue = np.load(data_path + 'mdno_RDM_20s_cue.npy')

SNR_dB_10s_delay = np.load(data_path + 'md_RDM_10s_delay.npy')
SNR_dB_20s_delay = np.load(data_path + 'mdno_RDM_20s_delay.npy')


def scatter_plot_SNR_RDM_ifMD():
    clrs =sns.color_palette("muted")
    # clrs = ['lightseagreen','coral','tab:purple']

    fig = plt.figure(figsize=(2.5,2.5))
    ax = fig.add_axes([0.2,0.15,0.75,0.75])
    plt.plot([-0.1,2],[-0.1,2],'--',color='grey')

    plt.scatter(SNR_dB_10s_cue[0:60],SNR_dB_20s_cue[0:60],marker="o",color=clrs[0],
                edgecolors='white',linewidths=0.5,label='cue period')
    plt.scatter(SNR_dB_10s_delay[0:60],SNR_dB_20s_delay[0:60],marker="o",color=clrs[1],
                edgecolors='white',linewidths=0.5,label='delay period')

    plt.xlim([-0.07,1.77])
    plt.ylim([-0.07,1.77])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.0,0.4,0.8,1.2,1.6], fontsize=12)
    plt.yticks([0.0,0.4,0.8,1.2,1.6], fontsize=12)
    # plt.xlabel('PFC-MD',fontsize=12)
    # plt.ylabel('no-MD',fontsize=12)
    #plt.title('PFC-MD vs no-MD exc (RDM)')
    #plt.legend()
    plt.savefig(figure_path + 'SNR_RDM_ifMD.png')
    fig.savefig(figure_path + 'SNR_RDM_ifMD.eps', format='eps', dpi=1000)
    plt.show()
#scatter_plot_SNR_RDM_ifMD()



dif_cue = SNR_dB_10s_cue-SNR_dB_20s_cue
dif_delay = SNR_dB_10s_cue-SNR_dB_20s_cue
res_cue = wilcoxon(dif_cue)
res_delay = wilcoxon(dif_delay)




####plot_SNR_RDM
def plot_SNR_RDM():
    SNR_cue_10s = np.load(data_path+'RDM_SNR_10s_cue_exc.npy')
    SNR_cue_20s  = np.load(data_path+'RDM_SNR_20s_cue_exc.npy')
    SNR_cue_30s  = np.load(data_path+'RDM_SNR_30s_cue_exc.npy')
    SNR_cue_30s = np.load(data_path+'md_RDM_10s_cue.npy')
    SNR_cue_10s = SNR_cue_10s
    SNR_cue_20s = SNR_cue_20s
    SNR_cue_30s = SNR_cue_30s

    SNR_10s_cue_mean = np.mean(SNR_cue_10s)
    SNR_20s_cue_mean = np.mean(SNR_cue_20s)
    SNR_30s_cue_mean = np.mean(SNR_cue_30s)
    SNR_10s_cue_std = np.std(SNR_cue_10s)/np.sqrt(SNR_cue_10s.shape[0])
    SNR_20s_cue_std = np.std(SNR_cue_10s)/np.sqrt(SNR_cue_20s.shape[0])
    SNR_30s_cue_std = np.std(SNR_cue_10s)/np.sqrt(SNR_cue_30s.shape[0])


    SNR_delay_10s = np.load(data_path+'RDM_SNR_10s_delay_exc.npy')
    SNR_delay_20s  = np.load(data_path+'RDM_SNR_20s_delay_exc.npy')
    SNR_delay_30s = np.load(data_path+'md_RDM_10s_delay.npy')

    SNR_delay_10s = SNR_delay_10s
    SNR_delay_20s = SNR_delay_20s
    SNR_delay_30s = SNR_delay_30s

    SNR_10s_delay_mean = np.mean(SNR_delay_10s)
    SNR_20s_delay_mean = np.mean(SNR_delay_20s)
    SNR_30s_delay_mean = np.mean(SNR_delay_30s)
    SNR_10s_delay_std = np.std(SNR_delay_10s)/np.sqrt(SNR_delay_10s.shape[0])
    SNR_20s_delay_std = np.std(SNR_delay_10s)/np.sqrt(SNR_delay_20s.shape[0])
    SNR_30s_delay_std = np.std(SNR_delay_10s)/np.sqrt(SNR_delay_30s.shape[0])



    IT_1 = [SNR_10s_cue_mean, SNR_10s_delay_mean]
    IT_std_1 = [SNR_10s_cue_std, SNR_10s_delay_std]

    IT_2 = [SNR_20s_cue_mean, SNR_20s_delay_mean]
    IT_std_2 = [SNR_20s_cue_std, SNR_20s_delay_std]

    IT_3 = [SNR_30s_cue_mean, SNR_30s_delay_mean]
    IT_std_3 = [SNR_30s_cue_std, SNR_30s_delay_std]

    clrs = sns.color_palette("Blues", 3)

    barWidth = 0.2
    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_axes([0.2, 0.15, 0.75, 0.75])
    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]


    plt.bar(br1, IT_1, yerr = IT_std_1,color =clrs[0], width = barWidth, label =str(0.3))
    plt.bar(br2, IT_2, yerr = IT_std_2,color =clrs[1], width = barWidth, label =str(0.67))
    plt.bar(br3, IT_3, yerr = IT_std_3,color =clrs[2], width = barWidth, label =str(0.9))


    plt.ylabel('SNR', fontsize = 12)
    plt.xticks([r + barWidth for r in range(len(IT_1))],
               ['cue', 'delay'],fontsize = 15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.title('SNR for RDM')
    plt.yticks([0.0,0.5,1.0,1.5,2.0],fontsize = 12)

    plt.legend(fontsize=8)
    plt.savefig(figure_path + 'SNR_RDM.png')
    fig.savefig(figure_path + 'SNR_RDM.eps', format='eps', dpi=1000)
    plt.show()

    p12_cue=ranksums(SNR_cue_10s, SNR_cue_20s)
    p13_cue=ranksums(SNR_cue_10s, SNR_cue_30s)
    p23_cue=ranksums(SNR_cue_20s, SNR_cue_30s)

    p12_delay=ranksums(SNR_delay_10s, SNR_delay_20s)
    p13_delay=ranksums(SNR_delay_10s, SNR_delay_30s)
    p23_delay=ranksums(SNR_delay_20s, SNR_delay_30s)
    print('cue: 0.3 and  0.67',p12_cue)
    print('cue: 0.3 and  0.9 ',p13_cue)
    print('cue: 0.67 and 0.9',p23_cue)

    print('delay: 0.3 and  0.67',p12_delay)
    print('delay: 0.3 and  0.9 ',p13_delay)
    print('delay: 0.67 and 0.9 ',p23_delay)

#plot_SNR_RDM()




#plot_SNR_HL_ifMD
SNR_dB_10s_cue = np.load(data_path+'md_HL_10s_cue.npy')
SNR_dB_20s_cue  = np.load(data_path+'mdno_HL_20s_cue.npy')

SNR_dB_10s_delay = np.load(data_path+'md_HL_10s_delay.npy')
SNR_dB_20s_delay  = np.load(data_path+'mdno_HL_20s_delay.npy')

SNR_dB_10s_delay = SNR_dB_10s_delay[0:60]
SNR_dB_20s_delay = SNR_dB_20s_delay[0:60]

def scatter_plot_SNR_HL_ifMD():
    clrs =sns.color_palette("muted")

    fig = plt.figure(figsize=(2.5,2.5))
    ax = fig.add_axes([0.2,0.15,0.75,0.75])
    plt.plot([-0.07,2],[-0.07,2],'--',color='grey')

    plt.scatter(SNR_dB_10s_cue[0:60],SNR_dB_20s_cue[0:60],marker="o",color=clrs[0],
                edgecolors='white', linewidths=0.5,label='cue period')
    plt.scatter(SNR_dB_10s_delay[0:60],SNR_dB_20s_delay[0:60],marker="o",color=clrs[1],
                edgecolors='white', linewidths=0.5, label='delay period')

    plt.xlim([-0.07,1.77])
    plt.ylim([-0.07,1.77])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0.0,0.4,0.8,1.2,1.6], fontsize=12)
    plt.yticks([0.0,0.4,0.8,1.2,1.6], fontsize=12)
    # plt.xlabel('PFC-MD',fontsize=12)
    # plt.ylabel('no-MD',fontsize=12)
    #plt.title('PFC-MD vs no-MD exc (HL)')
    #plt.legend()
    plt.savefig(figure_path + 'SNR_HL_ifMD.png')
    fig.savefig(figure_path + 'SNR_HL_ifMD.eps', format='eps', dpi=1000)
    plt.show()

#scatter_plot_SNR_HL_ifMD()

dif_cue = SNR_dB_10s_cue-SNR_dB_20s_cue
dif_delay = SNR_dB_10s_cue-SNR_dB_20s_cue
res_cue = wilcoxon(dif_cue)
res_delay = wilcoxon(dif_delay)

print('HL res_cue',res_cue)
print('HL res_delay',res_delay)



#plot_SNR_HL
def plot_SNR_HL():
    SNR_cue_10s = np.load(data_path+'HL_SNR_10s_cue_exc.npy')
    SNR_cue_20s  = np.load(data_path+'HL_SNR_20s_cue_exc.npy')
    SNR_cue_30s  = np.load(data_path+'md_HL_10s_cue.npy')

    SNR_cue_10s =SNR_cue_10s[0:60]
    SNR_cue_20s =SNR_cue_20s[0:60]
    SNR_cue_30s =SNR_cue_30s[0:60]

    SNR_10s_cue_mean = np.mean(SNR_cue_10s)
    SNR_20s_cue_mean = np.mean(SNR_cue_20s)
    SNR_30s_cue_mean = np.mean(SNR_cue_30s)
    SNR_10s_cue_std = np.std(SNR_cue_10s)/np.sqrt(SNR_cue_10s.shape[0])
    SNR_20s_cue_std = np.std(SNR_cue_10s)/np.sqrt(SNR_cue_20s.shape[0])
    SNR_30s_cue_std = np.std(SNR_cue_10s)/np.sqrt(SNR_cue_30s.shape[0])


    SNR_delay_10s = np.load(data_path+'HL_SNR_10s_delay_exc.npy')
    SNR_delay_20s  = np.load(data_path+'HL_SNR_20s_delay_exc.npy')
    SNR_delay_30s  = np.load(data_path+'md_HL_10s_delay.npy')
    SNR_delay_10s = SNR_delay_10s[0:60]
    SNR_delay_20s = SNR_delay_20s[0:60]
    SNR_delay_30s = SNR_delay_30s[0:60]


    SNR_10s_delay_mean = np.mean(SNR_delay_10s)
    SNR_20s_delay_mean = np.mean(SNR_delay_20s)
    SNR_30s_delay_mean = np.mean(SNR_delay_30s)
    SNR_10s_delay_std = np.std(SNR_delay_10s)/np.sqrt(SNR_delay_10s.shape[0])
    SNR_20s_delay_std = np.std(SNR_delay_10s)/np.sqrt(SNR_delay_20s.shape[0])
    SNR_30s_delay_std = np.std(SNR_delay_10s)/np.sqrt(SNR_delay_30s.shape[0])



    IT_1 = [SNR_10s_cue_mean, SNR_10s_delay_mean]
    IT_std_1 = [SNR_10s_cue_std, SNR_10s_delay_std]

    IT_2 = [SNR_20s_cue_mean, SNR_20s_delay_mean]
    IT_std_2 = [SNR_20s_cue_std, SNR_20s_delay_std]

    IT_3 = [SNR_30s_cue_mean, SNR_30s_delay_mean]
    IT_std_3 = [SNR_30s_cue_std, SNR_30s_delay_std]




    barWidth = 0.2
    fig = plt.figure(figsize=(2.5,2.5))
    ax = fig.add_axes([0.2, 0.15, 0.75, 0.75])
    clrs = sns.color_palette("Blues", 3)

    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, IT_1, yerr = IT_std_1,color =clrs[0], width = barWidth,label =str(0.6))
    plt.bar(br2, IT_2, yerr = IT_std_2,color =clrs[1], width = barWidth,label =str(0.75))
    plt.bar(br3, IT_3, yerr = IT_std_3,color =clrs[2], width = barWidth,label =str(0.9))

    plt.ylabel('SNR', fontsize = 12)
    plt.xticks([r + barWidth for r in range(len(IT_1))],
               ['cue', 'delay'],fontsize = 15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.title('SNR for HL')
    plt.yticks([0.0,0.5,1.0,1.5,2.0],fontsize = 12)

    plt.legend(fontsize=8)
    plt.savefig(figure_path + 'SNR_HL.png')
    fig.savefig(figure_path + 'SNR_HL.eps', format='eps', dpi=1000)
    plt.show()

    p12_cue=ranksums(SNR_cue_10s, SNR_cue_20s)
    p13_cue=ranksums(SNR_cue_10s, SNR_cue_30s)
    p23_cue=ranksums(SNR_cue_20s, SNR_cue_30s)

    p12_delay=ranksums(SNR_delay_10s, SNR_delay_20s)
    p13_delay=ranksums(SNR_delay_10s, SNR_delay_30s)
    p23_delay=ranksums(SNR_delay_20s, SNR_delay_30s)
    print('cue: 0.6 and  0.76',p12_cue)
    print('cue: 0.6 and  0.9 ',p13_cue)
    print('cue: 0.76 and 0.9',p23_cue)

    print('delay: 0.6 and  0.76',p12_delay)
    print('delay: 0.6 and  0.9 ',p13_delay)
    print('delay: 0.76 and 0.9 ',p23_delay)

#plot_SNR_HL()





#===========================================================
fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'SNR_lib/')
tools.mkdir_p(figure_path)

data_path = os.path.join(hp['root_path'], 'Datas','CC_plot/')
tools.mkdir_p(data_path)

#plot_CCA_RDM
def plot_CCA_RDM():
    corr_cue_1s=np.load(data_path+'corr_RDM_cue_1s.npy')
    corr_cue_2s=np.load(data_path+'corr_RDM_cue_2s.npy')
    corr_cue_3s=np.load(data_path+'corr_RDM_cue_3s.npy')
    corr_cue_1s=np.abs(corr_cue_1s)
    corr_cue_2s=np.abs(corr_cue_2s)
    corr_cue_3s=np.abs(corr_cue_3s)


    corr_RDM_cue_dic = {"corr_RDM_cue_1s": corr_cue_1s,
                    "corr_RDM_cue_2s": corr_cue_2s,
                    "corr_RDM_cue_3s": corr_cue_3s}
    savemat(data_path+"corr_RDM_cue_dic.mat",corr_RDM_cue_dic)

    p12_cue=ranksums(corr_cue_1s, corr_cue_2s)
    p13_cue=ranksums(corr_cue_1s,corr_cue_3s)
    p23_cue=ranksums(corr_cue_2s, corr_cue_3s)


    corr_cue_1_mean = np.mean(corr_cue_1s)
    corr_cue_2_mean = np.mean(corr_cue_2s)
    corr_cue_3_mean = np.mean(corr_cue_3s)
    corr_cue_1_std = np.std(corr_cue_1s)
    corr_cue_2_std = np.std(corr_cue_2s)
    corr_cue_3_std = np.std(corr_cue_3s)

    corr_delay_1s=np.load(data_path+'corr_RDM_delay_1s.npy')
    corr_delay_2s=np.load(data_path+'corr_RDM_delay_2s.npy')
    corr_delay_3s=np.load(data_path+'corr_RDM_delay_3s.npy')
    corr_delay_1s=np.abs(corr_delay_1s)
    corr_delay_2s=np.abs(corr_delay_2s)
    corr_delay_3s=np.abs(corr_delay_3s)


    corr_RDM_delay_dic = {"corr_RDM_delay_1s": corr_delay_1s,
                    "corr_RDM_delay_2s": corr_delay_2s,
                    "corr_RDM_delay_3s": corr_delay_3s}
    savemat(data_path+"corr_RDM_delay_dic.mat",corr_RDM_delay_dic )



    corr_delay_1_mean = np.mean(corr_delay_1s)
    corr_delay_2_mean = np.mean(corr_delay_2s)
    corr_delay_3_mean = np.mean(corr_delay_3s)
    corr_delay_1_std = np.std(corr_delay_1s)
    corr_delay_2_std = np.std(corr_delay_2s)
    corr_delay_3_std = np.std(corr_delay_3s)


    IT_1     = [corr_cue_1_mean, corr_delay_1_mean]
    IT_std_1 = [corr_cue_1_std, corr_delay_1_std]

    IT_2     = [corr_cue_2_mean, corr_delay_2_mean]
    IT_std_2 = [corr_cue_2_std,  corr_delay_2_std]

    IT_3     = [corr_cue_3_mean, corr_delay_3_mean]
    IT_std_3 = [corr_cue_3_std,  corr_delay_3_std]

    barWidth = 0.2
    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_axes([0.2, 0.15, 0.75, 0.75])
    clrs = sns.color_palette("Blues", 3)

    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, IT_1, yerr = IT_std_1,color =clrs[0], width = barWidth, label =str(0.1))
    plt.bar(br2, IT_2, yerr = IT_std_2,color =clrs[1], width = barWidth, label =str(0.5))
    plt.bar(br3, IT_3, yerr = IT_std_3,color =clrs[2], width = barWidth, label =str(0.9))

    plt.ylabel('Corrcoef', fontsize = 12)
    plt.xticks([r + barWidth for r in range(len(IT_1))],
               ['cue', 'delay'],fontsize = 15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.title('CCA for RDM')
    plt.yticks([0.0,0.2,0.4,0.6],fontsize = 12)
    #plt.legend(fontsize=8)
    plt.savefig(figure_path + 'CCA_RDM.png')
    fig.savefig(figure_path + 'CCA_RDM.eps', format='eps', dpi=1000)
    plt.show()


    p12_delay=ranksums(corr_delay_1s, corr_delay_2s)
    p13_delay=ranksums(corr_delay_1s, corr_delay_3s)
    p23_delay=ranksums(corr_delay_2s, corr_delay_3s)
    print('delay: 0.1 and  0.5',p12_delay)
    print('delay: 0.1 and  0.9 ',p13_delay)
    print('delay: 0.5 and 0.9 ',p23_delay)

#plot_CCA_RDM()

#plot_CCA_HL
def plot_CCA_HL():
    corr_cue_1s=np.load(data_path+'corr_HL_cue_1s.npy')
    corr_cue_2s=np.load(data_path+'corr_HL_cue_2s.npy')
    corr_cue_3s=np.load(data_path+'corr_HL_cue_3s.npy')
    corr_cue_1s=np.abs(corr_cue_1s)
    corr_cue_2s=np.abs(corr_cue_2s)
    corr_cue_3s=np.abs(corr_cue_3s)

    corr_HL_cue_dic = {"corr_HL_cue_1s": corr_cue_1s,
                        "corr_HL_cue_2s": corr_cue_2s,
                        "corr_HL_cue_3s": corr_cue_3s}
    savemat(data_path+"corr_HL_cue_dic.mat",corr_HL_cue_dic)


    corr_cue_1_mean = np.mean(corr_cue_1s)
    corr_cue_2_mean = np.mean(corr_cue_2s)
    corr_cue_3_mean = np.mean(corr_cue_3s)
    corr_cue_1_std = np.std(corr_cue_1s)/np.sqrt(corr_cue_1s.shape[0])
    corr_cue_2_std = np.std(corr_cue_2s)/np.sqrt(corr_cue_2s.shape[0])
    corr_cue_3_std = np.std(corr_cue_3s)/np.sqrt(corr_cue_3s.shape[0])

    corr_delay_1s=np.load(data_path+'corr_HL_delay_1s.npy')
    corr_delay_2s=np.load(data_path+'corr_HL_delay_2s.npy')
    corr_delay_3s=np.load(data_path+'corr_HL_delay_3s.npy')
    corr_delay_1s=np.abs(corr_delay_1s)
    corr_delay_2s=np.abs(corr_delay_2s)
    corr_delay_3s=np.abs(corr_delay_3s)



    corr_HL_delay_dic = {"corr_HL_delay_1s": corr_delay_1s,
                       "corr_HL_delay_2s": corr_delay_2s,
                       "corr_HL_delay_3s": corr_cue_3s}
    savemat(data_path+"corr_HL_delay_dic.mat",corr_HL_delay_dic)


    corr_delay_1_mean = np.mean(corr_delay_1s)
    corr_delay_2_mean = np.mean(corr_delay_2s)
    corr_delay_3_mean = np.mean(corr_delay_3s)
    corr_delay_1_std = np.std(corr_delay_1s)
    corr_delay_2_std = np.std(corr_delay_2s)
    corr_delay_3_std = np.std(corr_delay_3s)


    IT_1     = [corr_cue_1_mean, corr_delay_1_mean]
    IT_std_1 = [corr_cue_1_std, corr_delay_1_std]

    IT_2     = [corr_cue_2_mean, corr_delay_2_mean]
    IT_std_2 = [corr_cue_2_std,  corr_delay_2_std]

    IT_3     = [corr_cue_3_mean, corr_delay_3_mean]
    IT_std_3 = [corr_cue_3_std,  corr_delay_3_std]


    barWidth = 0.2
    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_axes([0.2, 0.15, 0.75, 0.75])
    clrs = sns.color_palette("Blues", 3)
    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, IT_1, yerr = IT_std_1,color =clrs[0],width = barWidth,
            label =str(0.6))
    plt.bar(br2, IT_2, yerr = IT_std_2,color =clrs[1],width = barWidth,
            label =str(0.7))
    plt.bar(br3, IT_3, yerr = IT_std_3,color =clrs[2],width = barWidth,
            label =str(0.8))


    plt.ylabel('Corrcoef', fontsize = 12)
    plt.xticks([r + barWidth for r in range(len(IT_1))],
               ['cue', 'delay'],fontsize = 15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.title('CCA for HL')
    plt.yticks([0.1,0.3,0.5,0.7,0.9],fontsize = 12)
    #plt.legend(fontsize=8)
    plt.savefig(figure_path + 'CCA_HL.png')
    fig.savefig(figure_path + 'CCA_HL.eps', format='eps', dpi=1000)
    plt.show()


    alternative='two-sided'
    p12_cue=ranksums(corr_cue_1s, corr_cue_2s,alternative=alternative)
    p13_cue=ranksums(corr_cue_1s,corr_cue_3s,alternative=alternative)
    p23_cue=ranksums(corr_cue_2s, corr_cue_3s,alternative=alternative)

    p12_delay=ranksums(corr_delay_1s, corr_delay_2s,alternative=alternative)
    p13_delay=ranksums(corr_delay_1s, corr_delay_3s,alternative=alternative)
    p23_delay=ranksums(corr_delay_2s, corr_delay_3s,alternative=alternative)
    print('cue:   0.6 and  0.7',p12_cue)
    print('cue:   0.6 and  0.8',p13_cue)
    print('cue:   0.7 and  0.8',p23_cue)

    print('delay: 0.6 and  0.7',p12_delay)
    print('delay: 0.6 and  0.8',p13_delay)
    print('delay: 0.7 and  0.8',p23_delay)
    print('corr_1_mean,corr_2_mean,corr_3_mean')


    diff_12=corr_cue_1s-corr_cue_2s
    res_12 = wilcoxon(diff_12, alternative='greater')

#plot_CCA_HL()



#======================== decode
data_path = os.path.join(hp['root_path'], 'Datas','SNR_lib/')

#decode_3tpye_neuron
def decode_3tpye_neuron():
    acc_decode_context_by_exc = np.load(data_path+'acc_decode_context_by_exc.npy')
    acc_decode_context_by_inh = np.load(data_path+'acc_decode_context_by_inh.npy')
    acc_decode_rule_by_exc = np.load(data_path+'acc_decode_rule_by_exc.npy')
    acc_decode_rule_by_inh = np.load(data_path+'acc_decode_rule_by_inh.npy')
    acc_decode_context_by_MD = np.load(data_path+'acc_decode_context_by_MD.npy')
    acc_decode_rule_by_MD = np.load(data_path+'acc_decode_rule_by_MD.npy')



    acc_decode_context_by_exc_mean = np.mean(acc_decode_context_by_exc)
    acc_decode_context_by_inh_mean = np.mean(acc_decode_context_by_inh)
    acc_decode_rule_by_exc_mean    = np.mean(acc_decode_rule_by_exc)
    acc_decode_rule_by_inh_mean    = np.mean(acc_decode_rule_by_inh)
    acc_decode_context_by_MD_mean = np.mean(acc_decode_context_by_MD)
    acc_decode_rule_by_MD_mean    = np.mean(acc_decode_rule_by_MD)



    acc_decode_context_by_exc_std = np.std(acc_decode_context_by_exc)
    acc_decode_context_by_inh_std = np.std(acc_decode_context_by_inh)
    acc_decode_rule_by_exc_std    = np.std(acc_decode_rule_by_exc)
    acc_decode_rule_by_inh_std    = np.std(acc_decode_rule_by_inh)
    acc_decode_context_by_MD_std = np.std(acc_decode_context_by_MD)
    acc_decode_rule_by_MD_std    = np.std(acc_decode_rule_by_MD)
    IT_1     = [acc_decode_rule_by_exc_mean, acc_decode_context_by_exc_mean]
    IT_std_1 = [acc_decode_rule_by_exc_std, acc_decode_context_by_exc_std]

    IT_2     = [acc_decode_rule_by_inh_mean, acc_decode_context_by_inh_mean]
    IT_std_2 = [acc_decode_rule_by_inh_std, acc_decode_context_by_inh_std]

    IT_3     = [acc_decode_rule_by_MD_mean, acc_decode_context_by_MD_mean]
    IT_std_3 = [acc_decode_rule_by_MD_std, acc_decode_context_by_MD_std]



    ######## plot
    barWidth = 0.15
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.2, 0.15, 0.75, 0.75])
    clrs = sns.color_palette("Set2")
    #clrs = sns.color_palette("muted")


    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, IT_1, yerr = IT_std_1,color =clrs[0], alpha=1,width = barWidth,
            label ='exc')
    plt.bar(br2, IT_2, yerr = IT_std_2,color =clrs[1], alpha=1,width = barWidth,
            label ='inh')
    plt.bar(br3, IT_3, yerr = IT_std_3,color =clrs[2], alpha=1,width = barWidth,
            label ='MD')

    plt.ylabel('Accuracy', fontsize = 12)
    plt.xticks([r + barWidth for r in range(len(IT_1))],
               ['Rule', 'Context'],fontsize = 15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('decode')
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize = 12)
    plt.legend(fontsize=8)
    plt.savefig(figure_path + 'decode_3tpye_neuron.png')
    fig.savefig(figure_path + 'decode_3tpye_neuron.eps', format='eps', dpi=1000)
    plt.show()
#decode_3tpye_neuron()



#decode_WithDiffUncert_3ytpe_neuron
def decode_WithDiffUncert_3ytpe_neuron():
    acc_decode_rule_by_exc_high= np.load(data_path+'acc_decode_rule_by_exc_high.npy')
    acc_decode_rule_by_inh_high= np.load(data_path+'acc_decode_rule_by_inh_high.npy')
    acc_decode_rule_by_exc_mid= np.load(data_path+'acc_decode_rule_by_exc_mid.npy')
    acc_decode_rule_by_inh_mid= np.load(data_path+'acc_decode_rule_by_inh_mid.npy')
    acc_decode_rule_by_exc_low= np.load(data_path+'acc_decode_rule_by_exc_low.npy')
    acc_decode_rule_by_inh_low= np.load(data_path+'acc_decode_rule_by_inh_low.npy')
    acc_decode_context_by_MD_high= np.load(data_path+'acc_decode_context_by_MD_high.npy')
    acc_decode_context_by_MD_mid= np.load(data_path+'acc_decode_context_by_MD_mid.npy')
    acc_decode_context_by_MD_low= np.load(data_path+'acc_decode_context_by_MD_low.npy')



    acc_decode_rule_by_exc_high_mean = np.mean(acc_decode_rule_by_exc_high)
    acc_decode_rule_by_inh_high_mean = np.mean(acc_decode_rule_by_inh_high)

    acc_decode_rule_by_exc_mid_mean = np.mean(acc_decode_rule_by_exc_mid)
    acc_decode_rule_by_inh_mid_mean = np.mean(acc_decode_rule_by_inh_mid)

    acc_decode_rule_by_exc_low_mean = np.mean(acc_decode_rule_by_exc_low)
    acc_decode_rule_by_inh_low_mean = np.mean(acc_decode_rule_by_inh_low)


    acc_decode_context_by_MD_high_mean = np.mean(acc_decode_context_by_MD_high)
    acc_decode_context_by_MD_mid_mean = np.mean(acc_decode_context_by_MD_mid)
    acc_decode_context_by_MD_low_mean = np.mean(acc_decode_context_by_MD_low)


    acc_decode_rule_by_exc_high_std = np.std(acc_decode_rule_by_exc_high)
    acc_decode_rule_by_inh_high_std = np.std(acc_decode_rule_by_inh_high)

    acc_decode_rule_by_exc_mid_std = np.std(acc_decode_rule_by_exc_mid)
    acc_decode_rule_by_inh_mid_std = np.std(acc_decode_rule_by_inh_mid)

    acc_decode_rule_by_exc_low_std = np.std(acc_decode_rule_by_exc_low)
    acc_decode_rule_by_inh_low_std = np.std(acc_decode_rule_by_inh_low)


    acc_decode_context_by_MD_high_std = np.std(acc_decode_context_by_MD_high)
    acc_decode_context_by_MD_mid_std = np.std(acc_decode_context_by_MD_mid)
    acc_decode_context_by_MD_low_std = np.std(acc_decode_context_by_MD_low)


    IT_1     = [acc_decode_rule_by_exc_high_mean, acc_decode_rule_by_inh_high_mean,
                acc_decode_context_by_MD_high_mean]

    IT_std_1 = [acc_decode_rule_by_exc_high_std, acc_decode_rule_by_inh_high_std,
                acc_decode_context_by_MD_high_std]

    IT_2     = [acc_decode_rule_by_exc_mid_mean, acc_decode_rule_by_inh_mid_mean,
                acc_decode_context_by_MD_mid_mean]

    IT_std_2 = [acc_decode_rule_by_exc_mid_std, acc_decode_rule_by_inh_mid_std,
                acc_decode_context_by_MD_mid_std]

    IT_3     = [acc_decode_rule_by_exc_low_mean, acc_decode_rule_by_inh_low_mean,
                acc_decode_context_by_MD_low_mean]

    IT_std_3 = [acc_decode_rule_by_exc_low_std, acc_decode_rule_by_inh_low_std,
                acc_decode_context_by_MD_low_std]

    ######## plot
    barWidth = 0.15
    fig = plt.figure(figsize=(3.3, 2.5))
    ax = fig.add_axes([0.2, 0.15, 0.75, 0.75])
    clrs = sns.color_palette("Greys")
    # clrs = sns.color_palette("muted")
    br1 = np.arange(len(IT_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]


    # Make the plot
    plt.bar(br1, IT_1, yerr = IT_std_1,color =clrs[3],width = barWidth,
            label ='high')
    plt.bar(br2, IT_2, yerr = IT_std_2,color =clrs[2],width = barWidth,
            label ='mid')
    plt.bar(br3, IT_3, yerr = IT_std_3,color =clrs[1],width = barWidth,
            label ='low')


    #plt.ylabel('Accuracy', fontsize = 12)
    plt.xticks([r + barWidth for r in range(len(IT_1))],
               ['Exc', 'Inh','MD'],fontsize = 15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.title('decode by diff uncertainty')
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize = 12)

    #plt.legend(fontsize=8)
    plt.savefig(figure_path + 'decode_WithDiffUncert_3ytpe_neuron.png')
    fig.savefig(figure_path + 'decode_WithDiffUncert_3ytpe_neuron.eps', format='eps', dpi=1000)
    plt.show()

    p12_cue_exc=ranksums(np.array(acc_decode_rule_by_exc_high), np.array(acc_decode_rule_by_exc_mid))
    p13_cue_exc=ranksums(np.array(acc_decode_rule_by_exc_high), np.array(acc_decode_rule_by_exc_low))
    p23_cue_exc=ranksums(np.array(acc_decode_rule_by_exc_mid), np.array(acc_decode_rule_by_exc_low))
    print('pvalue12_cue_exc:',p12_cue_exc[1])
    print('pvalue13_cue_exc:',p13_cue_exc[1])
    print('pvalue23_cue_exc:',p23_cue_exc[1])

    p12_cue_inh=ranksums(acc_decode_rule_by_inh_high, acc_decode_rule_by_inh_mid)
    p13_cue_inh=ranksums(acc_decode_rule_by_inh_high, acc_decode_rule_by_inh_low)
    p23_cue_inh=ranksums(acc_decode_rule_by_inh_mid, acc_decode_rule_by_inh_low)
    print('pvalue12_cue_inh:',p12_cue_inh[1])
    print('pvalue13_cue_inh:',p13_cue_inh[1])
    print('pvalue23_cue_inh:',p23_cue_inh[1])

decode_WithDiffUncert_3ytpe_neuron()






