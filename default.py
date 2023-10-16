import numpy as np
import os


def input_output_n():
    n_rule = 2
    n_present = 4
    n_ring_RDM = 64
    n_ring_HL = 2
    return n_ring_RDM,n_ring_HL,n_ring_RDM+n_ring_HL+n_present+n_rule, 4


def get_default_hp(rule_name, random_seed=None):

    root_path = os.path.abspath(os.path.join(os.getcwd(),"./"))
    save_dir = os.path.join(root_path, 'models_saved')
    figure_dir = os.path.join(root_path, 'z_figure')
    root_path_1 = os.path.abspath(os.path.join(os.getcwd(),"."))
    picture_dir = os.path.join(root_path_1, 'picture3/')

    root_path_data = os.path.abspath(os.path.join(os.getcwd(),"../"))

    print('### root_path',root_path)
    n_ring_RDM,n_ring_HL,n_input, n_output = input_output_n()


    if random_seed is None:
        seed = np.random.randint(100)
    else:
        seed = random_seed

    hp = {
        'root_path':root_path,
        'picture_dir':picture_dir,
        'figure_dir': figure_dir,


        'n_ring_RDM':n_ring_RDM,
        'n_ring_HL':n_ring_HL,
        'e_prop':0.8,
        'rule_name': rule_name,

        'batch_size_train': 64,
        'batch_size_test': 512,
        'rnn_type': 'RNN',
        'optimizer': 'adam',
        'activation': 'softplus',
        'tau': 20,
        'dt': 20,
        'alpha': 1,
        'initial_std': 0.3,
        'sigma_rec': 0.05,
        'sigma_x': 0.01,
        'l1_firing_rate': 0,
        'l2_firing_rate': 0,
        'l1_weight': 0.0,
        'l2_weight': 0.0,
        'n_input': n_input,
        'n_output': n_output,
        'n_rnn':256,
        'n_md':200,
        'learning_rate': 0.0005,
        'seed': seed,
        'rng': np.random.RandomState(seed),
        'is_EI': 'EI',
        'use_reset':'yes',
        'rdm_context':'rdm1',
        'hlcontext':'hl1',

        'scale_RDM':1.0,
        'scale_HL':1.0,
        'std_scale':0.0,
        'cue_scale':1.0,

        'stim_scale':1.0,
        'p_coh':0.92,
        'mask_start':'respOn',

        'cue_duration':800,
        'cue_delay':800,
        'stim':200,

        'response_time':200,
        'model_idx':1,
        'gamma_noise':1.0,
        'stim_std':1.0,
        'input_mask':'input_mask',
        'dropout':0.0,
        'scale_random':5.0,
        'init_hh':'init1',

        'remove_mask':'no',
        'stim_epoch':'no',
        'mask_type':'no',
        'add_mask':'no',
        'mode_mask':'train',
        'sparsity_HL':0.0,
        'sparsity_RDM':0.0,
        'get_SNR':'no',
        'zero_input':'no',
        'mask_test' :'no',
        'stim_cell':'no',
        'stim_value':0,

    }

    return hp
