import torch
from torch.utils.data import Dataset, DataLoader

import task
import tools
import pdb

class TaskDataset(Dataset):

    def __init__(self, context_name, hp, mode='train', **kwargs):
        '''provide name of the rules'''
        self.context_name = context_name
        self.hp = hp

        if mode == 'train':
            self.bach_size = hp['batch_size_train']
            self.task_mode = 'random'
        elif mode == 'test':
            self.bach_size = hp['batch_size_test']
            self.task_mode = 'random_validate'
        else:
            raise ValueError('Unknown mode: ' + str(mode))

        self.counter = 0

    def __len__(self):
        '''arbitrary'''
        return 10000000

    def __getitem__(self, index):

        self.trial = task.generate_trials(self.context_name, self.hp, self.task_mode, batch_size=self.bach_size,noise_on=True)

        '''model.x: trial.x,
                 model.y: trial.y,
                 model.cost_mask: trial.cost_mask,
                 model.seq_len: trial.seq_len,
                 model.initial_state: np.zeros((trial.x.shape[1], hp['n_rnn']))'''

        result = dict()
        result['inputs'] = torch.as_tensor(self.trial.x)
        result['target_outputs'] = torch.as_tensor(self.trial.y)
        result['cost_mask'] = torch.as_tensor(self.trial.cost_mask)
        result['cost_start_time'] = 0 # trial.cost_start_time
        result['cost_end_time'] = self.trial.max_seq_len
        result['seq_mask'] = tools.sequence_mask(self.trial.seq_len)
        result['initial_state'] = torch.zeros((self.trial.x.shape[1], self.hp['n_rnn']+self.hp['n_md']))
        result['epochs'] = self.trial.epochs

        result['strength_cue'] = self.trial.strength_cue
        result['context_RDM'] = self.trial.context_RDM
        result['context_HL'] = self.trial.context_HL

        result['batch_0'] = self.trial.batch_0
        result['mean_vis_0'] = self.trial.mean_vis_0
        result['mean_vis_1'] = self.trial.mean_vis_1
        result['mean_aud_0'] = self.trial.mean_aud_0
        result['mean_aud_1'] = self.trial.mean_aud_1

        result['gamma_bar_vis'] = self.trial.gamma_bar_vis
        result['gamma_bar_aud'] = self.trial.gamma_bar_aud
        result['c_vis'] = self.trial.c_vis
        result['c_aud'] = self.trial.c_aud



        result['target_choice'] = self.trial.target_choice
        #print('*',result['initial_state'].shape)


        return result


class TaskDatasetForRun(object):

    def __init__(self, context_name, hp, noise_on=True, mode='test', **kwargs):
        '''provide name of the rules'''
        self.context_name = context_name
        self.hp = hp
        self.kwargs = kwargs
        self.noise_on = noise_on
        self.mode = mode
        #print('**self.mode',self.mode)

    def __getitem__(self):

        self.trial = task.generate_trials(self.context_name, self.hp, self.mode, noise_on=self.noise_on, **self.kwargs)
        #print('self.context_name',self.context_name)

        result = dict()
        result['inputs'] = torch.as_tensor(self.trial.x)
        result['target_outputs'] = torch.as_tensor(self.trial.y)
        result['cost_mask'] = torch.as_tensor(self.trial.cost_mask)
        result['cost_start_time'] = 0 # trial.cost_start_time
        result['cost_end_time'] = self.trial.max_seq_len
        result['seq_mask'] = tools.sequence_mask(self.trial.seq_len)
        result['initial_state'] = torch.zeros((self.trial.x.shape[1], self.hp['n_rnn']+self.hp['n_md']))
        # print('**',result['initial_state'])
        # pdb.set_trace()


        return result


