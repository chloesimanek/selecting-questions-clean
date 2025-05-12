import numpy as np
import torch
from torch.utils import data
import torch
import random
from utils.utils import open_json, dump_json


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, seed=None):
        'Initialization'
        self.data = data
        self.seed = seed

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        # return self.data[index]
        'Generates one sample of data'
        data = self.data[index]
        observed_index = np.array([idx for idx in range(len(data['q_ids']))])
        if not self.seed:
            np.random.shuffle(observed_index)
        else:
            random.Random(index+self.seed).shuffle(observed_index)
        N = len(observed_index)
        target_index = observed_index[-N//5:]
        trainable_index = observed_index[:-N//5]

        # input_ans = data['ans'][trainable_index]
        input_label = data['labels'][trainable_index]
        input_question = data['q_ids'][trainable_index]
        output_label = data['labels'][target_index]
        output_question = data['q_ids'][target_index]

        # adding diffs 
        diffs = data['diffs'][trainable_index]

        user_id = data.get('user_id', index)  # Use index as fallback if user_id doesn't exist


        output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
                   'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
                  'user_id': user_id, # Noah: added user id for logging
                  'diffs':torch.FloatTensor(diffs)} 
        return output


class collate_fn(object):
    def __init__(self, n_question):
        self.n_question = n_question

    def __call__(self, batch):
        B = len(batch)
        input_labels = torch.zeros(B, self.n_question).long()
        output_labels = torch.zeros(B, self.n_question).long()
        #input_ans = torch.ones(B, self.n_question).long()
        input_mask = torch.zeros(B, self.n_question).long()
        output_mask = torch.zeros(B, self.n_question).float()
        user_ids = []  # Initialize user_ids list here

        diffs = torch.full((B, self.n_question), float('inf')).float()

        for b_idx in range(B):
            input_labels[b_idx, batch[b_idx]['input_question'].long(
            )] = batch[b_idx]['input_label'].long()
            #input_ans[b_idx, batch[b_idx]['input_question'].long()] = batch[b_idx]['input_ans'].long()
            input_mask[b_idx, batch[b_idx]['input_question'].long()] = 1
            output_labels[b_idx, batch[b_idx]['output_question'].long(
            )] = batch[b_idx]['output_label'].long()
            output_mask[b_idx, batch[b_idx]['output_question'].long()] = 1

            diffs[b_idx, batch[b_idx]['input_question'].long()] = batch[b_idx]['diffs'].float()

            user_ids.append(batch[b_idx].get('user_id', b_idx))  # Use batch index as fallback

        output = {'input_labels': input_labels, 'input_mask': input_mask,
                  'output_labels': output_labels, 'output_mask': output_mask,
                  'user_ids': user_ids, 'diffs': diffs}  # Add user_ids to batch
        # 'input_ans':input_ans,
        return output