import torch
import random
from TweetNormalizer import normalizeTweet
from torch.utils.data import Dataset
import emoji
import os
import numpy as np
from sklearn import metrics


def check_matrix(matrix, gold, pred):
  """Check matrix dimension."""
  if matrix.size == 1:
    tmp = matrix[0][0]
    matrix = np.zeros((2, 2))
    if (pred[1] == 0):
      if gold[1] == 0:  #true negative
        matrix[0][0] = tmp
      else:  #falsi negativi
        matrix[1][0] = tmp
    else:
      if gold[1] == 0:  #false positive
        matrix[0][1] = tmp
      else:  #true positive
        matrix[1][1] = tmp
  return matrix
def compute_f1(pred_values, gold_values):
  matrix = metrics.confusion_matrix(gold_values, pred_values)
  matrix = check_matrix(matrix, gold_values, pred_values)

  #positive label
  if matrix[0][0] == 0:
    pos_precision = 0.0
    pos_recall = 0.0
  else:
    pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

  if (pos_precision + pos_recall) != 0:
    pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
  else:
    pos_F1 = 0.0

  #negative label
  neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

  if neg_matrix[0][0] == 0:
    neg_precision = 0.0
    neg_recall = 0.0
  else:
    neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
    neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

  if (neg_precision + neg_recall) != 0:
    neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
  else:
    neg_F1 = 0.0

  f1 = (pos_F1 + neg_F1) / 2
  return f1

def pad_bert_custom_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    node_sets_sequence = []
    mask_sequence = []
    label_sequence = []
    filename_sequence = []
    for node_sets, mask, label, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        label_sequence.append(label)
        filename_sequence.append(filename)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=1)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return node_sets_sequence, mask_sequence, label_sequence, filename_sequence
def pad_clip_custom_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    node_sets_sequence = []
    mask_sequence = []
    label_sequence = []
    filename_sequence = []
    for node_sets, mask, label, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        label_sequence.append(label)
        filename_sequence.append(filename)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=49407)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return node_sets_sequence, mask_sequence, label_sequence, filename_sequence
def pad_testclip_custom_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    node_sets_sequence = []
    mask_sequence = []
    filename_sequence = []
    for node_sets, mask, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        filename_sequence.append(filename)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=49407)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    return node_sets_sequence, mask_sequence, filename_sequence
class BERTweetdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, test_file, tokenizer, device, max_len, type=None):
        self.train_file = train_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.type = type
        self.train_dataset, self.test_dataset = self.prepare_dataset()


    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        [self.train_file[id].update(
            {'encode': self.tokenizer(normalizeTweet(self.train_file[id]['text'].replace('\n', ' ')), return_tensors='pt', truncation=True, max_length=self.max_len).to(self.device)})
            for id in list(self.train_file.keys())]
        [self.test_file[id].update(
            {'encode': self.tokenizer(normalizeTweet(self.test_file[id]['text'].replace('\n', ' ')), return_tensors='pt', truncation=True, max_length=self.max_len).to(self.device)})
            for id in list(self.test_file.keys())]
        train_dataset = BERTweetdatasetloader(self.train_file, self.max_len, type=self.type)
        test_dataset = BERTweetdatasetloader(self.test_file, self.max_len, type=self.type)
        return train_dataset, test_dataset

class BERTweetevaldatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, test_file, dset, tokenizer, device, max_len, type=None):
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.device = device
        self.dset = dset
        self.max_len = max_len
        self.type = type
        self.test_dataset = self.prepare_dataset()


    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        [self.test_file[id].update(
            {'encode': self.tokenizer(normalizeTweet(self.test_file[id]['text'].replace('\n', ' ')), return_tensors='pt', truncation=True, max_length=self.max_len).to(self.device)})
            for id in list(self.test_file.keys())]
        if self.dset == 'test':
            test_dataset = BERTweettestdatasetloader(self.test_file, self.max_len, type=self.type)
        else:
            test_dataset = BERTweetdatasetloader(self.test_file, self.max_len, type=self.type)
        return test_dataset


class BERTweettestdatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, max_len, type=None):
        super(BERTweettestdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.type = type
        self.max_len = max_len

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        id = self.datadict[self.datakeys[index]]['encode'].data['input_ids']
        mask = self.datadict[self.datakeys[index]]['encode'].data['attention_mask']
        filename = self.datakeys[index]
        if id[0].size()[0] > 77:
            newid = id[0][:77].unsqueeze(0)
            newmask = mask[0][:77].unsqueeze(0)
            return newid, newmask, filename
        else:
            return id, mask, filename  # twtfsingdata.squeeze(0), filename
class BERTweetdatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, max_len, type=None):
        super(BERTweetdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.type = type
        self.max_len = max_len

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        id = self.datadict[self.datakeys[index]]['encode'].data['input_ids']
        mask = self.datadict[self.datakeys[index]]['encode'].data['attention_mask']
        filename = self.datakeys[index]
        label = int(self.datadict[self.datakeys[index]]['taskA'])
        label = torch.LongTensor([label])
        '''if self.type == 'pretrain':
            
        elif self.type == 'train':
            label = int(self.datadict[self.datakeys[index]]['taskA'])
            label = torch.FloatTensor(one_hot(np.array([label]), 3))'''
        if id[0].size()[0] > 77:
            newid = id[0][:77].unsqueeze(0)
            newmask = mask[0][:77].unsqueeze(0)
            return newid, newmask, label, filename
        else:
            return id, mask, label, filename  # twtfsingdata.squeeze(0), filename
