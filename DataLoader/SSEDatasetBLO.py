import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import numpy as np
import glob
import random
import imageio
import configs.config_sse as config
from configs.config_sse import IUPAC_VOCAB
import os


class SSEDataset(Dataset):
    def __init__(self, profile_data_path, sequence_data_path_prefix, label_data_path_prefix, mode='all', config=config):
        self.config = config
        self.profile_file_list = glob.glob(profile_data_path)
        if mode == 'high':
            self.profile_file_list = list(
                filter(lambda x: x.split('/')[-1].split('.npy')[0] in self.config.high_quality_list,
                       self.profile_file_list))
        elif mode == 'low':
            self.profile_file_list = list(
                filter(lambda x: x.split('/')[-1].split('.npy')[0] in self.config.low_quality_list,
                       self.profile_file_list))

        self.sequence_data_path = sequence_data_path_prefix
        self.label_data_path = label_data_path_prefix
        self.tokenizer = Tokenizer()
        super(SSEDataset, self).__init__()

    def __getitem__(self, index):
        profile_file = self.profile_file_list[index]
        filename = profile_file.split('/')[-1].split('.npy')[0]

        sequence_file = self.sequence_data_path + filename + '.fasta'
        label_file = self.label_data_path + filename + '.label'

        # load data
        profile_array = np.load(profile_file)
        sequence_str = open(sequence_file, 'r').readlines()[1].strip()
        token_ids = self.tokenizer.encode(sequence_str)
        label = np.loadtxt(label_file)

        high_quality = profile_file.split('/')[-1].split('.npy')[0] in self.config.high_quality_list,

        return filename, token_ids, profile_array, label, high_quality

    # one epoch 6125 samples
    def __len__(self):
        return len(self.profile_file_list)

    def collate_fn(self, batch):
        # gap model
        filename, input_ids, profile_array, label, high_quality = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        label = torch.from_numpy(pad_sequences(label, -1))
        profile_array = torch.from_numpy(pad_sequences(profile_array))
        high_quality = torch.from_numpy(pad_sequences(np.array(high_quality)))
        return {'filename': filename[0], 'sequence': input_ids, 'profile': profile_array, 'label': label,
                'high_quality': high_quality}


class Tokenizer(object):
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.vocab = IUPAC_VOCAB

    def tokenize(self, text: str):
        return [x for x in text]

    def encode(self, text: str):
        tokens = self.tokenize(text)
        token_ids = [self.vocab[token] for token in tokens if token != '\n']
        return np.array(token_ids, np.int64)


def pad_sequences(sequences, constant_value=0, dtype=None):
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array
