import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import numpy as np
import glob
import random
import imageio
import configs.config_sse as config
from configs.config_sse import IUPAC_VOCAB
from utils.profile2psm import bgfr

class SSEDataset(Dataset):
    def __init__(self, profile_data_path, sequence_data_path_prefix, label_data_path_prefix):
        self.config = config
        self.profile_file_list = glob.glob(profile_data_path)
        self.sequence_data_path = sequence_data_path_prefix
        self.label_data_path = label_data_path_prefix
        self.tokenizer = Tokenizer()
        super(SSEDataset, self).__init__()

        self.count2file_dict = {}
        for index in range(len(self.profile_file_list)):
            profile_file = self.profile_file_list[index]
            filename = profile_file.split('/')[-1].split('.npy')[0]
            msa_file = self.config.msa_data_path_prefix + filename + '.a3m'
            msa_strs = list(map(lambda x: x.strip(), open(msa_file, 'r').readlines()[1::2]))
            npz, msa_c = self.discretize(len(msa_strs))
            if msa_c not in self.count2file_dict:
                self.count2file_dict[msa_c] = []
            self.count2file_dict[msa_c].append((filename, npz))
        self.count2file_dict_keys = list(self.count2file_dict.keys())

        # kmax = 0
        # for k, v in self.count2file_dict.items():
        #     kmax = max(k, kmax)
        #     print(k, len(v))

    def discretize(self, msa_c):
        msa_c = min(msa_c, 2047)
        msa_c = msa_c // 4
        npz = np.zeros((512))
        npz[msa_c] = 1
        return npz, msa_c

    def __getitem__(self, index):
        filename, npz = random.choice(self.count2file_dict[self.count2file_dict_keys[index]])

        sequence_file = self.sequence_data_path + filename + '.fasta'
        label_file = self.label_data_path + filename + '.label'
        profile_file = config.real_profile_data_prefix + filename + '.npy'

        # load data
        profile_array = np.load(profile_file)
        profile_array = profile_array - bgfr
        profile_array[profile_array < 0] = 0

        sequence_str = open(sequence_file, 'r').readlines()[1].strip()
        token_ids = self.tokenizer.encode(sequence_str)
        label = np.loadtxt(label_file)

        return token_ids, profile_array, label, npz, filename

    # one epoch 6125 samples
    def __len__(self):
        return len(self.count2file_dict_keys)

    def collate_fn(self, batch):
        # gap model
        input_ids, profile_array, label, msa_c, filename = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 21))
        label = torch.from_numpy(pad_sequences(label, -1))
        profile_array = torch.from_numpy(pad_sequences(profile_array))
        return {'sequence': input_ids, 'profile': profile_array, 'label': label, 'msa_c': msa_c, 'filename': filename[0]}


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



class SSEDatasetVal(Dataset):
    def __init__(self, profile_data_path, sequence_data_path_prefix, label_data_path_prefix):
        self.config = config
        self.profile_file_list = glob.glob(profile_data_path)
        self.sequence_data_path = sequence_data_path_prefix
        self.label_data_path = label_data_path_prefix
        self.tokenizer = Tokenizer()
        super(SSEDatasetVal, self).__init__()

    def discretize(self, msa_c):
        msa_c = min(msa_c, 2047)
        msa_c = msa_c // 4
        npz = np.zeros((512))
        npz[msa_c] = 1
        return npz, msa_c

    def __getitem__(self, index):
        profile_file = self.profile_file_list[index]
        filename = profile_file.split('/')[-1].split('.npy')[0]
        sequence_file = self.sequence_data_path + filename + '.fasta'
        label_file = self.label_data_path + filename + '.label'

        # load data
        profile_array = np.load(profile_file)
        profile_array = profile_array - bgfr
        profile_array[profile_array < 0] = 0

        msa_file = self.config.msa_data_path_prefix + filename + '.a3m'
        msa_strs = list(map(lambda x: x.strip(), open(msa_file, 'r').readlines()[1::2]))
        npz, msa_c = self.discretize(len(msa_strs))

        sequence_str = open(sequence_file, 'r').readlines()[1].strip()
        token_ids = self.tokenizer.encode(sequence_str)
        label = np.loadtxt(label_file)

        return token_ids, profile_array, label, npz, filename

    # one epoch 6125 samples
    def __len__(self):
        return len(self.profile_file_list)

    def collate_fn(self, batch):
        # gap model
        input_ids, profile_array, label, msa_c, filename = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 21))
        label = torch.from_numpy(pad_sequences(label, -1))
        profile_array = torch.from_numpy(pad_sequences(profile_array))
        return {'sequence': input_ids, 'profile': profile_array, 'label': label, 'msa_c': msa_c, 'filename': filename[0]}