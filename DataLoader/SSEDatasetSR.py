import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import random
import imageio
from configs.config_sse import IUPAC_VOCAB
import configs.config_sse as config_sse
from utils.profile2psm import bgfr


class SSEDataset(Dataset):
    def __init__(self, fake_profile_data_path, sequence_data_path_prefix, label_data_path_prefix, mode='all',
                 enable_downsample=True, config=config_sse):
        # mode: all low high
        self.mode = mode
        self.config = config
        self.fake_profile_file_list = glob.glob(fake_profile_data_path)
        self.post_fix = '.'+self.fake_profile_file_list[0].split('.')[-1]
        if mode == 'high':
            self.fake_profile_file_list = list(
                filter(lambda x: x.split('/')[-1].split(self.post_fix)[0] in self.config.high_quality_list,
                       self.fake_profile_file_list))
            print('high msa number:', len(self.fake_profile_file_list))
        elif mode == 'low':
            self.fake_profile_file_list = list(
                filter(lambda x: x.split('/')[-1].split(self.post_fix)[0] in self.config.low_quality_list,
                       self.fake_profile_file_list))
            print('low msa number:', len(self.fake_profile_file_list))

        self.sequence_data_path = sequence_data_path_prefix
        self.label_data_path = label_data_path_prefix
        self.tokenizer = Tokenizer()
        self.enable_downsample = enable_downsample
        super(SSEDataset, self).__init__()

    def get_low_profile(self, msa_list):
        def get_dict(x):
            pos_dict = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9, "L": 10,
                        "K": 11,
                        "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19}
            if x not in pos_dict.keys():
                return 20
            else:
                return pos_dict[x]

        # percent = random.randint(10, 20) / 100.0
        # num_msa_for_profile = int(len(msa_list) * percent)
        num_msa_for_profile = random.choice(config_sse.low_quality_num)
        msa_list = random.choices(msa_list, k=num_msa_for_profile)

        msa_list = np.array(list(map(lambda st: [get_dict(x) for x in st], msa_list)))
        profile = np.zeros((msa_list.shape[1], 20))
        for i in range(20):
            frequency = (msa_list == i).sum(axis=0)
            profile[:, i] += frequency
        divider = profile.sum(axis=1, keepdims=True)
        divider[divider == 0] = 1
        profile = profile / divider
        return profile

    def __getitem__(self, index):
        fake_profile_file = self.fake_profile_file_list[index]
        filename = fake_profile_file.split('/')[-1].split(self.post_fix)[0]

        sequence_file = self.sequence_data_path + filename + '.fasta'
        label_file = self.label_data_path + filename + '.label'

        # load data
        if 'fake_profile_6125_FAIR' in fake_profile_file:
            real_profile_file = fake_profile_file.replace('fake_profile_6125_FAIR', 'uniref90/uniref90_profile_6125_nogap_real')
        else:
            real_profile_file = fake_profile_file.replace('fake_profile_bc40_FAIR', 'uniref90/uniref90_bc40_real_nogap')

        fake_profile_array = np.load(fake_profile_file)
        real_profile_array = np.load(real_profile_file)

        if self.enable_downsample:
            if self.mode == 'high' or (self.mode == 'all' and filename in self.config.high_quality_list):
                msa_file = self.config.msa_data_path_prefix + filename + '.a3m'
                msa_strs = list(map(lambda x: x.strip(), open(msa_file, 'r').readlines()[1::2]))
                low_profile_array = self.get_low_profile(msa_strs)
            else:
                low_profile_array = real_profile_array
        else:
            low_profile_array = real_profile_array
        high_quality = filename in self.config.high_quality_list
        sequence_str = open(sequence_file, 'r').readlines()[1].strip()
        token_ids = self.tokenizer.encode(sequence_str)
        label = np.loadtxt(label_file)

        return filename, token_ids, fake_profile_array, real_profile_array, label, low_profile_array, high_quality

    # one epoch 6125 samples
    def __len__(self):
        return len(self.fake_profile_file_list)

    def collate_fn(self, batch):
        # gap model
        filename, input_ids, fake_profile_array, real_profile_array, label, low_profile, high_quality = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        label = torch.from_numpy(pad_sequences(label, -1))
        fake_profile_array = torch.from_numpy(pad_sequences(fake_profile_array))
        real_profile_array = torch.from_numpy(pad_sequences(real_profile_array))
        low_profile = torch.from_numpy(pad_sequences(low_profile))
        return {'filename': filename[0], 'sequence': input_ids, 'fake_profile': fake_profile_array, 'real_profile': real_profile_array, 'label': label,
                'low_profile': low_profile, 'high_quality': high_quality}


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
