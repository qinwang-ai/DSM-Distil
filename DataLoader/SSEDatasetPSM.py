from torch.utils.data import DataLoader, RandomSampler, Dataset
import numpy as np
import glob
import random
import torch
import imageio
from configs.config_sse import IUPAC_VOCAB
from utils.profile2psm import get_profile, bgfr, get_psm
import configs


class SSEDataset(Dataset):
    def __init__(self, real_psm_data_path, real_profile_path_prefix, sequence_data_path_prefix, label_data_path_prefix, mode='all', enable_downsample=True, config=configs.config_sse, dd_downsample=True):
        # mode: all low high
        self.mode = mode
        if type(real_psm_data_path) is str:
            self.psm_file_list = glob.glob(real_psm_data_path)
        else:
            self.psm_file_list = real_psm_data_path
        self.dd_downsample = dd_downsample
        if mode == 'high':
            self.psm_file_list = list(
                filter(lambda x: x.split('/')[-1].split('.npy')[0] in config.high_quality_list,
                       self.psm_file_list))
            print('high msa number:', len(self.psm_file_list))
        elif mode == 'low':
            self.psm_file_list = list(
                filter(lambda x: x.split('/')[-1].split('.npy')[0] in config.low_quality_list,
                       self.psm_file_list))
            print('low msa number:', len(self.psm_file_list))

        self.sequence_data_path = sequence_data_path_prefix
        self.real_profile_path_prefix = real_profile_path_prefix
        self.label_data_path = label_data_path_prefix
        self.tokenizer = Tokenizer()
        self.enable_downsample = enable_downsample
        self.config = config
        super(SSEDataset, self).__init__()

    def get_low_psm(self, msa_list):

        if self.dd_downsample:
            num_msa_for_psm = random.choice(self.config.low_quality_num)
        else:
            percent = random.randint(10, 20) / 100.0
            num_msa_for_psm = int(len(msa_list) * percent)

        msa_list = random.choices(msa_list, k=num_msa_for_psm)
        profile, profile_norm = get_profile(msa_list)

        pssm = get_psm(profile, bgfr, len(msa_list))
        return pssm, profile_norm

    def __getitem__(self, index):

	    # psm
        psm_file_real = self.psm_file_list[index]
        psm_file_fake = psm_file_real.replace('real', 'fake')
        filename = psm_file_real.split('/')[-1].split('.npy')[0]

        # profile
        profile_file_real = self.real_profile_path_prefix + filename + '.tif'
        profile_file_fake = self.real_profile_path_prefix.replace('real', 'fake') + filename + '.tif'

        sequence_file = self.sequence_data_path + filename + '.fasta'
        label_file = self.label_data_path + filename + '.label'

        # load psm
        real_psm_array = np.load(psm_file_real)
        fake_psm_array = np.load(psm_file_fake)

        # load profile
        real_profile_array = imageio.imread(profile_file_real)
        fake_profile_array = imageio.imread(profile_file_fake)

        if self.enable_downsample and (self.mode == 'high' or (self.mode == 'all' and filename in self.config.high_quality_list)):
            msa_file = self.config.msa_data_path_prefix + filename + '.a3m'
            msa_strs = list(map(lambda x: x.strip(), open(msa_file, 'r').readlines()[1::2]))
            low_psm_array, low_profile_array = self.get_low_psm(msa_strs)
        else:
            low_psm_array, low_profile_array = real_psm_array, real_profile_array

        high_quality = filename in self.config.high_quality_list
        sequence_str = open(sequence_file, 'r').readlines()[1].strip()
        token_ids = self.tokenizer.encode(sequence_str)
        label = np.loadtxt(label_file)

        return filename, token_ids, fake_psm_array, real_psm_array, fake_profile_array, real_profile_array, label, low_psm_array, low_profile_array, high_quality

    # one epoch 6125 samples
    def __len__(self):
        return len(self.psm_file_list)

    def collate_fn(self, batch):
        # gap model
        filename, input_ids, fake_psm_array, real_psm_array, fake_profile_array, real_profile_array, label, low_psm_array, low_profile_array, high_quality = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        label = torch.from_numpy(pad_sequences(label, -1))

        real_psm_array = torch.from_numpy(pad_sequences(real_psm_array))
        fake_psm_array = torch.from_numpy(pad_sequences(fake_psm_array))
        real_profile_array = torch.from_numpy(pad_sequences(real_profile_array))
        fake_profile_array = torch.from_numpy(pad_sequences(fake_profile_array))
        low_profile_array = torch.from_numpy(pad_sequences(low_profile_array))

        low_psm_array = torch.from_numpy(pad_sequences(low_psm_array))
        return {'filename': filename[0], 'sequence': input_ids, 'bert_psm': fake_psm_array, 'real_psm': real_psm_array,
                'bert_profile': fake_profile_array, 'real_profile': real_profile_array,
                'low_profile': low_profile_array,
                'label': label,
                'low_psm': low_psm_array, 'high_quality': high_quality}


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
