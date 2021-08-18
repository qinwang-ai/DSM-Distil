from torch.utils.data import Dataset
import numpy as np
import glob
import random
import torch
from configs.config_sse import IUPAC_VOCAB
from utils.profile2psm import get_profile, get_bg_profile, get_psm


class PreTrainDataset(Dataset):
    def __init__(self, name_path, sequence_data_path_prefix, psm_path_prefix, msa_data_path_prefix):
        # mode: all low high
        self.name_list = list(map(lambda x:x.strip(), open(name_path).readlines()))

        print('num of samples', len(self.name_list))
        self.sequence_data_path_prefix = sequence_data_path_prefix
        self.psm_path_prefix = psm_path_prefix
        self.msa_data_path_prefix = msa_data_path_prefix
        self.tokenizer = Tokenizer()
        super(PreTrainDataset, self).__init__()

    def get_low_psm(self, msa_list):
        from configs.config_sse_bc40 import low_quality_num

        num_msa_for_psm = random.choice(low_quality_num)

        msa_list = random.choices(msa_list, k=num_msa_for_psm)
        profile, profile_norm = get_profile(msa_list)
        bgfr = get_bg_profile(msa_list)

        pssm = get_psm(profile, bgfr, len(msa_list))
        return pssm, profile_norm

    def __getitem__(self, index):

	    # psm
        filename = self.name_list[index]

        sequence_file = self.sequence_data_path_prefix + filename + '.fasta'
        psm_file = self.psm_path_prefix + filename + '.npy'

        sequence_str = open(sequence_file, 'r').readlines()[1].strip()
        psm_array = np.load(psm_file)

        msa_file = self.msa_data_path_prefix + filename + '.a3m'
        msa_strs = list(map(lambda x: x.strip(), open(msa_file, 'r').readlines()[1::2]))
        low_psm_array, low_profile_array = self.get_low_psm(msa_strs)
        bert_psm_array, bert_profile_array = self.get_low_psm(msa_strs)

        token_ids = self.tokenizer.encode(sequence_str)

        return filename, token_ids, psm_array, low_psm_array, bert_psm_array

    # one epoch 6125 samples
    def __len__(self):
        return len(self.name_list)

    def collate_fn(self, batch):
        # gap model
        filename, token_ids, psm_array, low_psm_array, bert_psm_array = tuple(zip(*batch))

        token_ids = torch.from_numpy(pad_sequences(token_ids, 0))
        psm_array = torch.from_numpy(pad_sequences(psm_array))
        low_psm_array = torch.from_numpy(pad_sequences(low_psm_array))
        bert_psm_array = torch.from_numpy(pad_sequences(bert_psm_array))
        return {'filename': filename[0], 'sequence': token_ids,'psm_array': psm_array, 'low_psm_array': low_psm_array,
                'bert_psm_array': bert_psm_array}


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
