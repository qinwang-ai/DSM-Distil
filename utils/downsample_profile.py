import numpy as np
from utils.profile2psm import get_profile
from configs.DataPathRef90.bc40 import *
import random
from utils.sample_msa_from_pssm import test_names
from configs.DataPathRef90.bc40 import real_profile_path_prefix


def get_low_psm(msa_list):
    num_msa_for_psm = 2
    msa_list = random.choices(msa_list, k=num_msa_for_psm)
    profile, profile_norm = get_profile(msa_list)
    return profile_norm

high_pssms = list(map(lambda x:'/data/proli/data/uniref90/uniref90_bc40_real_nogap/valid/%s.npy' % x, test_names))

save_path_prefix = '/data/proli/raw_data/visualization/low_pssm/'

if __name__ == '__main__':
    for high_pssm in high_pssms:
        filename = high_pssm.split('/')[-1].split('.npy')[0]
        original_msa_file = msa_data_path_prefix + filename + '.a3m'
        msa_strs = list(map(lambda x: x.strip(), open(original_msa_file, 'r').readlines()[1::2]))

        low_pssm = get_low_psm(msa_strs)
        low_pssm += 0.001
        low_pssm = low_pssm/low_pssm.sum(axis=-1, keepdims=True)

        save_file = save_path_prefix + filename + '.npy'
        print(save_file)
        np.save(save_file, low_pssm)

