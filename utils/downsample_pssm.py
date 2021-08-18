import numpy as np
from utils.profile2psm import get_bg_profile, get_profile, get_psm
from configs.DataPathRef90.bc40 import *
import random
from utils.sample_msa_from_pssm import test_names


def get_low_psm(msa_list):
    num_msa_for_psm = 20
    msa_list = random.choices(msa_list, k=num_msa_for_psm)
    profile, profile_norm = get_profile(msa_list)
    bgfr = get_bg_profile(msa_list)

    pssm = get_psm(profile, bgfr, len(msa_list))
    return pssm, profile_norm

high_pssms = list(map(lambda x:'/data/proli/data/ref90_psm_bc40_real/valid/%s.npy' % x, test_names))
save_path_prefix = '/data/proli/raw_data/visualization/low_pssm/'

if __name__ == '__main__':
    for high_pssm in high_pssms:
        filename = high_pssm.split('/')[-1].split('.npy')[0]
        original_msa_file = msa_data_path_prefix + filename + '.a3m'
        msa_strs = list(map(lambda x: x.strip(), open(original_msa_file, 'r').readlines()[1::2]))

        low_pssm, _ = get_low_psm(msa_strs)
        save_file = save_path_prefix + filename + '.npy'
        print(save_file)
        np.save(save_file, low_pssm)

