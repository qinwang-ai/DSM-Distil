import numpy as np
import glob
import os
import torch
from tqdm import tqdm

dataset = 'valid'
kind = 'real'

# profile_data_path = '/data/proli/data/uniref90/uniref90_profile_6125_nogap_%s/%s_tiff/*.tif' % (kind, dataset)
profile_data_path = '/data/proli/data/uniref90/uniref90_bc40_%s_nogap/%s_tiff/*.tif' % (kind, dataset)
# profile_data_path = '/data/proli/data/uniref90/uniref90_cb513_%s/%s_tiff/*.tif' % (kind, dataset)

msa_data_path_prefix = '/data/proli/raw_data/bc40_a3m_%s/'%kind
# msa_data_path_prefix = '/data/proli/raw_data/uniref90_6125_matchlength/'

# save path
# psm_data_path_prefix = '/data/proli/data/ref90_psm_6125_%s_fix_back/' % kind
psm_data_path_prefix = '/data/proli/data/ref90_psm_bc40_%s_fix_back/' % kind



bgfr = np.array([0.07805, 0.05129, 0.04487, 0.05364, 0.01925, 0.04264, 0.06295,
                 0.07377, 0.02199, 0.05142, 0.09019, 0.05744, 0.02243, 0.03856,
                 0.05203, 0.0712, 0.05841, 0.0133, 0.03216, 0.06441])

def get_profile(msa_list):
    def get_dict(x):
        pos_dict = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9, "L": 10, "K": 11,"M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19}
        if x not in pos_dict.keys():
            return 20
        else:
            return pos_dict[x]

    #import ipdb; ipdb.set_trace()
    _msa_list = np.array(list(map(lambda st: [get_dict(x) for x in st], msa_list)))
    profile = np.zeros((_msa_list.shape[1], 20))
    for i in range(20):
        frequency = (_msa_list == i).sum(axis=0)
        profile[:, i] += frequency
    summ = profile.sum(axis=1, keepdims=True)
    summ[summ==0]=20
    profile_norm = profile /summ
    return profile, profile_norm


def get_psm(profile_array, N):
    pseudocount = 1
    score = (profile_array + pseudocount) / (N + 20 * pseudocount)
    pssm = np.log(score / bgfr)
    return pssm


if __name__ == '__main__':
    file_list = glob.glob(profile_data_path)
    print('file amount:', len(file_list))
    for file in tqdm(file_list):
        filename = file.split('/')[-1].split('.tif')[0]
        msa_file = msa_data_path_prefix + filename + '.a3m'
        save_path = psm_data_path_prefix + dataset+'/'+filename + '.npy'
        if os.path.exists(save_path):
            print('skip...%s already exists', save_path)
            # continue
        msa_strs = list(map(lambda x: x.strip(), open(msa_file, 'r').readlines()[1::2]))
        profile_array, _ = get_profile(msa_strs)

        pssm = get_psm(profile_array, len(msa_strs))

        np.save(save_path, pssm)
