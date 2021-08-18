import numpy as np
from utils.profile2psm import bgfr, get_profile, get_psm
from configs.DataPathRef90.bc40 import *
from utils.rebuttal.kl_divergence_metricl import psm2profile

test_names = list(map(lambda x: x.split(' ')[0].strip(), open('./bc40_fasta_leq_120.txt', 'r').readlines()))
# test_names = ['4jm1A']

dict_table = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']



# a3m_file_prefix = 'enhance_'
# pssm_files = list(map(lambda x:'/data/proli/raw_data/visualization/enhanced_pssm/%s.npy'% x, test_names))

# a3m_file_prefix = 'high_'
# pssm_files = list(map(lambda x: '/data/proli/data/ref90_psm_bc40_real/valid/%s.npy' % x, test_names))

a3m_file_prefix = 'low_'
pssm_files = list(map(lambda x:'/data/proli/raw_data/visualization/low_pssm/%s.npy' % x, test_names))




save_msa_file_prefix = '/data/proli/raw_data/visualization/bc40_msa_from_pssm/' + a3m_file_prefix

if __name__ == '__main__':
    for pssm_file in pssm_files:
        filename = pssm_file.split('/')[-1].split('.npy')[0]
        save_msa_file = save_msa_file_prefix + filename + '.a3m'
        original_msa_file = msa_data_path_prefix + filename + '.a3m'

        # inverse
        msa_strs = list(map(lambda x: x.strip(), open(original_msa_file, 'r').readlines()[1::2]))

        pssm_array = np.load(pssm_file)

        # pssm_array = np.exp(pssm_array) * bgfr
        # pssm_array = pssm_array - np.expand_dims(pssm_array.min(axis=1), axis=1)
        # profile = pssm_array / np.expand_dims(pssm_array.sum(axis=1), axis=1)

        profile = psm2profile(pssm_array)

        f = open(save_msa_file, 'w')
        msa_list = []
        for i in range(2000):
            ans = ''
            for j in range(profile.shape[0]):
                char = np.random.choice(dict_table, p=profile[j, :])
                ans += char
            msa_list.append(ans)

        for msa in msa_list:
            f.write('>' + filename + '\n')
            f.write(msa + '\n')
        print(save_msa_file, 'saved')
        f.close()













