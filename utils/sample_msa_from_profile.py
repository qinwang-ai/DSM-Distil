import numpy as np
import scipy
from utils.profile2psm import bgfr, get_profile, get_psm
from configs.DataPathRef90.bc40 import *
from utils.rebuttal.kl_divergence_metricl import psm2profile

test_names = list(map(lambda x: x.split(' ')[0].strip(), open('./bc40_fasta_leq_120.txt', 'r').readlines()))
test_names = ['1bzgA']

dict_table = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

a3m_file_prefix = 'lo_'
pssm_files = list(map(lambda x: '/data/proli/raw_data/visualization/low_pssm/%s.npy' % x, test_names))

# a3m_file_prefix = 'en_'
# pssm_files = list(map(lambda x:'/data/proli/raw_data/visualization/enhanced_pssm/%s.npy'% x, test_names))

# a3m_file_prefix = 'ba_'
# pssm_files = list(map(lambda x:'/data/proli/raw_data/visualization/bagging_pssm/%s.npy'% x, test_names))
#
# a3m_file_prefix = 'gt_'
# pssm_files = list(map(lambda x: '/data/proli/data/uniref90/uniref90_bc40_real_nogap/valid/%s.npy' % x, test_names))


save_msa_file_prefix = '/data/proli/raw_data/visualization/bc40_msa_from_pssm/' + a3m_file_prefix

if __name__ == '__main__':
    for profile_file in pssm_files:
        filename = profile_file.split('/')[-1].split('.npy')[0]
        save_msa_file = save_msa_file_prefix + filename + '.a3m'

        profile = np.load(profile_file)

        profile[profile < profile.mean()*0.25] = profile.min()
        profile = profile / profile.sum(axis=-1, keepdims=True)

        # CAUTION: normalize if this is PSSM otherwise comment this
        # N = profile.shape[0]
        # profile = (profile * N +1)/(N+20)
        # profile = profile / profile.sum(axis=-1, keepdims=True)

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

        print(save_msa_file, profile.shape, 'saved')
        f.close()
