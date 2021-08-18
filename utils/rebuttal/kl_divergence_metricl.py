import torch.nn.functional as F
import torch
import glob
import torch.nn as nn
import numpy as np

low_pssm_list = glob.glob('/data/proli/raw_data/visualization/low_pssm/*.npy')
bgfr = np.array([0.07805, 0.05129, 0.04487, 0.05364, 0.01925, 0.04264, 0.06295,
                 0.07377, 0.02199, 0.05142, 0.09019, 0.05744, 0.02243, 0.03856,
                 0.05203, 0.0712, 0.05841, 0.0133, 0.03216, 0.06441])


def psm2profile(psm):
    N = psm.shape[0]
    score = np.exp(psm) * bgfr
    profile = score * (N + 20) - 1

    if profile.min() < 0:
        profile += -profile.min()
    profile /= profile.sum(axis=1, keepdims=True)

    return profile


if __name__ == '__main__':
    kl_low = []
    kl_enh = []
    kl_func = nn.KLDivLoss(reduction='mean')
    for f in low_pssm_list:
        name = f.split('/')[-1]
        low_f = f
        high_f = '/data/proli/data/ref90_psm_bc40_real/valid/' + name
        enh_f = '/data/proli/raw_data/visualization/enhanced_pssm/'+name

        low_pssm = np.load(low_f)
        high_pssm = np.load(high_f)
        enh_pssm = np.load(enh_f)

        low_profile = torch.tensor(psm2profile(low_pssm))
        high_profile = torch.tensor(psm2profile(high_pssm))
        enh_profile = torch.tensor(psm2profile(enh_pssm))

        kl_loss = kl_func(F.log_softmax(high_profile, dim=1), low_profile)
        kl_low.append(kl_loss.item())
        kl_loss = kl_func(F.log_softmax(high_profile, dim=1), enh_profile)
        kl_enh.append(kl_loss.item())
    print('kl_low:', np.array(kl_low).mean(), 'kl_enh:', np.array(kl_enh).mean())




