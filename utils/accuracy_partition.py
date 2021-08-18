import numpy as np
import configs.config_sse_bc40 as config
# import configs.config_sse as config
import glob


def main(acc_file):
    fake_profile_file_list = list(glob.glob(config.profile_fake_data_path.replace('train', 'valid')))
    filenames = list(map(lambda x:x.split('/')[-1], fake_profile_file_list))
    filenames = list(map(lambda x:x.split('.npy')[0], filenames))

    accs = []
    for x in open(acc_file).readlines():
        name, acc = x.split(' ')[0].strip(), x.split(' ')[-1].strip()
        if name in config.low_quality_list and name in filenames:
            accs.append(float(acc))

    print(len(accs), np.array(accs).mean())


if __name__ == '__main__':
    # analysis_file = '/data/proli/Bioinfo/logs/our_acc3_bc40.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/our_acc8_bc40.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/our_acc3_6125.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/our_acc8_6125.txt'

    # analysis_file = '/data/proli/Bioinfo/logs/real_profile3_bc40.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/real_profile3_6125.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/real_profile8_6125.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/real_profile8_bc40.txt'

    # analysis_file = '/data/proli/Bioinfo/logs/real_psm8_bc40.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/real_psm8_6125.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/real_psm3_6125.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/real_psm3_bc40.txt'

    # analysis_file = '/data/proli/Bioinfo/logs/bagging_acc3_bc40.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/bagging_acc3_6125.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/bagging_acc8_bc40.txt'

    # analysis_file = '/data/proli/Bioinfo/logs/real_dsm_acc3_bc40.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/real_dsm_acc3_6125.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/real_dsm_acc8_bc40.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/real_dsm_acc3_6125.txt'

    analysis_file = '/data/proli/Bioinfo/logs/our_wo_spf_acc3_bc40.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/our_wo_prior_acc3_bc40.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/our_wo_distil_acc3_bc40.txt'

    # analysis_file = '/data/proli/Bioinfo/logs/af2_acc3.txt'
    # analysis_file = '/data/proli/Bioinfo/logs/af2_acc8.txt'
    main(analysis_file)
