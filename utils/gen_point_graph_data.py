import numpy as np

# our_real = [(msa_count, our_acc, real_acc), (msa_count, our_acc, real_acc), (msa_count, our_acc, real_acc)]
# our_bagging = [(msa_count, our_acc, bagging_acc), (msa_count, our_acc, bagging_acc), (msa_count, our_acc, baggine_acc)]

meff_file = '/data/proli/data/bc40_num'

def main(our_acc_file, other_acc_file):
    meff_dict = {}
    ans_dict = {}
    for x in open(meff_file).readlines():
        name, meff = x.split(' ')[0].strip(), x.split(' ')[-1].strip()
        meff_dict[name] = float(meff)

    print('meff list count', len(meff_dict))

    f_our = open(our_acc_file).readlines()
    for x in f_our:
        name, acc = x.split(' ')[0].strip(), x.split(' ')[-1].strip()
        if name not in ans_dict:
            ans_dict[name] = {'our_acc':0, 'other_acc': 0, 'meff_value':0}
        ans_dict[name]['our_acc'] = float(acc)
        ans_dict[name]['meff_value'] = meff_dict[name]

    f_other = open(other_acc_file).readlines()
    for x in f_other:
        name, acc = x.split(' ')[0].strip(), x.split(' ')[-1].strip()
        if name in ans_dict:
            ans_dict[name]['other_acc'] = float(acc)

    ans_list = []
    for key in ans_dict.keys():
        acc0 = ans_dict[key]['our_acc']
        acc1 = ans_dict[key]['other_acc']
        meff = ans_dict[key]['meff_value']
        ans_list.append((meff, acc0, acc1))

    # np.save('/data/proli/Bioinfo/data4draw_points/ourvsreal3.npy', ans_list)
    # np.save('/data/proli/Bioinfo/data4draw_points/ourvsbagging3.npy', ans_list)
    # np.save('/data/proli/Bioinfo/data4draw_points/ourvsaf3.npy', ans_list)

    # np.save('/data/proli/Bioinfo/data4draw_points/ourvsreal8.npy', ans_list)
    np.save('/data/proli/Bioinfo/data4draw_points/ourvsbagging8.npy', ans_list)
    # np.save('/data/proli/Bioinfo/data4draw_points/ourvsaf8.npy', ans_list)


if __name__ == '__main__':
    # our_file = '/data/proli/Bioinfo/logs/our_acc3_bc40.txt'
    # other_file = '/data/proli/Bioinfo/logs/real_psm3_bc40.txt'
    # other_file = '/data/proli/Bioinfo/logs/bagging_acc3_bc40.txt'
    # other_file = '/data/proli/Bioinfo/logs/af2_acc3.txt'

    our_file = '/data/proli/Bioinfo/logs/our_acc8_bc40.txt'
    # other_file = '/data/proli/Bioinfo/logs/real_psm8_bc40.txt'
    other_file = '/data/proli/Bioinfo/logs/real_profile8_bc40.txt'
    # other_file = '/data/proli/Bioinfo/logs/bagging_acc8_bc40.txt'
    # other_file = '/data/proli/Bioinfo/logs/af2_acc8.txt'


    main(our_file, other_file)
