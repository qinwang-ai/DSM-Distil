import numpy as np
if __name__ == '__main__':
    f1 = open('../logs/real_psm.txt').readlines()
    f2 = open('../logs/real_profile.txt').readlines()
    f2_file = list(map(lambda x: x.split(' ')[0], f2))
    ans = []
    i=1
    for line in f1:
        filename = line.split(' ')[0]
        psm_acc = line.split(' ')[-1]
        order = f2_file.index(filename)
        profile_acc = f2[order].split(' ')[-1]
        ans.append("%s %03d %03d %.3f %.3f" % (filename, i, order, float(psm_acc.strip()), float(profile_acc.strip())))
        i+=1
    f = open('./psm_profile_performance_analysis.txt', 'w')
    f.writelines([s + '\n' for s in ans])


# filename pssm_order profile_order pssm_acc profile_acc
