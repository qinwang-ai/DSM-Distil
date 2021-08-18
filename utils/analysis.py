import numpy as np
if __name__ == '__main__':
    f1 = open('./real_profile.txt').readlines()
    f2 = open('./real_psm.txt').readlines()
    f2 = list(map(lambda x: x.split(' ')[0], f2))
    ans = []
    i=1
    for line in f1:
        filename = line.split(' ')[0]
        order = f2.index(filename)
        ans.append((filename, i, order))
        i+=1
    f = open('./psm_profile_performance_analysis.txt', 'w')
    data = list(map(lambda x: ' '.join([str(t) for t in x]), ans))
    f.writelines([s + '\n' for s in data])


