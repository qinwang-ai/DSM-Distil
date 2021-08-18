num_file = '../uni90_bc40_num_low'

high_quality_list = list(filter(lambda x: int(x.split(' ')[1]) > 10, open(num_file, 'r').readlines()))
low_quality_list = list(
    filter(lambda x: int(x.split(' ')[1]) > 0 and x not in high_quality_list, open(num_file, 'r').readlines()))


if __name__ == '__main__':
    f = open('./bc40_msa_leq_10.txt', 'w')
    for name in low_quality_list:
        f.write(name)

