import numpy as np
import os
import random
if __name__ == '__main__':
    bc40_list = open('cl_bc40_list').readlines()
    bc40_list = list(map(lambda x:x.strip(), bc40_list))
    amount = 10000
    num_valid = int(amount*0.2)

    random.shuffle(bc40_list)

    f = open('cl_bc40_list_valid', 'w')
    f.write('\n'.join(bc40_list[:num_valid]))

    f = open('cl_bc40_list_train', 'w')
    f.write('\n'.join(bc40_list[num_valid:]))
