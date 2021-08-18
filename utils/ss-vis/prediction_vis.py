import numpy as np
def main():
    f = '/data/proli/raw_data/visualization/enhanced_ss/1ka8A.ss.npy'
    # f = '/data/proli/raw_data/visualization/low_ss/1ka8A.ss.npy'
    a = list(np.load(f))
    str=''
    for i in a:
        if i == 0:
            str+='H'
        elif i == 1:
            str+='E'
        else:
            str+='C'
    print(str)

if __name__ == '__main__':
    main()