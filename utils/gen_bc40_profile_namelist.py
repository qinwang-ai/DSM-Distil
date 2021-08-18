import glob
def main():
    file_list=[]
    for file in glob.glob("/data/proli/data/uniref90/uniref90_bc40_real_nogap/valid/*.npy"):
        filename = file.split('/')[-1].split('.npy')[0]
        file_list.append(filename)

    f = open('./bc40_real_low_list', 'w')
    f.writelines('\n'.join(file_list))



if __name__ == "__main__":
    main()