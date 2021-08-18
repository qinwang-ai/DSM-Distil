import os
from shutil import copyfile
casp11_list = list(map(lambda x: x.strip(), open('/bigdata/proli/data/casp11_list', 'r').readlines()))

if __name__ == '__main__':
    for filename in casp11_list:
        source = '/bigdata/proli/data/pfam_casp_fake/valid_tiff/'+filename+'.tif'
        target = '/bigdata/proli/data/pfam_casp11_fake/valid_tiff/'+filename+'.tif'
        copyfile(source, target)
