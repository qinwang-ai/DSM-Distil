sequence_data_path_prefix = '/bigdata/proli/data/casp_fasta/valid/'
profile_real_data_path = '/bigdata/proli/data/pfam_casp11_real/train_tiff/*.tif'
profile_fake_data_path = '/bigdata/proli/data/pfam_casp11_fake/train_tiff/*.tif'
label_data_path_prefix = '/bigdata/proli/data/sse-label/casp_label/train/'

high_quality_profile_list = list(filter(lambda x:int(x.split(' ')[1])>60, open('/bigdata/proli/data/casp_num','r').readlines()))

low_quality_profile_list = list(filter(lambda x:int(x.split(' ')[1])>0 and x not in high_quality_profile_list, open('/bigdata/proli/data/casp_num','r').readlines()))

high_quality_profile_list = list(map(lambda x:x.split(' ')[0].strip(), high_quality_profile_list))
low_quality_profile_list = list(map(lambda x:x.split(' ')[0].strip(), low_quality_profile_list))
