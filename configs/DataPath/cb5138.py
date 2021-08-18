sequence_data_path_prefix = '/bigdata/proli/data/CB513_fasta/valid/'
profile_real_data_path = '/bigdata/proli/data/pfam_CB513_real/train_tiff/*.tif'
profile_fake_data_path = '/bigdata/proli/data/pfam_CB513_fake/train_tiff/*.tif'
label_data_path_prefix = '/bigdata/proli/data/sse-label/CB5138/train/'
# SSEDatasetPSM require
high_quality_profile_list = list(filter(lambda x:int(x.split(' ')[1])>60, open('/bigdata/proli/data/CB513_num','r').readlines()))
# only for cb518
low_quality_profile_list = list(filter(lambda x:int(x.split(' ')[1])>5 and x not in high_quality_profile_list, open('/bigdata/proli/data/CB513_num','r').readlines()))

high_quality_profile_list = list(map(lambda x:x.split(' ')[0].strip(), high_quality_profile_list))
low_quality_profile_list = list(map(lambda x:x.split(' ')[0].strip(), low_quality_profile_list))
