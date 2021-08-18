sequence_data_path_prefix = '/bigdata/proli/data/cameo_fasta/valid/'
profile_real_data_path = '/bigdata/proli/data/pfam_cameo_real/train_tiff/*.tif'
profile_fake_data_path = '/bigdata/proli/data/pfam_cameo_fake/train_tiff/*.tif'
label_data_path_prefix = '/bigdata/proli/data/sse-label/cameo_label/train/'
# SSEDatasetPSM require
psm_real_data_path_prefix = '/bigdata/proli/data/pfam_psm_6125_real/'
high_quality_profile_list = list(filter(lambda x:int(x.split(' ')[1])>1, open('/bigdata/proli/data/cameo_num','r').readlines()))
high_quality_profile_list = list(map(lambda x:x.split(' ')[0].strip(), high_quality_profile_list))