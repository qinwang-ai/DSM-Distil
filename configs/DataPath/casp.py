sequence_data_path_prefix = '/bigdata/proli/data/casp_fasta/valid/'
profile_real_data_path = '/bigdata/proli/data/pfam_casp_real/train_tiff/*.tif'
profile_fake_data_path = '/bigdata/proli/data/pfam_casp_fake/train_tiff/*.tif'
label_data_path_prefix = '/bigdata/proli/data/sse-label/casp_label/train/'
# SSEDatasetPSM require
# psm_real_data_path_prefix = '/bigdata/proli/data/casp_psm_real/'
high_quality_profile_list = list(filter(lambda x:int(x.split(' ')[1])>60, open('/bigdata/proli/data/casp_num','r').readlines()))
high_quality_profile_list = list(map(lambda x:x.split(' ')[0].strip(), high_quality_profile_list))