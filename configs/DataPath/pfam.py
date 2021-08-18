sequence_data_path_prefix = '/bigdata/proli/raw_data/6125_fasta/'
profile_real_data_path = '/bigdata/proli/data/pfam_profile_6125_real/train_tiff/*.tif'
profile_fake_data_path = '/bigdata/proli/data/pfam_profile_6125_fake/train_tiff/*.tif'


# phfam 3
label_data_path_prefix = '/bigdata/proli/data/sse-label/6125_label/train/'

# phfam 8
# label_data_path_prefix = '/bigdata/proli/data/sse-label/pfam8/train/'


# downsmapling online profile calculation require
msa_data_path_prefix = '/bigdata/proli/pfam_6125_matchlength/'
# SSEDatasetPSM require
psm_real_data_path_prefix = '/bigdata/proli/data/pfam_psm_6125_real/'
high_quality_profile_list = list(filter(lambda x:int(x.split(' ')[1])>60, open('./6125_pfam_a3m_num.txt','r').readlines()))
high_quality_profile_list = list(map(lambda x:x.split(' ')[0].strip(), high_quality_profile_list))