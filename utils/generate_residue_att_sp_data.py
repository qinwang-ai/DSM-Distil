import numpy as np
from tqdm import tqdm
import configs.config_sse as config
code_dir = '/data/proli/Bioinfo/'

def main():
	real_low_file = code_dir + 'logs/low_dsm_acc8_bc40.txt'
	fake_p_file = code_dir + 'logs/fake_dsm_acc8_bc40.txt'
	real_low = open(real_low_file).readlines()
	fakep = open(fake_p_file).readlines()

	real_low = list(map(lambda x:x.split(' '), real_low))
	fakep = list(map(lambda x:x.split(' '), fakep))

	real_low_dict = {}
	for (filename, errs) in real_low:
		real_low_dict[filename] = list(map(lambda x: float(x), errs.split(',')))

	fake_p_dict = {}
	for (filename, errs) in fakep:
		fake_p_dict[filename] = list(map(lambda x: float(x), errs.split(',')))

	# blosum_dict = {}
	# for (filename, errs) in blosum:
	# 	blosum_dict[filename] = list(map(lambda x: float(x), errs.split(',')))

	ans ={}
	for (filename, errs) in real_low_dict.items():
		fp_errs = fake_p_dict[filename]
		# blo_errs = blosum_dict[filename]
		data = []
		for i in range(len(errs)):
			loss_arr = np.array([errs[i], fp_errs[i]])
			loss_soft = 1 - loss_arr/loss_arr.sum()

			data.append(loss_soft)
			# data.append(int(errs[i] > fp_errs[i]))

		ans[filename] = np.array(data)

	np.save(config.sp_att_data_path, ans, allow_pickle=True)
	print(config.sp_att_data_path, 'saved....')


# real_low: 0 [MIN_ERR, MAX_ERR]
# fake_p: 1
# blosum: 2
if __name__ == '__main__':
	main()