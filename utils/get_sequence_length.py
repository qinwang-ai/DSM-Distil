from configs.config_sse import sequence_data_path_prefix
if __name__ == '__main__':
    f = open('/data/proli/data/bc40_count500_num', 'r').readlines()
    f2 = open('./bc40_fasta_leq_120.txt', 'w')
    for l in f:
        filename = l.split(' ')[0].strip()
        count = int(l.split(' ')[1].strip())
        fasta = open(sequence_data_path_prefix + filename + '.fasta').read()
        if len(fasta) <= 120 and count > 450:
            f2.write("%s %d %d\n" % (filename, count, len(fasta)))
    f2.close()

