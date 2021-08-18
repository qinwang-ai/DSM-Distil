# coding=utf-8

import os
import torch
from torch.utils.data import DataLoader
from DataLoader.SSEDatasetSR import SSEDataset
from models.Teacher import Teacher
from tqdm import tqdm
import configs.config_sse as config
import configs.config_sse_bc40 as config_bc_40
import torch.nn as nn
import numpy as np
import random
import torch.nn.utils as utils

def try_get_pretrained(ssenet, scratch=False):
    ssenet_path = config.pretrain_path + 'bio_teacher.pth'

    ssenet.init_weights()

    if not scratch:
        if os.path.exists(ssenet_path):
            ssenet.load_state_dict(torch.load(ssenet_path))

    return ssenet.cuda()


def parse_batch(batch):
    sequence = torch.tensor(batch['sequence'], dtype=torch.int64).cuda()
    profile = torch.tensor(batch['real_profile'], dtype=torch.float32).cuda()
    label = torch.tensor(batch['label'], dtype=torch.int64).cuda()
    filename = batch['filename']
    return sequence, profile, label, filename


def train(sse_loader, val_loader, teacher, is_test=False):
    optimizer = torch.optim.Adam(teacher.parameters(), lr=config.gen_lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    prev_best = 0

    for epoch in range(1, config.epochs + 1):

        # train
        if not is_test:
            teacher.train()
            summary = []
            scheduler.step()
            for batch in tqdm(sse_loader):
                sequence, profile, label, _ = parse_batch(batch)

                optimizer.zero_grad()
                pred = teacher(sequence, profile)  # 16 x 44 x 3
                pred_no_pad = pred[label != -1, :]
                label_no_pad = label[label != -1]
                errCE = ce_loss_func(pred_no_pad, label_no_pad)
                errCE.backward()
                optimizer.step()

                pred_label = torch.argmax(pred_no_pad, dim=-1)
                acc = (pred_label == label_no_pad).sum().float() / pred_label.shape[0]
                summary.append((errCE.item(), acc.item()))

            # statistic
            summary = np.array(summary).mean(axis=0)
            print('Epoch %d' % epoch, 'CE_loss: %0.2f, acc: %0.2f' % (summary[0], summary[1]))

        # validation
        teacher.eval()
        summary = []

        for batch in tqdm(val_loader):
            sequence, profile, label, filename = parse_batch(batch)

            pred = teacher(sequence, profile)
            pred_no_pad = pred[label != -1, :]
            label_no_pad = label[label != -1]
            errCE = ce_loss_func(pred_no_pad, label_no_pad)

            pred_label = torch.argmax(pred_no_pad, dim=-1)
            acc = (pred_label == label_no_pad).sum().float() / pred_label.shape[0]
            summary.append((filename, errCE.item(), acc.item()))

        # statistic
        summary_np = np.array(summary)
        _summary = summary_np[:, 1:].astype(np.float).mean(axis=0)
        if not is_test and _summary[-1] > prev_best:
            prev_best = _summary[-1]
            save_model()
        print('[EVAL]', 'CE_loss: %0.3f, curr_acc: %0.3f, best_acc:%0.3f' % (_summary[0], _summary[1], prev_best))

        if is_test:
            summary = sorted(summary, key=lambda x:x[-1])
            f = open('logs/real_profile.txt', 'w')
            data = list(map(lambda x:' '.join([str(t) for t in x]), summary))
            f.writelines([s + '\n' for s in data])
            break


def save_model():
    print('update model..')
    ssenet_path = config.pretrain_path + 'bio_teacher.pth'
    torch.save(teacher.state_dict(), ssenet_path)


if __name__ == '__main__':
    is_test = False
    sse_dataset = SSEDataset(config.profile_real_data_path, config.sequence_data_path_prefix,
                             config.label_data_path_prefix, mode='high')

    sse_val_dataset = SSEDataset(config.profile_real_data_path.replace('train', 'valid'),
                                 config.sequence_data_path_prefix,
                                 config.label_data_path_prefix.replace('train', 'valid'),
                                 mode='high', config=config)

    sse_loader = DataLoader(sse_dataset, batch_size=config.batch_size, num_workers=config.batch_size,
                            collate_fn=sse_dataset.collate_fn, shuffle=True)
    sse_val_loader = DataLoader(sse_val_dataset, batch_size=1, num_workers=config.batch_size,
                                collate_fn=sse_dataset.collate_fn)
    # loss function
    ce_loss_func = nn.CrossEntropyLoss()

    teacher = Teacher(config.embed_dim + config.profile_width)

    # try load pretrained model
    teacher = try_get_pretrained(teacher, scratch=True)

    train(sse_loader, sse_val_loader, teacher, is_test=is_test)


# DSM epoch 10 863
# Prof epoch 20 850
