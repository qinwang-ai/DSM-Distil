# coding=utf-8

import os
import torch
from torch.utils.data import DataLoader
from DataLoader.SSEDataset4TrainDSM import SSEDataset, SSEDatasetVal
from models.SSENetV3 import SSENet_Profile as SSENet
from tqdm import tqdm
import configs.config_sse as config
import torch.nn as nn
import numpy as np
import random
import torch.nn.utils as utils


def try_get_pretrained(ssenet, scratch=False):
    ssenet_path = config.pretrain_path + 'ssenet_dsm_net.pth'

    ssenet.init_weights()

    if not scratch:
        if os.path.exists(ssenet_path):
            ssenet.load_state_dict(torch.load(ssenet_path))

    return ssenet.cuda()


def parse_batch(batch):
    sequence = torch.tensor(batch['sequence'], dtype=torch.int64).cuda()
    profile = torch.tensor(batch['profile'], dtype=torch.float32).cuda()

    # CB513 ONLY
    # profile = torch.tensor(batch['profile'], dtype=torch.float32).cuda()[:,:,:-1]
    label = torch.tensor(batch['label'], dtype=torch.int64).cuda()
    msa_c = torch.tensor(batch['msa_c'], dtype=torch.float).cuda()
    return sequence, profile, label, msa_c


def train(sse_loader, val_loader, ssenet, is_test=False):
    optimizer = torch.optim.Adam(ssenet.parameters(), lr=1e-4, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    prev_best = 0
    for epoch in range(1, config.epochs + 1):
        # train
        if not is_test:
            ssenet.train()
            summary = []
            scheduler.step()
            for batch in tqdm(sse_loader):
                sequence, profile, label, msa_c = parse_batch(batch)

                optimizer.zero_grad()
                pred,_, _ = ssenet(profile, msa_c)  # 16 x 44 x 3
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
        ssenet.eval()
        summary = []

        for batch in tqdm(val_loader):
            sequence, profile, label, msa_c = parse_batch(batch)

            pred,_,_ = ssenet(profile, msa_c)
            pred_no_pad = pred[label != -1, :]
            label_no_pad = label[label != -1]
            errCE = ce_loss_func(pred_no_pad, label_no_pad)

            pred_label = torch.argmax(pred_no_pad, dim=-1)
            acc = (pred_label == label_no_pad).sum().float() / pred_label.shape[0]
            summary.append((0, errCE.item(), acc.item()))
        # statistic
        summary_np = np.array(summary)
        _summary = summary_np[:, 1:].astype(np.float).mean(axis=0)
        if not is_test and _summary[-1] > prev_best:
            prev_best = _summary[-1]
            save_model()
        print('[EVAL]', 'CE_loss: %0.3f, curr_acc: %0.3f, best_acc:%0.3f' % (_summary[0], _summary[1], prev_best))


def save_model():
    print('update model..')
    ssenet_path = config.pretrain_path + 'ssenet_dsm_net.pth'
    torch.save(ssenet.state_dict(), ssenet_path)

if __name__ == '__main__':
    is_test = False
    sse_dataset = SSEDatasetVal(config.profile_real_data_path, config.sequence_data_path_prefix,
                             config.label_data_path_prefix)
    config_bc_40 = config
    sse_val_dataset = SSEDatasetVal(config_bc_40.profile_real_data_path.replace('train', 'valid'),
                                 config_bc_40.sequence_data_path_prefix,
                                 config_bc_40.label_data_path_prefix.replace('train', 'valid'))

    sse_loader = DataLoader(sse_dataset, batch_size=config.batch_size, num_workers=config.batch_size,
                            collate_fn=sse_dataset.collate_fn, shuffle=True)
    sse_val_loader = DataLoader(sse_val_dataset, batch_size=1, num_workers=config.batch_size,
                                collate_fn=sse_dataset.collate_fn)
    # loss function
    ce_loss_func = nn.CrossEntropyLoss()

    ssenet = SSENet(config.profile_width)

    # try load pretrained model
    ssenet = try_get_pretrained(ssenet, scratch=False)

    train(sse_loader, sse_val_loader, ssenet, is_test=is_test)
