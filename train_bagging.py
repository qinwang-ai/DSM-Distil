# coding=utf-8

import os
import torch
from torch.utils.data import DataLoader
from DataLoader.SSEDatasetPSMSimple import SSEDataset
from models.SSENet import SSENet
from models.Generator import Generator
from tqdm import tqdm
import configs.config_sse as config
import configs.config_sse_bc40 as config_bc_40
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def try_get_pretrained(ssenet, generator, scratch=config.train_scratch):
    ssenet_path = '../module/ssenet_real_ref90_psm.pth'
    generator_path = '../module/generator_bagging.pth'

    ssenet.init_weights()
    generator.init_weights()

    if not scratch:
        if os.path.exists(ssenet_path):
            ssenet.load_state_dict(torch.load(ssenet_path))
            print('got pretrained ssenet')

        if os.path.exists(generator_path):
            generator.load_state_dict(torch.load(generator_path))
            print('got pretrained generator')

    return ssenet.cuda(), generator.cuda()


def parse_batch(batch):
    sequence = torch.tensor(batch['sequence'], dtype=torch.int64).cuda()
    label = torch.tensor(batch['label'], dtype=torch.int64).cuda()
    real_psm = torch.tensor(batch['real_psm'], dtype=torch.float32).cuda()
    low_psm = torch.tensor(batch['low_psm'], dtype=torch.float32).cuda()
    filename = batch['filename']
    return filename, sequence, low_psm, real_psm, label


def get_ce_loss(sequence, psm, label):
    pred = ssenet(sequence, psm)  # 16 x 44 x 3
    pred_no_pad = pred[label != -1, :]
    label_no_pad = label[label != -1]
    errCE = ce_loss_func(pred_no_pad, label_no_pad)
    pred_label = torch.argmax(pred_no_pad, dim=-1)
    acc = (pred_label == label_no_pad).sum().float() / pred_label.shape[0]
    return errCE, acc


def get_mse_loss(sequence, low_psm, real_psm, label):
    low_psm = low_psm[label != -1, :]
    real_psm = real_psm[label != -1, :]

    mse_loss = F.mse_loss(low_psm, real_psm)
    return mse_loss


def train_gen(sse_loader, val_loader, generator, epochs=100):
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.001, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=40, gamma=0.1)
    prev_best = -1000

    for epoch in range(1, epochs):
        # train
        generator.train()
        summary = []
        scheduler.step()
        for batch in tqdm(sse_loader):
            _, sequence, low_psm, real_psm, label = parse_batch(batch)

            # Train SSE + adversrial
            optimizer_gen.zero_grad()
            fake_psm = generator(sequence, low_psm)
            MSE_loss = get_mse_loss(sequence, fake_psm, real_psm, label)
            MSE_loss.backward()
            optimizer_gen.step()

            summary.append(MSE_loss.item())

        # statistic
        summary = np.array(summary).mean()
        print('Epoch %d' % epoch, 'MSE_loss: %07f' % summary)

        # validation
        # generator.eval()
        # summary = []
        # for batch in tqdm(val_loader):
        #     _, sequence, low_psm, real_psm, label = parse_batch(batch)
        #     profile_fake = generator(sequence, low_psm)
        #     errMSE = get_mse_loss(sequence, profile_fake, real_psm, label)
        #     summary.append(errMSE.item())

        curr_acc = test_sse(val_loader, generator, ssenet, 1)
        # statistic
        if curr_acc > prev_best:
            prev_best = curr_acc
            save_model(generator, '../module/generator_bagging.pth')
        print('[EVAL]', 'curr_acc: %0.3f, best_acc:%0.3f' % (curr_acc, prev_best))


def update_profile(fake_profile, profile_gt, high_quality):
    fake_profile[high_quality == 1, :, :] = profile_gt[high_quality == 1, :, :]
    return fake_profile


def test_sse(val_loader, generator, ssenet, epochs):

    f = open('logs/bagging_acc3_6125.txt', 'w')
    generator.eval()
    for epoch in range(1, epochs + 1):
        # validation
        ssenet.eval()
        summary = []
        for batch in tqdm(val_loader):
            filename, sequence, low_profile, profile_gt, label = parse_batch(batch)
            profile_fake = generator(sequence, low_profile)

            errMSE = get_mse_loss(sequence, profile_fake, profile_gt, label)
            errCE, acc = get_ce_loss(sequence, profile_fake.softmax(dim=-1), label)

            summary.append((errCE.item(), errMSE.item(), acc.item()))
            # f.write('%s %f\n' % (filename, acc.item()))
        # statistic
        summary = np.array(summary).mean(axis=0)
        print('[EVAL]', 'CE_loss: %0.3f, MSE_loss: %0.3f, curr_acc: %0.3f' % (
            summary[0], summary[1], summary[2]))
        return summary[-1]


def save_model(net, path):
    print('update model..')
    torch.save(net.state_dict(), path)


if __name__ == '__main__':
    is_test = True
    # is_test = False
    sse_loader = None
    if not is_test:
        sse_dataset = SSEDataset(config.psm_real_data_path,
                                 config.sequence_data_path_prefix,
                                 config.label_data_path_prefix, mode='high', config=config)
        sse_loader = DataLoader(sse_dataset, batch_size=config.batch_size, num_workers=config.batch_size,
                                collate_fn=sse_dataset.collate_fn, shuffle=True)
        val_mode = 'high'
    else:
        val_mode = 'low'

    # config_tmp = config_bc_40
    config_tmp = config
    sse_val_dataset = SSEDataset(config_tmp.psm_real_data_path.replace('train', 'valid'),
                                 config_tmp.sequence_data_path_prefix,
                                 config_tmp.label_data_path_prefix.replace('train', 'valid'),
                                 mode='low', config=config_tmp)

    sse_val_loader = DataLoader(sse_val_dataset, batch_size=1, num_workers=config.batch_size,
                                collate_fn=sse_val_dataset.collate_fn, shuffle=False)
    # loss function
    ce_loss_func = nn.CrossEntropyLoss()

    ssenet = SSENet(input_dim=config.embed_dim + config.profile_width)
    generator = Generator(input_dim=config.embed_dim + config.profile_width)

    # try load pretrained model
    ssenet, generator = try_get_pretrained(ssenet, generator, scratch=False)

    if is_test:
        test_sse(sse_val_loader, generator, ssenet, epochs=1)
    else:
        train_gen(sse_loader, sse_val_loader, generator, epochs=100)
