# coding=utf-8
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pretrain.PreTrainDataset import PreTrainDataset
from models.Generator import Generator
from models.SSENet import SSENet
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import configs.config_sse_bc40 as config


def try_get_pretrained(teacher, generator, scratch=False):
    teacher_path = '../module/teacher_our_ref90.pth'
    generator_path = '../module/generator_our_ref90_init.pth'

    teacher.init_weights()
    generator.init_weights()

    if not scratch:
        if os.path.exists(teacher_path):
            teacher.load_state_dict(torch.load(teacher_path))
            print('load teacher')

        # if os.path.exists(generator_path):
        #     generator.load_state_dict(torch.load(generator_path))
        #     print('load generator')

    return teacher.cuda(), generator.cuda()


def save_model():
    generator_path = '../module/generator_our_ref90_init.pth'
    torch.save(generator.state_dict(), generator_path)


def parse_batch(batch):
    sequence = torch.tensor(batch['sequence'], dtype=torch.int64).cuda()
    psm_array = torch.tensor(batch['psm_array'], dtype=torch.float32).cuda()
    low_psm_array = torch.tensor(batch['low_psm_array'], dtype=torch.float32).cuda()
    bert_psm_array = torch.tensor(batch['bert_psm_array'], dtype=torch.float32).cuda()
    return sequence, psm_array, low_psm_array, bert_psm_array


def get_pred(sequence, profile, net):
    pred = net(sequence, profile)  # 16 x 44 x 3
    pred_no_pad = pred[label != -1, :]
    return pred_no_pad


def get_mse_loss(sequence, profile, profile_gt):
    profile = profile[label != -1, :]
    profile_gt = profile_gt[label != -1, :]

    mse_loss = F.mse_loss(profile, profile_gt)
    return mse_loss


def train(sse_loader, generator, teacher, epochs=1000):
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=config.gen_lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=100, gamma=0.1)

    teacher.train()
    for epoch in range(1, epochs + 1):

        # train
        generator.train()
        summary = []
        scheduler.step()
        for batch in tqdm(sse_loader):
            sequence, psm_array, low_psm_array, bert_psm_array = parse_batch(batch)

            # Train SSE + adversrial
            optimizer_gen.zero_grad()

            profile = torch.cat([psm_array, bert_psm_array], dim=2)
            x = generator(sequence, profile)

            # MSE loss
            errMSE = 0.1 * get_mse_loss(sequence, x, psm_array)

            # get x for contrastive learning
            f_x = get_pred(sequence, x, teacher)

            # get f_positive
            f_pos = get_pred(sequence, psm_array, teacher)

            # get f_negative
            f_neg_1 = get_pred(sequence, low_psm_array,teacher)
            f_neg_2 = get_pred(sequence, bert_psm_array, teacher)

            # triplet loss
            d_x_pos_x = F.kl_div(F.log_softmax(f_pos.detach(), dim=1), F.softmax(f_x, dim=1))
            d_x_neg_x_1 = F.kl_div(F.log_softmax(f_neg_1.detach(), dim=1), F.softmax(f_x, dim=1))
            d_x_neg_x_2 = F.kl_div(F.log_softmax(f_neg_2.detach(), dim=1), F.softmax(f_x, dim=1))
            d_x_neg_x = (d_x_neg_x_1 + d_x_neg_x_2) / 2

            triplet_loss = 5 * max(d_x_pos_x - d_x_neg_x + 0.6, torch.tensor(0))

            (errMSE + triplet_loss).backward()

            optimizer_gen.step()

            summary.append((errMSE.item(), triplet_loss.item()))

        # statistic
        summary = np.array(summary).mean(axis=0)
        print('Epoch %d' % epoch,
              'errMSE: %0.3f, triplet_loss: %0.3f' % (summary[0], summary[1]))
        save_model()


if __name__ == '__main__':
    train_name_path = './cl_bc40_list'
    sequence_data_path_prefix = '/data/proli/data/bc40_fasta/'
    psm_path_prefix = '/data/proli/data/ref90_psm_bc40_pretrain/'
    msa_data_path_prefix = '/data/proli/raw_data/cl_bc40_pseudo_a3m/'

    sse_dataset = PreTrainDataset(train_name_path, sequence_data_path_prefix, psm_path_prefix, msa_data_path_prefix)
    sse_loader = DataLoader(sse_dataset, batch_size=config.batch_size, num_workers=config.batch_size,
                            collate_fn=sse_dataset.collate_fn, shuffle=True)

    # loss function
    mse_func = nn.MSELoss()
    kl_func = nn.KLDivLoss()
    ce_loss_func = nn.CrossEntropyLoss

    teacher = SSENet(input_dim=config.embed_dim + config.profile_width)
    generator = Generator()

    # try load pretrained model
    teacher, generator = try_get_pretrained(teacher, generator, scratch=False)
    train(sse_loader, generator, teacher)
