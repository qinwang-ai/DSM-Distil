# coding=utf-8
import os
import configs.config_sse_bc40 as config_40
import torch
from torch.utils.data import DataLoader
from DataLoader.SSEDatasetPSM import SSEDataset
from models.SSENet import SSENet
from models.Generator import Generator
from tqdm import tqdm
import configs.config_sse as config
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def try_get_pretrained(teacher, student, generator, scratch=False):
    teacher_path = '../module/teacher_our_ref90.pth'
    student_path = '../module/student_our_ref90.pth'
    generator_path = '../module/generator_our_ref90.pth'

    student.init_weights()
    teacher.init_weights()
    generator.init_weights()

    if not scratch:
        if os.path.exists(teacher_path):
            teacher.load_state_dict(torch.load(teacher_path))
            print('load teacher')

        if os.path.exists(student_path):
            student.load_state_dict(torch.load(student_path))
            print('load student')
        
        if os.path.exists(generator_path):
            generator.load_state_dict(torch.load(generator_path))
            print('load generator')
    return teacher.cuda(), student.cuda(), generator.cuda()

def save_model():
    print('update model..')
    student_path = '../module/student_our_ref90.pth'
    generator_path = '../module/generator_our_ref90.pth'
    torch.save(student.state_dict(),  student_path)
    torch.save(generator.state_dict(), generator_path)


def parse_batch(batch):
    sequence = torch.tensor(batch['sequence'], dtype=torch.int64).cuda()
    profile = torch.tensor(batch['bert_psm'], dtype=torch.float32).cuda()
    label = torch.tensor(batch['label'], dtype=torch.int64).cuda()
    profile_gt = torch.tensor(batch['real_psm'], dtype=torch.float32).cuda()
    low_profile = torch.tensor(batch['low_psm'], dtype=torch.float32).cuda()
    high_quality = torch.tensor(batch['high_quality'], dtype=torch.int64).cuda()
    return sequence, profile, low_profile, profile_gt, label, high_quality


def remove_pad_keep_dim(data, sequence):
    ans = []
    for i in range(data.shape[0]):
        ans.append(data[i, sequence[i] !=0, :])
    return ans

def get_ce_loss(sequence, profile, label, net):
    pred = net(sequence, profile)  # 16 x 44 x 3
    pred_no_pad = pred[label != -1, :]
    label_no_pad = label[label != -1]
    errCE = ce_loss_func(pred_no_pad, label_no_pad)
    pred_label = torch.argmax(pred_no_pad, dim=-1)
    acc = (pred_label == label_no_pad).sum().float() / pred_label.shape[0]
    return errCE, acc, pred_no_pad


def get_mse_loss(sequence, profile, profile_gt, high_quality):
    sequence = sequence[high_quality == 1, :]
    profile = profile[high_quality == 1, :, :]
    profile_gt = profile_gt[high_quality == 1, :, :]

    profile = profile[label != -1, :]
    profile_gt = profile_gt[label != -1, :]

    mse_loss = F.mse_loss(profile, profile_gt)
    return mse_loss


def update_profile(fake_profile, profile_gt, high_quality):
    fake_profile[high_quality == 1, :, :] = profile_gt[high_quality == 1, :, :]
    return fake_profile


def train(sse_loader, val_loader, generator, teacher, student, epochs=config.epochs):
    optimizer_sse = torch.optim.Adam(student.parameters(), lr=config.gen_lr, weight_decay=1e-8)
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=config.gen_lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=100, gamma=0.1)

    teacher.train()
    for epoch in range(1, epochs + 1):

        if epochs != 1:
            # train
            student.train()
            generator.train()
            summary = []
            scheduler.step()
            for batch in tqdm(sse_loader):
                sequence, bert_psm, low_psm, real_psm, label, high_quality = parse_batch(batch)

                # Train SSE + adversrial
                optimizer_sse.zero_grad()
                optimizer_gen.zero_grad()

                # get x for distill
                profile = torch.cat([bert_psm, low_psm], dim=2)
                enhance_psm = generator(sequence, profile)

                errCE, acc, x_d = get_ce_loss(sequence, enhance_psm, label, student)

                # get x for contrastive learning
                errCE, acc, x_c = get_ce_loss(sequence, enhance_psm, label, teacher)

                # get f_positive
                _, _, x_pos = get_ce_loss(sequence, real_psm, label, teacher)

                # get f_negative
                _, _, x_neg_1 = get_ce_loss(sequence, low_psm, label, teacher)
                _, _, x_neg_2 = get_ce_loss(sequence, bert_psm, label, teacher)

                # MSE loss
                errMSE = 0.1 * get_mse_loss(sequence, enhance_psm, real_psm, high_quality)

                # triplet loss
                d_x_pos_x = F.kl_div(F.log_softmax(x_pos.detach(), dim=1), F.softmax(x_c, dim=1))
                d_x_neg_x_1 = F.kl_div(F.log_softmax(x_neg_1.detach(), dim=1), F.softmax(x_c, dim=1))
                d_x_neg_x_2 = F.kl_div(F.log_softmax(x_neg_2.detach(), dim=1), F.softmax(x_c, dim=1))
                d_x_neg_x = (d_x_neg_x_1 + d_x_neg_x_2) / 2

                triplet_loss = 5 * max(d_x_pos_x - d_x_neg_x + 0.6, torch.tensor(0))

                # Distill Loss
                _d_x_pos_x = F.kl_div(F.log_softmax(x_pos.detach(), dim=1), F.softmax(x_d, dim=1))
                errDistil = errCE + 10 * _d_x_pos_x

                (errMSE + triplet_loss + errDistil).backward()

                optimizer_sse.step()
                optimizer_gen.step()

                summary.append((errDistil.item(), errMSE.item(), triplet_loss.item(), acc.item()))

            # statistic
            summary = np.array(summary).mean(axis=0)
            print('Epoch %d' % epoch,
                  'errDistil: %0.2f, errMSE: %0.2f, triplet_loss: %0.2f, acc: %0.2f' % (
                  summary[0], summary[1], summary[2], summary[3]))

        # validation
        student.eval()
        generator.eval()
        summary = []
        for batch in tqdm(val_loader):
            sequence, bert_psm, low_psm, real_psm, label, high_quality = parse_batch(batch)
            profile = torch.cat([bert_psm, real_psm], dim=2)

            psm = generator(sequence, profile)

            errCE, acc, _ = get_ce_loss(sequence, psm, label, student)

            high_quality = np.ones(high_quality.shape)
            errMSE = 0.1 * get_mse_loss(sequence, psm, real_psm, high_quality)

            summary.append((errCE.item(), errMSE.item(), acc.item()))

        # statistic
        summary = np.array(summary).mean(axis=0)
        global prev_best
        if summary[-1] > prev_best:
            prev_best = summary[-1]
            if epochs != 1:
                save_model()
        print('[EVAL]', 'CE_loss: %0.3f, errMSE: %0.3f, curr_acc: %0.3f, best_acc:%0.3f' % (
            summary[0], summary[1], summary[2], prev_best))



if __name__ == '__main__':
    is_test = False
    sse_loader = None
    prev_best = 0

    for i in range(10):
        if not is_test:
            sse_dataset = SSEDataset(config.psm_real_data_path,
                                     config.real_profile_path_prefix,
                                     config.sequence_data_path_prefix,
                                 config.label_data_path_prefix, mode='high')
            sse_loader = DataLoader(sse_dataset, batch_size=config.batch_size, num_workers=config.batch_size,
                                    collate_fn=sse_dataset.collate_fn, shuffle=True)

        sse_val_dataset = SSEDataset(config_40.psm_real_data_path.replace('train', 'valid'),
                                     config_40.real_profile_path_prefix.replace('train', 'valid'),
                                     config_40.sequence_data_path_prefix,
                                     config_40.label_data_path_prefix.replace('train', 'valid'),
                                     mode='low', enable_downsample=False, config=config_40)

        # sse_val_dataset = SSEDataset(config.psm_real_data_path.replace('train', 'valid'),
        #                              config.real_profile_path_prefix.replace('train', 'valid'),
        #                              config.sequence_data_path_prefix,
        #                              config.label_data_path_prefix.replace('train', 'valid'),
        #                              mode='low', enable_downsample=False, config=config)


        sse_val_loader = DataLoader(sse_val_dataset, batch_size=1, num_workers=config.batch_size,
                                    collate_fn=sse_val_dataset.collate_fn)
        # loss function
        ce_loss_func = nn.CrossEntropyLoss()
        kl_func = nn.KLDivLoss()

        teacher = SSENet(input_dim=config.embed_dim + config.profile_width)
        student = SSENet(input_dim=config.embed_dim + config.profile_width)
        generator = Generator()

        # try load pretrained model
        teacher, student, generator = try_get_pretrained(teacher, student, generator, scratch=False)
        if is_test:
            train(sse_loader, sse_val_loader, generator, teacher, student, epochs=1)
        else:
            train(sse_loader, sse_val_loader, generator, teacher, student)
            break
