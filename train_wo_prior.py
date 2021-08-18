# coding=utf-8
import os
import torch
from torch.utils.data import DataLoader
from DataLoader.SSEDatasetATTV2 import SSEDataset
from models.Teacher import Teacher as SSENet
from models.GradeNet import SSENet as GradeNet
from models.Student import Student
from tqdm import tqdm
import configs.config_sse as config
import configs.config_sse_bc40 as config_bc_40
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random

def try_get_pretrained_raw(ssenet_lowreal, ssenet_fake):
    ssenet_low_real_path = config.pretrain_path + 'ssenet_raw_lowprofile.pth'
    ssenet_fake_path = config.pretrain_path + 'ssenet_raw_fakeprofile.pth'

    ssenet_lowreal.load_state_dict(torch.load(ssenet_low_real_path))
    ssenet_fake.load_state_dict(torch.load(ssenet_fake_path))

    ssenet_lowreal.eval()
    ssenet_fake.eval()
    return ssenet_lowreal.cuda(), ssenet_fake.cuda()

def get_raw_features(sequence, low_profile, fake_profile):
    with torch.no_grad():
        pred_low_real, low_dsm = ssenet_lowreal(sequence, low_profile)  # 16 x 44 x 3
        pred_fake, fake_dsm = ssenet_fake(sequence, fake_profile)  # 16 x 44 x 3
    return pred_low_real.detach(), pred_fake.detach(), low_dsm.detach(), fake_dsm.detach()

def try_get_pretrained(teacher, student, scratch=False):
    teacher_path = '../module/bio_teacher.pth'
    student_path = '../module/bioinfo_student_wo_prior.pth'

    teacher.init_weights()
    student.init_weights()

    teacher.load_state_dict(torch.load(teacher_path))

    if not scratch:
        if os.path.exists(student_path):
            student.load_state_dict(torch.load(student_path))
            print('load bioinfo_student', student_path)

    return teacher.cuda(), student.cuda()

def save_model():
    print('update model..')
    student_path = '../module/bioinfo_student_wo_prior.pth'
    torch.save(student.state_dict(),  student_path)

def parse_batch(batch):
    filename = batch['filename']
    sequence = torch.tensor(batch['sequence'], dtype=torch.int64).cuda()
    label = torch.tensor(batch['label'], dtype=torch.int64).cuda()

    att_gt = torch.tensor(batch['att_gt'], dtype=torch.float32).cuda()
    real_profile = torch.tensor(batch['real_profile'], dtype=torch.float32).cuda()
    fake_profile = torch.tensor(batch['fake_profile'], dtype=torch.float32).cuda()
    low_profile = torch.tensor(batch['low_profile'], dtype=torch.float32).cuda()
    return filename, sequence, low_profile, real_profile, fake_profile, label, att_gt

def get_ce_loss(sequence, profile, feats, label, net):
    if isinstance(net, Student):
        pred, _ = net(sequence, profile, feats)  # 16 x 44 x 3
    else:
        pred, _ = net(sequence, profile)  # 16 x 44 x 3
    pred_no_pad = pred[label != -1, :]
    label_no_pad = label[label != -1]
    errCE = ce_loss_func(pred_no_pad, label_no_pad)
    pred_label = torch.argmax(pred_no_pad, dim=-1)
    acc = (pred_label == label_no_pad).sum().float() / pred_label.shape[0]
    return errCE, acc, pred_no_pad


def get_mxe_loss(profile, profile_gt, label):
    profile = profile[label != -1, :]
    profile_gt = profile_gt[label != -1, :]
    mxe_loss = F.mse_loss(profile, profile_gt)
    return mxe_loss


def get_grade_loss(sequence, feats, att_gt, gradenet, label):
    vec, _ = gradenet(sequence, feats) # 16 x 44 x 3
    vec_nopad, att_gt_nopad = vec[label != -1, :], att_gt[label != -1, :]
    err = F.mse_loss(torch.log(0.01+vec_nopad), torch.log(0.01+att_gt_nopad))

    att_gt_label = torch.argmax(att_gt_nopad, dim=-1)
    acc = (torch.argmax(vec_nopad, dim=-1) == att_gt_label).sum()/float(att_gt_label.shape[0])
    u, v = vec[:, :, 0].unsqueeze(dim=-1), vec[:, :, 1].unsqueeze(dim=-1)
    return err, (u,v), acc


def train(sse_loader, val_loader, teacher, student, epochs=config.epochs):

    optimizer = torch.optim.Adam(list(student.parameters()), lr=config.gen_lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    teacher.eval()
    for epoch in range(1, epochs + 1):

        if epochs != 1:
            # train
            student.train()

            summary = []
            scheduler.step()
            for batch in tqdm(sse_loader):
                filename, sequence, low_profile, real_profile, fake_profile, label, att_gt = parse_batch(batch)

                optimizer.zero_grad()

                # profile transform
                low_feats, fake_feats, low_dsm, fake_dsm = get_raw_features(sequence, low_profile, fake_profile)


                errGrade = torch.tensor(0)

                # Obtain SSE CE loss
                errCE, acc, x_d = get_ce_loss(sequence, low_dsm, low_feats, label, student)

                # f(ph)
                _, _, x_pos = get_ce_loss(sequence, real_profile, None, label, teacher)

                # Distill Loss
                _d_x_pos_x = F.kl_div(F.log_softmax(x_pos.detach(), dim=1), F.softmax(x_d, dim=1))
                errDistil = errCE + 10 * _d_x_pos_x

                errDistil.backward()
                optimizer.step()

                summary.append((errDistil.item(), errGrade.item(), acc.item()))

            # statistic
            summary = np.array(summary).mean(axis=0)
            print('Epoch %d' % epoch,
                  'errDistil: %0.3f, errGrade: %0.5f, acc: %0.2f' % (
                  summary[0], summary[1], summary[2]))

        # validation
        student.eval()

        _summary = []
        for batch in tqdm(val_loader):
            filename, sequence, low_profile, real_profile, fake_profile, label, att_gt = parse_batch(batch)

            # profile transform
            low_feats, fake_feats, low_dsm, fake_dsm = get_raw_features(sequence, low_profile, fake_profile)
            # feats = feats.softmax(dim=-1)
            grade_acc = torch.tensor(0)


            errCE_stu, acc, x_d = get_ce_loss(sequence, low_dsm, low_feats, label, student)

            _summary.append((filename, errCE_stu.item(), grade_acc.item(), acc.item()))

        # statistic
        summary_np = np.array(_summary)[:, 1:].astype(np.float)
        summary = summary_np.mean(axis=0)
        _std = np.array(summary_np).std(axis=0)
        global prev_best
        if summary[-1] > prev_best:
            prev_best = summary[-1]
            if epochs != 1:
                save_model()
        print('[EVAL]', 'CE_loss: %0.3f, grade_acc: %0.3f, curr_acc: %0.3f, best_acc:%0.3f, std: %0.3f' % (
            summary[0], summary[1], summary[2], prev_best, _std[-1]))

        if is_test:
            # summary = sorted(_summary, key=lambda x:x[-1])
            # f = open('logs/our_wo_prior_acc3_bc40.txt', 'w')
            # data = list(map(lambda x:' '.join([str(t) for t in x]), summary))
            # f.writelines([s + '\n' for s in data])
            break


if __name__ == '__main__':
    # is_test = False
    is_test = True
    sse_loader = None
    prev_best = 0

    for i in range(1):
        sse_dataset = SSEDataset(config.profile_fake_data_path,
                                 config.sequence_data_path_prefix,
                                 config.blosum_path_prefix,
                                 config.sp_att_data_path,
                                 config.real_profile_data_prefix,
                                 config.label_data_path_prefix, mode='high')

        config_tmp = config_bc_40
        # config_tmp = config
        sse_val_dataset = SSEDataset(config_tmp.profile_fake_data_path.replace('train', 'valid'),
                                     config_tmp.sequence_data_path_prefix,
                                     config_tmp.blosum_path_prefix.replace('train', 'valid'),
                                     config_tmp.sp_att_data_path,
                                     config_tmp.real_profile_data_prefix.replace('train', 'valid'),
                                     config_tmp.label_data_path_prefix.replace('train', 'valid'),
                                     mode='low', config=config_tmp, enable_downsample=False)
        if not is_test:
            sse_loader = DataLoader(sse_dataset, batch_size=config.batch_size, num_workers=config.batch_size,
                                    collate_fn=sse_dataset.collate_fn, shuffle=True)
        sse_val_loader = DataLoader(sse_val_dataset, batch_size=1, num_workers=config.batch_size,
                                    collate_fn=sse_dataset.collate_fn)

        ce_loss_func = nn.CrossEntropyLoss()
        kl_func = nn.KLDivLoss()
        teacher = SSENet(input_dim=config.embed_dim + config.profile_width)
        student = Student(input_dim=config.embed_dim + config.profile_width + 3)

        ssenet_lowreal = SSENet(config.embed_dim + config.profile_width)
        ssenet_fake = SSENet(config.embed_dim + config.profile_width)
        # get raw net
        ssenet_lowreal, ssenet_fake = try_get_pretrained_raw(ssenet_lowreal, ssenet_fake)

        # try load pretrained model
        teacher, student  = try_get_pretrained(teacher, student, scratch=False)
        if is_test:
            train(sse_loader, sse_val_loader, teacher, student, epochs=1)
        else:
            train(sse_loader, sse_val_loader, teacher, student, epochs=300)
            break

# low profile 0.762