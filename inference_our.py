# coding=utf-8
import os
import torch
from torch.utils.data import DataLoader
from DataLoader.SSEDatasetATT import SSEDataset
from models.SSENet import SSENet
from models.GradeNet import SSENet as GradeNet
from models.Generator import Generator
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
    pred_low_real, _ = ssenet_lowreal(sequence, low_profile)  # 16 x 44 x 3
    pred_fake, _ = ssenet_fake(sequence, fake_profile)  # 16 x 44 x 3
    return pred_low_real.detach(), pred_fake.detach()

def try_get_pretrained(teacher, student, generator, gradenet, scratch=False):
    teacher_path = '../module/bio_teacher.pth'
    student_path = '../module/bioinfo_student.pth'
    generator_path = '../module/bioinfo_generator.pth'
    gradenet_path = '../module/bioinfo_gradenet.pth'

    teacher.init_weights()
    student.init_weights()
    generator.init_weights()
    gradenet.init_weights()

    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path))
        print('load teacher')

    if not scratch:
        if os.path.exists(student_path):
            student.load_state_dict(torch.load(student_path))
            print('load bioinfo_student', student_path)

        if os.path.exists(generator_path):
            generator.load_state_dict(torch.load(generator_path))
            print('load bioinfo_generator', generator_path)

        if os.path.exists(gradenet_path):
            gradenet.load_state_dict(torch.load(gradenet_path))
            print('load bioinfo_gradenet', gradenet_path)

    return teacher.cuda(), student.cuda(), generator.cuda(), gradenet.cuda()

def save_model():
    print('update model..')
    student_path = '../module/bioinfo_student.pth'
    generator_path = '../module/bioinfo_generator.pth'
    gradenet_path = '../module/bioinfo_gradenet.pth'

    torch.save(student.state_dict(),  student_path)
    torch.save(generator.state_dict(), generator_path)
    torch.save(gradenet.state_dict(), gradenet_path)

def parse_batch(batch):
    filename = batch['filename']
    sequence = torch.tensor(batch['sequence'], dtype=torch.int64).cuda()
    label = torch.tensor(batch['label'], dtype=torch.int64).cuda()

    att_gt = torch.tensor(batch['att_gt'], dtype=torch.float32).cuda()
    real_profile = torch.tensor(batch['real_profile'], dtype=torch.float32).cuda()
    fake_profile = torch.tensor(batch['fake_profile'], dtype=torch.float32).cuda()
    low_profile = torch.tensor(batch['low_profile'], dtype=torch.float32).cuda()
    return filename, sequence, low_profile, real_profile, fake_profile, label, att_gt

def remove_pad_keep_dim(data, sequence):
    ans = []
    for i in range(data.shape[0]):
        ans.append(data[i, sequence[i] !=-1, :])
    return ans

def get_ce_loss(sequence, profile, label, net):
    pred, feat = net(sequence, profile)  # 16 x 44 x 3
    pred_no_pad = pred[label != -1, :]
    label_no_pad = label[label != -1]
    errCE = ce_loss_func(pred_no_pad, label_no_pad)
    pred_label = torch.argmax(pred_no_pad, dim=-1)
    acc = (pred_label == label_no_pad).sum().float() / pred_label.shape[0]
    return errCE, acc, pred_no_pad, pred_label

def normalize_profile(profile):
    profile = F.softmax(profile, dim=-1)
    return profile


def get_mxe_loss(profile, profile_gt, label):
    profile = normalize_profile(profile)
    profile = profile[label != -1, :]
    profile_gt = profile_gt[label != -1, :]
    mxe_loss = F.mse_loss(torch.log(profile+0.01), torch.log(profile_gt+0.01))
    return mxe_loss


def get_grade_loss(sequence, feats, att_gt, gradenet, label):
    # since sum is 2 so need divide 2
    vec, _ = gradenet(sequence, feats) # 16 x 44 x 3
    vec_nopad, att_gt_nopad = vec[label != -1, :], att_gt[label != -1, :]
    # err = ce_loss_func(vec_nopad, att_gt_nopad)
    err = F.mse_loss(torch.log(0.01+vec_nopad), torch.log(0.01+att_gt_nopad))

    att_gt_label = torch.argmax(att_gt_nopad, dim=-1)
    acc = (torch.argmax(vec_nopad, dim=-1) == att_gt_label).sum()/float(att_gt_label.shape[0])
    # vec = vec.softmax(dim=-1)
    u, v = vec[:, :, 0].unsqueeze(dim=-1), vec[:, :, 1].unsqueeze(dim=-1)
    return err, (u,v), acc


def train(sse_loader, val_loader, generator, teacher, student, gradenet, epochs=config.epochs):

    optimizer_stu = torch.optim.Adam(student.parameters(), lr=config.gen_lr, weight_decay=1e-8)
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=config.gen_lr, weight_decay=1e-8)
    optimizer_grade = torch.optim.Adam(gradenet.parameters(), lr=config.gen_lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=20, gamma=0.1)

    for epoch in range(1, epochs + 1):
        # validation
        student.eval()
        generator.eval()
        gradenet.eval()
        _summary = []
        f = open('./logs/our_psm_acc.txt', 'w')
        for batch in tqdm(val_loader):
            filename, sequence, low_profile, real_profile, fake_profile, label, att_gt = parse_batch(batch)

            # profile transform
            low_feats, fake_feats = get_raw_features(sequence, real_profile, fake_profile)
            feats = torch.cat([low_feats, fake_feats], dim=-1)
            # feats = feats.softmax(dim=-1)
            errGrade, (u, v), grade_acc = get_grade_loss(sequence, feats, att_gt, gradenet, label)

            profile = u * low_profile + v * fake_profile

            enhance_pro = generator(sequence, profile)

            # inference #
            _enhance_pro = enhance_pro.clone()
            _enhance_pro[_enhance_pro<_enhance_pro.mean()] = _enhance_pro.min()
            _enhance_pro = _enhance_pro.softmax(dim=-1)

            np.save('/data/proli/raw_data/visualization/enhanced_pssm/'+filename+'.npy', _enhance_pro.squeeze().cpu().detach().numpy())
            print(_enhance_pro.shape, 'saved..')
            # inference #


            errCE_stu, acc, x_stu, pred_label = get_ce_loss(sequence, enhance_pro, label, student)

            np.save('/data/proli/raw_data/visualization/enhanced_ss/'+filename+'.ss', pred_label.squeeze().cpu().detach().numpy())

            f.write('%s %f\n' % (filename, acc.item()))
            _summary.append((errCE_stu.item(), grade_acc.item(), acc.item()))

        # statistic
        summary = np.array(_summary).mean(axis=0)
        _std = np.array(_summary).std(axis=0)
        global prev_best
        if summary[-1] > prev_best:
            prev_best = summary[-1]
            if epochs != 1:
                save_model()
        print('[EVAL]', 'CE_loss: %0.3f, grade_acc: %0.3f, curr_sse_acc: %0.3f, best_acc:%0.3f, std: %0.3f' % (
            summary[0], summary[1], summary[2], prev_best, _std[-1]))



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

        # sse_val_dataset = SSEDataset(config_tmp.profile_fake_data_path.replace('train', 'valid'),
        #                              config_tmp.sequence_data_path_prefix,
        #                              config_tmp.blosum_path_prefix.replace('train', 'valid'),
        #                              config_tmp.sp_att_data_path,
        #                              config_tmp.real_profile_data_prefix.replace('train', 'valid'),
        #                              config_tmp.label_data_path_prefix.replace('train', 'valid'),
        #                              mode='low', config=config_tmp, enable_downsample=False)

        sse_val_dataset = SSEDataset('/data/proli/raw_data/visualization/low_pssm/*.npy',
                                     config_tmp.sequence_data_path_prefix,
                                     config_tmp.blosum_path_prefix.replace('train', 'valid'),
                                     config_tmp.sp_att_data_path,
                                     config_tmp.real_profile_data_prefix.replace('train', 'valid'),
                                     config_tmp.label_data_path_prefix.replace('train', 'valid'),
                                     mode='all', config=config_tmp, enable_downsample=False)

        if not is_test:
            sse_loader = DataLoader(sse_dataset, batch_size=config.batch_size, num_workers=config.batch_size,
                                    collate_fn=sse_dataset.collate_fn, shuffle=True)
        sse_val_loader = DataLoader(sse_val_dataset, batch_size=1, num_workers=config.batch_size,
                                    collate_fn=sse_dataset.collate_fn)

        ce_loss_func = nn.CrossEntropyLoss()
        kl_func = nn.KLDivLoss()
        teacher = SSENet(input_dim=config.embed_dim + config.profile_width)
        student = SSENet(input_dim=config.embed_dim + config.profile_width)
        gradenet = GradeNet(input_dim=config.num_labels*2)
        generator = Generator(input_dim=config.embed_dim + config.profile_width)

        ssenet_lowreal = SSENet(config.embed_dim + config.profile_width)
        ssenet_fake = SSENet(config.embed_dim + config.profile_width)
        # get raw net
        ssenet_lowreal, ssenet_fake = try_get_pretrained_raw(ssenet_lowreal, ssenet_fake)

        # try load pretrained model
        teacher, student, generator, gradenet = try_get_pretrained(teacher, student, generator, gradenet, scratch=False)
        if is_test:
            train(sse_loader, sse_val_loader, generator, teacher, student, gradenet, epochs=1)
        else:
            train(sse_loader, sse_val_loader, generator, teacher, student, gradenet, epochs=300)
            break

