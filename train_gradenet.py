# coding=utf-8
import os
import torch
from torch.utils.data import DataLoader
from DataLoader.SSEDatasetATT import SSEDataset
from models.GradeNet import SSENet as GradeNet
from tqdm import tqdm
from models.SSENet import SSENet
import configs.config_sse as config
import configs.config_sse_bc40 as config_bc_40
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def try_get_pretrained(gradenet, ssenet_lowreal, ssenet_fake, scratch=False):
    gradenet_path = '../module/ijcai_gradenet_sp.pth'
    ssenet_low_real_path = config.pretrain_path + 'ssenet_raw_lowprofile.pth'
    ssenet_fake_path = config.pretrain_path + 'ssenet_raw_fakeprofile.pth'

    gradenet.init_weights()

    if not scratch:
        if os.path.exists(gradenet_path):
            gradenet.load_state_dict(torch.load(gradenet_path))
            print('load ijcai_generator')

        ssenet_lowreal.load_state_dict(torch.load(ssenet_low_real_path))
        ssenet_fake.load_state_dict(torch.load(ssenet_fake_path))
        ssenet_lowreal.eval()
        ssenet_fake.eval()
    return gradenet.cuda(), ssenet_lowreal.cuda(), ssenet_fake.cuda()

def save_model():
    print('update model..')
    gradenet_path = '../module/ijcai_gradenet_sp.pth'
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


def get_grade_loss(sequence, feats, att_gt, gradenet, label):
    vec, _ = gradenet(sequence, feats) # 16 x 44 x 3
    vec_nopad, att_gt_nopad = vec[label != -1, :], att_gt[label != -1, :]
    # err = ce_loss_func(vec_nopad, att_gt_nopad)
    # err = F.l2_loss(vec_nopad, att_gt_nopad)
    err = F.mse_loss(vec_nopad, att_gt_nopad)
    att_gt_label = torch.argmax(att_gt_nopad, dim=-1)
    acc = (torch.argmax(vec_nopad, dim=-1) == att_gt_label).sum()/float(att_gt_label.shape[0])
    return err, vec, acc


def update_profile(fake_profile, profile_gt, high_quality):
    fake_profile[high_quality == 1, :, :] = profile_gt[high_quality == 1, :, :]
    return fake_profile

def get_raw_features(sequence, low_profile, fake_profile):
    pred_low_real, _ = ssenet_lowreal(sequence, low_profile)  # 16 x 44 x 3
    pred_fake, _ = ssenet_fake(sequence, fake_profile)  # 16 x 44 x 3
    return pred_low_real.detach(), pred_fake.detach()


prev_best = -100000

def train(sse_loader, val_loader, gradenet, epochs=300, is_test=False):
    optimizer_grade = torch.optim.Adam(gradenet.parameters(), lr=config.gen_lr, weight_decay=1e-8)

    for epoch in range(1, epochs + 1):

        if epochs != 1:
            # train
            gradenet.train()
            summary = []
            for batch in tqdm(sse_loader):
                filename, sequence, low_profile, real_profile, fake_profile, label, att_gt = parse_batch(batch)
                optimizer_grade.zero_grad()

                # profile transform
                low_profile, fake_profile = get_raw_features(sequence, low_profile, fake_profile)

                # GradeNet
                profile = torch.cat([low_profile, fake_profile], dim=-1)
                errGrade, u, acc = get_grade_loss(sequence, profile, att_gt, gradenet, label)

                profile0 = torch.cat([fake_profile, low_profile], dim=-1)
                errGrade0, u, acc = get_grade_loss(sequence, profile0, 1-att_gt, gradenet, label)

                (errGrade+errGrade0).backward()
                optimizer_grade.step()
                summary.append((errGrade.item(), acc.item()))

            # statistic
            summary = np.array(summary).mean(axis=0)
            print('Epoch %d' % epoch,
                  'errGrade: %0.7f, acc: %0.3f' % (summary[0], summary[1]))

        # validation
        gradenet.eval()
        _summary = []
        for batch in tqdm(val_loader):
            filename, sequence, low_profile, real_profile, fake_profile, label, att_gt = parse_batch(batch)

            # profile transform
            low_profile, fake_profile = get_raw_features(sequence, real_profile, fake_profile)

            # GradeNet
            profile = torch.cat([low_profile, fake_profile], dim=-1)
            errGrade, u, acc = get_grade_loss(sequence, profile, att_gt, gradenet, label)
            _summary.append((errGrade.item(), acc.item()))

        # statistic
        summary = np.array(_summary).mean(axis=0)
        global prev_best
        if summary[1] > prev_best and not is_test:
            prev_best = summary[1]
            if epochs != 1:
                save_model()
        print('[EVAL]', 'Grade_loss: %0.7f, acc: %0.3f, prev_best: %0.3f' % (summary[0], summary[1], prev_best))


if __name__ == '__main__':
    global ssenet_fake, ssenet_lowreal
    is_test = True
    sse_loader = None

    for i in range(1):
        sse_dataset = SSEDataset(config.profile_fake_data_path,
                                 config.sequence_data_path_prefix,
                                 config.blosum_path_prefix,
                                 config.sp_att_data_path,
                                 config.label_data_path_prefix, mode='high')
        sse_val_dataset = SSEDataset(config_bc_40.profile_fake_data_path.replace('train', 'valid'),
                                     config_bc_40.sequence_data_path_prefix,
                                     config_bc_40.blosum_path_prefix.replace('train', 'valid'),
                                     config_bc_40.sp_att_data_path,
                                     config_bc_40.label_data_path_prefix.replace('train', 'valid'),
                                     mode='low', config=config_bc_40)
        if not is_test:
            sse_loader = DataLoader(sse_dataset, batch_size=config.batch_size, num_workers=config.batch_size,
                                    collate_fn=sse_dataset.collate_fn, shuffle=True)
        sse_val_loader = DataLoader(sse_val_dataset, batch_size=1, num_workers=config.batch_size,
                                    collate_fn=sse_dataset.collate_fn)

        ce_loss_func = nn.CrossEntropyLoss()
        kl_func = nn.KLDivLoss()

        gradenet = GradeNet(input_dim=config.num_labels*2)
        ssenet_lowreal = SSENet(config.embed_dim + config.profile_width)
        ssenet_fake = SSENet(config.embed_dim + config.profile_width)

        # try load pretrained model
        gradenet, ssenet_lowreal, ssenet_fake = try_get_pretrained(gradenet, ssenet_lowreal, ssenet_fake, scratch=False)

        if is_test:
            train(sse_loader, sse_val_loader,gradenet, epochs=1, is_test=is_test)
        else:
            train(sse_loader, sse_val_loader,gradenet, is_test=is_test)
            break
