import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import random
import configs.config_sse as config


class SSENet(nn.Module):

    def __init__(self, input_dim, label=config.num_labels):
        super(SSENet, self).__init__()
        self.hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.lstm_num_layers
        self.labels = label
        V = len(config.IUPAC_VOCAB)
        self.embed = nn.Embedding(V, config.embed_dim, padding_idx=-1)

        # BiLSTM
        self.bilstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=config.dropout,
                              bidirectional=True, bias=True)

        # linear
        self.cnn_lstm_len = config.lstm_hidden_dim * 2  # + config.kernel_num * 3
        self.hidden2label1 = nn.Linear(self.cnn_lstm_len, self.cnn_lstm_len // 2)
        self.hidden2label2 = nn.Linear(self.cnn_lstm_len // 2, self.labels)

        # dropout
        self.dropout = nn.Dropout(config.dropout)

    def init_weights(self):
        init.xavier_normal_(self.bilstm.all_weights[0][0], gain=np.sqrt(2))
        init.xavier_normal_(self.bilstm.all_weights[0][1], gain=np.sqrt(2))
        init.xavier_uniform_(self.hidden2label1.weight, gain=1)
        init.xavier_uniform_(self.hidden2label2.weight, gain=1)
        init.xavier_uniform_(self.embed.weight.data, gain=1)

    def forward(self, sequence, profile):
        _sequence = torch.transpose(sequence, 0, 1)
        _profile = torch.transpose(profile, 0, 1)

        embed = self.embed(_sequence)  # 44 x 16 x 300

        input = torch.cat([embed, _profile], dim=2)  # [44 x 16 x 321]

        # BiLSTM
        bilstm_out, _ = self.bilstm(input)  # 44 x 16 x 600
        bilstm_out = torch.transpose(bilstm_out, 0, 1)  # 16 x 44 x 600

        # linear
        _x = self.hidden2label1(F.relu(bilstm_out))
        _x = self.dropout(F.relu(_x))
        x = self.hidden2label2(_x)
        out = x
        return out


class StretchNet(nn.Module):
    def __init__(self):
        super(StretchNet, self).__init__()
        self.bilstm = nn.LSTM(20, 128, num_layers=1, dropout=config.dropout,
                              bidirectional=True, bias=True)
        self.fc = nn.Linear(256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, profile, MSA_C=None):
        p, _ = self.bilstm(profile.transpose(0, 1))
        p = F.relu(p).transpose(0, 1)
        p = self.bn1(p.transpose(1, 2)).transpose(1, 2)
        p = self.fc(p)
        p = self.bn2(p.transpose(1, 2)).transpose(1, 2)
        N = F.sigmoid(p)

        profile = (1 - profile ** N) / N
        return profile, N

    def init_weights(self):
        init.xavier_uniform_(self.fc.weight, gain=1)
        init.xavier_normal_(self.bilstm.all_weights[0][0], gain=np.sqrt(2))


class SSENetCNN(nn.Module):

    def __init__(self, input_dim, label=config.num_labels):
        super(SSENetCNN, self).__init__()
        self.hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.lstm_num_layers * 2

        V = len(config.IUPAC_VOCAB)
        self.embed = nn.Embedding(V, config.embed_dim, padding_idx=-1)

        # BiLSTM
        self.bilstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=config.dropout,
                              bidirectional=True, bias=True)

        # CNN
        self.convs = nn.Sequential(nn.Conv1d(input_dim, config.kernel_num, 1, stride=1), nn.ReLU(),
                                   nn.Conv1d(config.kernel_num, config.kernel_num, 3, padding=1, stride=1), nn.ReLU(),
                                   nn.Conv1d(config.kernel_num, config.kernel_num, 3, padding=1, stride=1), nn.ReLU(),
                                   nn.Conv1d(config.kernel_num, config.kernel_num, 5, padding=2, stride=1), nn.ReLU(),
                                   )

        # linear
        self.cnn_lstm_len = config.lstm_hidden_dim * 2 + config.kernel_num
        self.hidden2label1 = nn.Linear(self.cnn_lstm_len, self.cnn_lstm_len // 2)
        self.hidden2label2 = nn.Linear(self.cnn_lstm_len // 2, label)

        # dropout
        self.dropout = nn.Dropout(config.dropout)


    def init_weights(self):
        init.xavier_normal_(self.bilstm.all_weights[0][0], gain=np.sqrt(2))
        init.xavier_normal_(self.bilstm.all_weights[0][1], gain=np.sqrt(2))
        init.xavier_uniform_(self.hidden2label1.weight, gain=1)
        init.xavier_uniform_(self.hidden2label2.weight, gain=1)
        init.xavier_uniform_(self.embed.weight.data, gain=1)
        for conv in self.convs:
            if isinstance(conv, nn.Conv1d):
                init.xavier_uniform_(conv.weight.data, gain=1)

    def forward(self, sequence, profile):

        # stretch
        _sequence = torch.transpose(sequence, 0, 1)
        _profile = torch.transpose(profile, 0, 1)

        embed = self.embed(_sequence)  # 44 x 16 x 300

        input = torch.cat([embed, _profile], dim=2)  # [44 x 16 x 321]

        # CNN
        cnn_x = torch.transpose(input, 0, 1).transpose(1, 2)  # [16 x 44 x 790]
        cnn_x = self.convs(cnn_x)

        # BiLSTM
        bilstm_out, _ = self.bilstm(input)  # 44 x 16 x 600
        bilstm_out = torch.transpose(bilstm_out, 0, 1).transpose(1, 2)  # 16 x 1200 x 44

        # CNN and BiLSTM CAT
        cnn_lstm = torch.cat([cnn_x, bilstm_out], dim=1)  # 16 x 1200 x 44
        cnn_lstm = torch.transpose(cnn_lstm, 1, 2)  # 16 x 44 x 1200

        # linear
        _x = self.hidden2label1(F.relu(cnn_lstm))
        _x = self.dropout(F.relu(_x))
        x = self.hidden2label2(_x)
        out = x
        return out
