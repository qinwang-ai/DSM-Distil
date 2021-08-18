import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import configs.config_sse as config


class Generator(nn.Module):

    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.lstm_num_layers

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
        self.hidden2label2 = nn.Linear(self.cnn_lstm_len // 2, config.profile_width)

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

    def forward(self, sequence, lstm_feats):
        # stretch
        _sequence = torch.transpose(sequence, 0, 1)
        lstm_feats = torch.transpose(lstm_feats, 0, 1)

        embed = self.embed(_sequence)  # 44 x 16 x 300

        input = torch.cat([embed, lstm_feats], dim=2)  # [44 x 16 x 321]

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
        _x = self.hidden2label1(cnn_lstm)
        _x = F.relu(_x)
        x = self.hidden2label2(_x)
        # out = x.view(profile.shape[0], -1, config.num_labels)  # 16 x 44 x 3
        out = x
        return out
