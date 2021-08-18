import torch
import configs.config_sse as config_sse
import torch.nn.init as init
import torch.nn as nn
import numpy as np

# This code is totally shit!


class Config(object):
	def __init__(self):
		super(Config, self).__init__()
		self.feature_size = 52
		self.pssm_dim = 20
		self.batch_size = 32
		self.cnn_layer_num = 3
		self.cnn_window_size = 3
		self.bot_window_size = 3
		self.node_size = 109
		self.dropout_rate = 0
		self.blstm_layer_num = 2
		self.lstm_hidden_size = 512
		self.mid_w_size = self.cnn_window_size
		self.lr = 0.0001
		self.device = 'cuda'
		self.window_sizes = [self.mid_w_size] * int(self.cnn_layer_num - 2)
		self.top_window_size = self.mid_w_size
		self.output_dropout_rate = 0
		self.lstm_dropout_rate = 0
		self.fc1_dim = self.pssm_dim
		self.fc1_dropout_rate = 0


class BaggingNet(nn.Module):
	def __init__(self):
		super(BaggingNet, self).__init__()
		config = Config()

		self.conv_bot = nn.Sequential(nn.Conv1d(in_channels=config.feature_size,
		                                        out_channels=config.node_size,
		                                        kernel_size=config.bot_window_size,
		                                        padding=int((config.bot_window_size - 1) / 2)),
		                              #   nn.BatchNorm1d(num_features=config.node_size),
		                              nn.Dropout(config.dropout_rate),
		                              nn.ReLU()
		                              )
		self.convs = nn.ModuleList([
			nn.Sequential(nn.Conv1d(in_channels=config.node_size,
			                        out_channels=config.node_size,
			                        kernel_size=h,
			                        padding=int((h - 1) / 2)),
			              #  nn.BatchNorm1d(num_features=config.node_size),
			              nn.Dropout(config.dropout_rate),
			              nn.ReLU()
			              )
			for h in config.window_sizes
		])
		self.conv_top = nn.Sequential(nn.Conv1d(in_channels=config.node_size,
		                                        out_channels=config.node_size,
		                                        kernel_size=config.top_window_size,
		                                        padding=int((config.top_window_size - 1) / 2)),
		                              #   nn.BatchNorm1d(num_features=config.node_size),
		                              nn.Dropout(config.output_dropout_rate),
		                              nn.ReLU()
		                              )

		self.bilstm = nn.LSTM(
			input_size=config.feature_size,
			hidden_size=config.lstm_hidden_size,
			num_layers=config.blstm_layer_num,
			batch_first=True,
			dropout=config.lstm_dropout_rate,
			bidirectional=True
		)

		self.fc1 = nn.Sequential(
			# nn.Sigmoid(),
			# nn.Dropout(config.fc1_dropout_rate),
			nn.Linear(config.node_size +
			          config.lstm_hidden_size * 2, config.fc1_dim),
			nn.Dropout(config.fc1_dropout_rate),
			nn.ReLU()
		)
		V = len(config_sse.IUPAC_VOCAB)
		self.embed = nn.Embedding(V, config_sse.embed_dim, padding_idx=0)

		self.device = config.device
		self.lstm_hidden_size = config.lstm_hidden_size
		self.feature_size = config.feature_size
		self.batch_size = config.batch_size
		self.blstm_layer_num = config.blstm_layer_num

	def init_hidden(self, batch_size):
		return (torch.zeros(self.blstm_layer_num * 2, batch_size, self.lstm_hidden_size).to(self.device),
		        torch.zeros(self.blstm_layer_num * 2, batch_size, self.lstm_hidden_size).to(self.device))

	def forward(self, seq, pssm):
		embed = self.embed(seq)  # 44 x 16 x 300
		x = torch.cat([embed, pssm], dim=-1)
		x = torch.transpose(x, 2,1)
		# x.size() = [32,42,700]
		cnnout = self.conv_bot(x)  # [32,100,700]
		for conv in self.convs:
			cnnout = conv(cnnout)
		cnnout = self.conv_top(cnnout)  # [32,100,700]
		cnnout = cnnout.permute(0, 2, 1)  # [32,700,100]

		x = x.permute(0, 2, 1)  # [32,700,42]
		hidden_states = self.init_hidden(x.shape[0])
		bilstm_out, hidden_states = self.bilstm(
			x, hidden_states)  # bilstm_out=[32,700,1024]

		out = torch.cat((cnnout, bilstm_out), 2)
		out = self.fc1(out)

		return out

	def init_weights(self):
		init.xavier_normal_(self.bilstm.all_weights[0][0], gain=np.sqrt(2))
		init.xavier_normal_(self.bilstm.all_weights[0][1], gain=np.sqrt(2))
		init.xavier_uniform_(self.embed.weight.data, gain=1)
		# for conv in self.convs:
		# 	init.xavier_uniform_(conv.weight.data, gain=1)
