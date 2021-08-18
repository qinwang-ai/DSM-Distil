from models.SSENet import SSENetCNN, StretchNet
import torch.nn as nn
import torch
import configs.config_sse as config

class Student(nn.Module):

    def __init__(self, input_dim, label=config.num_labels):
        super(Student, self).__init__()
        self.ssenet = SSENetCNN(input_dim, label)

    def init_weights(self):
        self.ssenet.init_weights()

    def forward(self, sequence, profile, feats):
        # profile = torch.cat([profile, feats], dim=-1)
        return self.ssenet(sequence, profile), profile

