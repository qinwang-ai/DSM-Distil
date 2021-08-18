from models.SSENet import SSENet, StretchNet
import torch.nn as nn
import configs.config_sse as config

class Teacher(nn.Module):

    def __init__(self, input_dim, label=config.num_labels):
        super(Teacher, self).__init__()
        self.ssenet = SSENet(input_dim, label)
        self.stretchNet = StretchNet()

    def init_weights(self):
        self.ssenet.init_weights()
        self.stretchNet.init_weights()

    def forward(self, sequence, profile):
        profile, _ = self.stretchNet(profile)
        return self.ssenet(sequence, profile), profile

