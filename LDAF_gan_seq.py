import torch
import torch.nn as nn

class discriminator(nn.Module):
    def __init__(self, diseaseCount, seq_shape):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(diseaseCount + seq_shape, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.l1 = nn.Linear(32765, 64)

    def forward(self, data, lnc_seq):
        # print("d")
        lnc_seq = lnc_seq.float()
        lnc_seq = self.l1(lnc_seq)
        data_c = torch.cat((data, lnc_seq), 1)
        result = self.dis(data_c)
        return result


class generator(nn.Module):
    def __init__(self, diseaseCount, seq_shape):
        self.diseaseCount = diseaseCount
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(self.diseaseCount + seq_shape, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, diseaseCount),
            nn.Sigmoid()
        )
        self.l1 = nn.Linear(32765, 64)

    def forward(self, noise, lnc_seq):
        lnc_seq = lnc_seq.float()
        lnc_seq = self.l1(lnc_seq)
        G_input = torch.cat([noise, lnc_seq], 1)
        result = self.gen(G_input)
        return result