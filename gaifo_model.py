import torch
from torch import nn

import nets
import utils


class Discriminator(nn.Module):
    def __init__(self, nb_frames, lr):
        super(Discriminator, self).__init__()
        self.discriminator_net = nets.ConvNet(nb_frames)
        self.optimizer = torch.optim.Adam(self.discriminator_net.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.discriminator_net(x)

    def evaluate(self, e_x, a_x):
        batch_size = e_x.shape[0]
        e_p = self.discriminator_net(e_x)
        a_p = self.discriminator_net(a_x)
        p = torch.cat([e_p, a_p], dim=0).view((-1,))
        target = torch.zeros((batch_size * 2), device=utils.device())
        target[batch_size:] = 1.
        loss = self.criterion(p, target)
        metrics = {
            'Discriminator loss': loss.item()
        }
        return metrics, loss

    def update(self, e_x, a_x):
        self.train()
        self.optimizer.zero_grad()

        metrics, loss = self.evaluate(e_x, a_x)
        loss.backward()

        self.optimizer.step()
        self.eval()

        return metrics





