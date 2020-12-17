import torch
import torch.nn as nn
from pdb import set_trace as st

class Encoder(nn.Module):
    def __init__(self, z_dim, channel_dim, fixed_num_of_contact):
        super().__init__()

        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Linear(channel_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
        )
        self.out = nn.Linear(32 * fixed_num_of_contact, z_dim)

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)


class Transition(nn.Module):
    def __init__(self, z_dim,
                       action_dim=0,
                       trans_type = 'linear', hidden_size = 32):
        super().__init__()
        if trans_type == 'linear':
            self.out = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        elif trans_type == 'mlp':
            self.out = nn.Sequential(
                nn.Linear(z_dim + action_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, z_dim)
            )

    def forward(self, x, squash = False):
        if squash:
            x = nn.tanh(x)
        return self.out(x)


class Decoder(nn.Module):

    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim

        self.main = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, channel_dim),
            # nn.Tanh(),
        )

    def forward(self, z):
        x = z.view(-1, self.z_dim)
        output = self.main(x)
        return output
