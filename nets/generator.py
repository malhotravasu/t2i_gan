import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, batch_size, img_size, z_dim, text_embed_dim):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.z_dim = z_dim
        self.text_embed_dim = text_embed_dim

        self.concat = nn.Linear(z_dim + text_embed_dim, 64 * 8 * 4 * 4)

        # Defining the generator network architecture
        self.g_net = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, text_embed, z):
        
        concat = torch.cat((text_embed, z), 2) # (bs, 1, 228)
        
        concat = concat.permute(0, 2, 1)

        concat = self.concat(concat.squeeze(2))

        concat = concat.view(1, 512, 4, 4)

        d_net_out = self.g_net(concat)  # (bs, 3, 64, 64)
        output = d_net_out / 2. + 0.5

        return output
