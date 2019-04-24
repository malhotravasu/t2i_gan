import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self, batch_size, img_size, text_embed_dim):
        super(Discriminator, self).__init__()

        self.batch_size = batch_size
        self.img_size = img_size
        self.in_channels = 3
        self.text_embed_dim = text_embed_dim

        # Defining the discriminator network architecture
        self.d_net = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))

        # output_dim = (batch_size, 4, 4, 512)
        # text.size() = (batch_size, text_embed_dim)

        # Defining a linear layer to reduce the dimensionality of caption embedding
        # from text_embed_dim to text_reduced_dim
        #self.text_reduced_dim = nn.Linear(self.text_embed_dim, self.text_reduced_dim)

        self.cat_net = nn.Sequential(
            nn.Conv2d(512 + self.text_embed_dim, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))

        self.linear = nn.Linear(2 * 2 * 256, 1)

    def forward(self, img_batch, text_embed):
        
        d_net_out = self.d_net(img_batch)  # (bs, 512, 4, 4)
        
        text_embed = text_embed.permute(0, 2, 1)
        text_embed = text_embed.unsqueeze(3)
        text_embed = text_embed.expand(1, 128, 4, 4)
        
        concat_out = torch.cat((d_net_out, text_embed), 1)  # (bs, 512+128, 4, 4)

        concat_out = self.cat_net(concat_out)
        concat_out = concat_out.view(1, -1)
        concat_out = self.linear(concat_out)
        output = torch.sigmoid(concat_out)

        return output
