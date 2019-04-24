import os
import time
import datetime
import logging
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

import numpy as np
from nets.discriminator import Discriminator
from nets.generator import Generator

class GAN_CLS(object):
    def __init__(self, data_loader, SUPERVISED=True):

        self.data_loader = data_loader
        self.num_epochs = 1
        self.batch_size = 8

        self.log_step = 100
        self.sample_step = 100

        self.log_dir = 'logs'
        self.checkpoint_dir = 'checkpoints'
        self.sample_dir = 'sample'
        self.final_model = 'final_model'

        self.dataset = 'flowers/'
        self.model_name = 'GANText2Pic'
        self.img_size = 64
        self.z_dim = 100
        self.text_embed_dim = 128
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.l1_coeff = 50
        self.resume_epoch = 0
        self.SUPERVISED = SUPERVISED
        self.model_save_step = 10
        # Logger setting
        self.logger = logging.getLogger('__name__')
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        self.file_handler = logging.FileHandler(self.log_dir)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

        self.build_model()

    def build_model(self):
        """ A function of defining following instances :

        -----  Generator
        -----  Discriminator
        -----  Optimizer for Generator
        -----  Optimizer for Discriminator
        -----  Defining Loss functions

        """
        self.gen = Generator(self.batch_size,
                             self.img_size,
                             self.z_dim,
                             self.text_embed_dim)

        self.disc = Discriminator(self.batch_size,
                                  self.img_size,
                                  self.text_embed_dim)

        self.gen_optim = optim.Adam(self.gen.parameters(),
                                    lr=self.learning_rate,
                                    betas=(self.beta1, self.beta2))

        self.disc_optim = optim.Adam(self.disc.parameters(),
                                     lr=self.learning_rate,
                                     betas=(self.beta1, self.beta2))

        self.cls_gan_optim = optim.Adam(itertools.chain(self.gen.parameters(),
                                                        self.disc.parameters()),
                                        lr=self.learning_rate,
                                        betas=(self.beta1, self.beta2))

        print ('\n-------------  Generator Model Info  ---------------')
        self.print_network(self.gen, 'G')
        print ('------------------------------------------------\n')

        print ('\n-------------  Discriminator Model Info  ---------------')
        self.print_network(self.disc, 'D')
        print ('------------------------------------------------\n')

        self.gen.cuda()
        self.disc.cuda()
        self.criterion = nn.BCELoss().cuda()
        # self.CE_loss = nn.CrossEntropyLoss().cuda()
        # self.MSE_loss = nn.MSELoss().cuda()
        self.gen.train()
        self.disc.train()

    def print_network(self, model, name):
        """ A function for printing total number of model parameters """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        print(model)
        print(name)
        print("\nTotal number of parameters: {}".format(num_params))

    def load_checkpoints(self, resume_epoch):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_epoch))
        G_path = os.path.join(self.checkpoint_dir, '{}-G.ckpt'.format(resume_epoch))
        D_path = os.path.join(self.checkpoint_dir, '{}-D.ckpt'.format(resume_epoch))
        self.gen.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.disc.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def train_model(self):

        data_loader = self.data_loader

        print ('---------------  Model Training Started  ---------------')

        for epoch in range(self.num_epochs):
            for idx, batch in enumerate(data_loader):

                true_imgs = batch['true_imgs']

                true_embed = batch['true_embed']
                
                false_imgs = batch['false_imgs']

                real_labels = torch.ones(true_imgs.size(0))
                fake_labels = torch.zeros(true_imgs.size(0))

                true_imgs = Variable(true_imgs.float()).cuda()
                true_embed = Variable(true_embed.float()).cuda()
                false_imgs = Variable(false_imgs.float()).cuda()

                real_labels = Variable(real_labels).cuda()
                smooth_real_labels = Variable(real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # ---------------------------------------------------------------
                # 					  2. Training the generator
                # ---------------------------------------------------------------
                self.gen.zero_grad()
                z = Variable(torch.randn(true_imgs.size(0), self.z_dim)).unsqueeze(0).cuda()
                fake_imgs = self.gen(true_embed, z)
                fake_out = self.disc(fake_imgs, true_embed)
                true_out = self.disc(true_imgs, true_embed)

                # gen_loss = (self.criterion(fake_out, real_labels) +
                #            self.l1_coeff*nn.L1Loss(fake_imgs, true_imgs))

                gen_loss = self.criterion(fake_out, real_labels)

                gen_loss.backward(retain_graph=True)
                self.gen_optim.step()

                # ---------------------------------------------------------------
                # 					3. Training the discriminator
                # ---------------------------------------------------------------
                self.disc.zero_grad()
                false_out = self.disc(false_imgs, true_embed)
                disc_loss = (self.criterion(true_out, smooth_real_labels) +
                            self.criterion(fake_out, fake_labels) + 
                            self.criterion(false_out, fake_labels))

                disc_loss.backward(retain_graph=True)
                self.disc_optim.step()

                # self.cls_gan_optim.step()

                # Logging
                loss = {}
                loss['G_loss'] = gen_loss.item()
                loss['D_loss'] = disc_loss.item()
                if (idx % 100 == 0):
                    print("Batch No.:", idx//100, " | G Loss:", loss['G_loss'], ", D Loss:", loss['D_loss'])

        print ('---------------  Model Training Completed  ---------------')

        # Saving final model into final_model directory
        G_path = os.path.join(self.final_model, "-G")
        D_path = os.path.join(self.final_model, "-D")

        torch.save(self.gen.state_dict(), G_path+'.pt')
        torch.save(self.disc.state_dict(), D_path+'.pt')

        torch.save(self.gen, G_path+'-complete.pt')
        torch.save(self.disc, D_path+'-complete.pt')

