#!/usr/bin/env python

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__credits__ = "Pix2pix developer team"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
Implementation of X-Bridge method.
"""

import torch
import itertools
from .base_model import BaseModel
from . import networks


class XBridgeModel(BaseModel):
    """ This class implements the X-Bridge model, for learning a mapping from input images to output images given paired data.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The training objective is: GAN translation loss + L1 translation loss + GAN reconstruction loss + L1 reconstruction loss
        By default, I use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm='batch', netG='XBridge', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_G2', type=float, default = 0.1, help='weight for reconstruction loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G1_GAN', 'G2_GAN', 'G2_L1', 'G1_L1', 'D1_real', 'D1_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['E', 'DEC1', 'DEC2', 'D1', 'D2']
        else:  # during test time, only load G (composed of E, DEC1, and DEC2)
            self.model_names = ['E', 'DEC1', 'DEC2']
        # define networks (both generators (composed of encoder and decoders) and discriminators)
        self.netE = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netDEC1 = networks.define_Dec(opt.output_nc, opt.ngf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netDEC2 = networks.define_Dec(opt.output_nc, opt.ngf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define both discriminators; conditional GANs need to take both input and output images; Therefore, #channels for D1 is input_nc + output_nc
            self.netD1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G1 = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netDEC1.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999)) # translation path
            self.optimizer_G2 = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netDEC2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999)) # reconstruction path
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(), self.netD2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_A = self.netDEC2(self.netE(self.real_A))  # A->A = reconstruction
        self.fake_B = self.netDEC1(self.netE(self.real_A))  # A->B = image translation

    def backward_D1(self):
        """Calculate GAN loss for the translation discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD1(fake_AB.detach())
        self.loss_D1_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD1(real_AB)
        self.loss_D1_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5
        self.loss_D1.backward()

    def backward_D2(self):
        """Calculate GAN loss for the reconstruction discriminator"""
        # Fake
        pred_fake = self.netD2(self.fake_A.detach())
        self.loss_D2_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD2(self.real_A)
        self.loss_D2_real = self.criterionGAN(pred_real, True)
        loss_D2 = (self.loss_D2_real + self.loss_D2_fake) * 0.5
        loss_D2.backward()
        return loss_D2


    def backward_G1(self):
        """Calculate GAN and L1 loss for the translation generator"""
        # First, G(A) should fake the discriminator D1
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD1(fake_AB)
        self.loss_G1_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G1_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G1 = self.loss_G1_GAN + self.loss_G1_L1
        self.loss_G1.backward()

    def backward_G2(self):
        """Calculate GAN and L1 loss for the reconstruction generator"""
        # First, G(A) should fake the discriminator D2
        pred_fake = self.netD2(self.fake_A)
        self.loss_G2_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G2_L1 = self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G2 = (self.loss_G2_GAN + self.loss_G2_L1) * self.opt.lambda_G2
        self.loss_G2.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D1+D2
        self.set_requires_grad([self.netD1, self.netD2], True)  # enable backprop for Ds
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D1()               # calculate gratients for D1
        self.backward_D2()               # calculate gradients for D2
        self.optimizer_D.step()          # update Ds's weights

        # update G1
        self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G
        self.optimizer_G1.zero_grad()        # set G1's gradients to zero
        self.backward_G1()                   # calculate gradients for G1
        self.optimizer_G1.step()             # udpate G1's weights

        #update G2
        self.set_requires_grad([self.netD1, self.netD2], False)  # D requires no gradients when optimizing G
        self.optimizer_G2.zero_grad()        # set G2's gradients to zero
        self.backward_G2()                   # calculate gradients for G2
        self.optimizer_G2.step()             # udpate G2's weights