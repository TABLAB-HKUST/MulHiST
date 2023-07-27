from cProfile import label
from http.cookies import Morsel
from model import Generator, StyleGenerator
from model import Discriminator, MsImageDis
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

from torchvision import models


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""
        self.opt = config
        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
 
        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.model_name +'/logs'
        self.sample_dir = config.model_name + '/samples-' + config.model_name
        self.model_save_dir = config.model_name +'/models'
        self.result_dir = config.model_name +'/results-'+ str(config.test_iters)

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        
        if self.opt.netG == 'adain':
            self.G = StyleGenerator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        if self.opt.netG == 'basic':
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)

        if self.opt.netD == 'basic':
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        if self.opt.netD == 'm_scale':
                self.D = MsImageDis(self.opt)
      
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        # self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):  #torch.Size([16, 1, 4, 4]) torch.Size([16, 3, 256, 256])
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)   # torch.Size([16, 1, 4, 4])
        dydx = torch.autograd.grad(outputs=y,           # torch.Size([16, 196608])
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
    
    def classification_loss_(self, input, target, starget):
        logsoft = torch.nn.LogSoftmax(dim=1)  
        s = logsoft(input)    # after softmax torch.Size([8, num_cls, 15, 15])    target:torch.Size([8, num_cls, 1, 1]) 
        return F.binary_cross_entropy_with_logits(torch.mean(s,dim=[3,2]), starget)

    def train(self):
        # Set data loader.
        data_loader = self.data_loader
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
  

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        fake_label_1 = torch.ones([self.opt.batch_size, 1], dtype=torch.float32).to(self.device)
        fake_label_2 = torch.zeros([self.opt.batch_size, self.opt.c_dim], dtype=torch.float32).to(self.device)
        fake_label = torch.cat([fake_label_1, fake_label_2], dim=1)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)  
            except:
                batch_iterator = iter(data_loader)
                x_real, label_org = next(batch_iterator)                    
        
            # x_real, label_org = next(data_iter)  
            y_real = x_real.to(self.device)     #1to2 have to change  y_real
            c_org = label_org.clone()

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            c_trg = label_trg.clone()


            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # Logging.          

            loss = {}
            if (i) % self.n_critic == 0:
                # print(x_real.shape)
                
                if self.opt.netD == 'basic':
                    # Compute loss with real images.
                    out_src, out_cls = self.D(y_real)  #x_real
                    d_loss_real = - torch.mean(out_src)   # print(out_cls.size(), label_org.size())  #torch.Size([16, 3]) torch.Size([16, 3])
                    d_loss_cls = self.classification_loss(out_cls, label_org)
                    
                    # Compute loss with fake images.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake.detach())
                    d_loss_fake = torch.mean(out_src)
                
                    # Compute loss for gradient penalty.
                    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)   #torch.Size([16, 1, 1, 1])  0~1
                    x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                    out_src, _ = self.D(x_hat)
                    d_loss_gp = self.gradient_penalty(out_src, x_hat)

                    # Backward and optimize.
                    d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                
                elif self.opt.netD == 'm_scale':
                    # Compute loss with real images.
                    outs0 = self.D(y_real)#x_real
                    m_org = torch.zeros([self.opt.batch_size, 1], dtype=torch.float32).to(self.device) 
                    m_org = torch.cat([m_org, label_org], dim=1)
                    d_loss_real = 0.
                    for out0 in outs0:
                        d_loss_real += self.classification_loss_(out0, m_org.unsqueeze(2).unsqueeze(3),m_org)
                    # Compute loss with fake images.
                    x_fake = self.G(x_real, c_trg)
                    outs1 = self.D(x_fake)
                    d_loss_fake = 0.
                    for out1 in outs1: 
                        d_loss_fake += self.classification_loss_(out1, fake_label.unsqueeze(2).unsqueeze(3),fake_label)
                    
                    # inserted on 20220901      
                    d_loss_fake /= len(outs1)
                    d_loss_real /= len(outs1)
                    # Backward and optimize.
                    d_loss = d_loss_real + d_loss_fake 
                
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                if not self.opt.netD == 'm_scale':
                    loss['D/loss_cls'] = d_loss_cls.item()
                    loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
                
            # Original-to-target domain.
            if self.opt.netD == 'basic':
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg) * self.lambda_cls
            
            elif self.opt.netD == 'm_scale':
                m_trg = torch.zeros([self.opt.batch_size, 1], dtype=torch.int32).to(self.device) 
                m_trg = torch.cat([m_trg, label_trg], dim=1)
                g_loss_fake = 0.
                
                x_fake = self.G(x_real, c_trg)
                outs1 = self.D(x_fake)
                for out0 in outs1:
                    g_loss_fake += self.classification_loss_(out0, m_trg.unsqueeze(2).unsqueeze(3),m_trg)
                g_loss_fake /= len(outs1)
                g_loss_cls = 0
 
            # reconstruction loss.
            if self.opt.lambda_rec > 0:  
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst)) * self.opt.lambda_rec 
            else:
                g_loss_rec = 0.

            # identity loss
            if self.opt.lambda_idt > 0:
                x_reconst = self.G(x_real, c_org)
                g_loss_idt = torch.mean(torch.abs(x_real - x_reconst)) * self.opt.lambda_idt
            else:
                g_loss_idt = 0.


            # Backward and optimize.
            g_loss = g_loss_fake + g_loss_cls + g_loss_rec + g_loss_idt
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss['G/loss_fake'] = g_loss_fake.item()
            if self.opt.lambda_rec > 0:
                loss['G/loss_rec'] = g_loss_rec.item()

            if not self.opt.netD == 'm_scale':
                loss['G/loss_cls'] = g_loss_cls.item()
            if self.opt.lambda_idt > 0:
                loss['G/loss_idt'] = g_loss_idt.item()


            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
            
 
            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    
                    x_fake_list = [x_fixed]
                    
                    x_fake_list.append(self.G(x_fixed, torch.tensor([[1.,0.,0.,0.]]).repeat(self.opt.batch_size,1).to(self.device)))
                    x_fake_list.append(self.G(x_fixed, torch.tensor([[0.,1.,0.,0.]]).repeat(self.opt.batch_size,1).to(self.device)))
                    x_fake_list.append(self.G(x_fixed, torch.tensor([[0.,0.,1.,0.]]).repeat(self.opt.batch_size,1).to(self.device)))    
                    x_fake_list.append(self.G(x_fixed, torch.tensor([[0.,0.,0.,1.]]).repeat(self.opt.batch_size,1).to(self.device) ))      
                  
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) >= 50000 and (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                # D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                # torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        data_loader = self.data_loader
        
        with torch.no_grad():
            for i, (x_real, c_org, x_name) in enumerate(data_loader):
                
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
        
                x_fake_AF = self.G(x_real, torch.tensor([[1.,0.,0.,0.]]).to(self.device))
                x_fake_HE = self.G(x_real, torch.tensor([[0.,1.,0.,0.]]).to(self.device))
                x_fake_PAS = self.G(x_real, torch.tensor([[0.,0.,1.,0.]]).to(self.device))
                x_fake_MT = self.G(x_real, torch.tensor([[0.,0.,0.,1.]]).to(self.device))
                save_image(self.denorm(x_fake_AF.data.cpu()), os.path.join(self.result_dir +'/AF/', '{}-AF.jpg'.format(x_name[0])))
                save_image(self.denorm(x_fake_HE.data.cpu()), os.path.join(self.result_dir +'/HE/', '{}-HE.jpg'.format(x_name[0])))
                save_image(self.denorm(x_fake_PAS.data.cpu()), os.path.join(self.result_dir +'/PAS/', '{}-PAS.jpg'.format(x_name[0])))
                save_image(self.denorm(x_fake_MT.data.cpu()), os.path.join(self.result_dir +'/MT/', '{}-MT.jpg'.format(x_name[0])))
              
                # Save the translated images.
                # x_concat = torch.cat(x_fake_list, dim=3)
                # result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                # save_image(self.denorm(x_real.data.cpu()), os.path.join(self.result_dir, '{}.jpg'.format(x_name[0])))
               
                if i%500 ==0:
                    print('Saved real and fake images into {}...'.format(i))
