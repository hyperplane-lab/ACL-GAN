"""
Created by Yihao Zhao
"""
from networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

# the model for ACL-GAN
class aclgan_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(aclgan_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_AB = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain A
        self.gen_BA = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain B
        self.dis_A = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain A
        self.dis_B = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain B
        self.dis_2 = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator 2
#        self.dis_2B = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator 2 for domain B
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.z_1 = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.z_2 = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.z_3 = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_A.parameters()) + list(self.dis_B.parameters()) + list(self.dis_2.parameters())
        gen_params = list(self.gen_AB.parameters()) + list(self.gen_BA.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.alpha = hyperparameters['alpha']
        self.focus_lam = hyperparameters['focus_loss']

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_A.apply(weights_init('gaussian'))
        self.dis_B.apply(weights_init('gaussian'))
        self.dis_2.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        z_1 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        z_2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        z_3 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_1, _  = self.gen_AB.encode(x_a)
        c_2, s_2 = self.gen_BA.encode(x_a)
        c_4, s_4 = self.gen_AB.encode(x_b)
        # decode
        self.x_B_fake = self.gen_AB.decode(c_1, z_1)
        self.x_A_fake = self.gen_BA.decode(c_2, z_2)
        # recon
        self.x_A_recon = self.gen_BA.decode(c_2, s_2)
        self.x_B_recon = self.gen_AB.decode(c_4, s_4)
        #encode 2
        c_3, _ = self.gen_BA.encode(self.x_B_fake)
        self.x_A2_fake = self.gen_BA.decode(c_3, z_3)

        self.X_A_A1_pair = torch.cat((x_a, self.x_A_fake), -3)
        self.X_A_A2_pair = torch.cat((x_a, self.x_A2_fake), -3)

    def focus_translation(self, x_fg, x_bg, x_focus):
        x_map = (x_focus+1)/2
        x_map = x_map.repeat(1, 3, 1, 1)
        return torch.mul(x_fg, x_map) + torch.mul(x_bg, 1-x_map)

    def gen_update(self, x_a, x_b,  hyperparameters):
        self.gen_opt.zero_grad()

        focus_delta = hyperparameters['focus_delta']
        focus_lambda = hyperparameters['focus_loss']
        focus_lower = hyperparameters['focus_lower']
        focus_upper = hyperparameters['focus_upper']
        focus_epsilon = hyperparameters['focus_epsilon']
        #forward
        z_1 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        z_2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        z_3 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_1, _  = self.gen_AB.encode(x_a)
        c_2, s_2 = self.gen_BA.encode(x_a)
        c_4, s_4 = self.gen_AB.encode(x_b)
        # decode
        if focus_lambda > 0:
            x_B_fake, x_B_focus = self.gen_AB.decode(c_1, z_1).split(3, 1)
            x_A_fake, x_A_focus = self.gen_BA.decode(c_2, self.alpha * z_2).split(3, 1)
            x_B_fake = self.focus_translation(x_B_fake, x_a, x_B_focus)
            x_A_fake = self.focus_translation(x_A_fake, x_a, x_A_focus)
            # recon
            x_A_recon, x_A_recon_focus = self.gen_BA.decode(c_2, s_2).split(3,1)
            x_B_recon, x_B_recon_focus = self.gen_AB.decode(c_4, s_4).split(3,1)
#            x_A_recon = self.focus_translation(x_A_recon, x_a, x_A_recon_focus)
#            x_B_recon = self.focus_translation(x_B_recon, x_b, x_B_recon_focus)
        else:
            x_B_fake = self.gen_AB.decode(c_1, z_1)
            x_A_fake = self.gen_BA.decode(c_2, self.alpha * z_2)
            # recon
            x_A_recon = self.gen_BA.decode(c_2, s_2)
            x_B_recon = self.gen_AB.decode(c_4, s_4)

        #encode 2
        c_3, _ = self.gen_BA.encode(x_B_fake)
        if focus_lambda > 0:
            x_A2_fake, x_A2_focus = self.gen_BA.decode(c_3, z_3).split(3, 1)
            x_A2_fake = self.focus_translation(x_A2_fake, x_B_fake, x_A2_focus)
        else:
            x_A2_fake = self.gen_BA.decode(c_3, z_3)

        x_A_A1_pair = torch.cat((x_a, x_A_fake), -3)
        x_A_A2_pair = torch.cat((x_a, x_A2_fake), -3)

        # GAN loss
        self.loss_gen_adv_A = (self.dis_A.calc_gen_loss(x_A_fake) + \
                              self.dis_A.calc_gen_loss(x_A2_fake)) * 0.5
        self.loss_gen_adv_B = self.dis_B.calc_gen_loss(x_B_fake)
        self.loss_gen_adv_2 = self.dis_2.calc_gen_d2_loss(x_A_A1_pair, x_A_A2_pair)

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_A + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_B + \
                              hyperparameters['gan_cw'] * self.loss_gen_adv_2
        if focus_lambda > 0:
            x_B_focus = (x_B_focus + 1)/2
            x_A_focus = (x_A_focus + 1)/2
            x_A2_focus = (x_A2_focus + 1)/2
            self.loss_gen_focus_B_size = (F.relu(torch.sum(x_B_focus - focus_upper), inplace=True) ** 2) * focus_delta + \
                (F.relu(torch.sum(focus_lower - x_B_focus), inplace=True) ** 2) * focus_delta
            self.loss_gen_focus_B_digit = torch.sum(1 / (torch.abs(x_B_focus - 0.5) + focus_epsilon))
            self.loss_gen_focus_A_size = (F.relu(torch.sum(x_A_focus - focus_upper), inplace=True) ** 2) * focus_delta + \
                (F.relu(torch.sum(focus_lower - x_A_focus), inplace=True) ** 2) * focus_delta
            self.loss_gen_focus_A_digit = torch.sum(1 / (torch.abs(x_A_focus - 0.5) + focus_epsilon))
#            self.loss_gen_focus_A = torch.sum(1 / (torch.abs(x_A_focus - 0.5) + focus_epsilon))
            self.loss_gen_focus_A2_size = (F.relu(torch.sum(x_A2_focus - focus_upper), inplace=True) ** 2) * focus_delta + \
                (F.relu(torch.sum(focus_lower - x_A2_focus), inplace=True) ** 2) * focus_delta
            self.loss_gen_focus_A2_digit = torch.sum(1 / (torch.abs(x_A2_focus - 0.5) + focus_epsilon))
            self.loss_gen_total += focus_lambda * (self.loss_gen_focus_B_size + self.loss_gen_focus_B_digit + \
                            self.loss_gen_focus_A_size + self.loss_gen_focus_A_digit +\
                            self.loss_gen_focus_A2_size + self.loss_gen_focus_A2_digit)/ x_a.size(2) / x_a.size(3) / x_a.size(0) / 3
        self.loss_idt_A = self.recon_criterion(x_A_recon, x_a)
        self.loss_idt_B = self.recon_criterion(x_B_recon, x_b)
        self.loss_gen_total += hyperparameters['recon_x_w'] * self.loss_idt_A + \
                              hyperparameters['recon_x_w'] * self.loss_idt_B

#        print(self.loss_gen_focus_B, self.loss_gen_total)
#        print(self.loss_idt_A)
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        z_1 = Variable(self.z_1)
        z_2 = Variable(self.z_2)
        z_3 = Variable(self.z_3)
        x_A, x_B, x_A_fake, x_B_fake, x_A2_fake = [], [], [], [], []
        if self.focus_lam > 0:
            mask_A, mask_B, mask_A2, mask_recon = [], [], [], []
            x_A_recon = []
        else:
            x_A_recon, x_B_recon = [], []
        for i in range(x_a.size(0)):
            x_A.append(x_a[i].unsqueeze(0))
            x_B.append(x_b[i].unsqueeze(0))
            if self.focus_lam > 0:
                c_1, s_1 = self.gen_BA.encode(x_a[i].unsqueeze(0))
                img, mask = self.gen_BA.decode(c_1, z_1[i].unsqueeze(0)).split(3, 1)
                x_A_fake.append(self.focus_translation(img, x_a[i].unsqueeze(0), mask))
                mask_A.append(mask)

                img, mask = self.gen_BA.decode(c_1, s_1).split(3, 1)
#                x_A_recon.append(self.focus_translation(img, x_a[i].unsqueeze(0), mask))
                x_A_recon.append(img)
                mask_recon.append(mask)

                c_2, _ = self.gen_AB.encode(x_a[i].unsqueeze(0))
                x_b_img, mask = self.gen_AB.decode(c_2, z_2[i].unsqueeze(0)).split(3,1)
                x_b_img = self.focus_translation(x_b_img, x_a[i].unsqueeze(0), mask)
                x_B_fake.append(x_b_img)
                mask_B.append(mask)

                c_3, _ = self.gen_BA.encode(x_b_img)
                img, mask = self.gen_BA.decode(c_3, z_3[i].unsqueeze(0)).split(3, 1)
                x_A2_fake.append(self.focus_translation(img, x_b_img, mask))
                mask_A2.append(mask)

            else:
                c_1, s_1 = self.gen_BA.encode(x_a[i].unsqueeze(0))
                x_A_fake.append(self.gen_BA.decode(c_1, z_1[i].unsqueeze(0)))
                x_A_recon.append(self.gen_BA.decode(c_1, s_1))

                c_2, _ = self.gen_AB.encode(x_a[i].unsqueeze(0))
                x_B1 = self.gen_AB.decode(c_2, z_2[i].unsqueeze(0))
                x_B_fake.append(x_B1)

                c_3, _ = self.gen_BA.encode(x_B1)
                x_A2_fake.append(self.gen_BA.decode(c_3, z_3[i].unsqueeze(0)))

                c_4, s_4 = self.gen_AB.encode(x_b)
                x_B_recon.append(self.gen_AB.decode(c_4, s_4))

        if self.focus_lam > 0:
            x_A, x_B = torch.cat(x_A), torch.cat(x_B)
            x_A_fake, x_B_fake = torch.cat(x_A_fake), torch.cat(x_B_fake)
            mask_A, x_A2_fake = torch.cat(mask_A), torch.cat(x_A2_fake)
            mask_B, mask_recon = torch.cat(mask_B), torch.cat(mask_recon)
            mask_A2, x_A_recon = torch.cat(mask_A2), torch.cat(x_A_recon)
            self.train()
            return x_A, x_A_fake, mask_A, x_B_fake, mask_B, x_A2_fake, mask_A2, x_A_recon, mask_recon

        else:
            x_A, x_B = torch.cat(x_A), torch.cat(x_B)
            x_A_fake, x_B_fake = torch.cat(x_A_fake), torch.cat(x_B_fake)
            x_A_recon, x_A2_fake = torch.cat(x_A_recon), torch.cat(x_A2_fake)
            x_B_recon = torch.cat(x_B_recon)
            self.train()
            return x_A, x_A_fake, x_B_fake, x_A2_fake, x_A_recon, x_B, x_B_recon

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()

        focus_delta = hyperparameters['focus_delta']
        focus_lambda = hyperparameters['focus_loss']
        focus_epsilon = hyperparameters['focus_epsilon']
        #forward
        z_1 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        z_2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        z_3 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_1, _  = self.gen_AB.encode(x_a)
        c_2, s_2 = self.gen_BA.encode(x_a)
        c_4, s_4 = self.gen_AB.encode(x_b)
        # decode
        if focus_lambda > 0:
            x_B_fake, x_B_focus = self.gen_AB.decode(c_1, z_1).split(3, 1)
            x_A_fake, x_A_focus = self.gen_BA.decode(c_2, self.alpha * z_2).split(3, 1)
            x_B_fake = self.focus_translation(x_B_fake, x_a, x_B_focus)
            x_A_fake = self.focus_translation(x_A_fake, x_a, x_A_focus)
        else:
            x_B_fake = self.gen_AB.decode(c_1, z_1)
            x_A_fake = self.gen_BA.decode(c_2, self.alpha * z_2)

        #encode 2
        c_3, _ = self.gen_BA.encode(x_B_fake)
        if focus_lambda > 0:
            x_A2_fake, x_A2_focus = self.gen_BA.decode(c_3, z_3).split(3, 1)
            x_A2_fake = self.focus_translation(x_A2_fake, x_B_fake, x_A2_focus)
        else:
            x_A2_fake = self.gen_BA.decode(c_3, z_3)

        x_A_A1_pair = torch.cat((x_a, x_A_fake), -3)
        x_A_A2_pair = torch.cat((x_a, x_A2_fake), -3)

        # D loss
        self.loss_dis_A = (self.dis_A.calc_dis_loss(x_A_fake, x_a) + \
                           self.dis_A.calc_dis_loss(x_A2_fake, x_a)) * 0.5
        self.loss_dis_B = self.dis_B.calc_dis_loss(x_B_fake, x_b)
        self.loss_dis_2 = self.dis_2.calc_dis_loss(x_A_A1_pair, x_A_A2_pair)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_A + \
                                hyperparameters['gan_w'] * self.loss_dis_B + \
                                hyperparameters['gan_cw'] * self.loss_dis_2

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_AB.load_state_dict(state_dict['AB'])
        self.gen_BA.load_state_dict(state_dict['BA'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_A.load_state_dict(state_dict['A'])
        self.dis_B.load_state_dict(state_dict['B'])
        self.dis_2.load_state_dict(state_dict['2'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'AB': self.gen_AB.state_dict(), 'BA': self.gen_BA.state_dict()}, gen_name)
        torch.save({'A': self.dis_A.state_dict(), 'B': self.dis_B.state_dict(), '2': self.dis_2.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

