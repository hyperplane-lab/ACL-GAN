"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

modified by Yihao Zhao
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
from trainer import MUNIT_Trainer, UNIT_Trainer, haodong_Trainer, Breaking_Trainer, hdmix_Trainer, hdidt_Trainer, hdmasklarger_Trainer, hdmaskper_Trainer, hdmaskpermg_Trainer, hdmaskpermgidtno_Trainer, hdmaskidt_Trainer, hdencode_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=1, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether only save the output images or also save the input images")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='haodong', help="Breaking|haodong|MUNIT|UNIT")
parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
parser.add_argument('--inception_a', type=str, default='.', help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='.', help="path to the pretrained inception network for domain B")

opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Load the inception networks if we need to compute IS or CIIS
if opts.compute_IS or opts.compute_IS:
    inception = load_inception(opts.inception_b) if opts.a2b else load_inception(opts.inception_a)
    # freeze the inception models and set eval mode
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder(opts.input_folder, 1, False, new_size=config['new_size'], crop=False)

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
elif opts.trainer == 'Breaking':
    style_dim = config['gen']['style_dim']
    trainer = Breaking_Trainer(config)
elif opts.trainer == 'haodong':
    style_dim = config['gen']['style_dim']
    trainer = haodong_Trainer(config)
elif opts.trainer == 'hdmix':
    style_dim = config['gen']['style_dim']
    trainer = hdmix_Trainer(config)
elif opts.trainer == 'hdidt':
    style_dim = config['gen']['style_dim']
    trainer = hdidt_Trainer(config)
elif opts.trainer == 'hdmasklarger':
    style_dim = config['gen']['style_dim']
    trainer = hdmasklarger_Trainer(config)
elif opts.trainer == 'hdmaskper':
    style_dim = config['gen']['style_dim']
    trainer = hdmaskper_Trainer(config)
elif opts.trainer == 'hdmaskpermg':
    style_dim = config['gen']['style_dim']
    trainer = hdmaskpermg_Trainer(config)
elif opts.trainer == 'hdmaskpermgidtno':
    style_dim = config['gen']['style_dim']
    trainer = hdmaskpermgidtno_Trainer(config)
elif opts.trainer == 'hdmaskidt':
    style_dim = config['gen']['style_dim']
    trainer = hdmaskidt_Trainer(config)
elif opts.trainer == 'hdencode':
    style_dim = config['gen']['style_dim']
    trainer = hdencode_Trainer(config)
else:
    sys.exit("Only support hdencode|hdmaskidt|hdmaskpermgidtno|hdmaskpermg|hdmaskper|hdmix|hdidt|hdmasklarger|Breaking|haodong|MUNIT|UNIT")

def focus_translation(x_fg, x_bg, x_focus):
    x_map = (x_focus+1)/2
    x_map = x_map.repeat(1, 3, 1, 1)
    return (torch.mul((x_fg+1)/2, x_map) + torch.mul((x_bg+1)/2, 1-x_map))*2-1

if opts.trainer == 'MUNIT' or opts.trainer == 'UNIT':
    try:
        state_dict = torch.load(opts.checkpoint)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])


    trainer.cuda()
    trainer.eval()
    encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
    decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

elif opts.trainer == 'Breaking':
    try:
        state_dict = torch.load(opts.checkpoint)
        trainer.gen_1.load_state_dict(state_dict['1'])
        trainer.gen_2.load_state_dict(state_dict['2'])
        trainer.gen_3.load_state_dict(state_dict['3'])
        trainer.gen_4.load_state_dict(state_dict['4'])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
        trainer.gen_1.load_state_dict(state_dict['1'])
        trainer.gen_2.load_state_dict(state_dict['2'])
        trainer.gen_3.load_state_dict(state_dict['3'])
        trainer.gen_4.load_state_dict(state_dict['4'])
    trainer.cuda()
    trainer.eval()
    encode1 = trainer.gen_1.encode
    encode2 = trainer.gen_2.encode
    encode3 = trainer.gen_3.encode
    encode4 = trainer.gen_4.encode
    decode1 = trainer.gen_1.decode
    decode2 = trainer.gen_2.decode
    decode3 = trainer.gen_3.decode
    decode4 = trainer.gen_4.decode
elif opts.trainer == 'haodong' or opts.trainer == 'hdmix' or opts.trainer == 'hdidt' or opts.trainer == 'hdmasklarger' or opts.trainer == 'hdmaskper' or opts.trainer == 'hdmaskpermg' or opts.trainer == 'hdmaskpermgidtno' or opts.trainer == 'hdmaskidt' or opts.trainer == 'hdencode':
    try:
        state_dict = torch.load(opts.checkpoint)
        trainer.gen_AB.load_state_dict(state_dict['AB'])
        trainer.gen_BA.load_state_dict(state_dict['BA'])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
        trainer.gen_AB.load_state_dict(state_dict['AB'])
        trainer.gen_BA.load_state_dict(state_dict['BA'])
    
    
    trainer.cuda()
    trainer.eval()
    Gab = trainer.gen_AB.encode if opts.a2b else trainer.gen_BA.encode # encode function
    Dab = trainer.gen_AB.decode if opts.a2b else trainer.gen_BA.decode # decode functions
    Gba = trainer.gen_BA.encode if opts.a2b else trainer.gen_BA.encode # encode function
    Dba = trainer.gen_BA.decode if opts.a2b else trainer.gen_BA.decode # decode functions

if opts.compute_IS:
    IS = []
    all_preds = []
if opts.compute_CIS:
    CIS = []

if opts.trainer == 'MUNIT':
    # Start testing
    style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if opts.compute_CIS:
            cur_preds = []
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)
        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            if opts.compute_IS or opts.compute_CIS:
                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
            if opts.compute_IS:
                all_preds.append(pred)
            if opts.compute_CIS:
                cur_preds.append(pred)
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(opts.output_folder+"_%02d"%j,basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if opts.compute_CIS:
            cur_preds = np.concatenate(cur_preds, 0)
            py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
    if opts.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    if opts.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
    if opts.compute_CIS:
        print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))

elif opts.trainer == 'UNIT':
    # Start testing
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)

        outputs = decode(content)
        outputs = (outputs + 1) / 2.
        # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
        basename = os.path.basename(names[1])
        path = os.path.join(opts.output_folder,basename)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
elif opts.trainer == 'Breaking':
    # Start testing
    style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if opts.compute_CIS:
            cur_preds = []
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content1, _ = encode1(images)
        content2, _ = encode2(images)
        content3, _ = encode3(images)
        content4, _ = encode4(images)
        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs1 = decode1(content1, s)
            outputs2 = decode2(content2, s)
            outputs3 = decode3(content3, s)
            outputs4 = decode4(content4, s)
            if config['focus_loss']>0:
                img,mask = outputs1.split(3,1)
                outputs1 = focus_translation(img, images, mask)
                img,mask = outputs2.split(3,1)
                outputs2 = focus_translation(img, images, mask)
                img,mask = outputs3.split(3,1)
                outputs3 = focus_translation(img, images, mask)
                img,mask = outputs4.split(3,1)
                outputs4 = focus_translation(img, images, mask)
            outputs1 = (outputs1 + 1) / 2.
            outputs2 = (outputs2 + 1) / 2.
            outputs3 = (outputs3 + 1) / 2.
            outputs4 = (outputs4 + 1) / 2.
#            if opts.compute_IS or opts.compute_CIS:
#                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
#                if opts.compute_IS:
#                    all_preds.append(pred)
#                if opts.compute_CIS:
#                    cur_preds.append(pred)
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path1 = os.path.join(opts.output_folder+"_1_%02d"%j,basename)
            path2 = os.path.join(opts.output_folder+"_2_%02d"%j,basename)
            path3 = os.path.join(opts.output_folder+"_3_%02d"%j,basename)
            path4 = os.path.join(opts.output_folder+"_4_%02d"%j,basename)
            if not os.path.exists(os.path.dirname(path1)):
                os.makedirs(os.path.dirname(path1))
            if not os.path.exists(os.path.dirname(path2)):
                os.makedirs(os.path.dirname(path2))
            if not os.path.exists(os.path.dirname(path3)):
                os.makedirs(os.path.dirname(path3))
            if not os.path.exists(os.path.dirname(path4)):
                os.makedirs(os.path.dirname(path4))
            vutils.save_image(outputs1.data, path1, padding=0, normalize=True)
            vutils.save_image(outputs2.data, path2, padding=0, normalize=True)
            vutils.save_image(outputs3.data, path3, padding=0, normalize=True)
            vutils.save_image(outputs4.data, path4, padding=0, normalize=True)
        if opts.compute_CIS:
            cur_preds = np.concatenate(cur_preds, 0)
            py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
    if opts.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    if opts.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
    if opts.compute_CIS:
        print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))

elif opts.trainer == 'haodong' or opts.trainer == 'hdmix' or opts.trainer == 'hdmasklarger' or opts.trainer == 'hdmaskper' or opts.trainer == 'hdmaskpermg' or opts.trainer == 'hdmaskpermgidtno' or opts.trainer == 'hdmaskidt' or opts.trainer == 'hdencode':#    # Start testing
    count_max = 0
    style_fixed = Variable(torch.randn(opts.num_style*3, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if i>=3000:
            break
        if opts.compute_CIS:
            cur_preds = []
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
#        content, _ = encode(images)
#        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
        content, _ = Gab(images)
        content_til, _ = Gba(images)
        style = style_fixed*2 if opts.synchronized else Variable(torch.randn(opts.num_style*3, style_dim, 1, 1).cuda(), volatile=True)*2
        for j in range(opts.num_style):
#            s = style[j].unsqueeze(0)
#            outputs = decode(content, s)
#            if config['focus_loss']>0:
#                img,mask = outputs.split(3,1)
#                outputs = focus_translation(img, images, mask)
#            outputs = (outputs + 1) / 2.
#            if opts.compute_IS or opts.compute_CIS:
#                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
#            if opts.compute_IS:
#                all_preds.append(pred)
#            if opts.compute_CIS:
#                cur_preds.append(pred)
#            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
#            basename = os.path.basename(names[1])
#            path = os.path.join(opts.output_folder+"_%02d"%j,basename)
#            if not os.path.exists(os.path.dirname(path)):
#                os.makedirs(os.path.dirname(path))
#            vutils.save_image(outputs.data, path, padding=0, normalize=True)
            s = style[j*3].unsqueeze(0)
            outputs = Dab(content, s)
            if config['focus_loss']>0:
                img,mask = outputs.split(3,1)
                outputs = focus_translation(img, images, mask)
                outputs_mask = mask.expand(-1,3,-1,-1)
            content_hat, _ = Gba(outputs)
            s2 = style[j*3+1].unsqueeze(0)
            outputs_hat = Dba(content_hat, s2)
            if config['focus_loss']>0:
                img,mask = outputs_hat.split(3,1)
                outputs_hat = focus_translation(img, outputs, mask)
            s3 = style[j*3+2].unsqueeze(0)
            outputs_til = Dba(content_til, s3)
            if config['focus_loss']>0:
                img,mask = outputs_til.split(3,1)
                outputs_til = focus_translation(img, images, mask)
            cnt = torch.mean(outputs_til-images)
#            if (cnt<0.07):
#                continue
            outputs = (outputs + 1) / 2.
            outputs_hat = (outputs_hat+1)/2.
            outputs_til = (outputs_til+1)/2.
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(opts.output_folder+"/_%02d_ori"%j,basename)
            path_bar = os.path.join(opts.output_folder+"/_%02d_bar"%j,basename)
            path_hat = os.path.join(opts.output_folder+"/_%02d_hat"%j,basename)
            path_til = os.path.join(opts.output_folder+"/_%02d_til"%j,basename)
#            if not os.path.exists(os.path.dirname(path)):
#                os.makedirs(os.path.dirname(path))
            if not os.path.exists(os.path.dirname(path_bar)):
                os.makedirs(os.path.dirname(path_bar))
#            if not os.path.exists(os.path.dirname(path_hat)):
#                os.makedirs(os.path.dirname(path_hat))
#            if not os.path.exists(os.path.dirname(path_til)):
#                os.makedirs(os.path.dirname(path_til))

            vutils.save_image(outputs.data, path_bar, padding=0, normalize=True)
#            vutils.save_image(images.data, path, padding=0, normalize=True)
#            vutils.save_image(outputs_hat.data, path_hat, padding=0, normalize=True)
#            vutils.save_image(outputs_til.data, path_til, padding=0, normalize=True)

            if config['focus_loss']>0:
                path_mask = os.path.join(opts.output_folder+"/_%02d_mask"%j,basename)
                if not os.path.exists(os.path.dirname(path_mask)):
                    os.makedirs(os.path.dirname(path_mask))
                vutils.save_image(outputs_mask.data, path_mask, padding=0, normalize=True)

        if opts.compute_CIS:
            cur_preds = np.concatenate(cur_preds, 0)
            py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
    print(count_max)
    if opts.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    if opts.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
    if opts.compute_CIS:
        print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))
elif opts.trainer == 'hdidt' :
    style_fixed = Variable(torch.zeros(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if opts.compute_CIS:
            cur_preds = []
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)
        style = style_fixed if opts.synchronized else Variable(torch.zeros(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            if config['focus_loss']>0:
                img,mask = outputs.split(3,1)
                outputs = focus_translation(img, images, mask)
            outputs = (outputs + 1) / 2.
            if opts.compute_IS or opts.compute_CIS:
                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
            if opts.compute_IS:
                all_preds.append(pred)
            if opts.compute_CIS:
                cur_preds.append(pred)
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(opts.output_folder+"_%02d"%j,basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if opts.compute_CIS:
            cur_preds = np.concatenate(cur_preds, 0)
            py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
    if opts.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    if opts.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
    if opts.compute_CIS:
        print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))

else:
    pass
