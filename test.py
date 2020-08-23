"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

modified by Yihao Zhao
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer, aclmaskpermgidtno_Trainer, aclmaskidt_Trainer, aclgan_Trainer, aclencode_Trainer, Breaking_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='aclgan', help="aclgan")
opts = parser.parse_args()



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'aclgan':
    style_dim = config['gen']['style_dim']
    trainer = aclgan_Trainer(config)
else:
    sys.exit("Only support aclgan")

if opts.trainer == 'aclgan':
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
    encode = trainer.gen_AB.encode if opts.a2b else trainer.gen_BA.encode # encode function
    style_encode = trainer.gen_AB.encode if opts.a2b else trainer.gen_BA.encode # encode function
    decode = trainer.gen_AB.decode if opts.a2b else trainer.gen_BA.decode # decode function
    Gba = trainer.gen_BA.encode
    Dba = trainer.gen_BA.decode


def focus_translation(x_fg, x_bg, x_focus):
    x_map = (x_focus+1)/2
    x_map = x_map.repeat(1, 3, 1, 1)
    return (torch.mul((x_fg+1)/2, x_map) + torch.mul((x_bg+1)/2, 1-x_map))*2-1



if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Variable(transform(Image.open(opts.input).convert('RGB')).unsqueeze(0).cuda())
    style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None

    # Start testing
    content, _ = encode(image)

    if opts.trainer == 'aclgan':
        style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
        if opts.style != '':
            _, style = style_encode(style_image)
        else:
            style = style_rand
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            if config['focus_loss']>0:
                img,mask = outputs.split(3,1)
                outputs_img = img
                outputs = focus_translation(img, image, mask)
                outputs_mask = mask.expand(-1,3,-1,-1)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, 'output{:03d}.jpg'.format(j))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if config['focus_loss']>0:
                path_mask = os.path.join(opts.output_folder, 'output{:03d}_mask.jpg'.format(j))
                path_img = os.path.join(opts.output_folder, 'output{:03d}_img.jpg'.format(j))
                if not os.path.exists(os.path.dirname(path_mask)):
                    os.makedirs(os.path.dirname(path_mask))
                if not os.path.exists(os.path.dirname(path_img)):
                    os.makedirs(os.path.dirname(path_img))
                vutils.save_image(outputs_mask.data, path_mask, padding=0, normalize=True)
                vutils.save_image(outputs_img.data, path_img, padding=0, normalize=True)

    
    else:
        pass

    if not opts.output_only:
        # also save input images
        vutils.save_image(image.data, os.path.join(opts.output_folder, 'input.jpg'), padding=0, normalize=True)

