"""
created by Yihao Zhao
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
from trainer import aclgan_Trainer
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
parser.add_argument('--num_style',type=int, default=19, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether only save the output images or also save the input images")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='aclgan', help="aclgan")
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
if opts.trainer == 'aclgan':
    style_dim = config['gen']['style_dim']
    trainer = aclgan_Trainer(config)
else:
    sys.exit("Only support aclgan")

def focus_translation(x_fg, x_bg, x_focus):
    x_map = (x_focus+1)/2
    x_map = x_map.repeat(1, 3, 1, 1)
    return (torch.mul((x_fg+1)/2, x_map) + torch.mul((x_bg+1)/2, 1-x_map))*2-1

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
    decode = trainer.gen_AB.decode if opts.a2b else trainer.gen_BA.decode # decode functions

if opts.compute_IS:
    IS = []
    all_preds = []
if opts.compute_CIS:
    CIS = []

if opts.trainer == 'aclgan':#    # Start testing
    style_fixed = Variable(torch.randn(opts.num_style*2, style_dim, 1, 1).cuda(), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        if (i>=100):
            break
        if opts.compute_CIS:
            cur_preds = []
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)
        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style*2, style_dim, 1, 1).cuda(), volatile=True)
        for j in range(opts.num_style):
            s = style[j*2].unsqueeze(0)
            s2 = style[j*2+1].unsqueeze(0)
            outputs = decode(content, s)
            outputs2 = decode(content, s2)
            if config['focus_loss']>0:
                img,mask = outputs.split(3,1)
                outputs = focus_translation(img, images, mask)
                img,mask = outputs2.split(3,1)
                outputs2 = focus_translation(img, images, mask)
            outputs = (outputs + 1) / 2.
            outputs2 = (outputs2 + 1) / 2.
            if opts.compute_IS or opts.compute_CIS:
                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
            if opts.compute_IS:
                all_preds.append(pred)
            if opts.compute_CIS:
                cur_preds.append(pred)
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(opts.output_folder+"_0","_%02d"%j+basename)
            path2 = os.path.join(opts.output_folder+"_1","_%02d"%j+basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            if not os.path.exists(os.path.dirname(path2)):
                os.makedirs(os.path.dirname(path2))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
            vutils.save_image(outputs2.data, path2, padding=0, normalize=True)
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
