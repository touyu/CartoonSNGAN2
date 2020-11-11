import os, time, pickle, argparse, networks, utils

from torch import device
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import scipy.stats as stats
from data.train import CreateDataLoader as CreateTrainDataLoader
from data.test import CreateDataLoader as CreateTestDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='project_name',  help='')
parser.add_argument('--train_root', required=False, default='datasets/train',  help='train dataset path')
parser.add_argument('--test_root', required=False, default='datasets/test',  help='test dataset path')
parser.add_argument('--vgg_model', required=False, default='vgg19-dcbb9e9d.pth', help='pre-trained VGG19 model path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--pre_train_epoch', type=int, default=10)
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--con_lambda', type=float, default=10, help='lambda for content loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--latest_generator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--latest_discriminator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--n_dis', type=int, default='5', help='discriminator trainging count per generater training count')
args = parser.parse_args()

landscape_dataloader = CreateTrainDataLoader(args, "landscape")
anime_dataloader = CreateTrainDataLoader(args, "anime")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_gen():
    mu, sigma = 1, 0.005
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

    maskS = args.input_size // 4

    mask1 = torch.cat(
        [torch.rand(1, 1, maskS, maskS).ge(0.5).float() for _ in range(args.batch_size)], 0)
    mask = mask1
    # mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(args.batch_size // 2)], 0)
    # mask = torch.cat([mask1, mask2], 0)

    # mask = torch.cat([torch.ones(1, 1, maskS, maskS) for _ in range(args.batch_size)], 0)
    print(mask.shape)

    return mask.to(device)

for i, (lcimg, lhimg, lsimg) in enumerate(landscape_dataloader):
    lcimg, lhimg, lsimg = lcimg.to(device), lhimg.to(device), lsimg.to(device)
    mask = mask_gen()
    mask_imgs = lhimg * mask
    hint = torch.cat((mask_imgs, mask), 1)

    for i, mask_img in enumerate(mask_imgs):
        plt.imsave(f"mask_results/{i}.png", (mask_img.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    break
