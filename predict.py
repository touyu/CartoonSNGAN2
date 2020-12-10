import os, time, pickle, argparse, networks, utils
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
parser.add_argument('--train_root', required=False, default='datasets/train',  help='train datasets path')
parser.add_argument('--test_root', required=False, default='datasets/test',  help='test datasets path')
parser.add_argument('--vgg_model', required=False, default='vgg19-dcbb9e9d.pth', help='pre-trained VGG19 model path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--train_epoch', type=int, default=400)
parser.add_argument('--pre_train_epoch', type=int, default=30)
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--con_lambda', type=float, default=10, help='lambda for content loss')
parser.add_argument('--beta1', type=float, default=0, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
parser.add_argument('--latest_generator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--latest_discriminator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--n_dis', type=int, default='5', help='discriminator trainging count per generater training count')
args = parser.parse_args()

def mask_gen():
    mu, sigma = 1, 0.005
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

    maskS = args.input_size // 4

    mask1 = torch.cat(
        [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(args.batch_size // 2)], 0)
    mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(args.batch_size // 2)], 0)
    mask = torch.cat([mask1, mask2], 0)

    # mask = torch.cat([torch.ones(1, 1, maskS, maskS) for _ in range(args.batch_size)], 0)

    mask = torch.cat(
        [torch.rand(1, 1, maskS, maskS).ge(0.9).float() for _ in range(args.batch_size)], 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return mask.to(device)

outputPath = f"predict_results/{args.name}"
os.makedirs(outputPath+"/predicts", exist_ok=True)
os.makedirs(outputPath+"/lines", exist_ok=True)
os.makedirs(outputPath+"/originals", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

landscape_test_dataloader = CreateTestDataLoader(args, "landscape")

generator = networks.Generator(args.ngf)

if args.latest_generator_model != '':
    if torch.cuda.is_available():
        generator.load_state_dict(torch.load(args.latest_generator_model))
    else:
        # cpu mode
        generator.load_state_dict(torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage))


generator.to(device)
generator.eval()

i = 0
with torch.no_grad():
    for lcimg, lhimg, lsimg in landscape_test_dataloader:
        lcimg, lhimg, lsimg = lcimg.to(device), lhimg.to(device), lsimg.to(device)
        mask = mask_gen()
        hint = torch.cat((lhimg * mask, mask), 1)
        gen_imgs = generator(lsimg, hint)
        for j, gen_img in enumerate(gen_imgs):
            plt.imsave(f"{outputPath}/originals/{i}.png", (lcimg[j].cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            plt.imsave(f"{outputPath}/predicts/{i}.png", (gen_img.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            line = transforms.ToPILImage()((lsimg[j].cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            plt.imsave(f"{outputPath}/lines/{i}.png", line, cmap="gray")
            i += 1
            print(i)
