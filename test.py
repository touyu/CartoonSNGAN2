import os, time, pickle, argparse, networks, utils
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
from dataloader import CreateDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='project_name',  help='')
parser.add_argument('--train_root', required=False, default='data/train',  help='sec data path')
# parser.add_argument('--src_data', required=False, default='src_data',  help='sec data path')
# parser.add_argument('--tgt_data', required=False, default='tgt_data',  help='tgt data path')
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


landscape_dataloader = CreateDataLoader(args, "landscape")
anime_dataloader = CreateDataLoader(args, "anime")

# for i, (acimg, ahimg) in enumerate(anime_dataloader):
#     print(i, acimg.shape)

for i, ((acimg, ahimg), (lcimg, lhimg, lsimg)) in enumerate(zip(anime_dataloader, landscape_dataloader)):
    print(i, acimg.shape)
    print(i, lcimg.shape)
    path = os.path.join(f'test_results/{i}_lcimg.png')
    plt.imsave(path, (lcimg[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    path = os.path.join(f'test_results/{i}_lhimg.png')
    plt.imsave(path, (lhimg[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    path = os.path.join(f'test_results/{i}_lsimg.png')
    lsimg = (lsimg[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
    # torchvision.utils.save_image(lsimg[0], path)
    pilTrans = transforms.ToPILImage()
    lsimg = pilTrans(lsimg)
    plt.imsave(path, lsimg, cmap="gray")

# src_data = "src_data"
# tgt_data = "tgt_data"
# batch_size = 8
# input_size = 256
# device = 'cuda'

# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]

# src_transform = transforms.Compose([
#     transforms.Resize((input_size, input_size)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

# hint_transform = transforms.Compose([
#     transforms.Resize(input_size // 4),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

# tgt_transform = transforms.Compose([
#     transforms.Resize((input_size, input_size)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

# # def jitter(x):
# #     ran = random.uniform(0.7, 1)
# #     return x * ran + 1 - ran

# sketch_transform = transforms.Compose([
#     transforms.Resize(input_size, Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.ToPILImage('L'),
#     transforms.ToTensor(),
#     # transforms.Lambda(jitter),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_loader_src = utils.data_load(os.path.join('data', src_data), 'train_color', src_transform, batch_size, shuffle=False, drop_last=True)
# train_loader_hint = utils.data_load(os.path.join('data', src_data), 'train_color', hint_transform, batch_size, shuffle=False, drop_last=True)
# train_loader_sketch = utils.data_load(os.path.join('data', src_data), 'train_line', sketch_transform, batch_size, shuffle=False, drop_last=True)
# train_loader_tgt = utils.data_load(os.path.join('data', tgt_data), 'train', tgt_transform, batch_size, shuffle=True, drop_last=True)

# for i, (x, _) in enumerate(train_loader_sketch):
#     print(x.shape)
#     path = os.path.join(f'test_results/sketch_{i}.png')
#     plt.imsave(path, (x[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2)

# for i, ((x, _), (h, _), (y, _)) in enumerate(zip(train_loader_src, train_loader_hint, train_loader_tgt)):
#     if i % 5 == 0:
#         print(i, x.shape)

    # e = y[:, :, :, input_size:]
    # y = y[:, :, :, :input_size]
    # x, h, y, e = x.to(device), h.to(device), y.to(device), e.to(device)
    # for i, (x, h) in enumerate(zip(x, h)):
    #     # print(result[0].shape)
    #     path = os.path.join(f'test_results/x_{i}.png')
    #     plt.imsave(path, (x.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    #     path = os.path.join(f'test_results/h_{i}.png')
    #     plt.imsave(path, (h.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
    
    # break



# class ImageFolder(data.Dataset):
#     def __init__(self, root, transform=None, vtransform=None):
#         imgs = make_dataset(root, dataset_type)
#         if len(imgs) == 0:
#             raise (RuntimeError("Found 0 images in folders."))
#         self.root = root
#         self.imgs = imgs
#         self.transform = transform
#         self.vtransform = vtransform

#     def __getitem__(self, index):
#         fname = self.imgs[index]
#         if self.dataset_type == "anime":
#             Cimg = color_loader(os.path.join(self.root, 'color', fname))
#             Simg = sketch_loader(os.path.join(self.root, 'line', fname))
#         else:
#             Cimg = color_loader(os.path.join(self.root, 'landscape_color', fname))
#             Simg = sketch_loader(os.path.join(self.root, 'landscape_line', fname))
#         Cimg, Simg = RandomCrop(512)(Cimg, Simg)
#         if random.random() < 0.5:
#             Cimg, Simg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)

#         Cimg, Vimg, Simg = self.transform(Cimg), self.vtransform(Cimg), self.stransform(Simg)

#         return Cimg, Vimg, Simg

# def make_dataset(root, dataset_type):
#     images = []

#     if dataset_type == "anime":
#         path = os.walk(os.path.join(root, 'color'))
#     else:
#         path = os.walk(os.path.join(root, 'landscape_color'))

#     for _, __, fnames in sorted(path):
#         for fname in fnames:
#             if is_image_file(fname):
#                 images.append(fname)
#     return images

# def color_loader(path):
#     return Image.open(path).convert('RGB')
