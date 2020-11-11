from PIL.Image import NONE
from torchvision import transforms
from PIL import Image
import random
import torch.utils.data as data
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def CreateDataLoader(args, dataset_type="landscape"):
    random.seed(2333)

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size) , Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    hint_transform = transforms.Compose([
        transforms.Resize(args.input_size // 4, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    sketch_transform = transforms.Compose([
        transforms.Resize(args.input_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = ImageFolder(root=args.test_root, transform=transform, hint_transform=hint_transform, sketch_transform=sketch_transform, dataset_type=dataset_type)
    return data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, hint_transform=None, sketch_transform=None, dataset_type=None):
        imgs = make_dataset(root, dataset_type)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.root = root
        self.imgs = imgs
        self.dataset_type = dataset_type
        self.transform = transform
        self.hint_transform = hint_transform
        self.sketch_transform = sketch_transform

    def __getitem__(self, index):
        fname = self.imgs[index]
        if self.dataset_type == "anime":
            Cimg = color_loader(os.path.join(self.root, 'anime_color', fname))
            # Simg = sketch_loader(os.path.join(self.root, 'line', fname))
            # Cimg, Simg = RandomCrop(512)(Cimg, Simg)
            # if random.random() < 0.5:
            #     Cimg = Cimg.transpose(Image.FLIP_LEFT_RIGHT)
            Cimg, Himg = self.transform(Cimg), self.hint_transform(Cimg)
            return Cimg, Himg
        else:
            Cimg = color_loader(os.path.join(self.root, 'landscape_color', fname))
            Simg = sketch_loader(os.path.join(self.root, 'landscape_line', fname))
            # Cimg, Simg = RandomCrop(512)(Cimg, Simg)
            # if random.random() < 0.5:
            #     Cimg, Simg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)
            Cimg, Himg, Simg = self.transform(Cimg), self.hint_transform(Cimg), self.sketch_transform(Simg)
            return Cimg, Himg, Simg

    def __len__(self):
        return len(self.imgs)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root, dataset_type):
    images = []

    if dataset_type == "anime":
        path = os.walk(os.path.join(root, 'anime_color'))
    else:
        path = os.walk(os.path.join(root, 'landscape_color'))

    for _, __, fnames in sorted(path):
        for fname in fnames:
            if is_image_file(fname):
                images.append(fname)
    return images


def color_loader(path):
    return Image.open(path).convert('RGB')


def sketch_loader(path):
    return Image.open(path).convert('L')