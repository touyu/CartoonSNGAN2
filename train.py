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
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
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

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

def prepare_result():
     # results save path
    if not os.path.isdir(os.path.join(args.name + '_results', 'Reconstruction')):
        os.makedirs(os.path.join(args.name + '_results', 'Reconstruction'))
    if not os.path.isdir(os.path.join(args.name + '_results', 'Transfer')):
        os.makedirs(os.path.join(args.name + '_results', 'Transfer'))

def mask_gen():
    mu, sigma = 1, 0.005
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

    maskS = args.input_size // 4

    mask1 = torch.cat(
        [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(args.batch_size // 2)], 0)
    mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(args.batch_size // 2)], 0)
    mask = torch.cat([mask1, mask2], 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return mask.to(device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    prepare_result()

    # data_loader
    landscape_dataloader = CreateTrainDataLoader(args, "landscape")
    anime_dataloader = CreateTrainDataLoader(args, "anime")
    landscape_test_dataloader = CreateTestDataLoader(args, "landscape")
    anime_test_dataloader = CreateTestDataLoader(args, "anime")

    generator = networks.Generator(args.ngf)
    if args.latest_generator_model != '':
        if torch.cuda.is_available():
            generator.load_state_dict(torch.load(args.latest_generator_model))
        else:
            # cpu mode
            generator.load_state_dict(torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage))

    discriminator = networks.Discriminator(args.in_ndc, args.out_ndc, args.ndf)
    if args.latest_discriminator_model != '':
        if torch.cuda.is_available():
            discriminator.load_state_dict(torch.load(args.latest_discriminator_model))
        else:
            discriminator.load_state_dict(torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage))

    VGG = networks.VGG19(init_weights=args.vgg_model, feature_mode=True)

    generator.to(device)
    discriminator.to(device)
    VGG.to(device)

    generator.train()
    discriminator.train()

    VGG.eval()

    G_optimizer = optim.Adam(generator.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
    # G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
    # D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)

    print('---------- Networks initialized -------------')
    utils.print_network(generator)
    utils.print_network(discriminator)
    utils.print_network(VGG)
    print('-----------------------------------------------')

    BCE_loss = nn.BCELoss().to(device)
    L1_loss = nn.L1Loss().to(device)
    MSELoss = nn.MSELoss().to(device)

    pre_train_hist = {}
    pre_train_hist['Recon_loss'] = []
    pre_train_hist['per_epoch_time'] = []
    pre_train_hist['total_time'] = []

    """ Pre-train reconstruction """
    if args.latest_generator_model == '':
        print('Pre-training start!')
        start_time = time.time()
        for epoch in range(args.pre_train_epoch):
            epoch_start_time = time.time()
            Recon_losses = []
            for lcimg, lhimg, lsimg in landscape_dataloader:
                lcimg, lhimg, lsimg = lcimg.to(device), lhimg.to(device), lsimg.to(device)

                # train generator G
                G_optimizer.zero_grad()

                x_feature = VGG((lcimg + 1) / 2)

                mask = mask_gen()
                hint = torch.cat((lhimg * mask, mask), 1)
                gen_img = generator(lsimg, hint)
                G_feature = VGG((gen_img + 1) / 2)

                Recon_loss = 10 * L1_loss(G_feature, x_feature.detach())
                Recon_losses.append(Recon_loss.item())
                pre_train_hist['Recon_loss'].append(Recon_loss.item())

                Recon_loss.backward()
                G_optimizer.step()

            per_epoch_time = time.time() - epoch_start_time
            pre_train_hist['per_epoch_time'].append(per_epoch_time)
            print('[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), args.pre_train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Recon_losses))))

            # Save
            if (epoch+1) % 5 == 0:
                with torch.no_grad():
                    generator.eval()
                    for n, (lcimg, lhimg, lsimg) in enumerate(landscape_dataloader):
                        lcimg, lhimg, lsimg = lcimg.to(device), lhimg.to(device), lsimg.to(device)
                        mask = mask_gen()
                        hint = torch.cat((lhimg * mask, mask), 1)
                        g_recon = generator(lsimg, hint)
                        result = torch.cat((lcimg[0], g_recon[0]), 2)
                        path = os.path.join(args.name + '_results', 'Reconstruction', args.name + '_train_recon_' + f'epoch_{epoch}_' + str(n + 1) + '.png')
                        plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                        if n == 4:
                            break

                    for n, (lcimg, lhimg, lsimg) in enumerate(landscape_test_dataloader):
                        lcimg, lhimg, lsimg = lcimg.to(device), lhimg.to(device), lsimg.to(device)
                        mask = mask_gen()
                        hint = torch.cat((lhimg * mask, mask), 1)
                        g_recon = generator(lsimg, hint)
                        result = torch.cat((lcimg[0], g_recon[0]), 2)
                        path = os.path.join(args.name + '_results', 'Reconstruction', args.name + '_test_recon_' + f'epoch_{epoch}_' + str(n + 1) + '.png')
                        plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                        if n == 4:
                            break

        total_time = time.time() - start_time
        pre_train_hist['total_time'].append(total_time)
        with open(os.path.join(args.name + '_results',  'pre_train_hist.pkl'), 'wb') as f:
            pickle.dump(pre_train_hist, f)
        torch.save(generator.state_dict(), os.path.join(args.name + '_results', 'generator_pretrain.pkl'))

    else:
        print('Load the latest generator model, no need to pre-train')

    train_hist = {}
    train_hist['Disc_loss'] = []
    train_hist['Gen_loss'] = []
    train_hist['Con_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []
    print('training start!')
    start_time = time.time()

    real = torch.ones(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
    fake = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
    for epoch in range(args.train_epoch):
        epoch_start_time = time.time()
        generator.train()
        Disc_losses = []
        Gen_losses = []
        Con_losses = []
        for i, ((acimg, _), (lcimg, lhimg, lsimg)) in enumerate(zip(anime_dataloader, landscape_dataloader)):
            acimg, lcimg, lhimg, lsimg = acimg.to(device), lcimg.to(device), lhimg.to(device), lsimg.to(device)

            if i % args.n_dis == 0:
                 # train G
                print("TrainG")
                G_optimizer.zero_grad()

                mask = mask_gen()
                hint = torch.cat((lhimg * mask, mask), 1)
                gen_img = generator(lsimg, hint)
                D_fake = discriminator(gen_img)
                D_fake_loss = BCE_loss(D_fake, real)

                x_feature = VGG((lcimg + 1) / 2)
                G_feature = VGG((gen_img + 1) / 2)
                Con_loss = args.con_lambda * L1_loss(G_feature, x_feature.detach())

                Gen_loss = D_fake_loss + Con_loss
                Gen_losses.append(D_fake_loss.item())
                train_hist['Gen_loss'].append(D_fake_loss.item())
                Con_losses.append(Con_loss.item())
                train_hist['Con_loss'].append(Con_loss.item())

                Gen_loss.backward()
                G_optimizer.step()
                # G_scheduler.step()

            print("TrainD")
            # train D
            D_optimizer.zero_grad()

            D_real = discriminator(acimg)
            D_real_loss = BCE_loss(D_real, real) # Hinge Loss (?)

            mask = mask_gen()
            hint = torch.cat((lhimg * mask, mask), 1)

            gen_img = generator(lsimg, hint)
            D_fake = discriminator(gen_img)
            D_fake_loss = BCE_loss(D_fake, fake)

            # D_edge = Discriminator(e)
            # D_edge_loss = BCE_loss(D_edge, fake)

            # Disc_loss = D_real_loss + D_fake_loss + D_edge_loss
            Disc_loss = D_real_loss + D_fake_loss
            Disc_losses.append(Disc_loss.item())
            train_hist['Disc_loss'].append(Disc_loss.item())

            Disc_loss.backward()
            D_optimizer.step()

    #     G_scheduler.step()
    #     D_scheduler.step()

        per_epoch_time = time.time() - epoch_start_time
        train_hist['per_epoch_time'].append(per_epoch_time)
        print(
        '[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % ((epoch + 1), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Disc_losses)),
            torch.mean(torch.FloatTensor(Gen_losses)), torch.mean(torch.FloatTensor(Con_losses))))

        if epoch % 2 == 1 or epoch == args.train_epoch - 1:
            with torch.no_grad():
                generator.eval()
                for n, (lcimg, lhimg, lsimg) in enumerate(landscape_dataloader):
                    lcimg, lhimg, lsimg = lcimg.to(device), lhimg.to(device), lsimg.to(device)
                    mask = mask_gen()
                    hint = torch.cat((lhimg * mask, mask), 1)
                    g_recon = generator(lsimg, hint)
                    result = torch.cat((lcimg[0], g_recon[0]), 2)
                    path = os.path.join(args.name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + args.name + '_train_' + str(n + 1) + '.png')
                    plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                    if n == 4:
                        break

                for n, (lcimg, lhimg, lsimg) in enumerate(landscape_test_dataloader):
                    lcimg, lhimg, lsimg = lcimg.to(device), lhimg.to(device), lsimg.to(device)
                    mask = mask_gen()
                    hint = torch.cat((lhimg * mask, mask), 1)
                    g_recon = generator(lsimg, hint)
                    result = torch.cat((lcimg[0], g_recon[0]), 2)
                    path = os.path.join(args.name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + args.name + '_test_' + str(n + 1) + '.png')
                    plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                    if n == 4:
                        break

                torch.save(generator.state_dict(), os.path.join(args.name + '_results', 'generator_latest.pkl'))
                torch.save(generator.state_dict(), os.path.join(args.name + '_results', 'discriminator_latest.pkl'))

    total_time = time.time() - start_time
    train_hist['total_time'].append(total_time)

    print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
    print("Training finish!... save training results")

    torch.save(generator.state_dict(), os.path.join(args.name + '_results',  'generator_param.pkl'))
    torch.save(discriminator.state_dict(), os.path.join(args.name + '_results',  'discriminator_param.pkl'))
    with open(os.path.join(args.name + '_results',  'train_hist.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)


if __name__ == '__main__':
    main()
    