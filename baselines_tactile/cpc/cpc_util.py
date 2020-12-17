from tqdm import tqdm
import numpy san p
import glob
from os.path import join, exists
import os

import torch
from torchvision.utils import save_image
from torchvision.datasets.folder import default_loader

from model import FCN_mse


fcn = FCN_mse(2).cuda()
fcn.load_state_dict(torch.load('/home/wilson/causal-infogan/data/FCN_mse'))
fcn.eval()

def apply_fcn_mse(img):
    o = fcn(img.cuda()).detach()
    return torch.clamp(2 * (o - 0.5), -1 + 1e-3, 1 - 1e-3)


def save_nearest_neighbors(encoder, train_loader, test_loader,
                           epoch, folder_name, k=100, thanard_dset=False,
                           metric='l2'):
    assert metric in ['l2', 'dotproduct']
    encoder.eval()
    train_batch = next(iter(train_loader))[0][:5]
    test_batch = next(iter(test_loader))[0][:5]

    with torch.no_grad():
        batch = torch.cat((train_batch, test_batch), dim=0)
        batch = apply_fcn_mse(batch) if thanard_dset else batch.cuda()
        z = encoder(batch) # 10 x z_dim
        zz = (z ** 2).sum(-1).unsqueeze(1) # z^Tz, 10 x 1

        pbar = tqdm(total=len(train_loader.dataset) + len(test_loader.dataset))
        pbar.set_description('Computing NN')
        dists = []
        for loader in [train_loader, test_loader]:
            for x, _ in loader:
                x = apply_fcn_mse(x) if thanard_dset else x.cuda()
                zx = encoder(x) # b x z_dim

                if metric == 'l2':
                    zzx = torch.matmul(z, zx.t()) # z_1^Tz_2, 10 x b
                    zxzx = (zx ** 2).sum(-1).unsqueeze(0) #zx^Tzx, 1 x b
                    dist = zz - 2 * zzx + zxzx # norm squared distance, 10 x b
                elif metric == 'dotproduct':
                    dist = -torch.matmul(z, zx.t())
                dists.append(dist.cpu())
                pbar.update(x.shape[0])
        dists = torch.cat(dists, dim=1) # 10 x dset_size
        topk = torch.topk(dists, k + 1, dim=1, largest=False)[1]

        pbar.close()

    folder_name = join(folder_name, 'nn_epoch{}'.format(epoch))
    if not exists(folder_name):
        os.makedirs(folder_name)

    train_size = len(train_loader.dataset)
    for i in range(10):
        imgs = []
        for idx in topk[i]:
            if idx >= train_size:
                imgs.append(train_loader.dataset[idx - train_size][0])
            else:
                imgs.append(test_loader.dataset[idx][0])
        imgs = torch.stack(imgs, dim=0)
        if thanard_dset:
            imgs = apply_fcn_mse(imgs).cpu()
        save_image(imgs * 0.5 + 0.5, join(folder_name, 'nn_{}.png'.format(i)), nrow=10)


def save_recon(decoder, train_loader, test_loader, encoder, epoch,
               folder_name, thanard_dset=False):
    decoder.eval()
    encoder.eval()

    train_batch = next(iter(train_loader))[0][:16]
    test_batch = next(iter(test_loader))[0][:16]
    if thanard_dset:
        train_batch, test_batch = apply_fcn_mse(train_batch), apply_fcn_mse(test_batch)
    else:
        train_batch, test_batch = train_batch.cuda(), test_batch.cuda()

    with torch.no_grad():
        train_z, test_z = encoder(train_batch), encoder(test_batch)
        train_recon, test_recon = decoder(train_z), decoder(test_z)

    real_imgs = torch.cat((train_batch, test_batch), dim=0)
    recon_imgs = torch.cat((train_recon, test_recon), dim=0)
    imgs = torch.stack((real_imgs, recon_imgs), dim=1)
    imgs = imgs.view(-1, *real_imgs.shape[1:]).cpu()

    folder_name = join(folder_name, 'reconstructions')
    if not exists(folder_name):
        os.makedirs(folder_name)

    filename = join(folder_name, 'recon_epoch{}.png'.format(epoch))
    save_image(imgs * 0.5 + 0.5, filename, nrow=8)


def save_interpolation(n_interp, decoder, start_images, goal_images,
                       encoder, epoch, folder_name):
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        z_start = encoder(start_images)
        z_goal = encoder(goal_images)
        z_dim = z_start.shape[1]

        lambdas = np.linspace(0, 1, n_interp + 2)
        zs = torch.stack([(1 - lambda_) * z_start + lambda_ * z_goal
                          for lambda_ in lambdas], dim=1)  # n x n_interp+2 x z_dim
        zs = zs.view(-1, z_dim)  # n * (n_interp+2) x z_dim

        imgs = decoder(zs).cpu()

    folder_name = join(folder_name, 'interpolations')
    if not exists(folder_name):
        os.makedirs(folder_name)

    filename = join(folder_name, 'interp_epoch{}.png'.format(epoch))
    save_image(imgs * 0.5 + 0.5, filename, nrow=n_interp + 2)


def save_run_dynamics(decoder, encoder, trans, start_images,
                      train_loader, epoch, folder_name, root,
                      include_actions=False, thanard_dset=False):
    decoder.eval()
    encoder.eval()

    dset = train_loader.dataset
    transform = dset.transform
    with torch.no_grad():
        actions, images = [], []
        n_ep = 5
        for i in range(n_ep):
            class_name = [name for name, idx in dset.class_to_idx.items() if idx == i]
            assert len(class_name) == 1
            class_name = class_name[0]

            a = np.load(join(root, 'train_data', class_name, 'actions.npy'))
            a = torch.FloatTensor(a)
            actions.append(a)
            ext = 'jpg' if thanard_dset else 'png'
            img_files = glob.glob(join(root, 'train_data', class_name, '*.{}'.format(ext)))
            img_files = sorted(img_files)
            image = torch.stack([transform(default_loader(f)) for f in img_files], dim=0)
            images.append(image)
        min_length = min(min([img.shape[0] for img in images]), 10)
        actions = [a[:min_length] for a in actions]
        images = [img[:min_length] for img in images]
        actions, images = torch.stack(actions, dim=0), torch.stack(images, dim=0)
        images = images.view(-1, *images.shape[2:])
        images = apply_fcn_mse(images) if thanard_dset else images.cuda()
        images = images.view(n_ep, min_length, *images.shape[1:])
        actions = actions.cuda()

        zs = [encoder(images[:, 0])]
        z_dim = zs[0].shape[1]
        for i in range(min_length - 1):
            inp = torch.cat((zs[-1], actions[:, i]), dim=1) if include_actions else zs[-1]
            zs.append(trans(inp))
        zs = torch.stack(zs, dim=1)
        zs = zs.view(-1, z_dim)
        recon = decoder(zs)
        recon = recon.view(n_ep, min_length, *images.shape[2:])

        all_imgs = torch.stack((images, recon), dim=1)
        all_imgs = all_imgs.view(-1, *all_imgs.shape[3:])

        folder_name = join(folder_name, 'run_dynamics')
        if not exists(folder_name):
            os.makedirs(folder_name)

        filename = join(folder_name, 'dyn_epoch{}.png'.format(epoch))
        save_image(all_imgs * 0.5 + 0.5, filename, nrow=min_length)
