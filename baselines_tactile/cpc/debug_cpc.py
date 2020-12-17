from os.path import join
import torch
import torch.utils.data as data
from torchvision import transforms
from model import FCN_mse

batch_size = 32
include_actions = False
encoder = torch.load(join('out', name, 'encoder.pt'), map_location='cuda')
trans = torch.load(join('out', name, 'trans.pt'), map_location='cuda')

def apply_fcn_mse(img):
    o = fcn(img.cuda()).detach()
    return torch.clamp(2 * (o - 0.5), -1 + 1e-3, 1 - 1e-3)

def filter_background(x):
    x[:, (x < 0.3).any(dim=0)] = 0.0
    return x

def dilate(x):
    x = x.squeeze(0).numpy()
    x = grey_dilation(x, size=3)
    x = x[None, :, :]
    return torch.from_numpy(x)

if thanard_dset:
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
else:
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        filter_background,
        lambda x: x.mean(dim=0)[None, :, :],
        dilate,
        transforms.Normalize((0.5,), (0.5,)),
    ])

train_dset = ImagePairs(root=join(root, 'train_data'), include_actions=include_actions,
                        thanard_dset=thanard_dset, transform=transform, n_frames_apart=1)
train_loader = data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=2)
neg_train_dset = ImageFolder(join(root, 'train_data'), transform=transform)
neg_train_inf = data.DataLoader(neg_train_dset, batch_size=50, shuffle=True,
                                pin_memory=True, num_workers=2)
with torch.no_grad():
    batch = next(iter(train_loader))
    if include_actions:
        (obs, _, actions), (obs_pos, _, _) = batch
        actions = actions.cuda()
    else:
        (obs, _), (obs_pos, _) = batch
        actions = None
    bs = obs.shape[0]

    if thanard_dset:
        obs, obs_pos = apply_fcn_mse(obs), apply_fcn_mse(obs_pos)
        obs_neg = apply_fcn_mse(next(neg_train_inf)[0])
    else:
        obs, obs_pos = obs.cuda(), obs_pos.cuda() # b x 1 x 64 x 64
        obs_neg = next(iter(neg_train_inf))[0].cuda() # n x 1 x 64 x 64

    z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim
    z_neg = encoder(obs_neg)  # n x z_dim

    z = torch.cat((z, actions), dim=1) if include_actions else z
    z_next = trans(z)  # b x z_dim

    dist_pos = torch.norm(z_pos - z_next, dim=1)
    xx = (z_next ** 2).sum(1).unsqueeze(1)
    xy = torch.matmul(z_next, z_neg.t())
    yy = (z_neg ** 2).sum(1).unsqueeze(0)
    dist_neg = torch.sqrt(xx - 2 * xy + yy)

    print('pos', dist_pos)
    print('neg', dist_neg)

    print('pos dotprod', (z_next * z_pos).sum(1))
    print('neg dtoprod', torch.bmm(z_next.unsqueeze(1), z_neg.t().unsqueeze(0).repeat(bs, 1, 1)).cpu().numpy())


    z_next = z_next.unsqueeze(1)  # b x 1 x z_dim
    z_pos = z_pos.unsqueeze(2)  # b x z_dim x 1
    pos_log_density = torch.bmm(z_next, z_pos).squeeze(-1)  # b x 1

    z_neg = z_neg.t().unsqueeze(0).repeat(bs, 1, 1)  # b x z_dim x n
    neg_log_density = torch.bmm(z_next, z_neg).squeeze(1)  # b x n

    loss = torch.cat((torch.zeros(bs, 1).cuda(), neg_log_density - pos_log_density), dim=1)  # b x n+1
    loss = torch.logsumexp(loss, dim=1)
    print('loss', loss.cpu().numpy())
