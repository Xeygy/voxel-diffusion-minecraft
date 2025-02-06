import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.utils.data import Dataset
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

from smalldiffusion import (
    ScheduleDDPM, samples, training_loop, MappedDataset, Unet3D,
    img_train_transform, img_normalize
)

def to_shape(a, shape):
    all_pads = []
    for idx, (tgt, cur) in enumerate(zip(shape, a.shape)):
        pad_amt = tgt - cur
        pad_before_after = (pad_amt//2, pad_amt//2 + pad_amt%2)
        all_pads.append(pad_before_after)
    return np.pad(a, all_pads,
                  mode = 'constant')

class VoxDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        
        if self.transform:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.data)

def visualize_house(vox, im_file=None):
    '''
    expects x, y, z, c
    '''
    vox = vox.cpu().numpy()
    vox = np.clip(vox, 0, 1)
    solid = vox[:, :, :, -1] != 0
    vox[:, :, :, -1] = 0.7 * (vox[:, :, :, -1] > 0)

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(solid,
              facecolors=vox)
    plt.savefig(im_file, transparent=True) 

def save_samples(curr_ep, model, schedule, ema, accelerator, sample_batch_size, sdir='saved'):
    # Sample
    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                        batchsize=sample_batch_size, accelerator=accelerator)
        os.makedirs(sdir, exist_ok=True)
        # save_image(img_normalize(make_grid(x0)), f'{sdir}/samples{curr_ep}.png')
        # reshape b, c, x, y, z -> b, x, y, z, c
        x0 = x0.permute(0, 2, 3, 4, 1)
        visualize_house(x0[0], f'{sdir}/samples1_{curr_ep}.png')
        visualize_house(x0[1], f'{sdir}/samples2_{curr_ep}.png')
        torch.save(model.state_dict(), f'{sdir}/checkpoint{curr_ep}.pth')

def main(train_batch_size=32, epochs=300, sample_batch_size=64):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    NAME = 'vox_3d_' + timestr
    # Setup
    a = Accelerator()

    houses_np = np.load('data/2104_houses_sub_30.npy')
    houses_np = to_shape(houses_np, (2104, 32, 32, 32, 4))
    # b, x, y, z, c -> b, c, x, y, z
    houses_np = np.transpose(houses_np, (0, 4, 1, 2, 3))
    houses_np = houses_np.astype(np.float32)
    dataset = VoxDataset(houses_np)
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
    model = Unet3D(in_dim=32, in_ch=4, out_ch=4, ch=64)

    # Train
    ema = EMA(model.parameters(), decay=0.999)
    ema.to(a.device)

    epoch_save, prev_ep = True, 0
    for ns in training_loop(loader, model, schedule, epochs=epochs, lr=1e-3, accelerator=a):
        ns.pbar.set_postfix(loss={ns.loss.item():.5})
        ema.update()

        curr_ep = ns.pbar.n
        if epoch_save and curr_ep % 5 == 1:
            save_samples(curr_ep, model, schedule, ema, a, sample_batch_size, sdir=NAME)
            print()
            
        if curr_ep == prev_ep:
            epoch_save = False
        else:
            epoch_save = True
        prev_ep = curr_ep
    
    save_samples(curr_ep, model, schedule, ema, a, sample_batch_size, sdir=NAME)

if __name__=='__main__':
    main()