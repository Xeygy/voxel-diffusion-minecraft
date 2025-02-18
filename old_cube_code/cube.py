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
    ScheduleDDPM, ScheduleLogLinear, samples, training_loop, MappedDataset, Unet3D, 
    img_train_transform, img_normalize, Scaled, DiT,
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
    expects x, y, z, 4
    '''
    vox = vox.cpu().numpy()
    vox = np.clip(vox, 0, 1)
    # expand to x, y, z, 4 by copying the last channel
    solid = vox[:, :, :, -1] != 0
    vox[:, :, :, -1] = 0.7 * (vox[:, :, :, -1] > 0)

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)
    ax.voxels(solid,
              facecolors=vox)
    plt.savefig(im_file, transparent=True) 
    plt.close()

def save_samples(curr_ep, model, schedule, ema, accelerator, sample_batch_size, sdir='saved'):
    # Sample
    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                        batchsize=sample_batch_size, accelerator=accelerator)
        os.makedirs(sdir, exist_ok=True)
        print(x0.min(), x0.max())

        # reshape b, c, x, y -> b, 4, x, y, z
        x0 = x0.permute(0, 2, 3, 1)
        x0 = x0.reshape(sample_batch_size, 32, 32, 32, 4)

        # from (-1, 1) to (0, 1)
        x0 = (x0 + 1) / 2
        print(x0.shape)
        visualize_house(x0[0], f'{sdir}/samples_{curr_ep}_1.png')
        visualize_house(x0[1], f'{sdir}/samples_{curr_ep}_2.png')
        torch.save(model.state_dict(), f'{sdir}/checkpoint_{curr_ep}.pth')

def main(train_batch_size=32, epochs=300, sample_batch_size=64):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    NAME = 'cube' + timestr

    # Setup
    a = Accelerator()

    # 32 x 32 x 32 with a 16x16x16 cube of a random color in the center
    DSIZE = 2000
    cubes = torch.zeros(DSIZE, 4, 32, 32, 32)
    for i in range(DSIZE):
        x = torch.randint(4, 12, (3,))
        color = torch.rand(4)
        color[3] = 1
        color = color.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        cubes[i, :, x[0]:x[0]+16, x[1]:x[1]+16, x[2]:x[2]+16] = color
    os.makedirs(NAME, exist_ok=True)
    cubes = cubes * 2 - 1

    # flatten the z dimension into the channel dimension
    # b, 4, x, y, z -> b, 4*z, x, y 
    cubes = cubes.permute(0, 1, 4, 2, 3)
    cubes = cubes.reshape(DSIZE, 4*32, 32, 32)

    # test visualization
    sample = cubes[0]
    # reshape c, x, y -> 4, x, y, z
    sample = sample.reshape(4, 32, 32, 32)
    sample = sample.permute(1, 2, 3, 0) 
    sample = (sample + 1) / 2
    visualize_house(sample, f'{NAME}/cube.png')
    loader = DataLoader(cubes, batch_size=train_batch_size, shuffle=True)

    schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.5, N=1000)
    model = DiT(in_dim=32, channels=128, patch_size=2, depth=12, head_dim=64, num_heads=6, mlp_ratio=4.0)

    # Train
    ema = EMA(model.parameters(), decay=0.99)
    ema.to(a.device)
    epoch_save, prev_ep = True, 0
    for ns in training_loop(loader, model, schedule, epochs=epochs, lr=1e-3, accelerator=a):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        ema.update()

        curr_ep = ns.pbar.n
        if epoch_save and curr_ep % 10 == 1:
            save_samples(curr_ep, model, schedule, ema, a, sample_batch_size, sdir=NAME)
            print()
            
        if curr_ep == prev_ep:
            epoch_save = False
        else:
            epoch_save = True
        prev_ep = curr_ep



if __name__=='__main__':
    main()