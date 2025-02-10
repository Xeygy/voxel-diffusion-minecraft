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
    solid = vox[:, :, :, -1] != 0
    vox[:, :, :, -1] = 0.8 * (vox[:, :, :, -1] > 0)

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)
    ax.voxels(solid,
              facecolors=vox)
    plt.savefig(im_file, transparent=True) 
    plt.close()
    
def save_samples(curr_ep, model, schedule, ema, accelerator, sample_batch_size, sdir='saved', render=False):
    # Sample
    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                        batchsize=sample_batch_size, accelerator=accelerator)
        os.makedirs(sdir, exist_ok=True)
        print(x0.min(), x0.max())

        # from (-1, 1) to (0, 1)
        sample = x0[0]
        sample = (sample + 1) / 2
        # image
        plt.imshow(sample.cpu().permute(1, 2, 0).numpy().clip(0, 1))
        plt.savefig(f'{sdir}/slice_{curr_ep}_1.png')
        plt.close()
        
        if render:
            # 3d (the render takes about 60 seconds)
            sample = from_mvi(sample)
            # c, x, y, z -> x, y, z, c
            sample = sample.permute(1, 2, 3, 0)
            visualize_house(sample, f'{sdir}/samples_{curr_ep}_1.png')
        torch.save(model.state_dict(), f'{sdir}/checkpoint_{curr_ep}.pth')

def from_mvi(mvi):
    '''
    expects c, x*z, y*z
    converts to c, x, y, z
    '''
    assert mvi.shape == (4, 192, 192)
    cxyz = torch.zeros((4, 32, 32, 32))
    for i in range(cxyz.shape[-1]):
        row = (i // 6) 
        col = (i % 6)
        cxyz[:, :, :, i] = mvi[:, row*32:(row+1)*32, col*32:(col+1)*32]
    return cxyz

def main(train_batch_size=8, epochs=300, sample_batch_size=2):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    NAME = 'cubemvi' + timestr

    # Setup
    a = Accelerator()

    # 32 x 32 x 32 with a 16x16x16 cube of a random color in the center
    # b, x, y, z, c
    DSIZE = 2000
    cubes = torch.zeros(DSIZE, 32, 32, 32, 4)
    for i in range(DSIZE):
        color = torch.rand(4)
        color[-1] = 1
        x = torch.randint(4, 12, (3,))
        cubes[i, x[0]:x[0]+16, x[1]:x[1]+16, x[2]:x[2]+16, :] = color
    cubes = cubes * 2 - 1
    # b, x, y, z, c -> b, z, c, x, y; slice along z
    cubes = cubes.permute(0, 3, 4, 1, 2)
    os.makedirs(NAME, exist_ok=True)

    print("slicing")
    slices = [make_grid(cube, nrow=6, padding=0) for cube in cubes]

    # test visualization
    sample = slices[0]
    sample = (sample + 1) / 2
    # c, x, y -> x, y, c
    plt.imshow(slices[0].permute(1, 2, 0))
    plt.imshow(make_grid(slices[:16], nrow=4).permute(1, 2, 0))
    plt.savefig(f'{NAME}/cube_slice.png')

    sample = from_mvi(sample)
    # c, x, y, z -> x, y, z, c
    sample = torch.tensor(sample)
    sample = sample.permute(1, 2, 3, 0)
    visualize_house(sample, f'{NAME}/cube.png')
    loader = DataLoader(slices, batch_size=train_batch_size, shuffle=True)

    schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
    model = DiT(in_dim=192, channels=4, patch_size=4, depth=12, head_dim=64, num_heads=12, mlp_ratio=4.0)

    # Train
    ema = EMA(model.parameters(), decay=0.99)
    ema.to(a.device)
    
    epoch_save, prev_ep = True, 0
    for ns in training_loop(loader, model, schedule, epochs=epochs, lr=1e-3, accelerator=a):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        ema.update()

        curr_ep = ns.pbar.n
        if epoch_save and curr_ep % 5 == 1:
            render = curr_ep % 20 == 1 or (curr_ep > 100 and curr_ep % 10 == 1) 
            save_samples(curr_ep, model, schedule, ema, a, sample_batch_size, sdir=NAME, render=render)
            print()
            
        if curr_ep == prev_ep:
            epoch_save = False
        else:
            epoch_save = True
        prev_ep = curr_ep

if __name__=='__main__':
    main()