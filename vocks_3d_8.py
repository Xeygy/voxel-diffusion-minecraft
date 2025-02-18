aimport torch
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
import argparse

from smalldiffusion import (
    ScheduleCosine, ScheduleLogLinear, samples, training_loop, MappedDataset, Unet3D, 
    img_train_transform, img_normalize, Scaled
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
    solid = vox[:, :, :, -1] > 0.8
    vox[:, :, :, -1] = 1 * vox[:, :, :, -1]

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(solid,
              facecolors=vox)
    ax.set_xlim(0, len(vox))
    ax.set_ylim(0, len(vox))
    ax.set_zlim(0, len(vox))
    plt.savefig(im_file, transparent=True) 
    plt.close()

def save_samples(curr_ep, model, schedule, ema, accelerator, sample_batch_size, sdir='saved'):
    # Sample
    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                        batchsize=sample_batch_size, accelerator=accelerator, fixed_seed=42)
        os.makedirs(sdir, exist_ok=True)
        print(x0.min(), x0.max())
        # reshape b, c, x, y, z -> b, x, y, z, c
        x0 = x0.permute(0, 2, 3, 4, 1)
        # from (-1, 1) to (0, 1)
        x0 = (x0 + 1) / 2
        for i in range(9):
            visualize_house(x0[i], f'{sdir}/samples_{curr_ep}_{i}.png')
        torch.save(model.state_dict(), f'{sdir}/checkpoint{curr_ep}.pth')

def main(train_batch_size=32, epochs=2000, sample_batch_size=64):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    NAME = 'vocks_' + timestr

    # Setup
    a = Accelerator()

    houses_np = np.load('data/2104_houses_sub_30.npy')
    houses_np = to_shape(houses_np, (2104, 32, 32, 32, 4))
    # b, x, y, z, c -> b, c, x, y, z
    houses_np = np.transpose(houses_np, (0, 4, 1, 2, 3))
    houses_np = houses_np.astype(np.float32)
    # from (0, 1) to (-1, 1)
    houses_np = (houses_np * 2) - 1
    print(houses_np.min(), houses_np.max())
    # crop 
    # houses_np = houses_np[:, :, 8:24, 8:24, 0:16]
    houses_np = houses_np[:, :, 12:20, 12:20, 0:8]
    houses_torch = torch.from_numpy(houses_np)
    os.makedirs(NAME, exist_ok=True)

    visualize_house((houses_torch[31].permute(1,2,3,0) + 1) / 2, f'{NAME}/sample1.png')
    visualize_house((houses_torch[81].permute(1,2,3,0) + 1) / 2, f'{NAME}/sample2.png')
    visualize_house((houses_torch[121].permute(1,2,3,0) + 1) / 2, f'{NAME}/sample3.png')
    
    houses_torch = houses_torch * 2 - 1

    dataset = VoxDataset(houses_torch)
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=30) #ScheduleCosine(max_beta=0.02, N=1000)
    model = Unet3D(in_dim=8, in_ch=4, out_ch=4, ch=64)

    print("sched", schedule.sample_sigmas(20))

    # Train
    ema = EMA(model.parameters(), decay=0.999)
    ema.to(a.device)

    epoch_save, prev_ep = True, 0
    for ns in training_loop(loader, model, schedule, epochs=epochs, lr=5e-4, accelerator=a):
        ns.pbar.set_postfix(loss={ns.loss.item():.5})
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
    
    save_samples(curr_ep, model, schedule, ema, a, sample_batch_size, sdir=NAME)

def generate_samples(model_path, train_batch_size=32, epochs=2000, sample_batch_size=64):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    NAME = 'vgen_' + timestr
    SAMPLE_CT = 25
    # Setup
    a = Accelerator()

    houses_np = np.load('data/2104_houses_sub_30.npy')
    houses_np = to_shape(houses_np, (2104, 32, 32, 32, 4))
    # b, x, y, z, c -> b, c, x, y, z
    houses_np = np.transpose(houses_np, (0, 4, 1, 2, 3))
    houses_np = houses_np.astype(np.float32)
    # from (0, 1) to (-1, 1)
    houses_np = (houses_np * 2) - 1
    print(houses_np.min(), houses_np.max())
    # crop 
    # houses_np = houses_np[:, :, 8:24, 8:24, 0:16]
    houses_np = houses_np[:, :, 12:20, 12:20, 0:8]
    houses_torch = torch.from_numpy(houses_np)
    os.makedirs(NAME, exist_ok=True)

    for i in range(SAMPLE_CT):
        visualize_house((houses_torch[i+200].permute(1,2,3,0) + 1) / 2, f'{NAME}/sample_{i}.png')
    
    houses_torch = houses_torch * 2 - 1

    dataset = VoxDataset(houses_torch)
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=30) #ScheduleCosine(max_beta=0.02, N=1000)
    model = Unet3D(in_dim=8, in_ch=4, out_ch=4, ch=64)
    model.load_state_dict(torch.load(model_path))
    model.to(a.device)

    *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                        batchsize=sample_batch_size, accelerator=a)
    # reshape b, c, x, y, z -> b, x, y, z, c
    x0 = x0.permute(0, 2, 3, 4, 1)
    # from (-1, 1) to (0, 1)
    x0 = (x0 + 1) / 2
    for i in range(SAMPLE_CT):
        visualize_house(x0[i], f'{NAME}/model_{i}.png')

if __name__=='__main__':
    # main()
    generate_samples('vocks_2025_8x8_houses/checkpoint1999.pth')
