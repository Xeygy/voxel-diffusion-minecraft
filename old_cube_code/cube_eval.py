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
from PIL import Image

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
    expects x, y, z, 1
    '''
    vox = vox.cpu().numpy()
    vox = np.clip(vox, 0, 1)
    # expand to x, y, z, 4 by copying the last channel
    vox = np.repeat(vox, 4, axis=-1)
    threshold = 0.5
    solid = vox[:, :, :, -1] > threshold
    vox[:, :, :, -1] = 0.7 * (vox[:, :, :, -1] > threshold)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)
    ax.voxels(solid,
              facecolors=vox)
    fig.canvas.draw()
    im = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
    plt.close()
    return im


def save_samples(curr_ep, model, schedule, ema, accelerator, sample_batch_size, sdir='saved'):
    # Sample
    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                        batchsize=sample_batch_size, accelerator=accelerator)
        os.makedirs(sdir, exist_ok=True)
        print(x0.min(), x0.max())
        # reshape b, x, y, z -> b, 1, x, y, z
        x0 = x0.unsqueeze(-1)
        # from (-1, 1) to (0, 1)
        x0 = (x0 + 1) / 2
        print(x0.shape)
        visualize_house(x0[0], f'{sdir}/samples_{curr_ep}_1.png')
        visualize_house(x0[1], f'{sdir}/samples_{curr_ep}_2.png')
        torch.save(model.state_dict(), f'{sdir}/checkpoint_{curr_ep}.pth')

def main(train_batch_size=32, epochs=300, sample_batch_size=64):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    NAME = 'qeval' + timestr

    # Setup
    a = Accelerator()

    # 32 x 32 x 32 with a 16x16x16 cube of a random color in the center
    DSIZE = 2000
    cubes = torch.zeros(DSIZE, 1, 32, 32, 32)
    for i in range(DSIZE):
        x = torch.randint(4, 12, (3,))
        cubes[i, :, x[0]:x[0]+16, x[1]:x[1]+16, x[2]:x[2]+16] = 1
    os.makedirs(NAME, exist_ok=True)
    cubes = cubes * 2 - 1

    # flatten the z dimension into the channel dimension
    # b, 1, x, y, z -> b, x, y, c
    cubes = cubes.squeeze(1)

    # test visualization
    sample = cubes[0].unsqueeze(-1)
    sample = (sample + 1) / 2
    loader = DataLoader(cubes, batch_size=train_batch_size, shuffle=True)

    print(cubes.shape)

    schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
    model = DiT(in_dim=32, channels=32, patch_size=2, depth=12, head_dim=64, num_heads=6, mlp_ratio=4.0)

    # load checkpoints
    # make a gif of the last 30 checkpoints
    images = []
    for n in range(30):
        model.load_state_dict(torch.load(f"fmnist/good_qube/checkpoint_{n * 10 + 1}.pth"))
        model.to(a.device)
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                            batchsize=sample_batch_size, accelerator=a)
        x0 = x0.unsqueeze(-1)
        # from (-1, 1) to (0, 1)
        x0 = (x0 + 1) / 2
        print("\r Saving image 1")
        im1 = visualize_house(x0[0], f'{NAME}/cube1.png')
        print("\r Saving image 2")
        im2 = visualize_house(x0[1], f'{NAME}/cube2.png')
        print("\r Saving image 3")
        im3 = visualize_house(x0[2], f'{NAME}/cube3.png')
        print("\r Saving image 4")
        im4 = visualize_house(x0[3], f'{NAME}/cube4.png')
        # concatenate images to a 2x2 grid
        grid = Image.new('RGB', (im1.width * 2, im1.height * 2))
        grid.paste(im1, (0, 0))
        grid.paste(im2, (im1.width, 0))
        grid.paste(im3, (0, im1.height))
        grid.paste(im4, (im1.width, im1.height))
        grid.save(f'{NAME}/cube_{n}.png')
    



if __name__=='__main__':
    main()