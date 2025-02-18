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
    plt.savefig(im_file) 


def main(train_batch_size=32, epochs=300, sample_batch_size=64):
    NAME = 'vox_3d'
    # Setup
    a = Accelerator()

    houses_np = np.load('data/2104_houses_sub_30.npy')
    houses_np = to_shape(houses_np, (2104, 32, 32, 32, 4))
    # b, x, y, z, c -> b, c, x, y, z
    houses_np = np.transpose(houses_np, (0, 4, 1, 2, 3))
    houses_np = houses_np.astype(np.float32)
    # get min and max x,y,z for non-zero values
    house = houses_np[0]
    for i in range(3):
        non_zero = np.nonzero(houses_np[:, :, :, :, i])
        min_i, max_i = np.min(non_zero), np.max(non_zero)
        houses_np[:, :, :, :, i] = (houses_np[:, :, :, :, i] - min_i) / (max_i - min_i)
    min_x, max_x = np.min(houses_np[:, :, :, :, 0]), np.max(houses_np[:, :, :, :, 0])


    dataset = VoxDataset(houses_np)
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    one = next(iter(loader))
    one = one.permute(0, 2, 3, 4, 1)
    print(one.shape, one.min(), one.max())
    visualize_house(one[1], 'image.png')
    # schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
    # model = Unet3D(in_dim=32, in_ch=4, out_ch=4, ch=64)

    # state_dict = torch.load('vox_3d/checkpoint50.pth')
    # model.load_state_dict(state_dict)
    # model.to(a.device)

    # *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
    #                     batchsize=sample_batch_size, accelerator=a)
    # # reshape b, c, x, y, z -> b, x, y, z, c
    # x0 = x0.permute(0, 2, 3, 4, 1)
    # for i in range(10):
    #     visualize_house(x0[i], f'50image{i}.png')

if __name__=='__main__':
    main()