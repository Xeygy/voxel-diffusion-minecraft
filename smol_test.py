import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from smalldiffusion import (
    ScheduleDDPM, samples, training_loop, MappedDataset, DiT,
    img_train_transform, img_normalize
)

def save_samples(curr_ep, model, schedule, ema, accelerator, sample_batch_size, sdir='saved'):
    # Sample
    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                        batchsize=sample_batch_size, accelerator=accelerator)
        save_dir = f'{sdir}/{curr_ep}'
        os.makedirs(save_dir, exist_ok=True)
        save_image(img_normalize(make_grid(x0)), f'{save_dir}/samples.png')
        torch.save(model.state_dict(), f'{save_dir}/checkpoint.pth')

def main(train_batch_size=1024, epochs=300, sample_batch_size=64):
    # Setup
    a = Accelerator()
    dataset = MappedDataset(FashionMNIST('datasets', train=True, download=True,
                                         transform=img_train_transform),
                            lambda x: x[0])
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
    model = DiT(in_dim=28, channels=1, patch_size=2, depth=6, head_dim=32, num_heads=6, mlp_ratio=4.0)

    # Train
    ema = EMA(model.parameters(), decay=0.99)
    ema.to(a.device)

    epoch_save, prev_ep = True, 0
    for ns in training_loop(loader, model, schedule, epochs=epochs, lr=1e-3, accelerator=a):
        ns.pbar.set_postfix(loss={ns.loss.item():.5})
        ema.update()

        curr_ep = ns.pbar.n
        if epoch_save:
            save_samples(curr_ep, model, schedule, ema, a, sample_batch_size, sdir='fmnist')
            
        if curr_ep == prev_ep:
            epoch_save = False
        else:
            epoch_save = True
        prev_ep = curr_ep
    
    save_samples(model, schedule, ema, a, sample_batch_size)

if __name__=='__main__':
    main()