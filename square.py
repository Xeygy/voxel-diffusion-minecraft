import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from smalldiffusion import (
    ScheduleLogLinear, samples, training_loop, MappedDataset, Unet, Scaled,
    img_train_transform, img_normalize
)

def main(train_batch_size=1024, epochs=300, sample_batch_size=64):
    # Setup
    a = Accelerator()
    # 32 x 32 with a 16x16 square of a random color in the center
    DS_SIZE = 2000
    squares = torch.zeros(DS_SIZE, 4, 32, 32)
    for i in range(DS_SIZE):
        color = torch.rand(4)
        color[3] = 1
        color = color.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = torch.randint(4, 12, (2,))
        squares[i, :, x[0]:x[0]+16, x[1]:x[1]+16] = color
    save_image(make_grid(squares[:64]), f'target.png')
    squares = squares * 2 - 1
    
    loader = DataLoader(squares, batch_size=train_batch_size, shuffle=True)
    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=800)
    model = Scaled(Unet)(32, 4, 4, ch=64, ch_mult=(1, 1, 2), attn_resolutions=(14,))

    # Train
    ema = EMA(model.parameters(), decay=0.999)
    ema.to(a.device)
    epoch_save, prev_ep = True, 0
    for ns in training_loop(loader, model, schedule, epochs=epochs, lr=7e-4, accelerator=a):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        ema.update()

        curr_ep = ns.pbar.n
        if epoch_save and curr_ep % 5 == 1:
            with ema.average_parameters():
                *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                                batchsize=sample_batch_size, accelerator=a)
                save_image(img_normalize(make_grid(x0)), f'samples{curr_ep}.png')
            
        if curr_ep == prev_ep:
            epoch_save = False
        else:
            epoch_save = True
        prev_ep = curr_ep


if __name__=='__main__':
    main()
