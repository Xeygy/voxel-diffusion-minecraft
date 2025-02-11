import torch
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

def main(train_batch_size=256, epochs=300, sample_batch_size=64):
    # Setup
    a = Accelerator()
    # 32 x 32 with a 16x16 square of a random color in the center
    squares = torch.zeros(10000, 3, 32, 32)
    for i in range(10000):
        color = torch.rand(3).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = torch.randint(8, 24, (2,))
        squares[i, :, x[0]:x[0]+16, x[1]:x[1]+16] = color
    squares = squares * 2 - 1
    save_image(make_grid(squares[:64]), f'target2.png')

    loader = DataLoader(squares, batch_size=train_batch_size, shuffle=True)
    schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
    model = DiT(in_dim=32, channels=3, patch_size=2, depth=12, head_dim=64, num_heads=6, mlp_ratio=4.0)

    # Train
    ema = EMA(model.parameters(), decay=0.99)
    ema.to(a.device)
    epoch_save, prev_ep = True, 0
    for ns in training_loop(loader, model, schedule, epochs=epochs, lr=1e-3, accelerator=a):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        ema.update()

        curr_ep = ns.pbar.n
        if epoch_save and curr_ep % 5 == 1:
            with ema.average_parameters():
                *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                                batchsize=sample_batch_size, accelerator=a)
                save_image(img_normalize(make_grid(x0)), f'samples{curr_ep}.png')
                torch.save(model.state_dict(), f'checkpoint{curr_ep}.pth')
            
        if curr_ep == prev_ep:
            epoch_save = False
        else:
            epoch_save = True
        prev_ep = curr_ep

    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                        batchsize=sample_batch_size, accelerator=a)
        save_image(img_normalize(make_grid(x0)), f'samples_end.png')
        torch.save(model.state_dict(), f'checkpoint_end.pth')



if __name__=='__main__':
    main()
