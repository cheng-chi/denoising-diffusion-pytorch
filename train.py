from typing_extensions import Required
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import click
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms, utils
from torch.utils.data import Dataset

class ImageOnlyDataset(Dataset):
    def __init__(self, in_dataset):
        self.in_dataset = in_dataset
    
    def __len__(self):
        return len(self.in_dataset)

    def __getitem__(self, idx):
        return self.in_dataset[idx][0]

@click.command()
@click.option('-g', '--gpu', type=int, default=0)
def main(gpu):
    image_size = 32
    image_shape = (image_size, image_size)
    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(image_shape),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    raw_dataset = CIFAR10(
        root='/local/crv/dataset/CIFAR10', 
        train=True,
        transform=transform,
        download=True)
    dataset = ImageOnlyDataset(raw_dataset)

    device = torch.device('cuda', gpu)
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    ).to(device)

    trainer = Trainer(
        diffusion,
        dataset,
        device,
        train_batch_size = 128,
        train_lr = 2e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True                        # turn on mixed precision
    )

    trainer.train()

if __name__ == '__main__':
    main()
