import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import math
import numpy as np
import cv2


class GANTrainer:

    def __init__(self, generator, discriminator):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

    def train(self, dataset, train_params):
        loader = DataLoader(dataset, batch_size=train_params.batch_size, shuffle=True)
        opt_disc = optim.Adam(self.discriminator.parameters(), lr=train_params.lr)
        opt_gen = optim.Adam(self.generator.parameters(), lr=train_params.lr)
        fixed_noise = torch.randn((train_params.batch_size, train_params.z_dim)).to(self.device)

        tqdm_length = int(math.ceil(dataset.data.shape[0]/train_params.batch_size))
        bar_format ='{l_bar}{bar:20}{r_bar}{bar:-20b}'

        step = 0
        lossD = 999
        lossG = 999
        for epoch in range(train_params.num_epochs):
            pbar = tqdm(enumerate(loader), total=tqdm_length, bar_format=bar_format)
            pbar.set_description(f"Epoch [{epoch}/{train_params.num_epochs}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}        ")
            for batch_idx, (real, _) in pbar:
                real = real.view(-1, 784).to(self.device)
                batch_size = real.shape[0]

                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                noise = torch.randn(batch_size, train_params.z_dim).to(self.device)
                fake = self.generator(noise)
                disc_real = self.discriminator(real).view(-1)
                lossD_real = train_params.criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = self.discriminator(fake).view(-1)
                lossD_fake = train_params.criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2
                self.discriminator.zero_grad()
                lossD.backward(retain_graph=True)
                opt_disc.step()

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                # where the second option of maximizing doesn't suffer from
                # saturating gradients
                output = self.discriminator(fake).view(-1)
                lossG = train_params.criterion(output, torch.ones_like(output))
                self.generator.zero_grad()
                lossG.backward()
                opt_gen.step()

                if batch_idx == 0:
                    self.save_log(train_params=train_params, real=real, noise=fixed_noise, step=step)
                    step += 1


    def save_log(self, train_params, real, noise, step):
        # write info to log and save images
        with torch.no_grad():
            fake = self.generator(noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            train_params.writer_fake.add_image(
                "Mnist Fake Images", img_grid_fake, global_step=step
            )
            train_params.writer_real.add_image(
                "Mnist Real Images", img_grid_real, global_step=step
            )

            self.save_images(img_grid_real, img_grid_fake, step, train_params)

    def save_images(self, real, fake, step, train_params):
        real = real.permute(1, 2, 0)
        fake = fake.permute(1, 2, 0)
        img = np.hstack([real, fake])
        img = img * 255
        print(step, real.shape, fake.shape)
        cv2.imwrite(f'{train_params.image_write_path}/step_{step}.png', img)