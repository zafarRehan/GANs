from gan import Generator, Discriminator
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from trainer import GANTrainer
from dataset import GANDataset
import os
import argparse, sys


class TrainingParameters:
    def __init__(self, args):
        self.lr = 3e-4
        self.z_dim = 64
        self.image_dim = 28 * 28 * 1  # 784
        self.batch_size = 128
        self.num_epochs = 50

        
        self.criterion = nn.BCELoss()
        self.writer_fake = SummaryWriter(f"logs/fake")
        self.writer_real = SummaryWriter(f"logs/real")
        self.image_write_path = f"{args.exp_name}/images"

        os.makedirs(self.image_write_path, exist_ok=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n", help="Experiment Name")
    args=parser.parse_args()


    tp = TrainingParameters(args)
    disc = Discriminator(tp.image_dim)
    gen = Generator(tp.z_dim, tp.image_dim)
    trainer = GANTrainer(gen, disc)
    gan_data = GANDataset()
    trainer.train(dataset=gan_data.dataset, train_params=tp)
