from gan import Generator, Discriminator
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from trainer import GANTrainer
from dataset import GANDataset
import os
import argparse, sys


class TrainingParameters:
    def __init__(self, args):
        self.LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 64
        self.CHANNELS_IMG = 1
        self.NOISE_DIM = 100
        self.NUM_EPOCHS = 5
        self.FEATURES_DISC = 64
        self.FEATURES_GEN = 64
        
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
    disc = Discriminator(tp.CHANNELS_IMG, tp.FEATURES_DISC)
    gen = Generator(tp.NOISE_DIM, tp.CHANNELS_IMG, tp.FEATURES_GEN)
    trainer = GANTrainer(gen, disc)
    gan_data = GANDataset(tp)
    trainer.train(dataset=gan_data.dataset, train_params=tp)
