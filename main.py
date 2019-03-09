import torch
import torch.optim as optim
import config
from dataloader import fetch_dataloaders
from train import train_evaluate
from model import MyAlexNet, loss_fn, metric
import utils
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--restore', default=None)
parser.add_argument('--train', default=False, action='store_true')
args = parser.parse_args()







