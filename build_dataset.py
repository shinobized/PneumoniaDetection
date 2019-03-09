"""Build dataset"""
import numpy as np
import os
import config
from PIL import Image
from shutil import copyfile
import utils

norm_data_dir = "./data/raw/normal"
pneu_dara_dir = "./data/raw/pneumonia"
max_size = 500

# get norm and pneu
norm = [os.path.join(norm_data_dir, file) for file in os.listdir(norm_data_dir)]
pneu = [os.path.join(pneu_dara_dir, file) for file in os.listdir(pneu_dara_dir)]

# for filepath in norm:
#     try:
#         img=Image.open(filepath)
#         img.verify()
#         img.close()
#     except: 
#         print("file broken")
#         utils.remove(filepath)

# random split
np.random.seed(1)
np.random.shuffle(norm)
np.random.shuffle(pneu)

N = min(len(norm), max_size, len(pneu))
n_tr = int(N*0.8)
n_val = int(N*0.1)

norm_split = {}
pneu_split = {}
norm_split['train'], norm_split['val'], norm_split['test'] = np.split(norm, [n_tr, n_tr+n_val])
pneu_split['train'], pneu_split['val'], pneu_split['test'] = np.split(pneu, [n_tr, n_tr+n_val])

# save images
def save_images(images, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, img in enumerate(images):
        save_path = os.path.join(save_dir, "{}.jpg".format(i))
        copyfile(img, save_path)

for split in ['train', 'val', 'test']:
    utils.remove(os.path.join(config.data_dir, split))
    norm_dir = os.path.join(config.data_dir, split, 'normal')
    pneu_dir = os.path.join(config.data_dir, split, 'pneumonia')
    save_images(norm_split[split], norm_dir)
    save_images(pneu_split[split], pneu_dir)