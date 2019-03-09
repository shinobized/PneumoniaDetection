import torch
import torch.nn.functional as F
import numpy as np
import utils
import config
from model import MyAlexNet
from PIL import Image
from dataloader import transform
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import time

def get_best_model():
    params = utils.load_params()
    model = MyAlexNet(params).to(device=params.device)
    checkpoint = os.path.join(config.model_dir, 'best.pth.tar')
    utils.load_checkpoint(checkpoint, model, params)
    return model, params

def predict(X, y=None):
    model, params = get_best_model()
    model.eval()
    with torch.no_grad():
        X = torch.cat([transform(Image.fromarray(x)).unsqueeze(0) for x in X])
        output = model(X)
        pre_output = model.pre_output
        prob = F.softmax(output, dim=1)

    prob = prob.data.cpu().numpy()
    pre_output = pre_output.data.cpu().numpy()
    y_pred = np.argmax(prob, axis = 1)
    prob = prob[:, 1]

    if y is not None:
        print("Accuracy:", np.mean(y == y_pred))
    return y_pred, prob, pre_output

def show(X, y, y_pred, prob):
    N = len(X)
    n_row = N // 5 + 1
    plt.figure(figsize = (10,10))
    for i in range(N):
        plt.subplot(n_row, 5, i+1)
        plt.imshow(np.array(X[i]))
        plt.axis('off')
        pred = 'norm' if y_pred[i] == 0 else 'pneu'
        true = 'norm' if y[i] == 0 else 'pneu'
        plt.title("P: {} ({:.2%}), T: {}".format(pred, prob[i], true), fontsize=10)
    plt.tight_layout()
    plt.show()

X, y = [], []
for label, name in enumerate(['normal', 'pneumonia']):
    for i in np.random.randint(0, 50, 10):
        img = "./data/test/{}/{}.jpg".format(name, i)
        X.append(np.array(Image.open(img).convert('RGB')))
        y.append(label)

y_pred, prob, pre_output = predict(X, y)
show(X, y, y_pred, prob)