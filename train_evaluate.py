from tqdm import tqdm
import numpy as np
import torch
import config
import os
from model import MyAlexNet, loss_fn, metric
from dataloader import fetch_dataloaders
import torch.optim as optim
import argparse
import utils

def train(model, dataloader, optimizer, loss_fn, metric, params):
    model.train()

    loss_avg = utils.RunningAverage()
    output = []
    y = []
    with tqdm(total=len(dataloader)) as t:
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(params.device)
            y_batch = y_batch.to(params.device)

            output_batch = model(X_batch)
            loss = loss_fn(output_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())
            y.append(y_batch.data.cpu().numpy())
            output.append(output_batch.data.cpu().numpy())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    output = np.concatenate(output, axis=0)
    y = np.concatenate(y, axis=0)
    metric_score = metric(output, y)  
    avg_loss = loss_avg()
    return avg_loss, metric_score

def evaluate(model, dataloader, loss_fn, metric, params):
    model.eval()

    loss_avg = utils.RunningAverage()
    output = []
    y = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(params.device)
            y_batch = y_batch.to(params.device)

            output_batch = model(X_batch)
            loss = loss_fn(output_batch, y_batch)
            loss_avg.update(loss.item())
            
            y.append(y_batch.data.cpu().numpy())
            output.append(output_batch.data.cpu().numpy())
    
    avg_loss = loss_avg() 
    output = np.concatenate(output, axis=0)
    y = np.concatenate(y, axis=0)
    metric_score = metric(output, y)   
    return avg_loss, metric_score

def train_evaluate(model, optimizer, dataloader_train, dataloader_val, loss_fn, metric, 
                    params, model_dir, restore_file=None):
    best_val_metric = float('-inf')
    start_epoch = 0

    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        print("Restoring parameters from {}".format(restore_path))
        checkpoint = utils.load_checkpoint(restore_path, model, params, optimizer)
        best_val_metric = checkpoint['best_val_metric']
        start_epoch = checkpoint['epoch']

    for e in range(params.num_epochs):
        print("epoch:", start_epoch + e + 1)
        tr_loss, tr_metric = train(model, dataloader_train, optimizer, loss_fn, metric, params)
        val_loss, val_metric = evaluate(model, dataloader_val, loss_fn, metric, params)
        
        is_best = (val_metric >= best_val_metric)

        utils.save_checkpoint({'epoch': start_epoch + e + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict(),
                               'best_val_metric': best_val_metric},
                               is_best=is_best,
                               checkpoint=model_dir)
        if is_best:
            best_val_metric = val_metric

        tqdm.write('val_acc: {}, best_val_acc: {}'.format(val_metric, best_val_metric))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--evaluate', default=False, action='store_true')
    args = parser.parse_args()

    params = utils.load_params()
    torch.manual_seed(config.root_seed)
    np.random.seed(config.root_seed)
    if params.cuda: torch.cuda.manual_seed(config.root_seed)

    dataloaders = fetch_dataloaders(config.data_dir, params)
    model = MyAlexNet(params).to(device=params.device)

    if args.train:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)
        restore_file = 'last' if args.restore else None
        train_evaluate(model, optimizer, dataloaders['train'], dataloaders['val'], loss_fn, metric, 
                        params, config.model_dir, restore_file=restore_file)
    if args.evaluate:
        checkpoint = os.path.join(config.model_dir, 'best.pth.tar')
        utils.load_checkpoint(checkpoint, model, params)
        test_loss, test_metric = evaluate(model, dataloaders['test'], loss_fn, metric, params)
        print("test acc:", test_metric)