from torch import optim
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import Dataset,DataLoader
import os
import sys

import utils

from model import Key2Text
# from dataset_tp import Key2TextDataset
from dataset import Key2TextDataset

curdir = os.path.dirname(__file__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--num-epoch", default=50, type=int)
    parser.add_argument("--save-interval", default=2, type=int)
    parser.add_argument("--save-dir", default=os.path.join(curdir, "models"))
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()
    return args

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for dic in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        if dic['target_token'] is None or len(dic['target_token'].shape)<2:
            continue
        input_ids, decoder_input_ids, labels = dic['key_token'], dic['target_token'][:,:-1], dic['target_token'][:,1:]

        input_ids, decoder_input_ids, labels = input_ids.to(device), decoder_input_ids.to(device), labels.to(device)
        loss = model.get_loss(input_ids, decoder_input_ids, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = input_ids.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        sys.stdout.flush()
    
def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for dic in metric_logger.log_every(data_loader, 20, header):
            input_ids, decoder_input_ids, labels = dic['key_token'], dic['target_token'][:,:-1], dic['target_token'][:,1:]
            input_ids, decoder_input_ids, labels = input_ids.to(device), decoder_input_ids.to(device), labels.to(device)
            loss = model.get_loss(input_ids, decoder_input_ids, labels)
            
            
            metric_logger.update(loss=loss.item())

def main(args):
    print(args)

    device = torch.device('cuda')

    print("Loading data")
    dataset_train = Key2TextDataset()
    data_loader_train = DataLoader(dataset_train, collate_fn=dataset_train.collate_fn, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    dataset_test = Key2TextDataset(split='valid')
    data_loader_test = DataLoader(dataset_test, collate_fn=dataset_test.collate_fn, batch_size=args.batch_size, pin_memory=True)

    print("Creating model")
    model = Key2Text()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    print("Start training")
    for epoch in range(args.start_epoch, args.num_epoch):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.print_freq)
        # torch.save(model, '/home/zhuoyang/NLP-Project/new_model/' + str(epoch) +'.pth')
        evaluate(model, data_loader_test, device)

if __name__ == "__main__":
    args = get_args()
    main(args)