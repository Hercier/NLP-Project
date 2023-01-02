from torch import optim
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import Dataset,DataLoader
import os
import sys

from transformers import BartTokenizer

import utils

from model import Key2Text
from dataset import Key2TextDataset

device = torch.device('cuda')

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

model = torch.load('/home/zhuoyang/shizhong/NLP-Project/23.1.2/10.pth').to(device)
# model = torch.load('/home/zhuoyang/NLP-Project/bart_model.pth').to(device)
# model = torch.load('/home/zhuoyang/NLP-Project/15.pth').to(device)
# inputs = tokenizer("club LOT lounge place said", return_tensors="pt")["input_ids"].to(device)
inputs = tokenizer(" weather fine play grass", return_tensors="pt")["input_ids"].to(device)
# print(inputs)
# inputs = torch.tensor([[0, 1437, 10676, 987, 1947, 1437, 2]]).to(device)
# inputs = tokenizer(" night food feel", return_tensors="pt")["input_ids"].to(device)
out_ids = model.model.generate(inputs, num_beams=80, min_length=0, max_length=32)
out_ids_my=model.constrain_search(inputs.to("cuda"),beam_size=10, min_length=10, max_length=32)
print(out_ids_my)
print(tokenizer.batch_decode(out_ids_my, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
print(tokenizer.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# A = torch.tensor([[0, 20125, 2350, 40436, 14391, 326, 4822, 3785, 50118, 2385, 2]])
# print(tokenizer.batch_decode(A, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
