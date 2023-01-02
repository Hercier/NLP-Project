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

model = torch.load('/home/zhuoyang/NLP-Project/new_model/15.pth').to(device)
#model = torch.load('/home/zhuoyang/NLP-Project/19.pth').to(device)
# inputs = tokenizer("club LOT lounge place said", return_tensors="pt")["input_ids"].to(device)
inputs = tokenizer(" black paint look", return_tensors="pt")["input_ids"].to(device)
#inputs = tokenizer(" place lucky enjoy", return_tensors="pt")["input_ids"].to(device)
print(inputs)
out_ids = model.model.generate(inputs, num_beams=50, min_length=0, max_length=32)
out_ids_my=model.constrain_search(inputs.to("cuda"),beam_size=10, min_length=10, max_length=32)
print(out_ids_my)
print(tokenizer.batch_decode(out_ids_my, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
print(tokenizer.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])