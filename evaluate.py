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

model = torch.load('/home/zhuoyang/NLP-Project/2.pth').to(device)
# inputs = tokenizer("club LOT lounge place said", return_tensors="pt")["input_ids"].to(device)
inputs = tokenizer(" thanks want real miss ", return_tensors="pt")["input_ids"].to(device)
out_ids = model.model.generate(inputs, num_beams=10, min_length=0, max_length=32)
out_ids_my=model.constrain_search(inputs.to("cuda"),beam_size=10, min_length=10, max_length=32)
print(out_ids_my)
print(tokenizer.batch_decode(out_ids_my, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
print(tokenizer.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
dataset_test = Key2TextDataset(split='test')
data_loader_test = DataLoader(dataset_test, collate_fn=dataset_test.collate_fn, batch_size=64, pin_memory=True)
suc_cnt=0
tot_cnt=1
for item in data_loader_test:
    key_b=item['key_token'].cuda()
    out_ids=model.model.generate(key_b, num_beams=10, min_length=0, max_length=32)
    tp_suc,tp_tot=utils.get_percentage(key_b.cpu(),out_ids.cpu())
    print(tokenizer.batch_decode(key_b.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    print(tokenizer.batch_decode(out_ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    print(tokenizer.batch_decode(item['target_token'].cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    suc_cnt+=tp_suc
    tot_cnt+=tp_tot
print(f"success rate: {suc_cnt/tot_cnt*100}%")
