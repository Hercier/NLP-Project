import transformers
import datasets
import datasets
from random import shuffle
from torch.utils.data import Dataset,DataLoader
import copy
import json
import os
import sys
import torch
import numpy as np
import nltk
import csv
import re
from transformers import (
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
#nltk.download('stopwords')
#print(nltk.corpus.stopwords.words('english'))
from numpy.random import default_rng
rng = default_rng()
model_name = "facebook/bart-base"
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
#dataset = datasets.load_dataset("bookcorpus")#
#dataset = datasets.load_from_disk("/home/zhuoyang/.cache/huggingface/datasets/bookcorpus/plain_text/1.0.0/eddee3cae1cc263a431aa98207d4d27fd8a73b0a9742f692af0e6c65afa4d75f/dataset_info.json") 
class yelpdata:
    def __init__(self):
        return
    def __call__(self,split):
        self.filename = os.path.join("./yelp_small", "{}.csv".format(split))
        #self.dataset=dataset[split]
        dataset=[]
        with open(self.filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                label = int(row[0])
                raw_text = row[1]
                dataset.append({'text':raw_text})
        return dataset
class newsdata:
    def __init__(self):
        self.dataset=[]
        self.filename=os.path.join("/home/zhuoyang", "news.2016.en.shuffled")
        f=open(self.filename,"r")
        
        for line in f:
            self.dataset.append({'text':line})
        self.train=int(0.7*len(self.dataset))
        self.valid=int(0.2*len(self.dataset))
        self.test=len(self.dataset)-self.train-self.valid
        self.train=self.train//300
        self.valid=self.valid//300
        self.test=self.test//300
    def __call__(self,split):
        if split=='train':
            return self.dataset[:self.train]
        if split=='valid':
            return self.dataset[self.train:self.train+self.valid]
        if split=='test':
            return self.dataset[self.train+self.valid:self.train+self.valid+self.test]
            
yelp=yelpdata()
class Key2TextDataset(Dataset):
    def __init__(self,
                 split="train",datacaller=yelp):
        self.split = split
        self.datacaller = datacaller
        #self.dataset=dataset[split]
        self.dataset=datacaller(split)
        self.stopwords = nltk.corpus.stopwords.words('english')
        #print(self.stopwords)
    def __len__(self):
        #return 200
        return len(self.dataset)
    def collate_fn(self,batch):
        max_len=0
        max_key_len=0
        for item in batch:
            if item is not None:
                max_len=max(max_len,len(item['target_token']))
                max_key_len=max(max_key_len,len(item['key_token']))
        if max_len==0:
            key_batch=[torch.zeros((1,1),dtype=torch.int32)]
            key_att_batch=[torch.zeros((1,1),dtype=torch.float32)]
            tgt_batch=[torch.zeros((1,1),dtype=torch.int32)]
            tgt_att_batch=[torch.zeros((1,1),dtype=torch.float32)]
        else:
            key_batch=[]
            key_att_batch=[]
            tgt_batch=[]
            tgt_att_batch=[]
            for item in batch:
                if item is not None:
                    key_batch.append(torch.cat((torch.LongTensor(item['key_token']),torch.ones(max_key_len-len(item['key_token']),dtype=torch.int32)),dim=-1).unsqueeze(0))
                    tgt_batch.append(torch.cat((torch.LongTensor(item['target_token']),torch.ones(max_len-len(item['target_token']),dtype=torch.int32)),dim=-1).unsqueeze(0))
                    key_att_batch.append(torch.cat((torch.Tensor(item['key_attention_mask']),torch.zeros(max_key_len-len(item['key_attention_mask']))),dim=-1).unsqueeze(0))
                    tgt_att_batch.append(torch.cat((torch.Tensor(item['target_attention_mask']),torch.zeros(max_len-len(item['target_attention_mask']))),dim=-1).unsqueeze(0))
            key_batch=torch.cat(key_batch,dim=0)
            key_att_batch=torch.cat(key_att_batch,dim=0)
            tgt_batch=torch.cat(tgt_batch,dim=0)
            tgt_att_batch=torch.cat(tgt_att_batch,dim=0)
        return {'target_token':tgt_batch,'target_attention_mask':tgt_att_batch,'key_token':key_batch,'key_attention_mask':key_att_batch}

    def __getitem__(self, index):
        line=self.dataset[index]['text'].replace('\\n',' ')
        splitted=line.split(' ')
        distilled=[]
        shorten_line=""
        for word in splitted:
            if word.lower() not in self.stopwords:
                distilled.append(re.sub(r'[^a-zA-Z\s]','', string=word))
            shorten_line+=word+" "
        if(len(distilled)<=2):
            return None
        numbers=np.random.choice(len(distilled), min(len(distilled),5),replace=False)
        key=""
        for i in numbers:
            key+=distilled[i]+" "
        token=tokenizer(shorten_line)
        line_token,line_mask=token['input_ids'],token['attention_mask']
        token=tokenizer(key)
        #token=tokenizer(key)
        key_token,key_mask=token['input_ids'],token['attention_mask']
        #print(line)
        #print(key)
        return {"target_token":line_token,"target_attention_mask":line_mask,"key_token":key_token,"key_attention_mask":key_mask}
        
def main():
    a="she she, She SHE she shE she."
    print(tokenizer(a))
    print(tokenizer("She she, she she she she she."))
    print(tokenizer("she she she."))
    print(tokenizer("me me me."))

    return 0
    train=Key2TextDataset()
    trainloader=DataLoader(train,batch_size=3,collate_fn=train.collate_fn,shuffle=True, pin_memory=True, num_workers=1)
    for i in range(4):
        print(train[i])
    for i,batch in zip(range(4),trainloader):
        print(batch)



if __name__=="__main__":
    main()