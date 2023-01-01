import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration
from functools import cmp_to_key
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
class beam_search_data:
    def __init__(self,token,cover,score):
        self.token=token
        self.cover=cover
        self.score=score
    def not_completed(self,tot_cnt):
        return len(self.cover)<tot_cnt
    def not_in(self,tot_cnt):
        comp=[]
        for i in range(tot_cnt):
            if i not in self.cover:
                comp.append(i)
        return comp
class Key2Text(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        #self.constrain_criterion=nn.NLLLoss(reduction='mean',ignore_index=1)

    def logits(self, input_ids, decoder_input_ids):
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    def get_loss(self, input_ids, decoder_input_ids, labels):
        ret=self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        logits=ret.logits
        logits=F.log_softmax(logits,dim=-1)
        onehot=F.one_hot(input_ids,num_classes=logits.shape[-1])
        logits,_=torch.max(logits,dim=-2,keepdim=True)
        #print(logits.shape,onehot.shape)
        return -(logits*onehot).sum()/onehot.shape[0]/onehot.shape[1]+ret.loss
    def get_constrain_loss(self, input_ids, decoder_input_ids):
        '''
        logits: batch_size * seq_length * vocab_size
        '''
        logits=self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
        logits=F.log_softmax(logits,dim=-1)
        onehot=F.one_hot(input_ids,num_classes=logits.shape[-1])
        logits,_=torch.max(logits,dim=-2,keepdim=True)
        #print(logits.shape,onehot.shape)
        return -(logits*onehot).sum()/onehot.shape[0]/onehot.shape[1]
    def constrain_search(self,input_ids,beam_size=10,min_length=0,max_length=32):
        self.eval()
        device=input_ids.device
        tot_con=input_ids.size(1)
        candidates=[[] for i in range(tot_con+1)]
        init_cover=[]
        if 0 in input_ids:
            init_cover=[0]
        else:
            init_cover=[]
        candidates[len(init_cover)].append(beam_search_data([torch.Tensor([0]).to(device)],cover=init_cover,score=0))
        best=beam_search_data([0],cover=[],score=-1e10)
        #print(input_ids)
        for length in range(max_length):

            new_list=[[] for i in range(tot_con+1)]
            for c_cnt in range(tot_con+1):
                for data in candidates[c_cnt]:
                    if(data.token[-1]==2):
                        continue
                    #print(tokenizer.batch_decode(torch.LongTensor([data.token]), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
                    ret=self.model(input_ids=input_ids,decoder_input_ids=torch.LongTensor([data.token]).to(device))
                    #print(ret.logits.shape)
                    last_logits=ret.logits[0][-1]
                    last_logits=F.log_softmax(last_logits, dim=-1)
                    #print(last_logits)
                    if(c_cnt==tot_con  and (data.score)/(length+1)>best.score and length>=min_length):
                        #print(torch.LongTensor([data.token]).to(device))
                        best=beam_search_data(data.token,cover=[],score=(data.score)/(length+1))
                    if data.not_completed(tot_con):
                        for k_ind in data.not_in(tot_con):# satisfy new constrain
                            new_data=beam_search_data(data.token+[input_ids[0][k_ind]],data.cover+[k_ind],data.score+last_logits[input_ids[0][k_ind]])
                            new_list[c_cnt+1].append(new_data)
                            print(tokenizer.batch_decode(torch.LongTensor([data.token+[input_ids[0][k_ind]]]), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

                    newchars=torch.argsort(last_logits)[-beam_size:]
                    for char in newchars:
                        new_data=beam_search_data(data.token+[char],data.cover,data.score+last_logits[char])
                        print(tokenizer.batch_decode(torch.LongTensor([data.token+[char]]), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
                        new_list[c_cnt].append(new_data)
                if len(new_list[c_cnt])>beam_size:
                    new_list[c_cnt]=sorted(new_list[c_cnt],
                    key=cmp_to_key(lambda a,b: b.score-a.score))[:beam_size]
            candidates=new_list
        print(tokenizer.batch_decode(torch.LongTensor([new_list[1][0].token]), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        #print(new_list[1][0].token)
        if best.score<-1e9:
            best=new_list[tot_con][0]
        return torch.LongTensor([best.token]).to(device)
if __name__ == "__main__":
    model = Key2Text()
    inputs = torch.tensor([[1,2,3,4,5],[2,3,4,5,6]])
    decoder_inputs = torch.tensor([[1,2,3,4,5,6,7,8],[2,3,4,5,6,7,8,9]])
    logits = model.logits(inputs, decoder_inputs)
    print(logits.shape)


