from transformers import BartTokenizer, BartForConditionalGeneration, BartModel
import torch

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# TXT = "My friends are <mask> but they eat too many carbs."
# input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
# print(input_ids)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")['input_ids']
print(inputs.shape)

label = torch.tensor([[1,2,3,4,5]])
outputs = model(input_ids=inputs, labels=label)
loss = outputs.logits
print(loss.shape)