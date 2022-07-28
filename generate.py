import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from fastai.text.all import *
import re
import fastai
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


print(torch.__version__)
print(transformers.__version__)
print( fastai.__version__)

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 
model = AutoModelWithLMHead.from_pretrained("skt/kogpt2-base-v2")
text = """ 옛날 옛날 어느 마을에 흥부와 놀부 형제가 """
input_ids = tokenizer.encode(text)
gen_ids = model.generate(torch.tensor([input_ids]),
                           max_length=128,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True
                        )
generated = tokenizer.decode(gen_ids[0,:].tolist())
print(generated)


import re

with open('./book_report_data.txt', 'r', encoding='utf-8') as f:
    data = f.read()
data=" ".join(data.split())
# print(len(data))
data = data.replace('\n|\t', ' ')
print(len(data))
# new_sent = data.replace(" ", '') # 띄어쓰기가 없는 문장 임의로 만들기
# # print(len(new_sent))
# spacing = Spacing()
# kospacing_sent = spacing(new_sent) 
# print(len(kospacing_sent))
data = re.sub('[-=+,#/\:^$@*\"※~&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》]','', data)
print(len(data))
data = re.sub('[a-zA-Z]' , '', data)
print(len(data))
# UnicodeDecodeError: 'cp949' codec can't decode byte 0xec in position 20: illegal multibyte sequence
# 해당 에러 ->  encoding='utf-8' 넣어 주기


#model input output tokenizer
class TransformersTokenizer(Transform):
   def __init__(self, tokenizer): self.tokenizer = tokenizer
   def encodes(self, x): 
       toks = self.tokenizer.tokenize(x)
       return tensor(self.tokenizer.convert_tokens_to_ids(toks))
   def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))

#split data
train=data[:int(len(data)*0.9)]
test=data[int(len(data)*0.9):]
splits = [[0],[1]]

#init dataloader
tls = TfmdLists([train,test], TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
dls = tls.dataloaders(bs=4, seq_len=128)

#gpt2 ouput is tuple, we need just one val
class DropOutput(Callback):
  def after_pred(self): self.learn.pred = self.pred[0]
      
      
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()

lr = learn.lr_find()
print(lr)
learn.fit_one_cycle(50, lr)


learn.model.save_pretrained("./models/kogpt2_bookreport_backup50")