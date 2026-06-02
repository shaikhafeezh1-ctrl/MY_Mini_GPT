import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_block import Block

print("Torch version:" , torch.__version__)
print("Cuda available:",     torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]

corpus = [s + " <END>" for s in corpus]
text= " ".join(corpus)
print(text)

words= list(set(text.split()))
print(words)

voc_size= len(words)
print(voc_size)  # 42

word2idx={w: i for i , w in enumerate(words)}
print("word2idx :" , word2idx)

idx2word = {i: w for w, i in word2idx.items()} 
print("idx2word : ", idx2word) 


data=torch.tensor([word2idx[w] for w in text.split()],dtype=torch.long)
print(data)

block_size=6 # It see preious 6 words to predict the next word
embedding_dim=32 # 'tea': 13 -> [0.1, 0.2, 0.3, ...] (32-dim vector)
n_heads=2 # Number of Multi-head attention heads
n_layers=2 # Number of transformer layers
lr=1e-3 # Learning rate
epochs=1500 # Number of training epochs

def get_batch(batch_size=16):
    ix=torch.randint(len(data)-block_size , (batch_size,)) # 62-6=56  (0-55) ,16
    #[12,34,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #16 random numbers between 0 and 55 for 1 batch
    # 12(t12,t13,t14,t15,t16,t17) 34(t34,t35,t36,t37,t38,t39) 5(t5,t6,t7,t8,t9,t10) ... 6 token(block_size=6)
    x=torch.stack([data[i:i+ block_size] for  i in ix])
    y=torch.stack([data[i+1:i+ block_size+1] for i in ix])
    # x=[12(t12,t13,t14,t15,t16,t17]
    # y=(t13,t14,t15,t16,t17,t18) 34(t35,t36,t37,t38,t39,t40) 5(t6,t7,t8,t9,t10,t11) ... 6 token(block_size=6)]
    
    return x,y

class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding=nn.Embedding(voc_size,embedding_dim) # (42,32)
            # 2 = hello - [32 floating point values] = [0.66,022,015,054,...] 32 values for each of the 42 words in the vocabulary
            # 13 = tea - [32 floating point values] = [0.1,0.2,0.3,...] 32 values for each of the 42 words in the vocabulary

            self.position_embedding=nn.Embedding(block_size,embedding_dim) # (6,32) 0->(0.1,0.2,0.3,...), 1->(0.4,0.5,0.6,...), 2->(0.7,0.8,0.9,...), 
            # 3->(1.0,1.1,1.2,...), 4->(1.3,1.4,1.5,...), 5->(1.6,1.7,1.8,...)


            # position_embedding means the position of the word in the sentence. For example, in the sentence "hello friends how are you", "hello" is at position 0,
            #  "friends" is at position 1, "how" is at position 2, "are" is at position 3
            # . The position embedding will help the model to understand the order of the words in the sentence.
            self.blocks=nn.Sequential(*[Block(embedding_dim,block_size,n_heads) for _ in range(n_layers)])

            self.ln_f=nn.LayerNorm(embedding_dim)
            self.head=nn.Linear(embedding_dim,voc_size)

        def forward(self, idx ,target=None):
            B, T =idx.shape
            tok_emb=self.token_embedding(idx) # (B,T,embedding_dim) 16,6,32

            pos_emp= self.position_embedding(torch.arange(T,device=idx.device)) # (T,embedding_dim) 6,32
            x=tok_emb + pos_emp # (B,T,embedding_dim) 16,6,32
            x=self.blocks(x) # (B,T,embedding_dim) 16,6,32
            x=self.ln_f(x) # (B,T,embedding_dim) 16,6,32
            logits=self.head(x) # (B,T,voc_size) 16,6,42
            loss=None
            if target is not None:
                B,T,C=logits.shape
                loss=F.cross_entropy(logits.view(B*T,C),target.view(B*T))
            return logits,loss

        def genrate(self, idx , max_new_tokens):

            for _ in range(max_new_tokens):
                idx_con = idx[:, -block_size:] # (B,T) -> (B,block_size) 16,6
                logits, _ = self(idx_con) # (B,T,voc_size) 16,6,42
                logits = logits[:, -1, :] # (B,voc_size) 16,42
                probs = F.softmax(logits, dim=-1) # (B,voc_size) 16,42
                next_token = torch.multinomial(probs,1) # (B,1) 16,1
                idx = torch.cat((idx, next_token), dim=1) # (B,T+1) 16,7 -> 16,8 -> 16,9 ...
            return idx

model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    xb , yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss= {loss.item(): .4f}") 

context =torch.tensor([[word2idx["hello"]]],dtype=torch.long)
out=model.genrate(context,max_new_tokens=15)

print(" ".join(idx2word[int(i)] for i in out[0]))

        


            