import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np


def SDPA(Q, K, V, d_k, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

class ViT(nn.Module):

    def __init__(self, d_model, num_heads):
        super(ViT, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
    
    def forward(self, x): 
        batch_size, seq_len, _ = x.size()
        
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        output, _ = SDPA(Q, K, V, self.d_k)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.fc(output)
        
        return output

batch_size = 2
seq_len = 4
d_model = 8
num_heads = 2
d_k = d_model // num_heads

# 임의의 Q, K, V 생성
Q = torch.rand(batch_size, num_heads, seq_len, d_k)
K = torch.rand(batch_size, num_heads, seq_len, d_k)
V = torch.rand(batch_size, num_heads, seq_len, d_k)

#랜덤 인풋 생성
x = torch.rand(batch_size, seq_len, d_model)

model = ViT(d_model, num_heads)
output = model(x)

#SDP는 ViT에서 각 패치간의 관계를 학습하여 이미지 전체의 맥락을 이해 (CNN은 로컬하게 어탠션은 글로벌하게)