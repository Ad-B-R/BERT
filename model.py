import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, dropout, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.segment_embed = nn.Embedding(2, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)        
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.segment_embed.weight, std=0.02)
    
    def forward(self, input_ids, token_type_ids=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        seg = token_type_ids if token_type_ids is not None else torch.zeros_like(input_ids)
        x = self.token_embed(input_ids) + self.pos_embedding(positions) + self.segment_embed(seg)
        return self.norm(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)

        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, attn_mask=None):
        d_k = query.shape[-1]
        attention_score = (query@key.transpose(-2,-1))/(math.sqrt(d_k))
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attention_score = attention_score.masked_fill(attn_mask == 0, float('-inf')) # Turns padded tokens into 0
        
        attention_score = attention_score.softmax(dim=-1)

        return attention_score @ value, attention_score

    def forward(self, x, attn_mask=None):
        # batch, seq_len, d_model
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        
        # RESHAPE       

        query = query.view(query.shape[0], -1, self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], -1, self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], -1, self.h, self.d_k).transpose(1,2)

        x, _ = self.attention(query=query, key=key, value=value, attn_mask=attn_mask)
        x = x.transpose(1,2)

        x = x.contiguous().view(x.shape[0], -1, self.d_model)
        
        return self.Wo(x)        

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(features))
        self.gamma = nn.Parameter(torch.ones(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma*(x-mean)/(std+1e-9) + self.beta
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)

    def forward(self, x, attn_mask):
        return self.ffn2(self.dropout(F.relu(self.ffn1(x))))

class ResidualConnection(nn.Module):
    def __init__(self, features, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(features)

    def forward(self, x, sublayer:nn.ModuleList):
        return x + (self.dropout(sublayer(self.layernorm(x))))

class Encoder(nn.Module):
    def __init__(self, features, dropout, attn, ffn, **kwargs):
        super().__init__(**kwargs)
        self.attention_block = attn
        self.ffn = ffn
        self.residual_connections = nn.ModuleList(ResidualConnection(features, dropout) for _ in range(2))

    def forward(self, x, attn_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, attn_mask))
        x = self.residual_connections[1](x, lambda x: self.ffn(x, attn_mask))
        return x

class BERT_Block(nn.Module):
    def __init__(self, embed, encoder, d_model, num_classes, dropout, **kwargs):
        super().__init__(**kwargs)
        self.embed = embed
        self.encoders = encoder
        self.d_model = d_model
        
        self.layernorm = LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask):
        x = self.dropout(self.embed(x))
        for encoder in self.encoders:
            x = encoder(x, attn_mask)
        
        x = self.layernorm(x)
        x = x[:,0,:]
        x = self.classifier(x)
        return x

