import math

import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dk = d_model

    def forward(self, x):
        # x: (batchsize, ctx_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        attn_score = F.softmax(score, dim=-1)
        output = torch.matmul(attn_score, V)
        return output


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, ctx_len, qkv_bias=False):
        super().__init__()
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.dk = d_model
        
        # 创建下三角'mask'，下三角为True，上三角为False
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(ctx_len, ctx_len, dtype=torch.bool)),
        )

    def forward(self, x):
        b, ctx_len, d_in = x.shape
        keys = self.k_proj(x)
        queries = self.q_proj(x)
        values = self.v_proj(x)

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        # 处理输入序列长度小于ctx的情况：x.ctx_len < self.ctx_len
        mask = self.mask.bool()[:ctx_len, :ctx_len]
        # mask取反后对上三角位置替换-inf
        masked_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(masked_scores / self.dk**0.5, dim=-1)
        context_vec = torch.matmul(attn_weights, values)
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, ctx_len, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(ctx_len, ctx_len, dtype=torch.bool)),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        # 通过view进行维度切分，d_in => num_heads * head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # [b,num_tokens, num_heads, head_dim]=>[b,num_heads,num_tokens,head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # [bkij@bkji]
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(~mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 算完再转换回 [b, num_tokens, num_heads, head_dim]
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将每个头vec集合到一起, self.d_out = num_heads * head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


if __name__ == "__main__":
    x = torch.rand(2, 3, 4)
    # self_attention = SelfAttention(d_model=4)
    # self_attention = CausalSelfAttention(d_model=4, ctx_len=3)
    self_attention = MultiHeadAttention(
        d_in=4, d_out=4, ctx_len=3, dropout=0.0, num_heads=2
    )
    output = self_attention(x)
    print(output.shape)
