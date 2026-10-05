import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0, kv_droprate=0.0):
        """
        Self-attention module for transformer architecture

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        embed_dim = dim
        num_heads = n_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"Embedding dimension {embed_dim} not divisible by number of heads {num_heads}")

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.kv_droprate = kv_droprate
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        cls_attn_weights = attention_weights[:, :, 0, :]

        cls_attn_weights = cls_attn_weights.mean(dim=1)

        attention_weights_drop = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights_drop, v)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attention_output)

        return output, cls_attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, n_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, cls_attn_weights = self.attn(self.norm1(x))

        x = x + attn_out

        x = x + self.mlp(self.norm2(x))

        return x, cls_attn_weights

