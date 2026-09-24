import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from embeddings import LinearEmbedding, SingleScaleEmbedding, TwoScaleEmbedding
from transformer import TransformerBlock
from pruning import topk_prune

class BasicViT(nn.Module):
    def __init__(self, channels=3, img_size=224, patch_size=16, embed_dim=768, num_classes=1000,
                n_layers=12, hidden_dim=3072, dropout=0.0, kv_dropout=0.0, heads=8, drop_at_layers=(0),
                 embedding_layer="linear", cnn_channels=(32, 64, 128), kernel_sizes=(3,3,3,3),
                 strides=(2,2,2,2), padding=(1,1,1,1), grid_size=14):
        super(BasicViT, self).__init__()

        # Attributes
        self.channels = channels
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.patch_dim = patch_size * patch_size * channels
        self.num_patches = (img_size // patch_size) ** 2
        self.dropout = nn.Dropout(dropout)
        self.kv_droprate = kv_dropout
        self.embed_dim = embed_dim
        self.embedding_layer = embedding_layer
        self.drop_at_layers = drop_at_layers
        self.cnn_channels = cnn_channels
        self.kernel_sizes=kernel_sizes
        self.strides=strides
        self.padding=padding
        self.grid_size=grid_size

        if embedding_layer == "single":
            self.embed = SingleScaleEmbedding(channels=channels, img_size=img_size, embed_dim=embed_dim, cnn_channels=self.cnn_channels,
                 kernel_sizes=self.kernel_sizes, strides=self.strides, padding=self.padding, grid_size=grid_size)
        elif embedding_layer == "two":
            self.embed = TwoScaleEmbedding(channels=channels, img_size=img_size, embed_dim=embed_dim, cnn_channels=self.cnn_channels,
                 kernel_sizes=self.kernel_sizes, strides=self.strides, padding=self.padding, grid_size=grid_size)
        else:
            self.embed = LinearEmbedding(channels=channels, img_size=img_size, embed_dim=embed_dim, patch_size=patch_size)

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            block = TransformerBlock(
                embed_dim=embed_dim,
                n_heads=heads,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            self.layers.append(block)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))


    def forward(self, x_in, prune_rate=0.0):
        x = self.embed(x_in)
        x = self.dropout(x)
        # Transformer layers
        for i in range(self.n_layers):

            x, cls_attn_weights = self.layers[i](x)

            if i in self.drop_at_layers and prune_rate > 0:
                x = topk_prune(x, cls_attn_weights, prune_rate)

        return self.head(x[:, 0, :])


