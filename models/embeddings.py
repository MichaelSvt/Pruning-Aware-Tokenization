import torch
import torch.nn as nn
from utils import generate_2d_sincos_embedding


class LinearEmbedding(nn.Module):
    def __init__(self, channels=3, img_size=224, embed_dim=768, patch_size=16):
        super(LinearEmbedding, self).__init__()

        # Attributes
        self.channels = channels
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * channels
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        cls_pos_embed = torch.zeros(1, embed_dim).unsqueeze(0)
        self.register_buffer('pos_embedding0', cls_pos_embed)

        self.patch_embedding = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        pos_embed_1 = generate_2d_sincos_embedding(img_size // patch_size, embed_dim, step=1.0).unsqueeze(0)
        self.register_buffer('pos_embedding1', pos_embed_1)

    def forward(self, x):
        B, C, H, W = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embedding0

        x1 = self.patch_embedding(x)
        x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        x1 = x1 + self.pos_embedding1

        return torch.cat([cls_tokens, x1], dim=1)


class SingleScaleEmbedding(nn.Module):
    def __init__(self, channels=3, img_size=224, embed_dim=768, cnn_channels=(32, 64, 128),
                 kernel_sizes=(3,3,3,3), strides=(2,2,2,2), padding=(1,1,1,1), grid_size=14):
        super(SingleScaleEmbedding, self).__init__()

        # Attributes
        self.channels = channels
        self.height = img_size
        self.width = img_size
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        cls_pos_embed = torch.zeros(1, embed_dim).unsqueeze(0)
        self.register_buffer('pos_embedding0', cls_pos_embed)

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(channels, cnn_channels[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=padding[0]),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=kernel_sizes[1], stride=strides[1], padding=padding[1]),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=kernel_sizes[2], stride=strides[2], padding=padding[2]),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[2], embed_dim, kernel_size=kernel_sizes[3], stride=strides[3], padding=padding[3]),
            nn.BatchNorm2d(embed_dim),
        )
        pos_embed_1 = generate_2d_sincos_embedding(grid_size, embed_dim, step=1.0).unsqueeze(0)
        self.register_buffer('pos_embedding1', pos_embed_1)

    def forward(self, x):
        B, C, H, W = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embedding0

        x1 = self.patch_embedding(x)
        x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        x1 = x1 + self.pos_embedding1

        return torch.cat([cls_tokens, x1], dim=1)


class TwoScaleEmbedding(nn.Module):
    def __init__(self, channels=3, img_size=224, embed_dim=768, cnn_channels=(32, 64, 128, 256),
                 kernel_sizes=(3,3,3,3,3), strides=(2,2,2,2,2), padding=(1,1,1,1,1), grid_size=14, drop_tokens=False):
        super(TwoScaleEmbedding, self).__init__()

        self.channels = channels
        self.height = img_size
        self.width = img_size
        self.embed_dim = embed_dim
        self.drop_tokens = drop_tokens

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        cls_pos_embed = torch.zeros(1, embed_dim).unsqueeze(0)
        self.register_buffer('pos_embedding0', cls_pos_embed)

        pos_embed_1 = generate_2d_sincos_embedding(grid_size, embed_dim, step=1.0).unsqueeze(0)
        self.register_buffer('pos_embedding1', pos_embed_1)
        self.scale_pos_embed1 = nn.Parameter(torch.rand(1, 1, embed_dim))

        pos_embed_2 = generate_2d_sincos_embedding(grid_size//2, embed_dim, step=2.0).unsqueeze(0)
        self.register_buffer('pos_embedding2', pos_embed_2)
        self.scale_pos_embed2 = nn.Parameter(torch.rand(1, 1, embed_dim))

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(channels, cnn_channels[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=padding[0]),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=kernel_sizes[1], stride=strides[1], padding=padding[1]),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=kernel_sizes[2], stride=strides[2], padding=padding[2]),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU()
        )

        self.patch_embedding1 = nn.Sequential(
            nn.Conv2d(cnn_channels[2], embed_dim, kernel_size=kernel_sizes[3], stride=strides[3], padding=padding[3]),
            nn.BatchNorm2d(embed_dim),
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(cnn_channels[2], cnn_channels[3], kernel_size=kernel_sizes[3], stride=strides[3], padding=padding[3]),
            nn.BatchNorm2d(cnn_channels[3]),  # Norm after every conv
        )

        self.patch_embedding2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(cnn_channels[3], embed_dim, kernel_size=kernel_sizes[3], stride=strides[3], padding=padding[3]),
            nn.BatchNorm2d(embed_dim),  # Norm after every conv
        )

    def forward(self, x):
        B, C, H, W = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embedding0
        x1 = self.conv_layer1(x)
        x2 = self.conv_layer2(x1)
        x1 = self.patch_embedding1(x1)
        x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)  # (B, 196, D)
        x1 = x1 + self.pos_embedding1 + self.scale_pos_embed1

        if self.drop_tokens:
            # Drop tokens corresponding to (0,0), (0,2), ..., (2,0), (2,2), ...
            mask = torch.ones(14, 14, dtype=torch.bool, device=x1.device)
            mask[::2, ::2] = False
            x1 = x1[:, mask.flatten(), :]  # (B, 147, D)

        x2 = self.patch_embedding2(x2)  # comment if we use only a single conv for the low res scale
        x2 = x2.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        x2 = x2 + self.pos_embedding2 + self.scale_pos_embed2

        return torch.cat([cls_tokens, x1, x2], dim=1)




