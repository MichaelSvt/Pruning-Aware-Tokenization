import torch
import torch.nn as nn


def generate_2d_sincos_embedding(grid_size, embed_dim, step=1.0):
    """
    Generates a 2D sine-cosine positional embedding for a grid of patches.

    Args:
        grid_size (int): The height/width of the grid (e.g., 16 or 8).
        embed_dim (int): The total embedding dimension (must be divisible by 4).
        step (float): The spatial step size. Crucial for multi-scale alignment!

    Returns:
        torch.Tensor: A tensor of shape [grid_size * grid_size, embed_dim]
    """
    assert embed_dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D sin-cos."

    # 1. Create a 2D grid of coordinates.
    # The 'step' parameter allows us to stretch the coordinates for lower resolutions.
    coords = torch.arange(0, grid_size * step, step, dtype=torch.float32)

    # grid_y and grid_x will both have shape (grid_size, grid_size)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')

    # Flatten the grids -> shape (grid_size * grid_size)
    grid_y = grid_y.flatten()
    grid_x = grid_x.flatten()

    # 2. Calculate the frequency (omega) values
    # We use embed_dim // 4 because X and Y each get half the dimensions,
    # and within that half, we need space for both sine and cosine.
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4 - 1)
    omega = 1.0 / (10000.0 ** omega)

    # 3. Compute outer products for X and Y
    # Shape: (grid_size * grid_size, embed_dim // 4)
    out_y = grid_y.unsqueeze(1) * omega.unsqueeze(0)
    out_x = grid_x.unsqueeze(1) * omega.unsqueeze(0)

    # 4. Apply sin and cos
    pos_emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)  # (..., embed_dim // 2)
    pos_emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)  # (..., embed_dim // 2)

    # 5. Concatenate Y and X embeddings to form the full D-dimensional vector
    # Shape: (grid_size * grid_size, embed_dim)
    pos_embed = torch.cat([pos_emb_y, pos_emb_x], dim=1)

    return pos_embed


def top_accuracy(output, target, topk=(1, 5)):
    """Computes the top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)

    # Get top-k predictions
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # shape: [batch_size, maxk]
    pred = pred.t()  # shape: [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # shape: [maxk, batch_size]

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc_k = correct_k * (100.0 / batch_size)
        results.append(acc_k.item())
    return results  # e.g., [top1_acc, top5_acc]

