import torch


def topk_prune(x, cls_attn_weights, kv_drop_rate):
    if kv_drop_rate <= 0:
        return x

    B, L, D = x.shape
    num_keep = int(L * (1 - kv_drop_rate))

    num_keep_attn = int(num_keep * 1.0)
    num_keep_div = num_keep - num_keep_attn

    cls_attn_weights_copy = cls_attn_weights.clone()
    cls_attn_weights_copy[:, 0] = float('inf')

    _, idx_attn = torch.topk(cls_attn_weights_copy, num_keep_attn, dim=-1)

    idx_attn_expanded = idx_attn.unsqueeze(-1).expand(-1, -1, D)
    x_attn = torch.gather(x, dim=1, index=idx_attn_expanded)

    mean_attn = x_attn.mean(dim=1, keepdim=True)

    distances = torch.norm(x - mean_attn, dim=-1)

    mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
    mask.scatter_(dim=1, index=idx_attn, value=True)
    distances.masked_fill_(mask, -float('inf'))

    _, idx_div = torch.topk(distances, num_keep_div, dim=-1)

    keep_idx = torch.cat([idx_attn, idx_div], dim=1)

    keep_idx, _ = torch.sort(keep_idx, dim=-1)

    keep_idx_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, D)
    x_reduced = torch.gather(x, dim=1, index=keep_idx_expanded)

    return x_reduced
