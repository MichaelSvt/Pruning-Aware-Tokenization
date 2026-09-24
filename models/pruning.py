import torch


def topk_prune(x, cls_attn_weights, kv_drop_rate):
    """
    Drops tokens by keeping 50% based on CLS attention, and 50% based on
    feature diversity (furthest from the mean of the attended tokens).
    """
    if kv_drop_rate <= 0:
        return x

    B, L, D = x.shape
    num_keep = int(L * (1 - kv_drop_rate))

    # Split the keep budget
    num_keep_attn = int(num_keep * 1.0)
    num_keep_div = num_keep - num_keep_attn

    # --- STAGE 1: SALIENCE (Attention) ---
    # Ensure CLS token is always kept by forcing infinite attention
    cls_attn_weights_copy = cls_attn_weights.clone()
    cls_attn_weights_copy[:, 0] = float('inf')

    # Select top tokens by attention
    _, idx_attn = torch.topk(cls_attn_weights_copy, num_keep_attn, dim=-1)  # [B, num_keep_attn]

    # Gather these tokens to compute their mean
    idx_attn_expanded = idx_attn.unsqueeze(-1).expand(-1, -1, D)
    x_attn = torch.gather(x, dim=1, index=idx_attn_expanded)  # [B, num_keep_attn, D]

    # Compute the mean of the attended tokens
    mean_attn = x_attn.mean(dim=1, keepdim=True)  # [B, 1, D]

    # --- STAGE 2: DIVERSITY (Distance) ---
    # Compute L2 distance of ALL tokens to this mean
    # Broadcasting mean_attn [B, 1, D] against x [B, L, D]
    distances = torch.norm(x - mean_attn, dim=-1)  # [B, L]

    # We must not re-select tokens from Stage 1.
    # Create a boolean mask and set the distance of Stage 1 tokens to -inf
    mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
    mask.scatter_(dim=1, index=idx_attn, value=True)
    distances.masked_fill_(mask, -float('inf'))

    # Select the tokens that are FURTHEST from the mean (highest remaining distance)
    _, idx_div = torch.topk(distances, num_keep_div, dim=-1)  # [B, num_keep_div]

    # --- STAGE 3: MERGE AND GATHER ---
    # Combine indices
    keep_idx = torch.cat([idx_attn, idx_div], dim=1)  # [B, num_keep]

    # Sort indices so the remaining tokens stay in their original relative sequence order.
    # This is crucial if you rely on positional relationships later.
    keep_idx, _ = torch.sort(keep_idx, dim=-1)

    # Gather the final merged tokens
    keep_idx_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, D)
    x_reduced = torch.gather(x, dim=1, index=keep_idx_expanded)

    return x_reduced
