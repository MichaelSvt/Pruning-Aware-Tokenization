import torch
import torch.nn as nn
from torchvision.transforms import Resize, ToTensor

import torch.nn.functional as F
from torch import Tensor

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

        # Check if embed_dim is divisible by num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"Embedding dimension {embed_dim} not divisible by number of heads {num_heads}")

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.kv_droprate = kv_droprate
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, return_aux=False):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # --- NEW for TopK, EViT, ToMe: Extract CLS attention ---
        # attention_weights shape: [Batch, Heads, Queries, Keys]
        # We want the 0th Query (CLS token) across all Keys
        cls_attn_weights = attention_weights[:, :, 0, :]  # Shape: [B, H, L]

        # Average across all heads to get a consensus on importance
        cls_attn_weights = cls_attn_weights.mean(dim=1)  # Shape: [B, L]

        # Continue normal attention
        attention_weights_drop = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights_drop, v)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attention_output)

        # Return both the features AND the CLS attention weights
        return output, cls_attn_weights

        #
        # # --- MODIFIED for Zero-TPrune: Extract FULL attention matrix averaged across heads ---
        # # attention_weights shape: [B, Heads, L, L] -> [B, L, L]
        # full_attn_weights = attention_weights.mean(dim=1)
        #
        # attention_weights_drop = self.dropout(attention_weights)
        # attention_output = torch.matmul(attention_weights_drop, v)
        #
        # attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        # output = self.out_proj(attention_output)
        #
        # return output, full_attn_weights
        #
        # # # --- MODIFIED for Zero-TPrune of ChatGPT: Extract FULL attention matrix averaged across heads ---
        # # Continue normal attention
        # attention_weights_drop = self.dropout(attention_weights)
        # attention_output = torch.matmul(attention_weights_drop, v)
        #
        # attention_output = (
        #     attention_output
        #     .transpose(1, 2)
        #     .contiguous()
        #     .view(
        #         batch_size,
        #         seq_len,
        #         self.embed_dim,
        #     )
        # )
        #
        # output = self.out_proj(
        #     attention_output
        # )
        #
        # if return_aux:
        #     # Keep the full matrices.
        #     #
        #     # attention_weights:
        #     #       [B, H, N, N]
        #     #
        #     # k:
        #     #       [B, H, N, Dh]
        #     #
        #     # cls_attn_weights:
        #     #       [B, N]
        #     cls_attn_weights = (
        #         attention_weights[:, :, 0, :]
        #         .mean(dim=1)
        #     )
        #
        #     return (
        #         output,
        #         cls_attn_weights,
        #         attention_weights,
        #         k,
        #     )
        #
        #     # Keep your current interface for normal layers.
        #
        # cls_attn_weights = (
        #     attention_weights[:, :, 0, :]
        #     .mean(dim=1)
        # )
        # return output, cls_attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, n_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        # self.mlp = FeedForward(embed_dim, hidden_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 1. Norm + Attention
        attn_out, cls_attn_weights = self.attn(self.norm1(x))

        # 2. Residual
        x = x + attn_out

        # 3. Norm + MLP + Residual
        x = x + self.mlp(self.norm2(x))

        return x, cls_attn_weights

    # def forward(self, x, return_aux=False):
    #
    #     if return_aux:
    #         (attn_out, cls_attn_weights, attention_weights, key,) = self.attn(self.norm1(x), return_aux=True,)
    #     else:
    #         attn_out, cls_attn_weights = self.attn(self.norm1(x), return_aux=False,)
    #
    #     # Attention residual
    #     x = x + attn_out
    #
    #     # MLP residual
    #     x = x + self.mlp(self.norm2(x))
    #
    #     if return_aux:
    #         return (
    #             x,
    #             cls_attn_weights,
    #             attention_weights,
    #             key,
    #         )
    #
    #     return x, cls_attn_weights


# class FeedForward(nn.Sequential):
#     def __init__(self, embed_dim, hidden_dim, dropout=0.1):
#         super().__init__(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, embed_dim),
#             nn.Dropout(dropout)
#         )

def merge_tokens_tome(x, cls_attn_weights, kv_drop_rate):
    """
    Token Merging (ToMe) via bipartite soft matching of token features.
    Acts as a drop-in replacement for top-K pruning functions.
    """
    if kv_drop_rate <= 0:
        return x

    B, L, D = x.shape
    num_keep = int(L * (1 - kv_drop_rate))
    r = L - num_keep  # Number of tokens to reduce/merge

    if r <= 0:
        return x

    # Separate CLS token from spatial tokens
    cls_token = x[:, 0:1, :]
    tokens = x[:, 1:, :]  # [B, N, D]
    N = tokens.shape[1]

    # Bipartite matching requires splitting tokens into two sets.
    # Therefore, we can merge at most 50% of the tokens in a single pass.
    r = min(r, N // 2)

    # 1. Split tokens into two sets: tokens_a and tokens_b.
    # Using alternating indices naturally preserves spatial layouts from the grid.
    tokens_a = tokens[:, ::2, :]
    tokens_b = tokens[:, 1::2, :]

    num_a = tokens_a.shape[1]
    num_b = tokens_b.shape[1]

    # 2. Compute Cosine Similarity between tokens_b and tokens_a
    a_norm = torch.nn.functional.normalize(tokens_a, dim=-1)
    b_norm = torch.nn.functional.normalize(tokens_b, dim=-1)

    # sim shape: [B, num_b, num_a]
    sim = torch.bmm(b_norm, a_norm.transpose(1, 2))

    # 3. Find the most similar token in tokens_a for each token in tokens_b
    max_sim, max_idx = sim.max(dim=-1)  # [B, num_b]

    # 4. Pick the top 'r' tokens from tokens_b that have the highest similarity to their match
    _, top_r_idx_in_b = max_sim.topk(r, dim=-1)  # [B, r]

    # Gather the actual tokens_b tokens being merged
    top_r_b_tokens = torch.gather(tokens_b, dim=1, index=top_r_idx_in_b.unsqueeze(-1).expand(-1, -1, D))  # [B, r, D]

    # Gather the corresponding target indices in tokens_a
    top_r_a_idx = torch.gather(max_idx, dim=1, index=top_r_idx_in_b)  # [B, r]

    # 5. Merge Step (Scatter Add)
    tokens_a_merged = tokens_a.clone()
    tokens_a_merged.scatter_add_(dim=1, index=top_r_a_idx.unsqueeze(-1).expand(-1, -1, D), src=top_r_b_tokens)

    # Keep track of how many tokens are merged into each tokens_a token for correct averaging
    A_counts = torch.ones(B, num_a, 1, device=x.device, dtype=x.dtype)
    b_counts = torch.ones(B, r, 1, device=x.device, dtype=x.dtype)
    A_counts.scatter_add_(dim=1, index=top_r_a_idx.unsqueeze(-1), src=b_counts)

    tokens_a_merged = tokens_a_merged / A_counts

    # 6. Keep the unmerged tokens from tokens_b
    b_mask = torch.ones(B, num_b, device=x.device, dtype=torch.bool)
    b_mask.scatter_(dim=1, index=top_r_idx_in_b, value=False)

    all_b_idx = torch.arange(num_b, device=x.device).unsqueeze(0).expand(B, -1)

    # Extract indices where b_mask is True.
    unmerged_b_idx = torch.masked_select(all_b_idx, b_mask).view(B, num_b - r)
    unmerged_b_tokens = torch.gather(tokens_b, dim=1, index=unmerged_b_idx.unsqueeze(-1).expand(-1, -1, D))

    # 7. Concatenate CLS, merged tokens_a, and unmerged tokens_b
    x_reduced = torch.cat([cls_token, tokens_a_merged, unmerged_b_tokens], dim=1)

    return x_reduced


def reorganise_tokens_evit(x, cls_attn_weights, kv_drop_rate):
    """
    EViT: Keeps the top tokens based on CLS attention and fuses the remaining
    "unimportant" tokens into a single new token via averaging.
    """
    if kv_drop_rate <= 0:
        return x

    B, L, D = x.shape
    num_keep = int(L * (1 - kv_drop_rate))

    # Safely clone to avoid in-place autograd errors
    cls_attn_weights_copy = cls_attn_weights.clone()

    # Force the 0th index (CLS) to have infinite importance so it is always kept
    cls_attn_weights_copy[:, 0] = float('inf')

    # Get sorted indices of all tokens by attention (descending)
    sorted_idx = torch.argsort(cls_attn_weights_copy, dim=-1, descending=True)

    # Split into keep and drop index budgets
    keep_idx = sorted_idx[:, :num_keep]
    drop_idx = sorted_idx[:, num_keep:]

    # --- STAGE 1: GATHER KEPT TOKENS ---
    # Sort keep_idx to maintain original relative spatial sequence order
    keep_idx, _ = torch.sort(keep_idx, dim=-1)

    keep_idx_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, D)
    x_keep = torch.gather(x, dim=1, index=keep_idx_expanded)  # [B, num_keep, D]

    # --- STAGE 2: FUSE DROPPED TOKENS ---
    drop_idx_expanded = drop_idx.unsqueeze(-1).expand(-1, -1, D)
    x_drop = torch.gather(x, dim=1, index=drop_idx_expanded)  # [B, L - num_keep, D]

    # Average the discarded tokens into a single new token
    x_fused = x_drop.mean(dim=1, keepdim=True)  # [B, 1, D]

    # --- STAGE 3: MERGE ---
    # Concatenate the kept tokens and the newly fused background token
    x_reduced = torch.cat([x_keep, x_fused], dim=1)  # [B, num_keep + 1, D]

    return x_reduced


def drop_tokens_by_attention_and_diversity(x, cls_attn_weights, kv_drop_rate):
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


def drop_tokens_by_attention(x, cls_attn_weights, kv_drop_rate):
    """
    Drops tokens based on the CLS attention weights from the preceding layer.
    """
    if kv_drop_rate <= 0:
        return x

    B, L, D = x.shape
    num_keep = int(L * (1 - kv_drop_rate))

    # --- ENSURE CLS TOKEN IS KEPT ---
    # Force the 0th index (CLS) to have infinite importance
    cls_attn_weights[:, 0] = float('inf')

    # Get indices of the tokens the CLS paid the most attention to
    _, keep_idx = torch.topk(cls_attn_weights, num_keep, dim=-1, sorted=True)  # [B, num_keep]

    # Gather the kept tokens from the actual feature tensor
    keep_idx_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, D)
    x_reduced = torch.gather(x, dim=1, index=keep_idx_expanded)

    return x_reduced


def zero_tprune_isi(x, full_attn_weights, kv_drop_rate):
    """
    Zero-TPrune with the I -> S -> I pattern.
    - Stage I1: Drops obvious background tokens quickly.
    - Stage S: Drops highly redundant/similar tokens.
    - Stage I2: Drops remaining unimportant tokens using PageRank.
    """
    if kv_drop_rate <= 0:
        return x

    B, L, D = x.shape
    total_drop = int(L * kv_drop_rate)

    # Standard budget allocation (20% / 40% / 40%)
    drop_i1 = int(total_drop * 0.2)
    drop_s = int(total_drop * 0.4)
    drop_i2 = total_drop - drop_i1 - drop_s

    # The attention matrix acts as the adjacency matrix (A)
    A = full_attn_weights.clone()  # [B, L, L]

    # ==========================================
    # STAGE I1: Lightweight Importance Pruning
    # ==========================================
    # Fast ranking using in-degree centrality (sum of incoming attention)
    i1_scores = A.sum(dim=1)  # [B, L]
    i1_scores[:, 0] = float('inf')  # Protect CLS token

    keep_i1 = L - drop_i1
    _, idx_i1 = torch.topk(i1_scores, keep_i1, dim=-1)

    # Sort to maintain spatial sequence order (CLS remains at index 0)
    idx_i1, _ = torch.sort(idx_i1, dim=-1)

    # Gather remaining tokens
    idx_i1_expanded = idx_i1.unsqueeze(-1).expand(-1, -1, D)
    x_i1 = torch.gather(x, 1, idx_i1_expanded)

    # --- Downsample the Attention Graph ---
    # 1. Gather rows (queries)
    A_i1 = torch.gather(A, 1, idx_i1.unsqueeze(-1).expand(-1, -1, L))
    # 2. Gather columns (keys)
    A_i1 = torch.gather(A_i1, 2, idx_i1.unsqueeze(1).expand(-1, keep_i1, -1))
    # 3. Re-normalize rows to sum to 1
    A_i1 = A_i1 / (A_i1.sum(dim=-1, keepdim=True) + 1e-9)

    # ==========================================
    # STAGE S: Similarity Pruning
    # ==========================================
    if drop_s > 0:
        x_i1_norm = torch.nn.functional.normalize(x_i1, p=2, dim=-1)
        sim_matrix = torch.bmm(x_i1_norm, x_i1_norm.transpose(1, 2))

        # Mask out self-similarity (diagonal)
        mask = torch.eye(keep_i1, device=x.device, dtype=torch.bool)
        sim_matrix.masked_fill_(mask, -1.0)

        max_sim, _ = torch.max(sim_matrix, dim=-1)  # [B, keep_i1]
        max_sim[:, 0] = -float('inf')  # Protect CLS token from being dropped

        keep_s = keep_i1 - drop_s
        # Keep tokens with the LOWEST maximum similarity (least redundant)
        _, idx_s = torch.topk(-max_sim, keep_s, dim=-1)
        idx_s, _ = torch.sort(idx_s, dim=-1)

        # Gather remaining tokens
        x_s = torch.gather(x_i1, 1, idx_s.unsqueeze(-1).expand(-1, -1, D))

        # --- Downsample the Attention Graph Again ---
        A_s = torch.gather(A_i1, 1, idx_s.unsqueeze(-1).expand(-1, -1, keep_i1))
        A_s = torch.gather(A_s, 2, idx_s.unsqueeze(1).expand(-1, keep_s, -1))
        A_s = A_s / (A_s.sum(dim=-1, keepdim=True) + 1e-9)
    else:
        x_s = x_i1
        A_s = A_i1
        keep_s = keep_i1

    # ==========================================
    # STAGE I2: Main Importance (Weighted Page Rank)
    # ==========================================
    if drop_i2 > 0:
        # Initialize PageRank vector uniformly
        V = torch.ones(B, keep_s, 1, device=x.device, dtype=x.dtype) / keep_s
        alpha = 0.85

        # Power iteration
        for _ in range(3):
            V = alpha * torch.bmm(A_s, V) + (1 - alpha) / keep_s

        i2_scores = V.squeeze(-1)  # [B, keep_s]
        i2_scores[:, 0] = float('inf')  # Protect CLS token

        keep_i2 = keep_s - drop_i2
        _, idx_i2 = torch.topk(i2_scores, keep_i2, dim=-1)
        idx_i2, _ = torch.sort(idx_i2, dim=-1)

        # Gather final tokens
        x_final = torch.gather(x_s, 1, idx_i2.unsqueeze(-1).expand(-1, -1, D))
    else:
        x_final = x_s

    return x_final


# # ------------------------------------------------------------------------------------
# # ----------------------------------Zero-TPrune ChatGPT-------------------------------
# # ------------------------------------------------------------------------------------

@torch.no_grad()
def zero_tprune(x, attn, key, prune_rate, wpr_iterations=30, vmin=0.01, vmax=0.7, s_prune_ratio=0.5,):
    """
    Zero-TPrune for a classification ViT.

    Args:
        x:
            Token embeddings, [B, N, D].
            Token 0 is assumed to be CLS.

        attn:
            Attention probabilities, [B, H, N, N].

        key:
            Key vectors, [B, H, N, Dh].

        prune_rate:
            Total fraction of tokens to remove at this pruning layer.
            Example: 0.2 means remove 20% of the non-CLS tokens.

        wpr_iterations:
            Number of WPR iterations for the final I-stage.

        vmin, vmax:
            VHF variance thresholds.
            Paper default: [0.01, 0.7].

        s_prune_ratio:
            Fraction of the total requested pruning assigned to the
            S-stage. The rest is performed by the final I-stage.

            Example:
                prune_rate = 0.20
                s_prune_ratio = 0.5

            -> remove 10% through S-stage
            -> remove 10% through I-stage

    Returns:
        x_pruned:
            Pruned tokens [B, N_new, D].
    """

    B, N, D = x.shape
    device = x.device

    if N <= 1 or prune_rate <= 0:
        return x

    # ------------------------------------------------------------
    # Number of tokens to remove.
    #
    # CLS is always preserved.
    # ------------------------------------------------------------

    num_non_cls = N - 1

    total_remove = int(round(num_non_cls * prune_rate))
    total_remove = min(total_remove, num_non_cls)

    if total_remove <= 0:
        return x

    # # Split total pruning between S-stage and I-stage.
    # s_remove = int(round(total_remove * s_prune_ratio))
    # s_remove = min(s_remove, total_remove)
    s_remove = 10                       # ----------------  HARD CODED  ----------------

    i_remove = total_remove - s_remove

    # ============================================================
    # I' STAGE
    # ============================================================
    #
    # The paper explicitly says:
    #
    #   - one round of voting
    #   - no token is pruned here
    #
    # For Zero-TPrune classification, CLS starts with sqrt(N)
    # times the importance of each other token.
    #
    # Ordinary tokens start at:
    #
    #       1 / N
    #
    # CLS starts at:
    #
    #       sqrt(N) / N
    #
    # exactly as specified in the paper.
    # ============================================================

    importance_pre = wpr_importance(
        attn,
        num_iterations=1,
        vmin=vmin,
        vmax=vmax,
        boost_cls=True,
    )

    # ============================================================
    # S STAGE
    # ============================================================

    if s_remove > 0:

        keep_indices_s = similarity_pruning(
            key=key,
            importance=importance_pre,
            num_remove=s_remove,
        )

        x = torch.gather(
            x,
            dim=1,
            index=keep_indices_s.unsqueeze(-1).expand(
                -1, -1, D
            ),
        )

        # --------------------------------------------------------
        # We need attention corresponding to the remaining tokens
        # for the final I-stage.
        # --------------------------------------------------------

        attn = gather_attention(
            attn,
            keep_indices_s,
        )

    else:

        keep_indices_s = torch.arange(
            N,
            device=device,
        ).unsqueeze(0).expand(B, -1)

    # ============================================================
    # FINAL I STAGE
    # ============================================================

    if i_remove > 0:

        importance_final = wpr_importance(
            attn,
            num_iterations=wpr_iterations,
            vmin=vmin,
            vmax=vmax,
            boost_cls=True,
        )

        # Never remove CLS.
        importance_final[:, 0] = float("inf")

        # Lowest importance tokens are removed.
        _, remove_idx = torch.topk(
            importance_final,
            k=i_remove,
            dim=1,
            largest=False,
        )

        keep_mask = torch.ones(
            B,
            x.shape[1],
            dtype=torch.bool,
            device=device,
        )

        keep_mask.scatter_(
            dim=1,
            index=remove_idx,
            value=False,
        )

        keep_indices_i = (
            torch.arange(
                x.shape[1],
                device=device,
            )
            .unsqueeze(0)
            .expand(B, -1)
        )

        keep_indices_i = keep_indices_i[
            keep_mask
        ].reshape(
            B,
            x.shape[1] - i_remove,
        )

        x = torch.gather(
            x,
            dim=1,
            index=keep_indices_i.unsqueeze(-1).expand(
                -1,
                -1,
                D,
            ),
        )

    return x


def wpr_importance(attn, num_iterations, vmin=0.01, vmax=0.7, boost_cls=True,):
    """
    Compute Zero-TPrune importance scores using:

        WPR + EIR + VHF

    Args:
        attn:
            [B, H, N, N]

        num_iterations:
            Number of WPR graph-propagation iterations.

        boost_cls:
            If True, use the classification-specific
            Zero-TPrune initialization where CLS has
            sqrt(N) times larger initial importance.
    """

    B, H, N, _ = attn.shape

    # ------------------------------------------------------------
    # Initialize graph signal.
    # ------------------------------------------------------------

    if boost_cls:

        # Every ordinary token:
        #       1 / N
        #
        # CLS:
        #       sqrt(N) / N
        #
        # Note that the paper describes the CLS score as
        # sqrt(N) times larger than the other tokens.
        scores = torch.full(
            (B, H, N),
            1.0 / N,
            dtype=attn.dtype,
            device=attn.device,
        )

        scores[:, :, 0] *= N ** 0.5

    else:

        scores = torch.full(
            (B, H, N),
            1.0 / N,
            dtype=attn.dtype,
            device=attn.device,
        )

    # ------------------------------------------------------------
    # WPR
    #
    # s_t = A^T s_{t-1}
    #
    # Attention matrix is interpreted as the adjacency matrix
    # of the attention graph.
    # ------------------------------------------------------------

    for _ in range(num_iterations):

        scores = torch.matmul(
            attn.transpose(-1, -2),
            scores.unsqueeze(-1),
        ).squeeze(-1)

    # ------------------------------------------------------------
    # VHF
    #
    # Calculate variance of importance distribution separately
    # for every head.
    # ------------------------------------------------------------

    variance = scores.var(
        dim=-1,
        unbiased=False,
    )  # [B, H]

    valid_heads = (
        (variance >= vmin)
        & (variance <= vmax)
    )

    # Make sure we never end up with zero valid heads.
    no_valid_heads = ~valid_heads.any(dim=1)

    if no_valid_heads.any():

        valid_heads = valid_heads.clone()

        valid_heads[no_valid_heads] = True

    valid = valid_heads.unsqueeze(-1).to(scores.dtype)

    # ------------------------------------------------------------
    # EIR
    #
    # sqrt(
    #       sum_h s_h^2 * valid_h
    #       --------------------
    #           sum_h valid_h
    # )
    # ------------------------------------------------------------

    numerator = (
        scores.square() * valid
    ).sum(dim=1)

    denominator = valid.sum(
        dim=1
    ).clamp_min(1.0)

    importance = torch.sqrt(
        numerator / denominator
    )

    return importance


def similarity_pruning(key, importance, num_remove,):
    """
    Zero-TPrune S-stage using:

        Sequential-U partition
        Key vectors
        cosine similarity

    key:
        [B, H, N, Dh]

    importance:
        [B, N]

    num_remove:
        Number of tokens to remove.

    CLS (token 0) is never considered for removal.

    Returns:
        keep_indices: [B, N - num_remove]
    """

    B, H, N, Dh = key.shape
    device = key.device

    num_candidates = N - 1

    if num_remove <= 0:
        return torch.arange(
            N,
            device=device,
        ).unsqueeze(0).expand(B, -1)

    num_remove = min(
        num_remove,
        num_candidates,
    )

    # ------------------------------------------------------------
    # We never prune CLS.
    # ------------------------------------------------------------

    candidate_indices = torch.arange(
        1,
        N,
        device=device,
    )

    candidate_importance = importance[
        :, 1:
    ]

    # ------------------------------------------------------------
    # Sort non-CLS tokens from most -> least important.
    # ------------------------------------------------------------

    order = torch.argsort(
        candidate_importance,
        dim=-1,
        descending=True,
    )

    ranked_indices = candidate_indices[
        None, :
    ].expand(B, -1).gather(
        1,
        order,
    )

    # ------------------------------------------------------------
    # Sequential-U:
    #
    #     important tokens -> B
    #     less important  -> A
    #
    # Groups are approximately equal sized.
    # ------------------------------------------------------------

    split = num_candidates // 2

    B_indices = ranked_indices[
        :, :split
    ]

    A_indices = ranked_indices[
        :, split:
    ]

    # If there is only one candidate token, there is no useful
    # bipartite comparison.
    if B_indices.shape[1] == 0:

        return torch.arange(
            N,
            device=device,
        ).unsqueeze(0).expand(B, -1)

    # ------------------------------------------------------------
    # Key representation.
    #
    # The paper finds Key vectors + cosine similarity to be
    # the best configuration.
    #
    # We average the heads to obtain one feature vector/token.
    # ------------------------------------------------------------

    token_key = key.mean(
        dim=1
    )  # [B, N, Dh]

    token_key = F.normalize(
        token_key,
        dim=-1,
    )

    A_key = torch.gather(
        token_key,
        1,
        A_indices.unsqueeze(-1).expand(
            -1,
            -1,
            Dh,
        ),
    )

    B_key = torch.gather(
        token_key,
        1,
        B_indices.unsqueeze(-1).expand(
            -1,
            -1,
            Dh,
        ),
    )

    # ------------------------------------------------------------
    # Pairwise cross-group cosine similarity.
    #
    # [B, |A|, |B|]
    # ------------------------------------------------------------

    similarity = torch.matmul(
        A_key,
        B_key.transpose(-1, -2),
    )

    # ------------------------------------------------------------
    # For every token in A:
    #
    #   find its most similar token in B
    # ------------------------------------------------------------

    best_similarity, _ = similarity.max(
        dim=-1
    )  # [B, |A|]

    # ------------------------------------------------------------
    # Select top-r most similar pairs.
    #
    # Corresponding A tokens are pruned.
    # ------------------------------------------------------------

    r = min(
        num_remove,
        A_indices.shape[1],
    )

    _, top_a = torch.topk(
        best_similarity,
        k=r,
        dim=-1,
        largest=True,
    )

    remove_indices = torch.gather(
        A_indices,
        1,
        top_a,
    )

    # ------------------------------------------------------------
    # Build keep mask.
    # ------------------------------------------------------------

    keep_mask = torch.ones(
        B,
        N,
        dtype=torch.bool,
        device=device,
    )

    keep_mask.scatter_(
        1,
        remove_indices,
        False,
    )

    all_indices = torch.arange(
        N,
        device=device,
    ).unsqueeze(0).expand(B, -1)

    keep_indices = all_indices[
        keep_mask
    ].reshape(
        B,
        N - r,
    )

    return keep_indices


def gather_attention(attn, indices):
    """
    Select both rows and columns of an attention matrix.

    attn:
        [B, H, N, N]

    indices:
        [B, N_new]
    """

    B, H, _, _ = attn.shape
    N_new = indices.shape[1]

    # Gather rows.
    attn = torch.gather(attn, dim=2,
        index=indices[:, None, :, None].expand(B, H, N_new, attn.shape[-1],),
    )

    # Gather columns.
    attn = torch.gather(attn,dim=3, index=indices[:, None, None, :].expand(
            B, H, N_new, N_new,),
    )

    return attn

# # ------------------------------------------------------------------------------------
# # ----------------------------------Zero-TPrune ChatGPT-------------------------------
# # ------------------------------------------------------------------------------------

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


# Vision Transformer (with regular attention) for tiny imagenet
class BasicCNNViT(nn.Module):
    def __init__(self, channels=3, img_size=224, patch_size=16, embed_dim=768, num_classes=1000,
                n_layers=12, hidden_dim=3072, dropout=0.0, kv_dropout=0.0, heads=8):
        super(BasicCNNViT, self).__init__()

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

        # self.drop_at_layers = [0, 1, 2, 3, 4, 5]
        # self.drop_at_layers = [2, 5, 8]
        # self.drop_at_layers = [1, 4, 7]
        # self.drop_at_layers = [0, 3, 6]
        # self.drop_at_layers = [2]
        self.drop_at_layers = [0]

        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.ReLU(),
        #     nn.Conv2d(64, 96, kernel_size=2, stride=2, padding=0),  # -> (B, 256, 28, 28) RF 15x15
        #     nn.ReLU(),
        #     nn.Conv2d(96, embed_dim, kernel_size=2, stride=2, padding=0),  # -> (B, embed_dim, 14, 14) RF 31x31
        # )

        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.ReLU(),
        #     nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),  # -> (B, 256, 28, 28) RF 15x15
        #     nn.ReLU(),
        #     nn.Conv2d(96, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, embed_dim, 14, 14) RF 31x31
        # )
        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.ReLU(),
        #     nn.Conv2d(32, 96, kernel_size=4, stride=2, padding=0),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.ReLU(),
        #     # nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),  # -> (B, 256, 28, 28) RF 15x15
        #     # nn.ReLU(),
        #     nn.Conv2d(96, embed_dim, kernel_size=3, stride=2, padding=0),  # -> (B, embed_dim, 14, 14) RF 31x31
        # )
        # # num_tokens = (14 ** 2) + 1
        # #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=0),  # -> (B, 64, 7, 7)   RF ?x?
        #     )
        #
        # num_tokens = (14 ** 2) + 1

        # ------------------------------------------------------------------------------------
        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.ReLU(),
        #     nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, embed_dim, 14, 14) RF 31x31
        # )

        # ------------------------------------------------------------------------------------
        #
        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=0),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # -> (B, 256, 28, 28) RF 15x15
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=2, stride=2, padding=0),  # -> (B, embed_dim, 14, 14) RF 31x31
        # )
        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 256, 28, 28) RF 15x15
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, embed_dim, 14, 14) RF 31x31
        # )
        #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 7, 7)   RF ?x?
        #     )
        #
        # self.patch_embedding3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 7, 7)   RF ?x?
        # )
        #
        # num_tokens = (14 ** 2) + 1

        # ------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------

        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  #
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  #
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  #
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),  #
        # )
        #
        # # self.patch_embedding2 = nn.Sequential(
        # #     nn.ReLU(),
        # #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=0),  # -> (B, 64, 7, 7)   RF ?x?
        # #     )
        #
        # num_tokens = (14 ** 2) + 1

        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=0),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # -> (B, 256, 28, 28) RF 15x15
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=2, stride=2, padding=0),  # -> (B, embed_dim, 14, 14) RF 31x31
        # )
        # num_tokens = (14 ** 2) + 1

        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=0),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.ReLU(),
        #     nn.Conv2d(16, 64, kernel_size=2, stride=2, padding=0),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=3, stride=3, padding=1),  # -> (B, embed_dim, 14, 14) RF 31x31
        # )
        # num_tokens = (14 ** 2) + 1

        # ------------------------------------------------------------------------------------
        #
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        # )
        # self.patch_embedding1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)     # -> (B, 128, 28, 28)
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        #     )     # -> (B, 128, 28, 28)
        # self.patch_embedding2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)     # -> (B, 128, 14, 14)
        #
        # self.conv_layer3 = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1))     # -> (B, 128, 14, 14)
        # self.patch_embedding3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)     # -> (B, 128, 7, 7)

        # ------------------------------------------------------------------------------------
        #
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(64),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # )
        # self.patch_embedding1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)     # -> (B, 128, 28, 28)
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 28, 28)
        # self.patch_embedding2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)     # -> (B, 128, 14, 14)
        # #
        # self.conv_layer3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 14, 14)
        # self.patch_embedding3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)     # -> (B, 128, 7, 7)

        # # ------------------------------------------------------------------------------------
        # # # # # # # # Double / Triple Scale
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(128),     # Norm after every conv
        # )
        # self.patch_embedding1 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 28, 28)   RF 3x3
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # )
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(192),     # Norm after every conv
        #     )     # -> (B, 128, 28, 28)
        #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(192, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 14, 14)
        # # #
        # self.conv_layer3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 14, 14)
        #
        # self.patch_embedding3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 7, 7)

        # # ------------------------------------------------------------------------------------
        # # ------------------------------------------------------------------------------------
        # # ------------------------------------------------------------------------------------
        # # # # # Single Scale
        # self.patch_embedding1 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16, padding=0)
        # #
        self.patch_embedding1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
            nn.BatchNorm2d(32),     # Norm after every conv
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
            nn.BatchNorm2d(64),     # Norm after every conv
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),     # Norm after every conv
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),     # Norm after every conv
            # nn.ReLU(),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(embed_dim),     # Norm after every conv
        )
        # #
        # self.patch_embedding1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(128),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     # nn.ReLU(),
        #     # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # )

        # self.patch_embedding1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(96),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(192),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(192, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     # nn.ReLU(),
        #     # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # )

        # self.patch_embedding1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(96),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(96, 288, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(288),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(288, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     # nn.ReLU(),
        #     # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # )

        # # ------------------------------------------------------------------------------------
        # # ------------------------------------------------------------------------------------
        # # ------------------------------------------------------------------------------------
        # # # # Simple Double  Scale
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(64),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(128),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 28, 28)   RF 3x3
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 14, 14)
        # #

        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(128),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 28, 28)   RF 3x3
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 14, 14)
        #

        # # # self.conv_layer3 = nn.Sequential(
        # # #     nn.ReLU(),
        # # #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        # # #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # # #     )     # -> (B, 128, 14, 14)
        # # #
        # # # self.patch_embedding3 = nn.Sequential(
        # # #     nn.ReLU(),
        # # #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        # # #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # # #     )     # -> (B, 128, 7, 7)

        # # ------------------------------------------------------------------------------------
        # # ------------------------------------------------------------------------------------
        # # ------------------------------------------------------------------------------------
        # # # # # Double / Triple Scale
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(128),  # Norm after every conv
        #     nn.ReLU(),
        # )
        # self.patch_embedding1 = nn.Sequential(
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 28, 28)   RF 3x3
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),  # Norm after every conv
        # )  # -> (B, 128, 28, 28)
        #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 14, 14)

        # # ------------------------------------------------------------------------------------
        # # ------------------------------------------------------------------------------------
        # # ------------------------------------------------------------------------------------
        # #
        # # # 10700
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(128),  # Norm after every conv
        # )
        # self.patch_embedding1 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=5, stride=4, padding=1),  # -> (B, 64, 28, 28)   RF 3x3
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 28, 28)
        #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 14, 14)
        #
        # self.smooth = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 14, 14)

        #
        # # # # 10800
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),  # Norm after every conv
        #
        # )
        # self.patch_embedding1 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=12, stride=8, padding=4),  # -> (B, 64, 28, 28)   RF 3x3
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(128),  # Norm after every conv
        # )  # -> (B, 128, 28, 28)
        #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=6, stride=4, padding=2),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 14, 14)
        #
        # self.conv_layer3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 28, 28)
        #
        # self.patch_embedding3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 14, 14)

        # #
        # self.conv_layer3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 14, 14)
        #
        # self.patch_embedding3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 7, 7)
        #
        # # # 11900
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # -> (B, 32, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),  # Norm after every conv
        #     # nn.ReLU(),
        # )
        # self.skip1 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(3, 32, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(32),  # Norm after every conv
        # )
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # -> (B, 64, 56, 56)   RF 3x3
        #     nn.BatchNorm2d(64),  # Norm after every conv
        #     # nn.ReLU(),
        # )
        # self.skip2 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(64),  # Norm after every conv
        # )
        #
        # self.conv_layer3 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # -> (B, 128, 28, 28)   RF 3x3
        #     nn.BatchNorm2d(128),  # Norm after every conv
        #     # nn.ReLU(),
        # )
        # self.skip3 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(128),  # Norm after every conv
        # )
        #
        # self.conv_layer4 = nn.Sequential(
        #     nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),  # -> (B, 192, 14, 14)   RF 3x3
        #     nn.BatchNorm2d(192),  # Norm after every conv
        #     # nn.ReLU(),
        # )
        # self.skip4 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(128, 192, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(192),  # Norm after every conv
        # )

        # ------------------------------------------------------------------------------------
        # # # # # # # # # SMALLER Double / Triple Scale
        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(128),  # Norm after every conv
        # )
        # self.patch_embedding1 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 28, 28)   RF 3x3
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 28, 28)
        #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 14, 14)


        # self.conv_layer2 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(128),  # Norm after every conv
        #     nn.ReLU(),
        # )  # -> (B, 128, 28, 28)
        #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.Conv2d(128, embed_dim, kernel_size=2, stride=2, padding=0),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.BatchNorm2d(embed_dim)
        # )  # -> (B, 128, 14, 14)

#
# for x1 (28x28) i apply 3 conv layers with k=3, s=2, p=1 (with batchnorm and relu)
# then i apply another conv(k=3, s=2, p=1),batchnorm  on x1 to get x2
# and conv(k=3, s=2, p=1),batchnorm,relu,conv(k=3, s=2, p=1),batchnorm to get x3
# and i take only x2 (196 tokens) and x3 (49 tokens)
#
# so I want to try a different configuration where x3 is computed with maxpooling

        # # #
        # self.conv_layer3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 14, 14)
        #
        # self.patch_embedding3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     )     # -> (B, 128, 7, 7)


        # ------------------------------------------------------------------------------------
        # # # # # SMALLER Single Scale
        # self.patch_embedding1 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16, padding=0)
        # # # #
        # self.patch_embedding1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(128),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     # nn.ReLU(),
        #     # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # )

        # self.patch_embedding1 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16, padding=0)
        # # #
        # self.patch_embedding1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(128),     # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),     # Norm after every conv
        #     # nn.ReLU(),
        #     # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # )

        # self.conv_layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(32),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(64),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(128),  # Norm after every conv
        # )
        # self.patch_embedding1 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 28, 28)   RF 3x3
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )
        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),  # Norm after every conv
        # )  # -> (B, 128, 28, 28)
        #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )  # -> (B, 128, 14, 14)

        # ------------------------------------------------------------------------------------
        # # # # # Much SMALLER 7x7 Single Scale
        # self.patch_embedding1 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16, padding=0)
        # # # #
        # self.patch_embedding1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(64),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 56, 56)   RF 3x3
        #     nn.BatchNorm2d(128),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 28, 28)  RF 7x7
        #     nn.BatchNorm2d(192),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 14, 14)  RF 7x7
        #     nn.BatchNorm2d(192),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(192, embed_dim, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 7, 7)  RF 7x7
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
        #     # nn.ReLU(),
        #     # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(embed_dim),     # Norm after every conv
        # )

        # ------------------------------------------------------------------------------------
        # # # 4 Scales

        #
        # self.conv_layer56 = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)
        #     nn.BatchNorm2d(16),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 56, 56)
        #     nn.BatchNorm2d(32),
        # )
        # self.patch_embedding56 = nn.Sequential(
        #     nn.Conv2d(32, embed_dim, kernel_size=8, stride=8, padding=0),  # -> (B, 64, 7, 7)
        #     nn.BatchNorm2d(embed_dim),
        # )
        #
        # self.conv_layer28 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 28, 28)
        #     nn.BatchNorm2d(64),
        # )
        #
        # self.patch_embedding28 = nn.Sequential(
        #     nn.Conv2d(64, embed_dim, kernel_size=4, stride=4, padding=0),   # -> (B, 64, 7, 7)
        #     nn.BatchNorm2d(embed_dim),
        # )
        # self.conv_layer14 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),     # -> (B, 64, 14, 14)
        #     nn.BatchNorm2d(128),
        # )
        #
        # self.patch_embedding14 = nn.Sequential(
        #     nn.Conv2d(128, embed_dim, kernel_size=2, stride=2, padding=0),      # -> (B, 64, 7, 7)
        #     nn.BatchNorm2d(embed_dim),
        # )
        #
        # self.conv_layer7 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),      # -> (B, 64, 7, 7)
        #     nn.BatchNorm2d(embed_dim),
        # )


        # self.conv_layer56 = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)
        #     nn.BatchNorm2d(16),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 56, 56)
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 28, 28)
        #     nn.BatchNorm2d(64),
        # )
        # self.patch_embedding56 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=4, stride=4, padding=0),  # -> (B, 64, 7, 7)
        #     nn.BatchNorm2d(embed_dim),
        # )

        #
        # self.conv_layer56 = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)
        #     nn.BatchNorm2d(16),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 56, 56)
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )
        #
        # self.patch_embedding56 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(32, embed_dim, kernel_size=8, stride=8, padding=0),  # -> (B, 64, 7, 7)
        #     nn.BatchNorm2d(embed_dim),
        # )
        #
        #
        # self.conv_layer28 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 28, 28)
        #     nn.BatchNorm2d(64),
        # )
        #
        # self.patch_embedding28 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(64, embed_dim, kernel_size=4, stride=4, padding=0),   # -> (B, 64, 7, 7)
        #     nn.BatchNorm2d(embed_dim),
        # )
        # self.conv_layer14 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),     # -> (B, 64, 14, 14)
        #     nn.BatchNorm2d(128),
        # )
        #
        # self.patch_embedding14 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=2, stride=2, padding=0),      # -> (B, 64, 7, 7)
        #     nn.BatchNorm2d(embed_dim),
        # )
        #
        # self.conv_layer7 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),      # -> (B, 64, 7, 7)
        #     nn.BatchNorm2d(embed_dim),
        # )

        # self.conv_layer7 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),      # -> (B, 64, 14, 14)
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=2, stride=2, padding=0),      # -> (B, 64, 14, 14)
        #     nn.BatchNorm2d(embed_dim),
        # )
        # self.norm_o1 = nn.LayerNorm(embed_dim)
        # self.norm_o2 = nn.LayerNorm(embed_dim)
        # self.norm_o3 = nn.LayerNorm(embed_dim)
        # self.norm_o4 = nn.LayerNorm(embed_dim)

        # ------------------------------------------------------------------------------------
        # self.patch_embedding1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, 112, 112)   RF 3x3
        #     nn.BatchNorm2d(64),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, 56, 56)  RF 7x7
        #     nn.BatchNorm2d(128),  # Norm after every conv
        #     nn.ReLU(),
        #     nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(embed_dim),  # Norm after every conv
            # nn.ReLU(),
            # nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(embed_dim),  # Norm after every conv
        # )


        # ------------------------------------------------------------------------------------

        # self.patch_embedding1 = nn.Sequential(
        #     nn.Conv2d(3, embed_dim, kernel_size=8, stride=8, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0))     # -> (B, 128, 28, 28)
        #
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0))     # -> (B, 128, 28, 28)

        # self.patch_embedding1 = nn.Conv2d(3, embed_dim, kernel_size=8, stride=8, padding=0)
        # self.patch_embedding2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0))     # -> (B, 128, 28, 28)

        #
        # self.conv_layer2 = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU())     # -> (B, 128, 28, 28)
        # self.patch_embedding2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)     # -> (B, 128, 14, 14)

        # self.conv_layer3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1))     # -> (B, 128, 14, 14)
        # self.patch_embedding3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)     # -> (B, 128, 7, 7)

        #
        # ------------------------------------------------------------------------------------

        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=12, padding=0))
        #
        # num_tokens = (15 ** 2) + 1
        #
        # self.patch_embedding = nn.Sequential(
        #     nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=14, padding=0))
        # self.patch_embedding2 = nn.Sequential(
        #     nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=14, padding=0))
        #
        # # num_tokens = 784 + 196 + 1
        # num_tokens = 196 + 49 + 1
        #
        # # self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, embed_dim))
        # #

        # ------------------------------------------------------------------------------------

        # pos_embed_1 = generate_2d_sincos_embedding(16, embed_dim, step=1.0)
        # # 2. Low-Res Grid (8x8) -> We use step=2.0 to align it spatially with grid 1
        # pos_embed_2 = generate_2d_sincos_embedding(8, embed_dim, step=2.0)
        # #
        # pos_embed_1 = generate_2d_sincos_embedding(14, embed_dim, step=1.0)
        # # 2. Low-Res Grid (8x8) -> We use step=2.0 to align it spatially with grid 1
        # pos_embed_2 = generate_2d_sincos_embedding(7, embed_dim, step=2.0)
        # pos_embed_3 = generate_2d_sincos_embedding(4, embed_dim, step=4.0)
        #
        # # 3. Combine them and add a zero-vector for the [CLS] token at the very beginning
        # cls_pos_embed = torch.zeros(1, embed_dim)
        #
        # # Shape: [1 + 256 + 64, embed_dim] -> [321, embed_dim]
        # combined_pos_embed = torch.cat([cls_pos_embed, pos_embed_1, pos_embed_2, pos_embed_3], dim=0)
        #
        # # Add batch dimension: [1, 321, embed_dim]
        # combined_pos_embed = combined_pos_embed.unsqueeze(0)
        #
        # # Register as a buffer instead of a Parameter.
        # # This means it will be saved in the state_dict, pushed to GPU automatically,
        # # but won't be updated by the optimizer (saves memory/compute).
        # self.register_buffer('pos_embedding', combined_pos_embed)

        # ------------------------------------------------------------------------------------

        # pos_embed_1 = generate_2d_sincos_embedding(16, embed_dim, step=1.0)
        # # 2. Low-Res Grid (8x8) -> We use step=2.0 to align it spatially with grid 1
        # pos_embed_2 = generate_2d_sincos_embedding(8, embed_dim, step=2.0)
        # #
        # pos_embed_1 = generate_2d_sincos_embedding(27, embed_dim, step=1.0)
        # # 2. Low-Res Grid (8x8) -> We use step=2.0 to align it spatially with grid 1
        # pos_embed_2 = generate_2d_sincos_embedding(13, embed_dim, step=2.0)
        #
        # # 3. Combine them and add a zero-vector for the [CLS] token at the very beginning
        # cls_pos_embed = torch.zeros(1, embed_dim)
        #
        # # Shape: [1 + 256 + 64, embed_dim] -> [321, embed_dim]
        # combined_pos_embed = torch.cat([cls_pos_embed, pos_embed_1, pos_embed_2], dim=0)
        #
        # # Add batch dimension: [1, 321, embed_dim]
        # combined_pos_embed = combined_pos_embed.unsqueeze(0)
        #
        # # Register as a buffer instead of a Parameter.
        # # This means it will be saved in the state_dict, pushed to GPU automatically,
        # # but won't be updated by the optimizer (saves memory/compute).
        # self.register_buffer('pos_embedding', combined_pos_embed)

        # ------------------------------------------------------------------------------------

        # pos_embed_1 = generate_2d_sincos_embedding(28, embed_dim, step=1.0)
        #
        # # 3. Combine them and add a zero-vector for the [CLS] token at the very beginning
        # cls_pos_embed = torch.zeros(1, embed_dim)
        #
        # # Shape: [1 + 256 + 64, embed_dim] -> [321, embed_dim]
        # combined_pos_embed = torch.cat([cls_pos_embed, pos_embed_1], dim=0)
        #
        # # Add batch dimension: [1, 321, embed_dim]
        # combined_pos_embed = combined_pos_embed.unsqueeze(0)
        #
        # # Register as a buffer instead of a Parameter.
        # # This means it will be saved in the state_dict, pushed to GPU automatically,
        # # but won't be updated by the optimizer (saves memory/compute).
        # self.register_buffer('pos_embedding', combined_pos_embed)
        # ------------------------------------------------------------------------------------

        # # pos_embed_2 = generate_2d_sincos_embedding(14, embed_dim, step=2.0).unsqueeze(0)
        # # pos_embed_3 = generate_2d_sincos_embedding(7, embed_dim, step=4.0).unsqueeze(0)
        pos_embed_1 = generate_2d_sincos_embedding(14, embed_dim, step=1.0).unsqueeze(0)

        # # 3. Combine them and add a zero-vector for the [CLS] token at the very beginning
        cls_pos_embed = torch.zeros(1, embed_dim).unsqueeze(0)

        self.register_buffer('pos_embedding0', cls_pos_embed)

        self.register_buffer('pos_embedding1', pos_embed_1)
        # # self.register_buffer('pos_embedding2', pos_embed_2)
        # # self.register_buffer('pos_embedding3', pos_embed_3)


        # self.pos_embedding1 = nn.Parameter(torch.randn(1, 49, embed_dim))
        # self.pos_embedding2 = nn.Parameter(torch.randn(1, 49, embed_dim))
        # self.pos_embedding3 = nn.Parameter(torch.randn(1, 49, embed_dim))

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.scale_pos_embed1 = nn.Parameter(torch.rand(1, 1, embed_dim))
        #
        # pos_embed_2 = generate_2d_sincos_embedding(7, embed_dim, step=2.0).unsqueeze(0)
        # self.register_buffer('pos_embedding2', pos_embed_2)
        # self.scale_pos_embed2 = nn.Parameter(torch.rand(1, 1, embed_dim))
        # #
        # pos_embed_3 = generate_2d_sincos_embedding(7, embed_dim, step=4.0).unsqueeze(0)
        # self.register_buffer('pos_embedding3', pos_embed_3)
        # self.scale_pos_embed3 = nn.Parameter(torch.rand(1, 1, embed_dim))

        # self.scale_pos_embed2 = nn.Parameter(torch.rand(1, 1, embed_dim))
        # self.scale_pos_embed3 = nn.Parameter(torch.rand(1, 1, embed_dim))
        # self.scale_pos_embed4 = nn.Parameter(torch.rand(1, 1, embed_dim))

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


    def forward(self, x, prune_rate=0.0, return_tokens=False):
        B, C, H, W = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)

        # ------------------------------------------------------------------------------------
        # x = self.patch_embedding(x)
        # x = x.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # x = torch.cat([cls_tokens, x], dim=1)
        # x = x + self.pos_embedding

        # ------------------------------------------------------------------------------------

        # x_shrunk2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        # x_shrunk2 = self.patch_embedding2(x_shrunk2)
        # x_shrunk2 = x_shrunk2.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        #
        # x = self.patch_embedding(x)
        # x = x.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        #
        # x = torch.cat([cls_tokens, x, x_shrunk2], dim=1)
        # x = x + self.pos_embedding

        # ------------------------------------------------------------------------------------
        # x1 = self.patch_embedding(x)
        # x2 = self.patch_embedding2(x1)
        # x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # x2 = x2.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # x = torch.cat([cls_tokens, x1, x2], dim=1)
        # x = x + self.pos_embedding
        # # ------------------------------------------------------------------------------------
        # x1 = self.patch_embedding(x)
        # x2 = self.patch_embedding2(x1)
        # x3 = self.patch_embedding3(x2)
        # x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # x2 = x2.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # x3 = x3.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # x = torch.cat([cls_tokens, x1, x2, x3], dim=1)
        # x = x + self.pos_embedding
        # ------------------------------------------------------------------------------------

        # x1 = self.conv_layer1(x)
        # x1 = self.patch_embedding1(x1)
        # x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        #
        # x = torch.cat([cls_tokens, x1], dim=1)
        # x = x + self.pos_embedding

        # # ------------------------------------------------------------------------------------
        # #
        # cls_tokens = cls_tokens + self.pos_embedding0
        # x1 = self.conv_layer1(x)
        # x1 = self.patch_embedding1(x1)
        # x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x1 = x1 + self.pos_embedding1 + self.scale_pos_embed1
        #
        # # N = x1.shape[1]
        # #
        # # k = x1.shape[1] - 100
        # # assert k <= N
        # #
        # # # print("x1:", x1.shape)
        # #
        # # idx = torch.rand(B, N, device=x1.device).argsort(dim=1)[:, :k]
        # # idx = idx.sort(dim=1).values
        # #
        # # assert idx.max() < N
        # # assert idx.min() >= 0
        # # assert idx.dtype == torch.long
        # #
        # # idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        # #
        # # x1_reduced = torch.gather(x1, 1, idx)
        # #
        # # x = torch.cat([cls_tokens, x1_reduced], dim=1)
        #
        # x = torch.cat([cls_tokens, x1], dim=1)

        # # ------------------------------------------------------------------------------------
        #
        # cls_tokens = cls_tokens + self.pos_embedding0
        # x1 = self.conv_layer1(x)
        # x1 = self.patch_embedding1(x1)
        # x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x1 = x1 + self.pos_embedding1
        #
        # N = x1.shape[1]
        #
        # k = x1.shape[1] - 100
        # assert k <= N
        #
        # # print("x1:", x1.shape)
        #
        # idx = torch.rand(B, N, device=x1.device).argsort(dim=1)[:, :k]
        # idx = idx.sort(dim=1).values
        #
        # assert idx.max() < N
        # assert idx.min() >= 0
        # assert idx.dtype == torch.long
        #
        # idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        #
        # x1_reduced = torch.gather(x1, 1, idx)
        #
        # x = torch.cat([cls_tokens, x1_reduced], dim=1)
        #
        # # x = torch.cat([cls_tokens, x1, x2], dim=1)
        # ------------------------------------------------------------------------------------

        # # # Simple Double Scale
        # cls_tokens = cls_tokens + self.pos_embedding0
        # x1 = self.conv_layer1(x)
        # x2 = self.conv_layer2(x1)
        # # x1 = self.patch_embedding1(x1)
        # # x1 = x1[:, :, 1:, 1:].permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x1 = x1 + self.pos_embedding1 + self.scale_pos_embed1
        # # x2 = self.patch_embedding2(x2)
        # x2 = x2.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # x2 = x2 + self.pos_embedding2 + self.scale_pos_embed2
        # # N = x1.shape[1]
        # # k = x1.shape[1] - x2.shape[1]
        # # assert k <= N
        # # idx = torch.rand(B, N, device=x1.device).argsort(dim=1)[:, :k]
        # # idx = idx.sort(dim=1).values
        # # assert idx.max() < N
        # # assert idx.min() >= 0
        # # assert idx.dtype == torch.long
        # # idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        # # x1_reduced = torch.gather(x1, 1, idx)
        # # x = torch.cat([cls_tokens, x1_reduced, x2], dim=1)
        # x = torch.cat([cls_tokens, x1, x2], dim=1)

        # # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # # #
        # # #   SINGLE SCALE
        cls_tokens = cls_tokens + self.pos_embedding0

        x1 = self.patch_embedding1(x)
        # x2 = self.patch_embedding2(x1)

        x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x2 = x2.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)

        x1 = x1 + self.pos_embedding1 + self.scale_pos_embed1
        # x2 = x2 + self.pos_embedding2
        # N = x1.shape[1]
        # k = x1.shape[1] - x2.shape[1]
        # assert k <= N
        # idx = torch.rand(B, N, device=x1.device).argsort(dim=1)[:, :k]
        # idx = idx.sort(dim=1).values
        # assert idx.max() < N
        # assert idx.min() >= 0
        # assert idx.dtype == torch.long
        # idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        # x1_reduced = torch.gather(x1, 1, idx)
        # x = torch.cat([cls_tokens, x1_reduced, x2], dim=1)
        # x = torch.cat([cls_tokens, x1, x2], dim=1)
        x = torch.cat([cls_tokens, x1], dim=1)

        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------

        # # Double Scale
        # cls_tokens = cls_tokens + self.pos_embedding0
        # x1 = self.conv_layer1(x)
        # x2 = self.conv_layer2(x1)
        # x1 = self.patch_embedding1(x1)
        # # x1 = x1[:, :, 1:, 1:].permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x1 = x1 + self.pos_embedding1 + self.scale_pos_embed1
        # x2 = self.patch_embedding2(x2)
        # x2 = x2.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # # x2 = x2 + self.pos_embedding2 + self.scale_pos_embed2
        # x2 = x2 + self.pos_embedding2 + self.scale_pos_embed2
        # # N = x1.shape[1]
        # # k = x1.shape[1] - x2.shape[1]
        # # assert k <= N
        # # idx = torch.rand(B, N, device=x1.device).argsort(dim=1)[:, :k]
        # # idx = idx.sort(dim=1).values
        # # assert idx.max() < N
        # # assert idx.min() >= 0
        # # assert idx.dtype == torch.long
        # # idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        # # x1_reduced = torch.gather(x1, 1, idx)
        # # x = torch.cat([cls_tokens, x1_reduced, x2], dim=1)
        # x = torch.cat([cls_tokens, x1, x2], dim=1)

        # # # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------

        # # # Triple Scale
        # cls_tokens = cls_tokens + self.pos_embedding0
        # x1 = self.conv_layer1(x)
        # x2 = self.conv_layer2(x1)
        # x3 = self.conv_layer3(x2)
        # x1 = self.patch_embedding1(x1)
        # x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x1 = x1 + self.pos_embedding1 + self.scale_pos_embed1
        # x2 = self.patch_embedding2(x2)
        # x2 = x2.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # # x2 = x2 + self.pos_embedding2 + self.scale_pos_embed2
        # x2 = x2 + self.pos_embedding1 + self.scale_pos_embed2
        # x3 = self.patch_embedding3(x3)
        # x3 = x3.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # # x3 = x3 + self.pos_embedding3 + self.scale_pos_embed3
        # x3 = x3 + self.pos_embedding1 + self.scale_pos_embed3
        #
        # # N = x1.shape[1]
        # # k = x1.shape[1] - x2.shape[1] - x3.shape[1]
        # # assert k <= N
        # # idx = torch.rand(B, N, device=x1.device).argsort(dim=1)[:, :k]
        # # idx = idx.sort(dim=1).values
        # # assert idx.max() < N
        # # assert idx.min() >= 0
        # # assert idx.dtype == torch.long
        # # idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        # # x1_reduced = torch.gather(x1, 1, idx)
        # # x = torch.cat([cls_tokens, x1_reduced, x2, x3], dim=1)
        #
        # x = torch.cat([cls_tokens, x1, x2, x3], dim=1)

        # # ------------------------------------------------------------------------------------
        # # # # # # # Feature Pyramid Network
        #
        # cls_tokens = cls_tokens + self.pos_embedding0
        #
        # x1 = self.conv_layer1(x)
        # x2 = self.conv_layer2(x1)
        # x1 = self.patch_embedding1(x1)
        # x2 = self.patch_embedding2(x2)
        #
        # x1 = self.smooth(x1 + x2)
        #
        # # x1 = x1[:, :, 1:, 1:].permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        # x1 = x1 + self.pos_embedding1 + self.scale_pos_embed1
        #
        # x2 = x2.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        # # x2 = x2 + self.pos_embedding2 + self.scale_pos_embed2
        # x2 = x2 + self.pos_embedding1 + self.scale_pos_embed2
        # # N = x1.shape[1]
        # # k = x1.shape[1] - x2.shape[1]
        # # assert k <= N
        # # idx = torch.rand(B, N, device=x1.device).argsort(dim=1)[:, :k]
        # # idx = idx.sort(dim=1).values
        # # assert idx.max() < N
        # # assert idx.min() >= 0
        # # assert idx.dtype == torch.long
        # # idx = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        # # x1_reduced = torch.gather(x1, 1, idx)
        # # x = torch.cat([cls_tokens, x1_reduced, x2], dim=1)
        # x = torch.cat([cls_tokens, x1, x2], dim=1)
        # # ------------------------------------------------------------------------------------
        #
        # # # # #   11900 SINGLE SCALE Wit Residuals
        # cls_tokens = cls_tokens + self.pos_embedding0
        #
        # x = nn.functional.relu(self.conv_layer1(x) + self.skip1(x))
        # x = nn.functional.relu(self.conv_layer2(x) + self.skip2(x))
        # x = nn.functional.relu(self.conv_layer3(x) + self.skip3(x))
        # x = self.conv_layer4(x) + self.skip4(x)
        #
        # x = x.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)    # (B, 785, 128)
        #
        # x = x + self.pos_embedding1 + self.scale_pos_embed1
        # x = torch.cat([cls_tokens, x], dim=1)

        # # ------------------------------------------------------------------------------------

        # x = x + self.pos_embedding
        x = self.dropout(x)
        # Transformer layers
        for i in range(self.n_layers):

            x, cls_attn_weights = self.layers[i](x)   # comment if using zero tprune

            # Check if this layer is scheduled for a drop
            if i in self.drop_at_layers and prune_rate > 0:
                # x = drop_tokens_by_attention(x, cls_attn_weights, prune_rate)
                x = drop_tokens_by_attention_and_diversity(x, cls_attn_weights, prune_rate)     # top k
                # x = reorganise_tokens_evit(x, cls_attn_weights, prune_rate)                   # EViT
                # x = merge_tokens_tome(x, cls_attn_weights, prune_rate)                          # ToMe.
                # x = zero_tprune_isi(x, cls_attn_weights, prune_rate)                          # Zero-TPrune Gemini

            #     (x, cls_attn_weights, attn, key,) = self.layers[i](x, return_aux=True,)
            #     x = zero_tprune(x, attn, key, prune_rate=prune_rate,)
            # else:
            #     x, cls_attn_weights = self.layers[i](x, return_aux=False,)

        # Output based on classification token
        if return_tokens:
            x = self.head[0](x)
            tokens = x[:, 1:, :]
            logits = self.head[1](x[:, 0, :])
            return logits, tokens
        return self.head(x[:, 0, :])


