import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ScaledSiLU(nn.Module):
    def __init__(self, inplace=False):
        super(ScaledSiLU, self).__init__()
        self.inplace = inplace
        self.scale_factor = 1.6791767923989418

    def forward(self, inputs):
        return F.silu(inputs, inplace=self.inplace) * self.scale_factor

    def extra_repr(self):
        str = 'scale_factor={}'.format(self.scale_factor)
        if self.inplace:
            str = str + ', inplace=True'
        return str


class ScaledSwiGLU(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ScaledSwiGLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = torch.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = ScaledSiLU()

    def forward(self, inputs):
        w = self.w(inputs)
        w_1 = w.narrow(-1, 0, self.out_channels)
        w_1 = self.act(w_1)
        w_2 = w.narrow(-1, self.out_channels, self.out_channels)
        out = w_1 * w_2
        return out


class SwiGLU(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SwiGLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = torch.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = torch.nn.SiLU()

    def forward(self, inputs):
        w = self.w(inputs)
        w_1 = w.narrow(-1, 0, self.out_channels)
        w_1 = self.act(w_1)
        w_2 = w.narrow(-1, self.out_channels, self.out_channels)
        out = w_1 * w_2
        return out


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)


class ScaledSmoothLeakyReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = SmoothLeakyReLU(0.2)
        self.scale_factor = 1.531320475574866

    def forward(self, x):
        return self.act(x) * self.scale_factor

    def extra_repr(self):
        return 'negative_slope={}, scale_factor={}'.format(self.act.alpha, self.scale_factor)


class ScaledSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1.8467055342154763

    def forward(self, x):
        return torch.sigmoid(x) * self.scale_factor


class GateActivation(torch.nn.Module):
    def __init__(self, lmax, mmax, num_channels):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels

        num_components = 0
        for l in range(1, self.lmax + 1):
            num_m_components = min((2 * l + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        expand_index = torch.zeros([num_components]).long()
        start_idx = 0
        for l in range(1, self.lmax + 1):
            length = min((2 * l + 1), (2 * self.mmax + 1))
            expand_index[start_idx : (start_idx + length)] = (l - 1)
            start_idx = start_idx + length
        self.register_buffer('expand_index', expand_index)

        self.scalar_act = torch.nn.SiLU()
        self.gate_act   = torch.nn.Sigmoid()

    def forward(self, gating_scalars, input_tensors):
        gating_scalars = self.gate_act(gating_scalars)
        gating_scalars = gating_scalars.reshape(gating_scalars.shape[0], self.lmax, self.num_channels)
        gating_scalars = torch.index_select(gating_scalars, dim=1, index=self.expand_index)
        input_tensors_scalars = input_tensors.narrow(1, 0, 1)
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)
        input_tensors_vectors = input_tensors.narrow(1, 1, input_tensors.shape[1] - 1)
        input_tensors_vectors = input_tensors_vectors * gating_scalars
        output_tensors = torch.cat((input_tensors_scalars, input_tensors_vectors), dim=1)
        return output_tensors


class S2Activation(torch.nn.Module):
    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()

    def forward(self, inputs, SO3_grid):
        to_grid_mat   = SO3_grid[self.lmax][self.mmax].get_to_grid_mat(device=None)
        from_grid_mat = SO3_grid[self.lmax][self.mmax].get_from_grid_mat(device=None)
        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        outputs = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
        return outputs


class SeparableS2Activation(torch.nn.Module):
    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = torch.nn.SiLU()
        self.s2_act     = S2Activation(self.lmax, self.mmax)

    def forward(self, input_scalars, input_tensors, SO3_grid):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[-1])
        output_tensors = self.s2_act(input_tensors, SO3_grid)
        outputs = torch.cat(
            (output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)),
            dim=1
        )
        return outputs


# =============================================================================
# HTR  —  Hierarchical Tensor Refinement
# =============================================================================

class HTR(nn.Module):
    def __init__(self, sphere_channels, edge_channels, lmax, hidden_channels=None):
        super().__init__()
        self.lmax = lmax
        self.edge_channels = edge_channels
        hidden_channels = hidden_channels or edge_channels
        self.degree_sizes = [2 * l + 1 for l in range(1, lmax + 1)]
        
        #Defines the Queries and Key matrix ( linear layer).. input is spherechanels, and output is hiddenchannels.. 
        # by mixing the different channels ( so m = 1 gets mixed together, m = 0 gets mixed together and so on for the different channels)
        
        self.W_vq = nn.Linear(sphere_channels, hidden_channels, bias=False)
        self.W_vk = nn.ModuleList([
            nn.Linear(sphere_channels, hidden_channels, bias=False)
            for _ in range(lmax)
        ])

        # Single linear, no nonlinearity — matches paper and GotenNet
        #self.gamma_w = nn.Linear(hidden_channels, edge_channels)
        # SiLU at end, NOT Sigmoid — matches GotenNet gamma_t
        
        
        # Option 1: closest to GotenNet — linear + single activation
        # Simple, projects hidden_C → edge_C with one nonlinearity
        self.gamma_w = nn.Sequential(
            nn.Linear(edge_channels, edge_channels),
            nn.SiLU(),
        )

        # SiLU at end, NOT Sigmoid — matches GotenNet gamma_t
        self.gamma_t = nn.Sequential(
            nn.Linear(edge_channels, edge_channels),
            nn.SiLU(),
            nn.Linear(edge_channels, edge_channels),
            nn.SiLU(),
        )

       # nn.init.xavier_uniform_(self.gamma_w.weight)
       # nn.init.zeros_(self.gamma_w.bias)
        #nn.init.xavier_uniform_(self.gamma_w[0].weight)   # index into Sequential
        #nn.init.zeros_(self.gamma_w[0].bias)
        # init each Linear inside the Sequential
        for module in self.gamma_w:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    #Vector rejection removes the component along the edge direction from each feature vector. Concretely, inside vector_rejection:
    # returns the part of node i's features that is orthogonal to the edge direction.
    #E dimension is size of number of edges. 
    
    # we remove the edge direction in some cases, because we already have it encoded it in the t_ij
    def vector_rejection(rep, rl):
        # rep: [E, 2l+1, C],  rl: [E, 2l+1]
        rl_u = rl.unsqueeze(-1)                        # [E, 2l+1, 1]
        proj = (rep * rl_u).sum(dim=1, keepdim=True)   # [E, 1, C]
        return rep - proj * rl_u

    def forward(self, t_ij, X_i, X_j, rl_ij):
        # rl_ij: [E, (L+1)^2-1]  — split by degree same as X
        
        # split Xi and Xj into their l = 0 component, l = 1 component and so on. 
        # What happens in the top layer of HTR ( see drawing)
        X_i_by_l  = torch.split(X_i,  self.degree_sizes, dim=1)
        X_j_by_l  = torch.split(X_j,  self.degree_sizes, dim=1)
        # We also split rl_ij ( direction between the two nodes) into the l degrees. 
        rl_ij_by_l = torch.split(rl_ij, self.degree_sizes, dim=1)

        # initializing zero tensor that will accumulate inner products across all degrees.
        w_ij = torch.zeros(t_ij.shape[0], self.W_vq.out_features,
                           device=t_ij.device, dtype=t_ij.dtype)
        
        # loop over all spherical harmonics degree ( first l =0, then l = 1... )
        for l_idx in range(self.lmax):
            # get the l part of the directional spherical embedding ( edge spherical embedding) 
            rl_l = rl_ij_by_l[l_idx]                  # [E, 2l+1]
            #self.W_vq = nn.Linear(sphere_channels, hidden_channels, bias=False)
            
            # applies the query matrix ( linear layer), and then projects to only get the part of the node that 
            # is orthogonal to the edge direction 
            qi_l = self.vector_rejection(
                self.W_vq(X_i_by_l[l_idx]), rl_l)
            # applies the keys matrix ( linear layer), and then projects to only get the part of the node that 
            # negatively orthogonal to the edge direction 
            kj_l = self.vector_rejection(
                self.W_vk[l_idx](X_j_by_l[l_idx]), -rl_l)
            # take the dotproduct between queiry ( from the target node) and the value ( from the edge nodes)
            # to see how much they overlap... remember.. the directional information was subtracted using vector_rejection..#
            #which means we are looking for similarities beynd how aligned they re
            # this is the  Agg(·)  function in the paper. 
            w_ij = w_ij + (qi_l * kj_l).sum(dim=1) / self.degree_sizes[l_idx]
        # Finally update the t_ij using the previous t_ij and the w_ij just made. ( gamma_w is a linear layer)
        # and gamma_t is a linear layer, then SiLU  
        #Their element-wise product: `gamma_t(t_ij)` acts as a **gate** — it controls how much of the new geometric signal `gamma_w(w_ij)` gets written into `t_ij`
        #`t_ij` is a **scalar/invariant** feature. It has shape `[E, edge_channels]` — just a flat vector of numbers with no spatial/m structure. It does **not** transform under rotations at all.
        delta = self.gamma_w(w_ij) * self.gamma_t(t_ij)
        return t_ij + delta

# =============================================================================
# GATAValueActivation  —  full version with t_ij geometric bias
# =============================================================================

class GATAValueActivation(nn.Module):
    """
    Full GATA value activation implementing your paper's Eq. 6 and 7.

    Eq. 6 — the combined input before splitting:
        sea_ij  +  (t_ij W_rs) ⊙ gamma_s(h_j) ⊙ phi(r^0_ij)

    where:
        sea_ij      = attention-weighted value (from so2_conv_1 extra m=0 output)
        t_ij W_rs   = learned expansion of the (HTR-refined) edge scalars
        gamma_s(h_j)= learned expansion of the neighbour's scalar (l=0) features
        phi(r^0_ij) = Gaussian smearing of raw distance — already absorbed into
                      t_ij via its initialization from edge_dist_feat

    Eq. 7 — the steerable update:
        Delta X^(l)_i = agg_j  o^d,(l)_ij * r^(l)_ij  +  o^t,(l)_ij * X^(l)_j

    Args:
        sphere_channels: node SO3 embedding width... from this we get h_j ( the l = 0 part of SO(3) embedding ), and X^l ( the l>0 part of the SO(3) embedding)
        hidden_channels: so2_conv_1 output width (= gate width C)
        edge_channels:   t_ij width
        lmax:            maximum degree
        mmax:            maximum order (for mmax-reduced SO2 basis)
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels:   int,
        lmax:            int,
        mmax:            int,
    ):
        super().__init__()
        self.lmax            = lmax
        self.mmax            = mmax
        self.hidden_channels = hidden_channels
        # S = total number of gate chunks: 1 (o_s) + lmax (o_d) + lmax (o_t)
        self.S               = 1 + 2 * lmax

        # W_rs: expands t_ij [E, edge_C] → [E, S*hidden_C]
        # This is the W_rs matrix in your paper's Eq. 6
        self.W_rs = nn.Linear(edge_channels, self.S * hidden_channels)

        # gamma_s: expands h_j (neighbour scalar features) [E, sphere_C] → [E, S*hidden_C]
        # This is gamma_s in your paper's Eq. 6
        self.gamma_s = nn.Sequential(
            nn.Linear(sphere_channels, self.S * hidden_channels),
            nn.SiLU(),
        )

        # X_j channel projection: sphere_C → hidden_C, no bias (equivariance)
        self.xj_proj = nn.Linear(sphere_channels, hidden_channels, bias=False)

        self.scalar_act = nn.SiLU()

        # full degree block sizes for splitting X_j and rl_ij
        self.full_degree_sizes    = [2 * l + 1 for l in range(1, lmax + 1)]
        # mmax-clipped sizes for the output (must match SO2_Convolution basis)
        self.reduced_degree_sizes = [min(2 * l + 1, 2 * mmax + 1) for l in range(1, lmax + 1)]

    def forward(
        self,
        attn_output: torch.Tensor,  # [E, S*hidden_C]  from so2_conv_1 extra m=0 (sea_ij approx)
        t_ij:        torch.Tensor,  # [E, edge_C]       scalar edge features, HTR-refined
        h_j:         torch.Tensor,  # [E, sphere_C]     l=0 features of neighbour node
        X_j:         torch.Tensor,  # [E, (L+1)^2-1, sphere_C]  neighbour steerable (l>=1)
        rl_ij:       torch.Tensor,  # [E, (L+1)^2-1]    edge SH coefficients (l>=1)
    ) -> torch.Tensor:              # [E, num_reduced_coeffs, hidden_C]

        C = self.hidden_channels

        # ── Eq. 6: geometric bias  (t_ij W_rs) ⊙ gamma_s(h_j) ───────────────
        # t_ij carries the refined edge context (distance + atom pair + geometry
        # accumulated by HTR).  W_rs expands it to gate space.
        # gamma_s(h_j) expands the neighbour's scalar features to the same space.
        # Their element-wise product means: "edge context gates neighbour identity"
        #Expands the edge scalar features into gate space. Asks: "what does the bond geometry/distance/chemistry say about each gate?"
        # remember h_j is the l = 0 part of the edge node... so we can apply nonlinearities 
        t_ij_bias = self.W_rs(t_ij) * self.gamma_s(h_j)       # [E, S*C]

        # ── Eq. 6: combine sea_ij with geometric bias ─────────────────────────
        # and attnd_ouput is from from so2_conv_1. 
        # This is what is s_ea in gotennet, 
        combined = attn_output + t_ij_bias                      # [E, S*C]

        # ── Eq. 6: split into o_s, {o_d^(l)}, {o_t^(l)} ─────────────────────
        #C is numbere of hidden_channels
        
        chunks   = combined.split(C, dim=-1)                    # S tensors of [E, C]
        #o_s: one chunk controlling the scalar (l=0) update
        o_s      = chunks[0]                                    # [E, C]
        #o_d_list: one chunk per degree l controlling how much the edge direction rl_ij^(l) contributes
        o_d_list = list(chunks[1 : 1 + self.lmax])             # lmax × [E, C]
        
        #o_t_list: one chunk per degree l controlling how much the neighbour's steerable features X_j^(l) contribute
        o_t_list = list(chunks[1 + self.lmax :])               # lmax × [E, C]

        # ── scalar output (l=0, Delta h) ──────────────────────────────────────
        #Apply SiLU to the scalar gate (fine — it's invariant), then add a dimension to make it [E, 1, C] so it can be concatenated with the steerable outputs later.
        out_scalar = self.scalar_act(o_s).unsqueeze(1)         # [E, 1, C]

        # ── project X_j channels ──────────────────────────────────────────────
        #from sphere_channels down to hidden_channels. 
        # by mixing same m´s over the channels. 
        X_j = self.xj_proj(X_j)                                # [E, (L+1)^2-1, C]

        # split both into per-degree blocks
        #This splits the `(L+1)^2-1` spatial dimension into per-degree blocks:

        #X_j:   [E, 3+5+7+9, C]  →  [E,3,C], [E,5,C], [E,7,C], [E,9,C]
        #rl_ij: [E, 3+5+7+9]     →  [E,3],   [E,5],   [E,7],   [E,9]
        X_j_by_l   = torch.split(X_j,   self.full_degree_sizes, dim=1)
        rl_ij_by_l = torch.split(rl_ij, self.full_degree_sizes, dim=1)

        # ── Eq. 7: per-degree steerable output ───────────────────────────────
        # Delta X^(l) = o_d^(l) * r^(l)_ij  +  o_t^(l) * X^(l)_j
        out_degrees = []
        for l_idx in range(self.lmax):
            m_width = self.reduced_degree_sizes[l_idx]         # mmax-clipped width.. to make sure it works with equiformerV2s mmax
            # And then only keep part of neighbours spherical embedding l up to mmax.. same with edge embedding. 
            Xj_l = X_j_by_l[l_idx][:, :m_width, :]            # [E, m_width, C]
            rl_l = rl_ij_by_l[l_idx][:, :m_width].unsqueeze(-1) # [E, m_width, 1]

            od_l = o_d_list[l_idx].unsqueeze(1)                # [E, 1, C]
            ot_l = o_t_list[l_idx].unsqueeze(1)                # [E, 1, C]

            dir_term    = od_l * rl_l                          # [E, m_width, C]
            
            #Each channel of `X_j^(l)` is scaled by the corresponding gate value.
            #This is: *"carry the neighbour's steerable features into the update,
            # scaled by how much the gate opens"*. Equivariant because `X_j` transforms correctly under rotation 
            # and we only multiply by an invariant scalar gate... so different number per channel. and per l
            # but for a specific l and channel, it gets multiplied with same number ot_l
            
            # same for direction term

            tensor_term = ot_l * Xj_l                          # [E, m_width, C]
            out_degrees.append(dir_term + tensor_term)
        ##Stack all degree outputs back together along the spatial dimension, then prepend the scalar (l=0) output. 
        # The result is a full steerable feature tensor ready to be passed to `so2_conv_2`.

        out_vectors = torch.cat(out_degrees, dim=1)            # [E, num_reduced-1, C]
        return torch.cat([out_scalar, out_vectors], dim=1)     # [E, num_reduced, C]
    
    



class GlobalNodeAttention(nn.Module):
    """
    All-to-all node attention operating on the invariant scalar channel only.
    Applied after all local transformer blocks, before the energy head.

    Design:
    - Operates only on l=0 (scalar) features → equivariance of steerable
      channels is fully preserved
    - Standard multi-head self-attention per graph (no cross-graph leakage)
    - Optional Euclidean RoPE: additive distance bias on attention logits,
      inspired by AllScAIP (Frank et al. 2024 / AllScAIP paper)
    - Per-graph masking: atoms from different structures never attend to
      each other (critical for batched training)

    Args:
        d_model:       channel width (= sphere_channels)
        num_heads:     number of attention heads
        dropout:       attention dropout
        use_rope:      whether to add distance bias to logits
        rope_dim:      number of Fourier features for distance encoding
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int   = 8,
        dropout:   float = 0.0,
        use_rope:  bool  = True,
        rope_dim:  int   = 16,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5
        self.use_rope  = use_rope
        self.rope_dim  = rope_dim

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

        # Euclidean RoPE: learned Fourier frequencies for pairwise distances
        # Each head gets its own scalar bias on the logit for pair (i,j)
        # bias_ij = w_heads · cos(omega * d_ij)  where omega are learned freqs
        if use_rope:
            self.rope_freqs  = nn.Parameter(torch.randn(rope_dim) * 0.1)
            self.rope_proj   = nn.Linear(rope_dim, num_heads, bias=False)

    def _euclidean_rope_bias(
        self,
        pos:   torch.Tensor,   # [N, 3]
        batch: torch.Tensor,   # [N]
        mask:  torch.Tensor,   # [B, N_max]  True where padded
    ) -> torch.Tensor:         # [B, H, N_max, N_max]
        """
        Compute additive distance bias for attention logits.
        bias[b, h, i, j] = rope_proj(cos(omega * ||r_i - r_j||))
        """
        B      = mask.shape[0]
        N_max  = mask.shape[1]
        device = pos.device

        # Build padded position tensor [B, N_max, 3]
        pos_pad = torch.zeros(B, N_max, 3, device=device, dtype=pos.dtype)
        for b in range(B):
            idx   = (batch == b).nonzero(as_tuple=True)[0]
            n_b   = idx.shape[0]
            pos_pad[b, :n_b] = pos[idx]

        # Pairwise distances [B, N_max, N_max]
        diff  = pos_pad.unsqueeze(2) - pos_pad.unsqueeze(1)   # [B, N, N, 3]
        dist  = diff.norm(dim=-1)                              # [B, N, N]

        # Fourier encoding: [B, N, N, rope_dim]
        omega   = self.rope_freqs.abs()                        # [rope_dim]
        fourier = torch.cos(dist.unsqueeze(-1) * omega)        # [B, N, N, rope_dim]

        # Project to per-head bias: [B, N, N, H] → [B, H, N, N]
        bias = self.rope_proj(fourier).permute(0, 3, 1, 2)    # [B, H, N, N]

        # Zero out padded positions
        pad_mask = mask.unsqueeze(1).unsqueeze(2) | mask.unsqueeze(1).unsqueeze(3)
        bias = bias.masked_fill(pad_mask, 0.0)

        return bias

    def forward(
        self,
        x:     torch.Tensor,   # [N, d_model]  — l=0 scalar features
        batch: torch.Tensor,   # [N]           — graph index per atom
        pos:   torch.Tensor,   # [N, 3]        — atom positions
    ) -> torch.Tensor:         # [N, d_model]

        N      = x.shape[0]
        B      = int(batch.max().item()) + 1
        device = x.device

        # ── Build padded tensor [B, N_max, d_model] ──────────────────────────
        sizes = torch.bincount(batch, minlength=B)   # [B]
        N_max = int(sizes.max().item())

        x_pad  = torch.zeros(B, N_max, self.d_model, device=device, dtype=x.dtype)
        mask   = torch.ones(B, N_max, dtype=torch.bool, device=device)  # True=padded

        # scatter atoms into padded tensor
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            idx  = (batch == b).nonzero(as_tuple=True)[0]
            n_b  = idx.shape[0]
            x_pad[b, :n_b]  = x[idx]
            mask[b, :n_b]   = False   # real atoms

        # ── QKV projections ───────────────────────────────────────────────────
        qkv = self.qkv_proj(x_pad)                          # [B, N_max, 3*d]
        q, k, v = qkv.chunk(3, dim=-1)                      # each [B, N_max, d]

        # reshape for multi-head: [B, H, N_max, head_dim]
        def split_heads(t):
            return t.view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # ── Attention logits ──────────────────────────────────────────────────
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, H, N, N]

        # Add Euclidean RoPE distance bias
        if self.use_rope:
            attn = attn + self._euclidean_rope_bias(pos, batch, mask)

        # Mask padded positions: set logits to -inf so softmax → 0
        key_mask = mask.unsqueeze(1).unsqueeze(2)            # [B, 1, 1, N_max]
        attn     = attn.masked_fill(key_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # ── Weighted sum ──────────────────────────────────────────────────────
        out = torch.matmul(attn, v)                          # [B, H, N_max, head_dim]
        out = out.transpose(1, 2).contiguous()               # [B, N_max, H, head_dim]
        out = out.view(B, N_max, self.d_model)               # [B, N_max, d_model]
        out = self.out_proj(out)                             # [B, N_max, d_model]

        # ── Unpad back to [N, d_model] ────────────────────────────────────────
        out_flat = torch.zeros(N, self.d_model, device=device, dtype=x.dtype)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            n_b = idx.shape[0]
            out_flat[idx] = out[b, :n_b]

        # ── Residual + LayerNorm ──────────────────────────────────────────────
        return self.norm(x + out_flat)