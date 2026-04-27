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
        # In _euclidean_rope_bias: to prevent NAN. 
        pos = pos.detach()
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
    
    
    
    
    
class GlobalNodeAttentionFull(nn.Module):
    """
    All-to-all node attention over the FULL SO3 embedding (all l degrees).
    Breaks rotational equivariance — only use as the final layer before
    the scalar energy head.

    Input:  x.embedding  [N, (lmax+1)^2, sphere_channels]
    Output: x.embedding  [N, (lmax+1)^2, sphere_channels]  (same shape)
    """

    def __init__(self, d_model, num_coeffs, num_heads=8, dropout=0.0,
                 use_rope=True, rope_dim=16):
        super().__init__()
        # d_model = sphere_channels, num_coeffs = (lmax+1)^2
        # We flatten the spatial and channel dims into one token dimension
        self.d_model    = d_model
        self.num_coeffs = num_coeffs
        self.flat_dim   = num_coeffs * d_model   # full flattened width
        self.num_heads  = num_heads
        self.head_dim   = self.flat_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.use_rope   = use_rope

        assert self.flat_dim % num_heads == 0

        self.qkv_proj = nn.Linear(self.flat_dim, 3 * self.flat_dim, bias=False)
        self.out_proj = nn.Linear(self.flat_dim, self.flat_dim, bias=False)
        self.norm     = nn.LayerNorm(self.flat_dim)
        self.dropout  = nn.Dropout(dropout)

        if use_rope:
            self.rope_freqs = nn.Parameter(torch.randn(rope_dim) * 0.1)
            self.rope_proj  = nn.Linear(rope_dim, num_heads, bias=False)

    def _rope_bias(self, pos, batch, mask):
        B, N_max = mask.shape
        pos_pad  = torch.zeros(B, N_max, 3, device=pos.device, dtype=pos.dtype)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            pos_pad[b, :idx.shape[0]] = pos[idx]
        diff  = pos_pad.unsqueeze(2) - pos_pad.unsqueeze(1)
        dist  = diff.norm(dim=-1)
        omega = self.rope_freqs.abs()
        fourier = torch.cos(dist.unsqueeze(-1) * omega)
        bias  = self.rope_proj(fourier).permute(0, 3, 1, 2)
        pad_mask = mask.unsqueeze(1).unsqueeze(2) | mask.unsqueeze(1).unsqueeze(3)
        return bias.masked_fill(pad_mask, 0.0)

    def forward(self, x_emb, batch, pos):
        # x_emb: [N, num_coeffs, sphere_channels]
        N  = x_emb.shape[0]
        B  = int(batch.max().item()) + 1
        device = x_emb.device

        # flatten spatial × channel into one vector per atom
        x_flat = x_emb.reshape(N, self.flat_dim)          # [N, num_coeffs*C]

        # pad into [B, N_max, flat_dim]
        sizes = torch.bincount(batch, minlength=B)
        N_max = int(sizes.max().item())
        x_pad = torch.zeros(B, N_max, self.flat_dim, device=device, dtype=x_flat.dtype)
        mask  = torch.ones(B, N_max, dtype=torch.bool, device=device)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            n_b = idx.shape[0]
            x_pad[b, :n_b] = x_flat[idx]
            mask[b, :n_b]  = False

        # QKV
        qkv = self.qkv_proj(x_pad)                        # [B, N_max, 3*flat]
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # attention logits
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        if self.use_rope:
            attn = attn + self._rope_bias(pos.detach(), batch, mask)

        key_mask = mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(key_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                        # [B, H, N_max, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N_max, self.flat_dim)
        out = self.out_proj(out)

        # unpad back to [N, flat_dim]
        out_flat = torch.zeros(N, self.flat_dim, device=device, dtype=x_flat.dtype)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            out_flat[idx] = out[b, :idx.shape[0]]

        # residual + norm, then reshape back
        out_flat = self.norm(x_flat + out_flat)            # [N, flat_dim]
        return out_flat.reshape(N, self.num_coeffs, self.d_model)
    
    
    
class GlobalNodeAttentionFullEquivariant(nn.Module):
    """
    All-to-all node attention over ALL spherical harmonic degrees.
    Preserves SO3 equivariance by computing attention weights from
    rotation-invariant norms of each degree's features, while using
    the full equivariant features as values.

    For each degree l:
        - Q_l, K_l  computed from  ||X^(l)||  (norm over 2l+1 dim) → invariant
        - V_l       computed from   X^(l)                           → equivariant
        - attention weights = softmax(Q_l K_l^T / sqrt(head_dim))   → invariant scalars
        - output    = attn_weights @ V_l                             → equivariant ✓

    Residual + LayerNorm applied per degree after attention.

    Args:
        sphere_channels:  channel width (= 128 in your config)
        lmax:             maximum degree (= 4 in your config)
        num_heads:        attention heads (default 8)
        dropout:          attention dropout
    """

    def __init__(
        self,
        sphere_channels: int,
        lmax:            int,
        num_heads:       int   = 8,
        dropout:         float = 0.0,
    ):
        super().__init__()
        assert sphere_channels % num_heads == 0

        self.sphere_channels = sphere_channels
        self.lmax            = lmax
        self.num_heads       = num_heads
        self.head_dim        = sphere_channels // num_heads
        self.scale           = self.head_dim ** -0.5

        # degree block sizes: l=0 → 1, l=1 → 3, l=2 → 5, l=3 → 7, l=4 → 9
        self.degree_sizes = [2 * l + 1 for l in range(lmax + 1)]

        # Per-degree Q, K projections (operate on norms → shape [N, sphere_channels])
        # V projection operates on the equivariant features directly
        # No bias in V to preserve equivariance (bias would break it for l>0)
        self.q_projs = nn.ModuleList([
            nn.Linear(sphere_channels, sphere_channels, bias=True)
            for _ in range(lmax + 1)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(sphere_channels, sphere_channels, bias=True)
            for _ in range(lmax + 1)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(sphere_channels, sphere_channels, bias=False)  # no bias for l>0
            for _ in range(lmax + 1)
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(sphere_channels, sphere_channels, bias=False)
            for _ in range(lmax + 1)
        ])

        # Per-degree LayerNorm applied to the norm of each degree block
        # (normalises the invariant summary, not the equivariant features)
        self.norms = nn.ModuleList([
            nn.LayerNorm(sphere_channels)
            for _ in range(lmax + 1)
        ])

        self.dropout = nn.Dropout(dropout)

    # ── Padding helpers ───────────────────────────────────────────────────────

    def _pad(self, x, batch, B, N_max, device):
        """Scatter flat [N, ...] tensor into padded [B, N_max, ...] tensor."""
        rest  = x.shape[1:]
        x_pad = torch.zeros(B, N_max, *rest, device=device, dtype=x.dtype)
        mask  = torch.ones(B, N_max, dtype=torch.bool, device=device)   # True = padded
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            n_b = idx.shape[0]
            x_pad[b, :n_b] = x[idx]
            mask[b, :n_b]  = False
        return x_pad, mask

    def _unpad(self, x_pad, batch, B, N, device, dtype):
        """Scatter padded [B, N_max, ...] back to flat [N, ...]."""
        rest     = x_pad.shape[2:]
        out_flat = torch.zeros(N, *rest, device=device, dtype=dtype)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            n_b = idx.shape[0]
            out_flat[idx] = x_pad[b, :n_b]
        return out_flat

    # ── Per-degree attention ──────────────────────────────────────────────────

    def _attend_one_degree(
        self,
        feat:   torch.Tensor,   # [N, 2l+1, sphere_channels]  equivariant features
        batch:  torch.Tensor,   # [N]
        B:      int,
        N_max:  int,
        mask:   torch.Tensor,   # [B, N_max]  True = padded
        l_idx:  int,
    ) -> torch.Tensor:          # [N, 2l+1, sphere_channels]
        """
        Run one round of all-to-all attention for degree l.

        Keys and queries are derived from the NORM of the features
        (invariant), values are the features themselves (equivariant).
        """
        N      = feat.shape[0]
        m_size = feat.shape[1]   # 2l+1
        device = feat.device

        # ── invariant summary: norm over the spatial (m) dimension ────────────
        # shape [N, sphere_channels] — rotation invariant for all l ✓
        feat_norm = feat.norm(dim=1)                                # [N, C]

        # ── Q, K from invariant norms ──────────────────────────────────────────
        q_flat = self.q_projs[l_idx](feat_norm)                     # [N, C]
        k_flat = self.k_projs[l_idx](feat_norm)                     # [N, C]

        # ── V from equivariant features ────────────────────────────────────────
        # project channels: [N, m, C] × W → [N, m, C]
        v_flat = self.v_projs[l_idx](feat)                          # [N, m, C]

        # ── pad Q, K, V ────────────────────────────────────────────────────────
        q_pad, _    = self._pad(q_flat, batch, B, N_max, device)    # [B, N_max, C]
        k_pad, _    = self._pad(k_flat, batch, B, N_max, device)
        v_pad, _    = self._pad(
            v_flat.reshape(N, m_size * self.sphere_channels),
            batch, B, N_max, device
        )   # [B, N_max, m*C]  — flatten m and C for padding convenience

        v_pad = v_pad.reshape(B, N_max, m_size, self.sphere_channels)

        # ── multi-head reshape ─────────────────────────────────────────────────
        def split_heads_inv(t):
            # t: [B, N_max, C]  →  [B, H, N_max, head_dim]
            return t.view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)

        def split_heads_equiv(t):
            # t: [B, N_max, m, C]  →  [B, H, N_max, m, head_dim]
            return t.view(B, N_max, m_size, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        q = split_heads_inv(q_pad)          # [B, H, N_max, head_dim]
        k = split_heads_inv(k_pad)          # [B, H, N_max, head_dim]
        v = split_heads_equiv(v_pad)        # [B, H, N_max, m, head_dim]

        # ── attention weights (invariant scalars) ──────────────────────────────
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, H, N, N]

        # mask padded positions
        key_mask = mask.unsqueeze(1).unsqueeze(2)                   # [B, 1, 1, N_max]
        attn     = attn.masked_fill(key_mask, float('-inf'))
        attn     = F.softmax(attn, dim=-1)
        attn     = torch.nan_to_num(attn, nan=0.0)                  # guard all-pad rows
        attn     = self.dropout(attn)
        # attn: [B, H, N_max, N_max]  — invariant scalar weights ✓

        # ── weighted sum of equivariant values ─────────────────────────────────
        # attn:      [B, H, N_max, N_max]
        # v:         [B, H, N_max, m, head_dim]
        # want out:  [B, H, N_max, m, head_dim]
        # einsum: sum over source atoms (dim -2 of attn, dim -3 of v)
        out = torch.einsum('bhij, bhjmd -> bhimd', attn, v)         # [B, H, N_max, m, head_dim]

        # ── merge heads ────────────────────────────────────────────────────────
        out = out.permute(0, 2, 3, 1, 4).contiguous()              # [B, N_max, m, H, head_dim]
        out = out.view(B, N_max, m_size, self.sphere_channels)      # [B, N_max, m, C]

        # ── output projection ─────────────────────────────────────────────────
        out = self.out_projs[l_idx](out)                            # [B, N_max, m, C]

        # ── unpad ──────────────────────────────────────────────────────────────
        out_flat = torch.zeros(N, m_size, self.sphere_channels,
                               device=device, dtype=feat.dtype)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            out_flat[idx] = out[b, :idx.shape[0]]

        # ── residual + layernorm ───────────────────────────────────────────────
        # LayerNorm over the channel dimension only (equivariant: same scale per m)
        # We normalise the norm-summary and use its scale for the residual
        # For l=0: standard residual + LN
        # For l>0: add residual, then normalise per-channel (not per-m)
        #   LN is applied to each m independently over the C dimension
        #   This is equivariant because the same normalisation is applied to all m
        out_flat = feat + out_flat                                   # residual
        # apply LN over C for each m position independently
        out_flat = self.norms[l_idx](out_flat)                       # [N, m, C] ✓

        return out_flat

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_emb:  torch.Tensor,   # [N, (lmax+1)^2, sphere_channels]
        batch:  torch.Tensor,   # [N]
    ) -> torch.Tensor:          # [N, (lmax+1)^2, sphere_channels]

        N      = x_emb.shape[0]
        B      = int(batch.max().item()) + 1
        device = x_emb.device

        sizes = torch.bincount(batch, minlength=B)
        N_max = int(sizes.max().item())

        # build the shared mask once — reused for every degree
        mask = torch.ones(B, N_max, dtype=torch.bool, device=device)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            mask[b, :idx.shape[0]] = False

        # split embedding into per-degree blocks
        feat_by_l = torch.split(x_emb, self.degree_sizes, dim=1)
        # e.g. for lmax=4: shapes [N,1,128], [N,3,128], [N,5,128], [N,7,128], [N,9,128]

        # run per-degree attention and collect updated blocks
        out_by_l = []
        for l_idx, feat_l in enumerate(feat_by_l):
            out_l = self._attend_one_degree(
                feat_l, batch, B, N_max, mask, l_idx
            )
            out_by_l.append(out_l)

        # reassemble full embedding
        return torch.cat(out_by_l, dim=1)   # [N, 25, sphere_channels]
    
    
    
    
    

class GlobalNodeAttentionFull(nn.Module):
    """
    All-to-all node attention over the FULL SO3 embedding (all l degrees).
    Breaks rotational equivariance — only use as the final layer before
    the scalar energy head.

    Input:  x.embedding  [N, (lmax+1)^2, sphere_channels]
    Output: x.embedding  [N, (lmax+1)^2, sphere_channels]  (same shape)
    """

    def __init__(self, d_model, num_coeffs, num_heads=8, dropout=0.0,
                 use_rope=True, rope_dim=16):
        super().__init__()
        # d_model = sphere_channels, num_coeffs = (lmax+1)^2
        # We flatten the spatial and channel dims into one token dimension
        self.d_model    = d_model
        self.num_coeffs = num_coeffs
        self.flat_dim   = num_coeffs * d_model   # full flattened width
        self.num_heads  = num_heads
        self.head_dim   = self.flat_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.use_rope   = use_rope

        assert self.flat_dim % num_heads == 0

        self.qkv_proj = nn.Linear(self.flat_dim, 3 * self.flat_dim, bias=False)
        self.out_proj = nn.Linear(self.flat_dim, self.flat_dim, bias=False)
        self.norm     = nn.LayerNorm(self.flat_dim)
        self.dropout  = nn.Dropout(dropout)

        if use_rope:
            self.rope_freqs = nn.Parameter(torch.randn(rope_dim) * 0.1)
            self.rope_proj  = nn.Linear(rope_dim, num_heads, bias=False)

    def _rope_bias(self, pos, batch, mask):
        B, N_max = mask.shape
        pos_pad  = torch.zeros(B, N_max, 3, device=pos.device, dtype=pos.dtype)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            pos_pad[b, :idx.shape[0]] = pos[idx]
        diff  = pos_pad.unsqueeze(2) - pos_pad.unsqueeze(1)
        dist  = diff.norm(dim=-1)
        omega = self.rope_freqs.abs()
        fourier = torch.cos(dist.unsqueeze(-1) * omega)
        bias  = self.rope_proj(fourier).permute(0, 3, 1, 2)
        pad_mask = mask.unsqueeze(1).unsqueeze(2) | mask.unsqueeze(1).unsqueeze(3)
        return bias.masked_fill(pad_mask, 0.0)

    def forward(self, x_emb, batch, pos):
        # x_emb: [N, num_coeffs, sphere_channels]
        N  = x_emb.shape[0]
        B  = int(batch.max().item()) + 1
        device = x_emb.device

        # flatten spatial × channel into one vector per atom
        x_flat = x_emb.reshape(N, self.flat_dim)          # [N, num_coeffs*C]

        # pad into [B, N_max, flat_dim]
        sizes = torch.bincount(batch, minlength=B)
        N_max = int(sizes.max().item())
        x_pad = torch.zeros(B, N_max, self.flat_dim, device=device, dtype=x_flat.dtype)
        mask  = torch.ones(B, N_max, dtype=torch.bool, device=device)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            n_b = idx.shape[0]
            x_pad[b, :n_b] = x_flat[idx]
            mask[b, :n_b]  = False

        # QKV
        qkv = self.qkv_proj(x_pad)                        # [B, N_max, 3*flat]
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, N_max, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # attention logits
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        if self.use_rope:
            attn = attn + self._rope_bias(pos.detach(), batch, mask)

        key_mask = mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(key_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                        # [B, H, N_max, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N_max, self.flat_dim)
        out = self.out_proj(out)

        # unpad back to [N, flat_dim]
        out_flat = torch.zeros(N, self.flat_dim, device=device, dtype=x_flat.dtype)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            out_flat[idx] = out[b, :idx.shape[0]]

        # residual + norm, then reshape back
        out_flat = self.norm(x_flat + out_flat)            # [N, flat_dim]
        return out_flat.reshape(N, self.num_coeffs, self.d_model)
    
class GlobalNodeAttentionHTR(nn.Module):
    """
    All-to-all node attention where keys and queries are computed from
    HTR-style inner products between steerable features and pairwise
    displacement directions — fully equivariant.

    For each degree l, the invariant score between atoms i and j is:
        s_ij^(l) = sum_m  X_i^(l,m) * Y_l^m(r_hat_ij)   (inner product)

    where Y_l^m(r_hat_ij) are the real spherical harmonics of the unit
    displacement vector from i to j.

    This is rotation-invariant because rotating R transforms both
    X_i^(l) and Y_l(r_hat_ij) by the same Wigner D-matrix, so their
    inner product is unchanged.
    """

    def __init__(
        self,
        sphere_channels: int,
        lmax:            int,
        num_heads:       int   = 8,
        dropout:         float = 0.0,
    ):
        super().__init__()
        assert sphere_channels % num_heads == 0

        self.sphere_channels = sphere_channels
        self.lmax            = lmax
        self.num_heads       = num_heads
        self.head_dim        = sphere_channels // num_heads
        self.scale           = self.head_dim ** -0.5
        self.degree_sizes    = [2 * l + 1 for l in range(lmax + 1)]

        # After computing inner products for each degree l, we get a
        # [N, N, C] invariant score tensor per degree.
        # We project these to Q and K spaces.
        # Input to q/k proj: sum over degrees of inner products → [N, N, C]
        # We use one shared projection across degrees (can be per-degree too)
        self.q_proj  = nn.Linear(sphere_channels, sphere_channels, bias=True)
        self.k_proj  = nn.Linear(sphere_channels, sphere_channels, bias=True)

        # Per-degree value projections (equivariant, no bias for l>0)
        self.v_projs = nn.ModuleList([
            nn.Linear(sphere_channels, sphere_channels, bias=(l == 0))
            for l in range(lmax + 1)
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(sphere_channels, sphere_channels, bias=False)
            for _ in range(lmax + 1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(sphere_channels)
            for _ in range(lmax + 1)
        ])

        self.dropout = nn.Dropout(dropout)

    def _compute_sh(self, rvec, l):
        """
        Compute real spherical harmonics Y_l^m of unit vectors rvec.
        rvec: [N, N, 3]  pairwise unit displacement vectors
        returns: [N, N, 2l+1]
        """
        # Use e3nn or your existing SO3 machinery
        # Here shown with e3nn for clarity
        from e3nn import o3
        N = rvec.shape[0]
        rvec_flat = rvec.reshape(-1, 3)                    # [N*N, 3]
        sh_flat   = o3.spherical_harmonics(
            l, rvec_flat, normalize=True
        )                                                   # [N*N, 2l+1]
        return sh_flat.reshape(N, N, 2 * l + 1)            # [N, N, 2l+1]

    def _htr_invariant(self, x_emb, pos, batch, B, N_max):
        """
        Compute HTR-style pairwise invariant scores.

        For each pair (i,j) and each degree l:
            score_ij^(l) = <X_i^(l), Y_l(r_hat_ij)>
                         = sum_m  X_i^(l,m,:) * Y_l^m(r_hat_ij)

        This inner product contracts the (2l+1) spatial dimension,
        leaving a [C]-dimensional invariant vector per pair per degree.

        We sum contributions across all degrees to get one [N, N, C]
        invariant score matrix.
        """
        N      = x_emb.shape[0]
        device = x_emb.device

        # pairwise displacement vectors [N, N, 3]
        diff    = pos.unsqueeze(1) - pos.unsqueeze(0)       # [N, N, 3]
        dist    = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        r_hat   = diff / dist                               # [N, N, 3] unit vectors

        # zero out self-pairs and cross-graph pairs
        same_graph = (batch.unsqueeze(1) == batch.unsqueeze(0))  # [N, N]
        self_mask  = torch.eye(N, dtype=torch.bool, device=device)
        valid_mask = same_graph & ~self_mask                # [N, N]

        # accumulate invariant score across degrees
        score = torch.zeros(N, N, self.sphere_channels,
                            device=device, dtype=x_emb.dtype)

        feat_by_l = torch.split(x_emb, self.degree_sizes, dim=1)

        for l_idx, feat_l in enumerate(feat_by_l):
            # feat_l: [N, 2l+1, C]
            # sh_l:   [N, N, 2l+1]  spherical harmonics of displacement
            sh_l = self._compute_sh(r_hat, l_idx)           # [N, N, 2l+1]

            # inner product over the (2l+1) spatial dimension:
            # einsum: imc, ijm -> ijc
            # = for each pair (i,j), dot X_i^(l) with Y_l(r_hat_ij)
            ip = torch.einsum('imc, ijm -> ijc', feat_l, sh_l)  # [N, N, C]

            score = score + ip / self.degree_sizes[l_idx]   # normalize by 2l+1

        # zero out invalid pairs
        score = score * valid_mask.unsqueeze(-1).float()    # [N, N, C]

        return score   # [N, N, C]  rotation-invariant pairwise scores

    def forward(self, x_emb, batch, pos):
        # x_emb: [N, 25, C]
        N      = x_emb.shape[0]
        B      = int(batch.max().item()) + 1
        device = x_emb.device

        # ── HTR-style invariant pairwise scores ───────────────────────
        # [N, N, C]  — captures directional alignment between
        # atom i's features and the direction toward atom j
        score_ij = self._htr_invariant(x_emb, pos, batch, B, N)

        # ── Q and K from invariant scores ─────────────────────────────
        # For atom i attending to atom j, the query comes from i's
        # average score profile, key from j's
        q_flat = self.q_proj(score_ij.mean(dim=1))          # [N, C]  avg over j
        k_flat = self.k_proj(score_ij.mean(dim=0))          # [N, C]  avg over i

        # ── Attention weights ─────────────────────────────────────────
        # [N, H, head_dim]
        def to_heads(t):
            return t.view(N, self.num_heads, self.head_dim)

        q = to_heads(q_flat)   # [N, H, head_dim]
        k = to_heads(k_flat)   # [N, H, head_dim]

        # [H, N, N]
        attn = torch.einsum('ihd, jhd -> hij', q, k) * self.scale

        # mask cross-graph pairs
        same_graph = (batch.unsqueeze(1) == batch.unsqueeze(0))  # [N, N]
        attn = attn.masked_fill(~same_graph.unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        # attn: [H, N, N]  invariant weights ✓

        # ── Per-degree equivariant value aggregation ──────────────────
        feat_by_l = torch.split(x_emb, self.degree_sizes, dim=1)
        out_by_l  = []

        for l_idx, feat_l in enumerate(feat_by_l):
            # feat_l: [N, 2l+1, C]
            v_l = self.v_projs[l_idx](feat_l)               # [N, 2l+1, C]

            # split C into heads: [N, 2l+1, H, head_dim]
            m   = self.degree_sizes[l_idx]
            v_l = v_l.view(N, m, self.num_heads, self.head_dim)

            # weighted sum: attn [H, N, N] × v [N, m, H, head_dim]
            # → out [N, m, H, head_dim]
            out_l = torch.einsum('hij, jmhd -> imhd', attn, v_l)

            # merge heads: [N, m, C]
            out_l = out_l.reshape(N, m, self.sphere_channels)
            out_l = self.out_projs[l_idx](out_l)

            # residual + layernorm (per m, over C)
            out_l = self.norms[l_idx](feat_l + out_l)

            out_by_l.append(out_l)

        return torch.cat(out_by_l, dim=1)   # [N, 25, C]
    
    