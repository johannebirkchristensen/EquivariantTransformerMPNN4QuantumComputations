
#E  = number of edges in the graph (all node pairs within cutoff)
#S  = 1 + 2*lmax  — number of gate chunks
#     for lmax=4: S = 9  (1 scalar + 4 direction + 4 tensor)
#C  = hidden_channels  — channel width inside attention
#num_heads = number of attention heads, e.g. 8
#alpha_C   = attn_alpha_channels, e.g. 32
#alpha_size = num_heads * alpha_C  = 8*32 = 256
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
import copy
from torch_geometric.utils import scatter
from math import ceil
from .activation import (
    ScaledSiLU,
    ScaledSwiGLU,
    SwiGLU,
    ScaledSmoothLeakyReLU,
    SmoothLeakyReLU,
    GateActivation,
    SeparableS2Activation,
    S2Activation,
    HTR,
    GATAValueActivation,
)
from EquiformerV2Functions.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    get_normalization_layer,
)
from EquiformerV2Functions.so2_ops import (
    SO2_Convolution,
    SO2_Linear,
)
from EquiformerV2Functions.so3 import (
    SO3_Embedding,
    SO3_Linear,
    SO3_LinearV2,
)
from EquiformerV2Functions.radial_function import RadialFunction
from EquiformerV2Functions.drop import (
    GraphDropPath,
    EquivariantDropoutArraySphericalHarmonics,
)


class SO2EquivariantGraphAttention(torch.nn.Module):
    """
    SO2EquivariantGraphAttention with full GATA value activation.

    Changes from original EquiformerV2:
      - SeparableS2Activation replaced by GATAValueActivation
      - extra_m0_output_channels of so2_conv_1 expanded to produce
        S = 1 + 2*lmax gate chunks (o_s, o_d^l, o_t^l)
      - forward() now takes t_ij, rl_ij as additional arguments
      - X_j (neighbour steerable) and h_j (neighbour scalar) are extracted
        before rotation and passed to GATAValueActivation
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        output_channels,
        lmax_list,
        mmax_list,
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        max_num_elements,
        edge_channels_list,
        edge_channels,           # ← NEW: width of t_ij (needed for GATAValueActivation)
        use_atom_edge_embedding=True,
        use_m_share_rad=False,
        activation='scaled_silu',
        use_s2_act_attn=False,
        use_attn_renorm=True,
        use_gate_act=False,
        use_sep_s2_act=True,
        alpha_drop=0.0,
    ):
        super(SO2EquivariantGraphAttention, self).__init__()

        self.sphere_channels     = sphere_channels
        self.hidden_channels     = hidden_channels
        self.num_heads           = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels     = output_channels
        self.lmax_list           = lmax_list
        self.mmax_list           = mmax_list
        self.num_resolutions     = len(self.lmax_list)
        self.lmax                = max(lmax_list)
        self.edge_channels       = edge_channels

        self.SO3_rotation    = SO3_rotation
        self.mappingReduced  = mappingReduced
        self.SO3_grid        = SO3_grid

        self.max_num_elements       = max_num_elements
        self.edge_channels_list     = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad        = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act    = use_gate_act
        self.use_sep_s2_act  = use_sep_s2_act

        assert not self.use_s2_act_attn

        # ── extra m=0 channels for so2_conv_1 ────────────────────────────────
        # Original layout:
        #   [0 : num_heads*alpha_C]            → x_0_alpha  (attention weights)
        #   [num_heads*alpha_C : + hidden_C]   → x_0_gating (S2Act scalar gate)
        #
        # New GATA layout (use_sep_s2_act=True):
        #   [0 : num_heads*alpha_C]            → x_0_alpha
        #   [+ S*hidden_C]                     → attn_output  (sea_ij approximation)
        #     where S = 1 + 2*lmax, split into o_s + lmax*o_d + lmax*o_t
        #
        # So extra_m0 grows from (alpha_C + hidden_C) to (alpha_C + S*hidden_C).
        # The extra (2*lmax - 1)*hidden_C params are the new per-degree gates.
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels += self.lmax * self.hidden_channels
            elif self.use_sep_s2_act:
                S = 1 + 2 * self.lmax
                extra_m0_output_channels += S * self.hidden_channels

        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [2 * self.sphere_channels * (self.lmax + 1)]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(self.lmax + 1) ** 2]).long()
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(False if not self.use_m_share_rad else True),
            edge_channels_list=(self.edge_channels_list if not self.use_m_share_rad else None),
            extra_m0_output_channels=extra_m0_output_channels,
        )

        # ── attention weight layers ───────────────────────────────────────────
        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act  = None
            self.alpha_dot  = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        # ── value activation ──────────────────────────────────────────────────
        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=self.lmax,
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
            )
        else:
            if self.use_sep_s2_act:
                # Full GATA value activation
                self.value_act = GATAValueActivation(
                    sphere_channels=self.sphere_channels,
                    hidden_channels=self.hidden_channels,
                    edge_channels=self.edge_channels,
                    lmax=self.lmax,
                    mmax=max(self.mmax_list),
                )
            else:
                self.value_act = S2Activation(
                    lmax=self.lmax,
                    mmax=max(self.mmax_list),
                )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=(self.num_heads if self.use_s2_act_attn else None),
        )

        self.proj = SO3_LinearV2(
            self.num_heads * self.attn_value_channels,
            self.output_channels,
            lmax=self.lmax_list[0],
        )

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        t_ij,     # [E, edge_channels]   scalar edge features, HTR-refined before call
        rl_ij,    # [E, (L+1)^2 - 1]    edge spherical harmonics, original frame
    ):
        # ── edge scalar features ──────────────────────────────────────────────
        if self.use_atom_edge_embedding:
            source_element   = atomic_numbers[edge_index[0]]
            target_element   = atomic_numbers[edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance

        # ── expand node embeddings to edges ───────────────────────────────────
        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

        # ── save X_j and h_j BEFORE rotation (original frame) ─────────────────
        # x_target.embedding shape: [E, (L+1)^2, sphere_channels]
        #
        # h_j: l=0 slice — the SCALAR (rotation-invariant) features of the
        # neighbour node.  Shape [E, sphere_channels].
        # This is "h_j" in the paper's gamma_s(h_j) term.
        h_j = x_target.embedding[:, 0, :].clone()             # [E, sphere_C]

        # X_j: l>=1 slices — the STEERABLE (equivariant) features of the
        # neighbour node.  Shape [E, (L+1)^2-1, sphere_channels].
        # This is X_j^(l) in your paper's Eq. 7.
        # Must be saved in the ORIGINAL (unrotated) frame — same frame as rl_ij.
        X_j = x_target.embedding[:, 1:, :].clone()            # [E, (L+1)^2-1, sphere_C]

        # Also save X_i (source steerable) for HTR — it is used to compute
        # the inner products w_ij.  We save it here; HTR is called from
        # TransBlockV2 BEFORE this forward(), so this save is for reference only.
        # (See TransBlockV2.forward() for the HTR call.)

        # ── build message embedding ───────────────────────────────────────────
        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(),
            x_target.num_channels * 2,
            device=x_target.device,
            dtype=x_target.dtype,
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(-1, (self.lmax + 1), 2 * self.sphere_channels)
            x_edge_weight = torch.index_select(x_edge_weight, dim=1, index=self.expand_index)
            x_message.embedding = x_message.embedding * x_edge_weight

        # ── rotate irreps into edge frame ─────────────────────────────────────
        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)

        # ── first SO(2) convolution ───────────────────────────────────────────
        x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # ── value activation ──────────────────────────────────────────────────
        alpha_size = self.num_heads * self.attn_alpha_channels
        C          = self.hidden_channels

        if self.use_gate_act:
            x_0_gating = x_0_extra.narrow(1, alpha_size, x_0_extra.shape[1] - alpha_size)
            x_0_alpha  = x_0_extra.narrow(1, 0, alpha_size)
            x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)

        elif self.use_sep_s2_act:
            # Full GATA path
            # x_0_extra layout:
            #   [0 : alpha_size]          → x_0_alpha  (attention weights)
            #   [alpha_size : alpha_size + S*C] → attn_output  (sea_ij approx, S=1+2*lmax)
            x_0_alpha   = x_0_extra[:, :alpha_size]             # [E, num_heads*alpha_C]
            attn_output = x_0_extra[:, alpha_size:]             # [E, S*C]

            # ── compute alpha for normalization ───────────────────────────────
            x_0_alpha_r = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha_r = self.alpha_norm(x_0_alpha_r)
            x_0_alpha_r = self.alpha_act(x_0_alpha_r)
            alpha = torch.einsum('bik, ik -> bi', x_0_alpha_r, self.alpha_dot)
            alpha = torch_geometric.utils.softmax(alpha, edge_index[1])  # [E, num_heads]

            # ── normalize attn_output across neighbours ───────────────────────
            alpha_scalar = alpha.mean(dim=1, keepdim=True)   # [E, 1]
            attn_output  = alpha_scalar * attn_output         # [E, S*C]
            # perhaps do not do this ???...dont normalize alpha ?
            x_message.embedding = self.value_act(
                attn_output=attn_output,   # sea_ij approximation
                t_ij=t_ij,                # HTR-refined edge scalars
                h_j=h_j,                  # neighbour scalar features
                X_j=X_j,                  # neighbour steerable features (original frame)
                rl_ij=rl_ij,             # edge spherical harmonics (original frame)
            )

        else:
            x_0_alpha = x_0_extra
            x_message.embedding = self.value_act(x_message.embedding, self.SO3_grid)

        # ── second SO(2) convolution ──────────────────────────────────────────
        x_message = self.so2_conv_2(x_message, x_edge)

        # ── attention weights ─────────────────────────────────────────────────
        if not self.use_sep_s2_act:
            #gate_act and s2_act branches compute alpha here
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum('bik, ik -> bi', x_0_alpha, self.alpha_dot)
            alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
            # else: alpha already computed above in use_sep_s2_act branch
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # ── weighted aggregation ──────────────────────────────────────────────
        attn = x_message.embedding
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)
        attn = attn * alpha
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        x_message.embedding = attn

        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced)
        x_message._reduce_edge(edge_index[1], len(x.embedding))

        out_embedding = self.proj(x_message)
        return out_embedding

# =============================================================================
# MoE-HTR Feed Forward Network
# =============================================================================
# Imports needed at top of activation.py:
# from torch_scatter import scatter   ← or torch_geometric.utils scatter


class EquivariantExpertFFN(nn.Module):
    """
    A single expert in the MoE-HTR FFN.

    This is a small equivariant feed-forward network where the nonlinearity
    is GATED by the chemical environment context c_i (aggregated from t_ij).

    The gate is invariant (computed from invariant t_ij) and the features
    it gates are equivariant → output is equivariant ✓

    Architecture per expert:
        x  →  SO3_linear_1  →  GatedAct(context_gate, x)  →  SO3_linear_2  →  x'

    The context gate replaces the fixed SeparableS2Activation with a
    LEARNED, ENVIRONMENT-SPECIFIC gating signal. Different experts learn
    to respond to different chemical patterns in c_i.

    Args:
        sphere_channels:  channel width C (= 128)
        hidden_channels:  internal width (= ffn_hidden_channels, e.g. 512)
        edge_channels:    width of c_i coming from t_ij aggregation (= 128)
        lmax:             maximum spherical harmonic degree (= 4)
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels:   int,
        lmax:            int,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax            = lmax
        self.degree_sizes    = [2 * l + 1 for l in range(lmax + 1)]
        self.total_coeffs    = sum(self.degree_sizes)   # = (lmax+1)^2 = 25

        # ── linear 1: expand sphere_channels → hidden_channels ───────────────
        # Applied per-coefficient position (same W across all m within each l)
        # This is SO3-equivariant: mixing channels at fixed (l, m)
        self.linear_1 = nn.Linear(sphere_channels, hidden_channels, bias=False)

        # ── gate projection: context c_i → gates for each degree ─────────────
        # For each degree l we need one scalar gate per hidden channel
        # Output: lmax+1 blocks of hidden_channels each
        # The scalar gate is the SAME across all m within a degree
        # (which is required for equivariance — you can't have different
        #  gates for m=+1 vs m=-1 or rotation equivariance breaks)
        self.gate_proj = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels * (lmax + 1)),
            nn.SiLU(),
        )

        # ── linear 2: project hidden_channels → sphere_channels ──────────────
        self.linear_2 = nn.Linear(hidden_channels, sphere_channels, bias=False)

        # ── scalar (l=0) branch gets its own SiLU — invariant so unrestricted ─
        self.scalar_act = nn.SiLU()

    def forward(
        self,
        x_emb: torch.Tensor,   # [N, (lmax+1)^2, sphere_channels]
        c_i:   torch.Tensor,   # [N, edge_channels]  aggregated neighbor context
    ) -> torch.Tensor:          # [N, (lmax+1)^2, sphere_channels]

        # ── linear 1: expand channels ─────────────────────────────────────────
        # [N, 25, C] → [N, 25, hidden_C]
        # Same linear applied to every (l, m) position — equivariant ✓
        h = self.linear_1(x_emb)   # [N, 25, hidden_C]

        # ── compute per-degree gates from context ─────────────────────────────
        # gates shape: [N, (lmax+1) * hidden_C]
        # split into lmax+1 blocks of [N, hidden_C]
        gates_flat = self.gate_proj(c_i)                      # [N, (L+1)*hidden_C]
        gates_by_l = gates_flat.split(self.hidden_channels, dim=-1)
        # gates_by_l[l]: [N, hidden_C] — one gate vector per degree

        # ── split h into per-degree blocks ────────────────────────────────────
        h_by_l = torch.split(h, self.degree_sizes, dim=1)
        # h_by_l[l]: [N, 2l+1, hidden_C]

        # ── apply gated nonlinearity per degree ───────────────────────────────
        out_by_l = []
        for l_idx, (h_l, gate_l) in enumerate(zip(h_by_l, gates_by_l)):
            # h_l:    [N, 2l+1, hidden_C]
            # gate_l: [N, hidden_C]  — invariant scalar gate ✓

            if l_idx == 0:
                # l=0 is a scalar — can apply any nonlinearity
                # Use SiLU on the features AND multiply by the context gate
                out_l = self.scalar_act(h_l) * gate_l.unsqueeze(1)
                # [N, 1, hidden_C]
            else:
                # l>0 is equivariant — gate must be an invariant scalar
                # gate_l is invariant (from t_ij) → multiply across m dim ✓
                # This is: each channel of the l-features gets scaled by
                # the corresponding gate value. Same gate for all m within
                # this l and channel → equivariance preserved ✓
                out_l = h_l * gate_l.unsqueeze(1)
                # [N, 2l+1, hidden_C]

            out_by_l.append(out_l)

        # ── reassemble and project back ────────────────────────────────────────
        h = torch.cat(out_by_l, dim=1)    # [N, 25, hidden_C]
        h = self.linear_2(h)               # [N, 25, sphere_C]

        return h


class MoEHTRFeedForward(nn.Module):
    """
    Mixture-of-Experts Feed Forward Network gated by HTR edge context.

    Replaces the fixed SeparableS2Activation in the FFN with K learned
    expert transformations. The routing weights are computed from c_i,
    the per-atom aggregation of t_ij — the HTR-refined edge scalar stream.

    WHY THIS IS SMART:
    ==================

    1. t_ij is already the richest chemical signal in your model.
       By layer k, t_ij has been refined k times by HTR, each time
       injecting inner products of the evolving steerable features with
       bond directions. Aggregating t_ij to nodes gives c_i — a summary
       of "what kind of geometric/chemical environment is atom i in,
       as understood by the model at this processing stage." This is
       the ideal routing signal.

    2. The FFN nonlinearity determines HOW features transform, not just
       WHETHER they pass through. A fixed nonlinearity (S2Activation)
       applies the same transformation strategy to an iron atom in BCC
       iron and an iron atom at an MgO interface — even though those
       environments require completely different feature transformations.
       The MoE lets different environments activate different strategies.

    3. Equivariance is preserved exactly. The routing weights w_i are
       computed from invariant t_ij → they are invariant scalars. The
       expert outputs are equivariant tensors. Invariant scalar × equivariant
       tensor = equivariant tensor. The weighted mixture is therefore
       equivariant. No approximations needed.

    4. The routing is SOFT (softmax, not top-K hard selection). This means:
       - Differentiable end-to-end
       - No discrete selection → no gradient issues
       - Experts can specialize gradually during training
       - Multiple experts can activate for ambiguous environments

    5. The additional parameters are concentrated where they matter most:
       the nonlinearity, which is the only non-equivariant part of the
       standard FFN. The linear layers remain shared (equivariant, efficient).

    Args:
        sphere_channels:  C = 128
        hidden_channels:  FFN internal width = 512
        edge_channels:    t_ij width = 128
        lmax:             max degree = 4
        num_experts:      K, number of expert transformations (default 4)
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels:   int,
        lmax:            int,
        num_experts:     int ,
    ):
        super().__init__()
        self.num_experts     = num_experts
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.edge_channels   = edge_channels
        self.lmax            = lmax

        # ── router: c_i → soft expert weights ─────────────────────────────────
        # A small MLP that maps the aggregated edge context to K weights.
        # Deeper than a single linear to allow non-linear routing decisions.
        # Why SiLU in the middle: allows the router to learn threshold-like
        # responses ("if this atom looks like coordination ~6 AND has d-orbital
        # character → activate expert 2").
        self.router = nn.Sequential(
            nn.Linear(edge_channels, edge_channels),
            nn.SiLU(),
            nn.Linear(edge_channels, num_experts),
        )

        # ── K expert FFNs ──────────────────────────────────────────────────────
        # Each expert is a full EquivariantExpertFFN.
        # They share the SAME input/output interface but have DIFFERENT weights.
        # During training they are encouraged to specialize through the routing:
        # if expert k consistently gets high weight for, say, octahedral metal
        # environments, its weights will gradually optimize for that case.
        # NEW — mixed: first half equivariant, second half non-equivariant
        
        n_equiv    = ceil(num_experts / 2)   # e.g. 4 experts → 2 equiv, 2 non-equiv
        n_nonequiv = num_experts - n_equiv
        self.experts = nn.ModuleList(
            [
                EquivariantExpertFFN(
                    sphere_channels = sphere_channels,
                    hidden_channels = hidden_channels,
                    edge_channels   = edge_channels,
                    lmax            = lmax,
                )
                for _ in range(n_equiv)
            ] + [
                NonEquivariantExpertFFN(
                    sphere_channels = sphere_channels,
                    hidden_channels = hidden_channels,
                    edge_channels   = edge_channels,
                    lmax            = lmax,
                )
                for _ in range(n_nonequiv)
            ]
        )

        # track which experts are equivariant for logging
        self.expert_is_equivariant = [True] * n_equiv + [False] * n_nonequiv
        #self.experts = nn.ModuleList([
       #     EquivariantExpertFFN(
       #         sphere_channels = sphere_channels,
       #         hidden_channels = hidden_channels,
       #         edge_channels   = edge_channels,
       #         lmax            = lmax,
       #     )
       #     for _ in range(num_experts)
       # ])
       

        # ── load balancing loss weight ─────────────────────────────────────────
        # Stores the routing entropy for optional load-balancing regularization.
        # If all atoms route to the same expert, diversity is lost.
        # You can add a -entropy term to the loss to encourage spread.
        self.last_routing_entropy = None

    def forward(
        self,
        x_emb:      torch.Tensor,   # [N, (lmax+1)^2, sphere_channels]
        t_ij:       torch.Tensor,   # [E, edge_channels]
        edge_index: torch.Tensor,   # [2, E]
    ) -> torch.Tensor:               # [N, (lmax+1)^2, sphere_channels]

        N      = x_emb.shape[0]
        device = x_emb.device

        # ── aggregate t_ij → c_i ──────────────────────────────────────────────
        # Mean aggregation over incoming edges.
        # This gives each atom a summary of its neighborhood's edge context,
        # refined by all previous HTR steps.
        # shape: [N, edge_channels]
        c_i = scatter(
            t_ij,
            edge_index[1],
            dim=0,
            dim_size=N,
            reduce='mean',
        )   # [N, edge_C]

        # ── compute routing weights ────────────────────────────────────────────
        # w: [N, K]  soft mixture weights, sum to 1 across K for each atom
        w = F.softmax(self.router(c_i), dim=-1)   # [N, K]

        # ── store routing entropy for optional load-balancing ──────────────────
        # H = -sum(w * log(w+eps))  — high entropy = all experts used equally
        # You can log this to W&B to monitor expert utilization
        #with torch.no_grad():
        #    entropy = -(w * (w + 1e-8).log()).sum(dim=-1).mean()
        #    self.last_routing_entropy = entropy.item()

        # REPLACE WITH:
        with torch.no_grad():
            entropy = -(w * (w + 1e-8).log()).sum(dim=-1).mean()
            self.last_routing_entropy = entropy.item()
            
            # track how much weight goes to non-equivariant experts
            n_equiv = sum(self.expert_is_equivariant)
            self.last_equiv_weight    = w[:, :n_equiv].sum(dim=-1).mean().item()
            self.last_nonequiv_weight = w[:, n_equiv:].sum(dim=-1).mean().item()
            
        # ── compute weighted mixture of expert outputs ─────────────────────────
        # Each expert takes (x_emb, c_i) and returns [N, 25, C]
        # We weight by w[:, k] and accumulate
        out = torch.zeros_like(x_emb)
        for k, expert in enumerate(self.experts):
            # expert output: [N, 25, sphere_C]
            expert_out = expert(x_emb, c_i)

            # w[:, k]: [N]  scalar weight for expert k at each atom
            # reshape to [N, 1, 1] to broadcast over (25, C)
            weight_k = w[:, k].view(N, 1, 1)

            # equivariant: invariant scalar × equivariant tensor ✓
            out = out + weight_k * expert_out

        return out
# ─────────────────────────────────────────────────────────────────────────────


class TransBlockV2(torch.nn.Module):
    """
    TransBlockV2 with full GATA + HTR.

    Per-block data flow:
        1. Norm
        2. HTR: refine t_ij using inner products of current X_i, X_j
        3. SO2EquivariantGraphAttention (GATA value activation uses updated t_ij)
        4. Residual
        5. Norm → FFN → Residual

    t_ij flows as a residual stream alongside x (node features).
    Each layer enriches t_ij with higher-order geometric information from
    the evolving steerable features before the attention step uses it.
    """

    def __init__(
        self,
        sphere_channels,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        ffn_hidden_channels,
        output_channels,
        lmax_list,
        mmax_list,
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        max_num_elements,
        edge_channels_list,
        edge_channels,           # ← NEW: width of t_ij
        use_atom_edge_embedding=True,
        use_m_share_rad=False,
        attn_activation='silu',
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation='silu',
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        norm_type='rms_norm_sh',
        alpha_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        num_experts = 4,
    ):
        super(TransBlockV2, self).__init__()

        max_lmax = max(lmax_list)
        self.norm_1 = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels)

        # ── HTR module ────────────────────────────────────────────────────────
        # One HTR per block.  It refines t_ij before each attention step.
        # sphere_channels: width of X_i, X_j steerable features
        # edge_channels:   width of t_ij
        # lmax:            how many degrees to compute inner products over
        self.htr = HTR(
            sphere_channels=sphere_channels,
            edge_channels=edge_channels,
            lmax=max_lmax,
        )

        self.ga = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            edge_channels=edge_channels,
            use_atom_edge_embedding=use_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        self.proj_drop = EquivariantDropoutArraySphericalHarmonics(proj_drop, drop_graph=False) if proj_drop > 0.0 else None

        self.norm_2 = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=sphere_channels)

        self.ffn = MoEHTRFeedForward(
            sphere_channels = sphere_channels,
            hidden_channels = ffn_hidden_channels,
            edge_channels   = edge_channels,
            lmax            = max(lmax_list),
            num_experts     = num_experts,
        )

        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(sphere_channels, output_channels, lmax=max_lmax)
        else:
            self.ffn_shortcut = None

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        batch,
        t_ij,    # [E, edge_channels]   scalar edge features (updated in place each block)
        rl_ij,   # [E, (L+1)^2 - 1]    edge spherical harmonics, original frame
    ):
        output_embedding = x

        # ── step 1: extract X_i and X_j for HTR ──────────────────────────────
        # We need the steerable features (l>=1) of source and target nodes
        # for the inner-product computation in HTR.
        # x.embedding shape: [N, (L+1)^2, sphere_channels]
        # We gather per-edge using edge_index before calling HTR.
        X_all = x.embedding[:, 1:, :]                         # [N, (L+1)^2-1, sphere_C]
        X_i_edges = X_all[edge_index[0]]                      # [E, (L+1)^2-1, sphere_C]
        X_j_edges = X_all[edge_index[1]]                      # [E, (L+1)^2-1, sphere_C]

        # ── step 2: HTR — refine t_ij ─────────────────────────────────────────
        # Paper: t_ij <- t_ij + gamma_w(w_ij) * gamma_t(t_ij)
        # where w_ij = sum_l (X_i^(l) W_vq)^T (X_j^(l) W_vk^(l))
        #
        # t_ij is updated in-place as a residual stream.
        # After this call t_ij carries the geometric inner-product information
        # from the CURRENT layer's steerable features, before attention.
        #t_ij = self.htr(t_ij, X_i_edges, X_j_edges)          # [E, edge_channels]
        # In TransBlockV2.forward(), step 2:
        t_ij = self.htr(t_ij, X_i_edges, X_j_edges, rl_ij)  # ← add rl_ij
        # ── step 3: norm + attention ──────────────────────────────────────────
        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_1(output_embedding.embedding)
        output_embedding = self.ga(
            output_embedding,
            atomic_numbers,
            edge_distance,
            edge_index,
            t_ij=t_ij,
            rl_ij=rl_ij,
        )

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        output_embedding.embedding = output_embedding.embedding + x_res

        # ── step 4: norm + FFN ────────────────────────────────────────────────
        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)
        #output_embedding = self.ffn(output_embedding)
        
        # MoE FFN needs t_ij and edge_index for routing
        moe_out = self.ffn(
            output_embedding.embedding,
            t_ij,
            edge_index,
        )
        output_embedding.embedding = moe_out
        
        
        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        if self.ffn_shortcut is not None:
            shortcut_embedding = SO3_Embedding(
                0,
                output_embedding.lmax_list.copy(),
                self.ffn_shortcut.in_features,
                device=output_embedding.device,
                dtype=output_embedding.dtype,
            )
            shortcut_embedding.set_embedding(x_res)
            shortcut_embedding.set_lmax_mmax(
                output_embedding.lmax_list.copy(),
                output_embedding.lmax_list.copy(),
            )
            shortcut_embedding = self.ffn_shortcut(shortcut_embedding)
            x_res = shortcut_embedding.embedding

        output_embedding.embedding = output_embedding.embedding + x_res

        # return both updated node features and updated t_ij
        # t_ij must be passed to the next block so it carries forward
        return output_embedding, t_ij
    
    
    


# ─────────────────────────────────────────────────────────────────────────────


class FeedForwardNetwork(torch.nn.Module):
    """
    FeedForwardNetwork: unchanged from original EquiformerV2.
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax_list,
        mmax_list,
        SO3_grid,
        activation='scaled_silu',
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
    ):
        super(FeedForwardNetwork, self).__init__()
        self.sphere_channels     = sphere_channels
        self.hidden_channels     = hidden_channels
        self.output_channels     = output_channels
        self.lmax_list           = lmax_list
        self.mmax_list           = mmax_list
        self.num_resolutions     = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid            = SO3_grid
        self.use_gate_act        = use_gate_act
        self.use_grid_mlp        = use_grid_mlp
        self.use_sep_s2_act      = use_sep_s2_act
        self.max_lmax            = max(self.lmax_list)

        self.so3_linear_1 = SO3_LinearV2(self.sphere_channels_all, self.hidden_channels, lmax=self.max_lmax)

        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(self.sphere_channels_all, self.hidden_channels, bias=True),
                    nn.SiLU(),
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            )
        else:
            if self.use_gate_act:
                self.gating_linear = torch.nn.Linear(self.sphere_channels_all, self.max_lmax * self.hidden_channels)
                self.gate_act = GateActivation(self.max_lmax, self.max_lmax, self.hidden_channels)
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = torch.nn.Linear(self.sphere_channels_all, self.hidden_channels)
                    self.s2_act = SeparableS2Activation(self.max_lmax, self.max_lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.max_lmax, self.max_lmax)

        self.so3_linear_2 = SO3_LinearV2(self.hidden_channels, self.output_channels, lmax=self.max_lmax)

    def forward(self, input_embedding):
        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(input_embedding.embedding.narrow(1, 0, 1))
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(input_embedding.embedding.narrow(1, 0, 1))

        input_embedding = self.so3_linear_1(input_embedding)

        if self.use_grid_mlp:
            input_embedding_grid = input_embedding.to_grid(self.SO3_grid, lmax=self.max_lmax)
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            input_embedding._from_grid(input_embedding_grid, self.SO3_grid, lmax=self.max_lmax)
            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat(
                    (gating_scalars, input_embedding.embedding.narrow(1, 1, input_embedding.embedding.shape[1] - 1)),
                    dim=1,
                )
        else:
            if self.use_gate_act:
                input_embedding.embedding = self.gate_act(gating_scalars, input_embedding.embedding)
            else:
                if self.use_sep_s2_act:
                    input_embedding.embedding = self.s2_act(gating_scalars, input_embedding.embedding, self.SO3_grid)
                else:
                    input_embedding.embedding = self.s2_act(input_embedding.embedding, self.SO3_grid)

        input_embedding = self.so3_linear_2(input_embedding)
        return input_embedding




class NonEquivariantExpertFFN(nn.Module):
    """
    Non-equivariant expert. Applies SiLU directly to ALL positions
    including l>0 — deliberately breaks equivariance.
    
    The model can learn to suppress this expert (w_k → 0) for environments
    where equivariance matters, or activate it where it helps.
    This gives us a diagnostic: which environments need strict equivariance?
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels:   int,
        lmax:            int,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(sphere_channels, hidden_channels, bias=True)  # bias OK — non-equivariant anyway
        self.linear_2 = nn.Linear(hidden_channels, sphere_channels, bias=True)
        self.act      = nn.SiLU()
        
        # context gate from c_i — same as equivariant expert
        # but here we apply it AFTER the nonlinearity, not as the nonlinearity
        self.gate_proj = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        x_emb: torch.Tensor,   # [N, (lmax+1)^2, sphere_channels]
        c_i:   torch.Tensor,   # [N, edge_channels]
    ) -> torch.Tensor:          # [N, (lmax+1)^2, sphere_channels]

        # linear 1 — same W applied to all (l, m) positions
        h = self.linear_1(x_emb)       # [N, 25, hidden_C]
        
        # SiLU applied to ALL positions — this is the equivariance violation
        # for l>0, different m components get different outputs after SiLU
        # because they have different values going in
        h = self.act(h)                 # [N, 25, hidden_C]  ← breaks equivariance for l>0
        
        # context gate from c_i — modulates how strongly this expert responds
        gate = self.gate_proj(c_i)      # [N, hidden_C]
        h    = h * gate.unsqueeze(1)    # [N, 25, hidden_C]
        
        h = self.linear_2(h)            # [N, 25, sphere_C]
        return h