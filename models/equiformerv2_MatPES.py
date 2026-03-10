"""
EquiformerV2 adapted for MatPES — Energy + Forces + Stress Prediction
======================================================================

KEY DIFFERENCES FROM MP20 VERSION:
------------------------------------
  regress_forces : False → True
      Forces = -dE/dr, computed via autograd through the energy prediction.
      This is the physically correct approach: the model predicts E(R),
      and forces are its negative gradient with respect to atomic positions.

  regress_stress : True (new)
      Stress = dE/d(strain), computed via the cell deformation Jacobian.
      Implemented via the "virials" method: stress = (1/V) * sum_ij r_ij ⊗ F_ij
      We use autograd on a strain variable appended to the positions.

  force_mult : scalar to scale force prediction (default 1.0)

WHY AUTOGRAD FOR FORCES?
--------------------------
  The equivariant architecture already respects rotational symmetry.
  Taking -dE/dr automatically gives forces that:
    (a) are equivariant (transform correctly under rotations)
    (b) are conservative (obey Newton's 3rd law for the learned PES)
    (c) require no separate force head — saving parameters

  This "energy-conserving" approach is standard in NN force fields
  (NequIP, MACE, SevenNet, EquiformerV2-OC20 all do this).

FORWARD SIGNATURE:
  Returns dict: {'energy': [B,1], 'forces': [N,3], 'stress': [B,6]}
"""

import math
import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BackboneInterface as BaseModel

from EquiformerV2Functions.so3 import (
    CoefficientMappingModule, SO3_Embedding, SO3_Grid,
    SO3_Rotation, SO3_LinearV2,
)
from EquiformerV2Functions.module_list import ModuleListInfo
from EquiformerV2Functions.so2_ops import SO2_Convolution
from EquiformerV2Functions.radial_function import RadialFunction
from EquiformerV2Functions.layer_norm import get_normalization_layer
from EquiformerV2Functions.transformer_block import (
    SO2EquivariantGraphAttention, FeedForwardNetwork, TransBlockV2,
)
from EquiformerV2Functions.input_block import EdgeDegreeEmbedding
from EquiformerV2Functions.edge_rot_mat import init_edge_rot_mat
from fairchem.core.models.escaip.utils.smearing import GaussianSmearing

try:
    from e3nn import o3
except ImportError:
    pass


_AVG_NUM_NODES_MATPES = 30.0   # MatPES structures average ~30 atoms/cell
_AVG_DEGREE_MATPES    = 12.0   # ~12 neighbours within 6 Å


@registry.register_model("equiformer_v2_matpes")
class EquiformerV2_MatPES(BaseModel, nn.Module):
    """
    EquiformerV2 universal MLIP for MatPES.

    Predicts:
      - energy_per_atom  (eV/atom)
      - forces           (eV/Å)   via autograd
      - stress           (eV/Å³)  via autograd strain
    """

    def __init__(
        self,
        use_pbc:                  bool  = True,
        regress_forces:           bool  = True,
        regress_stress:           bool  = True,
        otf_graph:                bool  = True,
        max_neighbors:            int   = 20,
        max_radius:               float = 6.0,
        max_num_elements:         int   = 100,
        num_layers:               int   = 6,
        sphere_channels:          int   = 128,
        attn_hidden_channels:     int   = 128,
        num_heads:                int   = 8,
        attn_alpha_channels:      int   = 32,
        attn_value_channels:      int   = 16,
        ffn_hidden_channels:      int   = 512,
        norm_type:                str   = 'rms_norm_sh',
        lmax_list:                list  = None,
        mmax_list:                list  = None,
        grid_resolution:          int   = 18,
        num_sphere_samples:       int   = 128,
        edge_channels:            int   = 128,
        use_atom_edge_embedding:  bool  = True,
        share_atom_edge_embedding:bool  = False,
        use_m_share_rad:          bool  = False,
        distance_function:        str   = 'gaussian',
        num_distance_basis:       int   = 512,
        attn_activation:          str   = 'scaled_silu',
        use_s2_act_attn:          bool  = False,
        use_attn_renorm:          bool  = True,
        ffn_activation:           str   = 'scaled_silu',
        use_gate_act:             bool  = False,
        use_grid_mlp:             bool  = False,
        use_sep_s2_act:           bool  = True,
        alpha_drop:               float = 0.05,
        drop_path_rate:           float = 0.05,
        proj_drop:                float = 0.0,
        weight_init:              str   = 'normal',
    ):
        super().__init__()

        if lmax_list is None:
            lmax_list = [4]
        if mmax_list is None:
            mmax_list = [2]

        self.use_pbc                  = use_pbc
        self.regress_forces           = regress_forces
        self.regress_stress           = regress_stress
        self.otf_graph                = otf_graph
        self.max_neighbors            = max_neighbors
        self.max_radius               = max_radius
        self.cutoff                   = max_radius
        self.max_num_elements         = max_num_elements
        self.num_layers               = num_layers
        self.sphere_channels          = sphere_channels
        self.attn_hidden_channels     = attn_hidden_channels
        self.num_heads                = num_heads
        self.attn_alpha_channels      = attn_alpha_channels
        self.attn_value_channels      = attn_value_channels
        self.ffn_hidden_channels      = ffn_hidden_channels
        self.norm_type                = norm_type
        self.lmax_list                = lmax_list
        self.mmax_list                = mmax_list
        self.grid_resolution          = grid_resolution
        self.num_sphere_samples       = num_sphere_samples
        self.edge_channels            = edge_channels
        self.use_atom_edge_embedding  = use_atom_edge_embedding
        self.share_atom_edge_embedding= share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad          = use_m_share_rad
        self.distance_function        = distance_function
        self.num_distance_basis       = num_distance_basis
        self.attn_activation          = attn_activation
        self.use_s2_act_attn          = use_s2_act_attn
        self.use_attn_renorm          = use_attn_renorm
        self.ffn_activation           = ffn_activation
        self.use_gate_act             = use_gate_act
        self.use_grid_mlp             = use_grid_mlp
        self.use_sep_s2_act           = use_sep_s2_act
        self.alpha_drop               = alpha_drop
        self.drop_path_rate           = drop_path_rate
        self.proj_drop                = proj_drop
        self.weight_init              = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device            = 'cpu'
        self.grad_forces       = regress_forces or regress_stress
        self.num_resolutions   = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels

        # ── Atomic embedding ─────────────────────────────────────
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # ── Distance basis ───────────────────────────────────────
        assert self.distance_function == 'gaussian'
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, 600, 2.0)
        self.edge_channels_list = [
            int(self.distance_expansion.num_output)
        ] + [self.edge_channels] * 2

        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] += 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding = None
            self.target_embedding = None

        # ── SO3 rotation ─────────────────────────────────────────
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        self.SO3_grid = ModuleListInfo(
            '({}, {})'.format(max(self.lmax_list), max(self.lmax_list))
        )
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m, resolution=self.grid_resolution,
                             normalization='component')
                )
            self.SO3_grid.append(SO3_m_grid)

        # ── Edge-degree embedding ────────────────────────────────
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels, self.lmax_list, self.mmax_list,
            self.SO3_rotation, self.mappingReduced, self.max_num_elements,
            self.edge_channels_list, self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE_MATPES,
        )

        # ── Transformer blocks ───────────────────────────────────
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransBlockV2(
                self.sphere_channels, self.attn_hidden_channels, self.num_heads,
                self.attn_alpha_channels, self.attn_value_channels,
                self.ffn_hidden_channels, self.sphere_channels,
                self.lmax_list, self.mmax_list, self.SO3_rotation,
                self.mappingReduced, self.SO3_grid, self.max_num_elements,
                self.edge_channels_list, self.block_use_atom_edge_embedding,
                self.use_m_share_rad, self.attn_activation, self.use_s2_act_attn,
                self.use_attn_renorm, self.ffn_activation, self.use_gate_act,
                self.use_grid_mlp, self.use_sep_s2_act, self.norm_type,
                self.alpha_drop, self.drop_path_rate, self.proj_drop,
            ))

        # ── Output ───────────────────────────────────────────────
        self.norm = get_normalization_layer(
            self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels
        )
        self.energy_block = FeedForwardNetwork(
            self.sphere_channels, self.ffn_hidden_channels, 1,
            self.lmax_list, self.mmax_list, self.SO3_grid,
            self.ffn_activation, self.use_gate_act, self.use_grid_mlp,
            self.use_sep_s2_act,
        )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    # ────────────────────────────────────────────────────────────
    # Graph construction (PBC, identical to MP20)
    # ────────────────────────────────────────────────────────────

    def generate_graph(self, data):
        pos    = data['pos']
        batch  = data['batch']
        cell   = data['cell']
        device = pos.device
        N_total = pos.shape[0]

        edge_index_list        = []
        edge_distance_list     = []
        edge_distance_vec_list = []

        for b_idx in range(cell.shape[0]):
            mask        = (batch == b_idx)
            node_global = torch.where(mask)[0]
            pos_b       = pos[mask]
            cell_b      = cell[b_idx]
            N           = pos_b.shape[0]

            images_range = torch.arange(-1, 2, device=device, dtype=torch.float32)
            gx, gy, gz   = torch.meshgrid(images_range, images_range, images_range,
                                          indexing='ij')
            offsets_frac = torch.stack([gx.reshape(-1), gy.reshape(-1),
                                        gz.reshape(-1)], dim=1)
            offsets_cart = offsets_frac @ cell_b

            src_list, dst_list, dvec_list = [], [], []
            for img_offset in offsets_cart:
                pos_img = pos_b + img_offset.unsqueeze(0)
                diff    = pos_img.unsqueeze(0) - pos_b.unsqueeze(1)
                dist    = torch.norm(diff, dim=2)
                is_zero = (img_offset.abs().sum() < 1e-6)
                within  = (dist < self.max_radius) & (dist > 1e-6) if is_zero \
                          else (dist < self.max_radius)
                src, dst = torch.where(within)
                if src.shape[0] == 0:
                    continue
                src_list.append(src)
                dst_list.append(dst)
                dvec_list.append(diff[src, dst])

            if not src_list:
                continue

            src_all  = torch.cat(src_list)
            dst_all  = torch.cat(dst_list)
            dvec_all = torch.cat(dvec_list)
            dist_all = torch.norm(dvec_all, dim=1)

            if self.max_neighbors is not None:
                keep = torch.zeros(len(src_all), dtype=torch.bool, device=device)
                for d in torch.unique(dst_all):
                    mask_d = dst_all == d
                    idx_d  = torch.where(mask_d)[0]
                    dist_d = dist_all[mask_d]
                    if len(idx_d) > self.max_neighbors:
                        _, si = torch.sort(dist_d)
                        keep[idx_d[si[:self.max_neighbors]]] = True
                    else:
                        keep[idx_d] = True
                src_all  = src_all[keep];  dst_all  = dst_all[keep]
                dvec_all = dvec_all[keep]; dist_all = dist_all[keep]

            src_global = node_global[src_all]
            dst_global = node_global[dst_all]
            edge_index_list.append(torch.stack([src_global, dst_global], dim=0))
            edge_distance_list.append(dist_all)
            edge_distance_vec_list.append(dvec_all)

        if edge_index_list:
            edge_index        = torch.cat(edge_index_list,        dim=1)
            edge_distance     = torch.cat(edge_distance_list,     dim=0)
            edge_distance_vec = torch.cat(edge_distance_vec_list, dim=0)
        else:
            edge_index        = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_distance     = torch.zeros(0,       device=device)
            edge_distance_vec = torch.zeros((0, 3),  device=device)

        neighbors = torch.zeros(N_total, dtype=torch.long, device=device)
        if edge_index.shape[1] > 0:
            unique, counts = torch.unique(edge_index[1], return_counts=True)
            neighbors[unique] = counts

        return edge_index, edge_distance, edge_distance_vec, None, None, neighbors

    # ────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        """
        Args:
            data : dict with keys
                   atomic_numbers [N], pos [N,3], cell [B,3,3],
                   batch [N], natoms [B], pbc [B,3]

        Returns:
            dict with:
              'energy' : [B, 1]   per-atom energy (normalized)
              'forces' : [N, 3]   forces in eV/Å  (if regress_forces)
              'stress' : [B, 6]   Voigt stress eV/Å³ (if regress_stress)
        """
        self.batch_size = len(data['natoms'])
        self.dtype      = data['pos'].dtype
        self.device     = data['pos'].device

        atomic_numbers = data['atomic_numbers'].long()
        num_atoms      = len(atomic_numbers)

        # ── Enable pos gradient for force computation ─────────────
        pos = data['pos']
        if self.regress_forces:
            pos = pos.requires_grad_(True)

        # ── Strain for stress computation ─────────────────────────
        if self.regress_stress:
            strain = torch.zeros(
                (self.batch_size, 3, 3),
                device=self.device, dtype=self.dtype, requires_grad=True
            )
            # Apply infinitesimal strain to positions: r' = r @ (I + ε)
            eye = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)
            deform = eye + strain                            # [B, 3, 3]
            # Per-atom deformation
            atom_deform = deform[data['batch']]              # [N, 3, 3]
            pos = torch.einsum('ni,nij->nj', pos, atom_deform)
            # Also transform cell
            data_cell = data['cell']
            data_cell = torch.einsum('bij,bjk->bik', data_cell, deform)
            data_pos_for_graph = pos
            data_cell_for_graph = data_cell
        else:
            data_pos_for_graph  = pos
            data_cell_for_graph = data['cell']

        # Temporarily update data dict for graph construction
        data_for_graph = dict(data)
        data_for_graph['pos']  = data_pos_for_graph
        data_for_graph['cell'] = data_cell_for_graph

        (
            edge_index, edge_distance, edge_distance_vec, _, _, neighbors
        ) = self.generate_graph(data_for_graph)

        edge_rot_mat = self._init_edge_rot_mat(data, edge_index, edge_distance_vec)
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        # ── Node embeddings ──────────────────────────────────────
        x = SO3_Embedding(
            num_atoms, self.lmax_list, self.sphere_channels,
            self.device, self.dtype,
        )
        offset_res = offset = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = \
                    self.sphere_embedding(atomic_numbers)[:, offset:offset+self.sphere_channels]
            offset     += self.sphere_channels
            offset_res += int((self.lmax_list[i] + 1) ** 2)

        # ── Edge features ────────────────────────────────────────
        edge_dist_feat = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            src_el = atomic_numbers[edge_index[0]]
            tgt_el = atomic_numbers[edge_index[1]]
            edge_dist_feat = torch.cat(
                (edge_dist_feat,
                 self.source_embedding(src_el),
                 self.target_embedding(tgt_el)), dim=1
            )

        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_dist_feat, edge_index
        )
        x.embedding = x.embedding + edge_degree.embedding

        # ── Transformer blocks ───────────────────────────────────
        for i in range(self.num_layers):
            x = self.blocks[i](
                x, atomic_numbers, edge_dist_feat, edge_index,
                batch=data['batch']
            )

        x.embedding = self.norm(x.embedding)

        # ── Output: per-atom energy → total → per-atom ────────────
        node_energy = self.energy_block(x).embedding.narrow(1, 0, 1).view(-1)

        energy_total = torch.zeros(
            self.batch_size, device=self.device, dtype=node_energy.dtype
        )
        energy_total.index_add_(0, data['batch'], node_energy)
        natoms_f         = data['natoms'].to(node_energy.dtype)
        energy_per_atom  = energy_total / natoms_f               # [B]
        energy_out       = energy_per_atom.unsqueeze(1)           # [B, 1]

        #out = {'energy': energy_out}
        out = {'energy': energy_out, 'pos': pos, 'energy_total': energy_total}
        # ── Forces via autograd ───────────────────────────────────
        if self.regress_forces:
            forces = -torch.autograd.grad(
                energy_total.sum(), pos,
                create_graph= True, #self.training,
                retain_graph= False, #self.regress_stress,
            )[0]
            out['forces'] = forces                               # [N, 3]

        # ── Stress via autograd on strain ─────────────────────────
        if self.regress_stress:
            # dE/dε / V  → stress [B, 3, 3] → Voigt [B, 6]
            # Cell volume per graph
            vol = torch.abs(torch.linalg.det(data['cell']))     # [B]
            # Gradient of total energy w.r.t. strain
            stress_full = torch.autograd.grad(
                energy_total.sum(), strain,
                create_graph=self.training,
            )[0]                                                  # [B, 3, 3]
            stress_full = stress_full / vol.view(-1, 1, 1)       # eV/Å³
            # Convert to Voigt [B, 6]: xx yy zz yz xz xy
            stress_voigt = torch.stack([
                stress_full[:, 0, 0],
                stress_full[:, 1, 1],
                stress_full[:, 2, 2],
                stress_full[:, 1, 2],
                stress_full[:, 0, 2],
                stress_full[:, 0, 1],
            ], dim=1)                                             # [B, 6]
            out['stress'] = stress_voigt

        return out

    # ────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────

    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias,   0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(module, (torch.nn.Linear, SO3_LinearV2, torch.nn.LayerNorm)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, (torch.nn.Linear, SO3_LinearV2)):
                        if 'weight' in parameter_name:
                            continue
                    gname = module_name + '.' + parameter_name
                    assert gname in named_parameters_list
                    no_wd_list.append(gname)
        return set(no_wd_list)