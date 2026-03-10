"""
EquiformerV2 for MatPES — Energy + Forces
==========================================

Force training design:
  - edge_rot_mat is DETACHED (rotation frames are not part of energy landscape)
  - grad flows through: pos → dvec → edge_distance → edge_dist_feat → blocks → energy
  - forces = -grad(energy_total, pos, create_graph=True)
  - single backward pass on combined loss: w_e*e_loss + w_f*f_loss
  - create_graph=True is required so that f_loss has a grad_fn back to model params
"""

import math
import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BackboneInterface as BaseModel

from EquiformerV2Functions.so3 import (
    CoefficientMappingModule, SO3_Embedding, SO3_Grid,
    SO3_Rotation, SO3_LinearV2,
)
from EquiformerV2Functions.module_list import ModuleListInfo
from EquiformerV2Functions.radial_function import RadialFunction
from EquiformerV2Functions.layer_norm import get_normalization_layer
from EquiformerV2Functions.transformer_block import FeedForwardNetwork, TransBlockV2
from EquiformerV2Functions.input_block import EdgeDegreeEmbedding
# baseline imported but not used directly — kept for reference
from EquiformerV2Functions.edge_rot_mat import init_edge_rot_mat as _baseline_init_edge_rot_mat
from fairchem.core.models.escaip.utils.smearing import GaussianSmearing


# ── Robust init_edge_rot_mat ──────────────────────────────────────────
# Replaces the baseline without touching edge_rot_mat.py.
# The baseline random-vector approach asserts vec_dot < 0.99, which fails
# for axis-aligned bonds common in crystals (MatPES).
# Fix: deterministically pick the cardinal axis least-aligned with each edge.
# The result is DETACHED — rotation frames define geometry, not the energy
# landscape. Grad flows through edge_distance (radial), not edge_rot_mat.
def init_edge_rot_mat(edge_distance_vec):
    # Work on detached vectors — rotation frame does not need grad
    ev = edge_distance_vec.detach()
    dist   = torch.sqrt(torch.sum(ev ** 2, dim=1, keepdim=True)).clamp(min=1e-8)
    norm_x = ev / dist                                        # [E, 3] unit vectors

    # For each edge, choose the cardinal axis with smallest |dot| with norm_x.
    # A unit vector can be close to at most one cardinal axis, so the minimum
    # is always <= 1/sqrt(3) ~ 0.577, well below the 0.99 threshold.
    eye     = torch.eye(3, device=ev.device, dtype=ev.dtype)
    best    = torch.argmin(torch.abs(norm_x), dim=1)          # [E]
    ref_vec = eye[best]                                        # [E, 3]

    norm_z = torch.cross(norm_x, ref_vec, dim=1)
    norm_z = norm_z / torch.sqrt(torch.sum(norm_z ** 2, dim=1, keepdim=True)).clamp(min=1e-8)

    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.sqrt(torch.sum(norm_y ** 2, dim=1, keepdim=True)).clamp(min=1e-8)

    # Same matrix convention as baseline
    norm_x_v = norm_x.view(-1, 3, 1)
    norm_y_v = -norm_y.view(-1, 3, 1)
    norm_z_v = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z_v, norm_x_v, norm_y_v], dim=2)
    return torch.transpose(edge_rot_mat_inv, 1, 2)  # already detached
# ─────────────────────────────────────────────────────────────────────


_AVG_DEGREE_MATPES = 12.0


@registry.register_model("equiformer_v2_matpes")
class EquiformerV2_MatPES(BaseModel, nn.Module):

    def __init__(
        self,
        use_pbc=True, regress_forces=True, regress_stress=False,
        otf_graph=True, max_neighbors=20, max_radius=6.0, max_num_elements=100,
        num_layers=6, sphere_channels=128, attn_hidden_channels=128, num_heads=8,
        attn_alpha_channels=32, attn_value_channels=16, ffn_hidden_channels=512,
        norm_type='rms_norm_sh', lmax_list=None, mmax_list=None, grid_resolution=18,
        num_sphere_samples=128, edge_channels=128, use_atom_edge_embedding=True,
        share_atom_edge_embedding=False, use_m_share_rad=False,
        distance_function='gaussian', num_distance_basis=512,
        attn_activation='scaled_silu', use_s2_act_attn=False, use_attn_renorm=True,
        ffn_activation='scaled_silu', use_gate_act=False, use_grid_mlp=False,
        use_sep_s2_act=True, alpha_drop=0.05, drop_path_rate=0.05, proj_drop=0.0,
        weight_init='normal',
    ):
        super().__init__()
        if lmax_list is None: lmax_list = [4]
        if mmax_list is None: mmax_list = [2]

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.regress_stress = regress_stress
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.norm_type = norm_type
        self.grid_resolution = grid_resolution
        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        self.block_use_atom_edge_embedding = False if share_atom_edge_embedding else use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.weight_init = weight_init
        self.device = 'cpu'
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * sphere_channels

        self.sphere_embedding = nn.Embedding(max_num_elements, self.sphere_channels_all)
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, 600, 2.0)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [edge_channels] * 2

        if share_atom_edge_embedding and use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] += 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding = None
            self.target_embedding = None

        self.SO3_rotation = nn.ModuleList([SO3_Rotation(lmax_list[i]) for i in range(self.num_resolutions)])
        self.mappingReduced = CoefficientMappingModule(lmax_list, mmax_list)
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(lmax_list), max(lmax_list)))
        for l in range(max(lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(l, m, resolution=grid_resolution, normalization='component'))
            self.SO3_grid.append(SO3_m_grid)

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels, lmax_list, mmax_list, self.SO3_rotation, self.mappingReduced,
            max_num_elements, self.edge_channels_list, self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE_MATPES,
        )
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransBlockV2(
                sphere_channels, attn_hidden_channels, num_heads, attn_alpha_channels,
                attn_value_channels, ffn_hidden_channels, sphere_channels, lmax_list, mmax_list,
                self.SO3_rotation, self.mappingReduced, self.SO3_grid, max_num_elements,
                self.edge_channels_list, self.block_use_atom_edge_embedding, use_m_share_rad,
                attn_activation, use_s2_act_attn, use_attn_renorm, ffn_activation, use_gate_act,
                use_grid_mlp, use_sep_s2_act, norm_type, alpha_drop, drop_path_rate, proj_drop,
            ))
        self.norm = get_normalization_layer(norm_type, lmax=max(lmax_list), num_channels=sphere_channels)
        self.energy_block = FeedForwardNetwork(
            sphere_channels, ffn_hidden_channels, 1, lmax_list, mmax_list, self.SO3_grid,
            ffn_activation, use_gate_act, use_grid_mlp, use_sep_s2_act,
        )
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def generate_graph(self, pos, batch, cell):
        device  = pos.device
        pos_det = pos.detach()   # detached copy for topology only
        edge_index_list, edge_distance_list, edge_distance_vec_list = [], [], []

        for b_idx in range(cell.shape[0]):
            mask        = (batch == b_idx)
            node_global = torch.where(mask)[0]
            pos_b       = pos_det[mask]
            cell_b      = cell[b_idx].detach()

            images_range = torch.arange(-1, 2, device=device, dtype=torch.float32)
            gx, gy, gz   = torch.meshgrid(images_range, images_range, images_range, indexing='ij')
            offsets_cart = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1) @ cell_b

            src_list, dst_list = [], []
            for img_offset in offsets_cart:
                pos_img = pos_b + img_offset.unsqueeze(0)
                diff    = pos_img.unsqueeze(0) - pos_b.unsqueeze(1)
                dist    = torch.norm(diff, dim=2)
                is_zero = img_offset.abs().sum() < 1e-6
                within  = (dist < self.max_radius) & (dist > 1e-6) if is_zero else (dist < self.max_radius)
                src, dst = torch.where(within)
                if src.shape[0] == 0: continue
                src_list.append(src); dst_list.append(dst)

            if not src_list: continue
            src_all = torch.cat(src_list)
            dst_all = torch.cat(dst_list)
            src_global = node_global[src_all]
            dst_global = node_global[dst_all]

            if self.max_neighbors is not None:
                dvec_det = pos_det[mask][dst_all] - pos_det[mask][src_all]
                dist_det = torch.norm(dvec_det, dim=1)
                keep = torch.zeros(len(src_all), dtype=torch.bool, device=device)
                for d in torch.unique(dst_all):
                    idx_d  = torch.where(dst_all == d)[0]
                    dist_d = dist_det[idx_d]
                    if len(idx_d) > self.max_neighbors:
                        _, si = torch.sort(dist_d)
                        keep[idx_d[si[:self.max_neighbors]]] = True
                    else:
                        keep[idx_d] = True
                src_global = src_global[keep]
                dst_global = dst_global[keep]

            edge_index_list.append(torch.stack([src_global, dst_global], dim=0))
            # IMPORTANT: use original pos (with grad) for dvec so that
            # edge_distance retains grad and forces can be computed
            dvec = pos[dst_global] - pos[src_global]
            edge_distance_list.append(torch.norm(dvec, dim=1))
            edge_distance_vec_list.append(dvec)

        if edge_index_list:
            edge_index        = torch.cat(edge_index_list, dim=1)
            edge_distance     = torch.cat(edge_distance_list, dim=0)
            edge_distance_vec = torch.cat(edge_distance_vec_list, dim=0)
        else:
            edge_index        = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_distance     = torch.zeros(0, device=device)
            edge_distance_vec = torch.zeros((0, 3), device=device)

        return edge_index, edge_distance, edge_distance_vec

    def forward(self, data):
        self.batch_size = len(data['natoms'])
        self.dtype      = data['pos'].dtype
        self.device     = data['pos'].device
        atomic_numbers  = data['atomic_numbers'].long()
        num_atoms       = len(atomic_numbers)

        pos = data['pos']  # has requires_grad=True during training

        edge_index, edge_distance, edge_distance_vec = self.generate_graph(
            pos, data['batch'], data['cell']
        )

        # Rotation matrices: detached (frame geometry, not energy landscape)
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        x = SO3_Embedding(num_atoms, self.lmax_list, self.sphere_channels, self.device, self.dtype)
        offset_res = offset = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[:, offset:offset + self.sphere_channels]
            offset += self.sphere_channels
            offset_res += int((self.lmax_list[i] + 1) ** 2)

        # edge_distance has grad → edge_dist_feat has grad → blocks → energy has grad
        edge_dist_feat = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            edge_dist_feat = torch.cat((
                edge_dist_feat,
                self.source_embedding(atomic_numbers[edge_index[0]]),
                self.target_embedding(atomic_numbers[edge_index[1]]),
            ), dim=1)

        x.embedding = x.embedding + self.edge_degree_embedding(
            atomic_numbers, edge_dist_feat, edge_index
        ).embedding

        for i in range(self.num_layers):
            x = self.blocks[i](x, atomic_numbers, edge_dist_feat, edge_index, batch=data['batch'])

        x.embedding = self.norm(x.embedding)
        node_energy = self.energy_block(x).embedding.narrow(1, 0, 1).view(-1)

        # Use scatter_add instead of index_add_ to avoid in-place op breaking grad graph
        energy_total = torch.zeros(self.batch_size, device=self.device, dtype=node_energy.dtype)
        energy_total = energy_total.scatter_add(0, data['batch'], node_energy)
        energy_out   = (energy_total / data['natoms'].to(node_energy.dtype)).unsqueeze(1)

        return {
            'energy':       energy_out,    # [B, 1]
            'energy_total': energy_total,  # [B]  — used for force autograd
            'pos':          pos,           # [N, 3] — the leaf with requires_grad
        }

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Linear, SO3_LinearV2)):
            if m.bias is not None: torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                torch.nn.init.normal_(m.weight, 0, 1 / math.sqrt(m.in_features))
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction): m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None: torch.nn.init.constant_(m.bias, 0)
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
                        if 'weight' in parameter_name: continue
                    gname = f'{module_name}.{parameter_name}'
                    assert gname in named_parameters_list
                    no_wd_list.append(gname)
        return set(no_wd_list)