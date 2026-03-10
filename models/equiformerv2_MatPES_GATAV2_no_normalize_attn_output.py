"""
EquiformerV2 for MatPES — Energy + Forces
with full GATA + HTR
==========================================

New vs previous version:
  - t_ij: per-edge scalar feature stream initialized in forward() and
    refined each layer by HTR before being used in GATAValueActivation.
  - HTR (Hierarchical Tensor Refinement): computes inner products of the
    evolving steerable features X_i^(l), X_j^(l) to enrich t_ij with
    layer-wise geometric context.
  - GATAValueActivation: now receives the full Eq. 6 inputs:
      sea_ij (attn_output) + (t_ij @ W_rs) * gamma_s(h_j)
    and splits into per-degree gates o_s, o_d^(l), o_t^(l).
  - TransBlockV2.forward() now returns (x, t_ij); the loop in forward()
    threads t_ij across all layers.

t_ij initialization (paper Eq. for t_ij,init):
    t_ij = (h_i + h_j) * (phi(r^0_ij) @ W_erp)
where:
    h_i, h_j  = l=0 slice of the initial SO3 node embedding [N, sphere_C]
    phi(r^0_ij) = Gaussian smearing of raw edge distances [E, num_rbf]
    W_erp     = learned linear projection [num_rbf → edge_channels]
    (h_i + h_j) encodes the chemical identity of the atom pair
    phi(...) @ W_erp encodes the distance

Force training design (unchanged):
  - edge_rot_mat is DETACHED
  - grad flows through pos → dvec → edge_distance → edge_dist_feat → blocks → energy
  - forces = -grad(energy_total, pos, create_graph=True)
"""

import math
import torch
import torch.nn as nn
from e3nn import o3

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BackboneInterface as BaseModel

from EquiformerV2Functions.so3 import (
    CoefficientMappingModule, SO3_Embedding, SO3_Grid,
    SO3_Rotation, SO3_LinearV2,
)
from EquiformerV2Functions.module_list import ModuleListInfo
from EquiformerV2Functions.radial_function import RadialFunction
from EquiformerV2Functions.layer_norm import get_normalization_layer
from NewFunctions.Gotennet_morethaninspired.transformer_block_no_attn_output_normalization_so_tij_no_dominate import FeedForwardNetwork, TransBlockV2
from EquiformerV2Functions.input_block import EdgeDegreeEmbedding
from fairchem.core.models.escaip.utils.smearing import GaussianSmearing


# ── Robust init_edge_rot_mat ──────────────────────────────────────────────────
def init_edge_rot_mat(edge_distance_vec):
    ev   = edge_distance_vec.detach()
    dist = torch.sqrt(torch.sum(ev ** 2, dim=1, keepdim=True)).clamp(min=1e-8)
    norm_x = ev / dist
    eye    = torch.eye(3, device=ev.device, dtype=ev.dtype)
    best   = torch.argmin(torch.abs(norm_x), dim=1)
    ref_vec = eye[best]
    norm_z = torch.cross(norm_x, ref_vec, dim=1)
    norm_z = norm_z / torch.sqrt(torch.sum(norm_z ** 2, dim=1, keepdim=True)).clamp(min=1e-8)
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.sqrt(torch.sum(norm_y ** 2, dim=1, keepdim=True)).clamp(min=1e-8)
    norm_x_v = norm_x.view(-1, 3, 1)
    norm_y_v = -norm_y.view(-1, 3, 1)
    norm_z_v = norm_z.view(-1, 3, 1)
    edge_rot_mat_inv = torch.cat([norm_z_v, norm_x_v, norm_y_v], dim=2)
    return torch.transpose(edge_rot_mat_inv, 1, 2)
# ─────────────────────────────────────────────────────────────────────────────


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

        self.use_pbc          = use_pbc
        self.regress_forces   = regress_forces
        self.regress_stress   = regress_stress
        self.max_neighbors    = max_neighbors
        self.max_radius       = max_radius
        self.cutoff           = max_radius
        self.max_num_elements = max_num_elements
        self.num_layers       = num_layers
        self.sphere_channels  = sphere_channels
        self.lmax_list        = lmax_list
        self.mmax_list        = mmax_list
        self.norm_type        = norm_type
        self.grid_resolution  = grid_resolution
        self.edge_channels    = edge_channels
        self.use_atom_edge_embedding      = use_atom_edge_embedding
        self.share_atom_edge_embedding    = share_atom_edge_embedding
        self.block_use_atom_edge_embedding = False if share_atom_edge_embedding else use_atom_edge_embedding
        self.use_m_share_rad      = use_m_share_rad
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads            = num_heads
        self.attn_alpha_channels  = attn_alpha_channels
        self.attn_value_channels  = attn_value_channels
        self.ffn_hidden_channels  = ffn_hidden_channels
        self.attn_activation      = attn_activation
        self.use_s2_act_attn      = use_s2_act_attn
        self.use_attn_renorm      = use_attn_renorm
        self.ffn_activation       = ffn_activation
        self.use_gate_act         = use_gate_act
        self.use_grid_mlp         = use_grid_mlp
        self.use_sep_s2_act       = use_sep_s2_act
        self.alpha_drop           = alpha_drop
        self.drop_path_rate       = drop_path_rate
        self.proj_drop            = proj_drop
        self.weight_init          = weight_init
        self.device               = 'cpu'
        self.num_resolutions      = len(lmax_list)
        self.sphere_channels_all  = self.num_resolutions * sphere_channels

        # ── spherical harmonics for rl_ij ─────────────────────────────────────
        _sh_irreps   = o3.Irreps.spherical_harmonics(max(lmax_list))
        self._sphere = o3.SphericalHarmonics(
            _sh_irreps, normalize=False, normalization='norm',
        )

        # ── node embedding ────────────────────────────────────────────────────
        self.sphere_embedding   = nn.Embedding(max_num_elements, self.sphere_channels_all)
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, 600, 2.0)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [edge_channels] * 2

        if share_atom_edge_embedding and use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] += 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding = None
            self.target_embedding = None

        # ── t_ij initialization components ───────────────────────────────────
        # Paper Eq. for t_ij,init:
        #     t_ij = (h_i + h_j)  *  (phi(r^0_ij) @ W_erp)
        #
        # W_erp projects the Gaussian smearing of raw distances
        # [E, num_rbf] → [E, edge_channels].
        # (h_i + h_j) has sphere_channels dims.
        # We therefore need a second projection to align channels:
        #     h_proj: sphere_channels → edge_channels
        # so the element-wise product makes sense dimensionally.
        num_rbf = int(self.distance_expansion.num_output)
        self.W_erp  = nn.Linear(num_rbf, edge_channels)          # phi(r) → edge_C
        self.h_proj = nn.Linear(sphere_channels, edge_channels)   # h_i/h_j → edge_C

        # ── SO3 machinery ─────────────────────────────────────────────────────
        self.SO3_rotation   = nn.ModuleList([SO3_Rotation(lmax_list[i]) for i in range(self.num_resolutions)])
        self.mappingReduced = CoefficientMappingModule(lmax_list, mmax_list)
        self.SO3_grid       = ModuleListInfo('({}, {})'.format(max(lmax_list), max(lmax_list)))
        for l in range(max(lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(l, m, resolution=grid_resolution, normalization='component'))
            self.SO3_grid.append(SO3_m_grid)

        # ── edge degree embedding ─────────────────────────────────────────────
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels, lmax_list, mmax_list, self.SO3_rotation, self.mappingReduced,
            max_num_elements, self.edge_channels_list, self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE_MATPES,
        )

        # ── transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransBlockV2(
                sphere_channels=sphere_channels,
                attn_hidden_channels=attn_hidden_channels,
                num_heads=num_heads,
                attn_alpha_channels=attn_alpha_channels,
                attn_value_channels=attn_value_channels,
                ffn_hidden_channels=ffn_hidden_channels,
                output_channels=sphere_channels,
                lmax_list=lmax_list,
                mmax_list=mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                max_num_elements=max_num_elements,
                edge_channels_list=self.edge_channels_list,
                edge_channels=edge_channels,        # ← t_ij width
                use_atom_edge_embedding=self.block_use_atom_edge_embedding,
                use_m_share_rad=use_m_share_rad,
                attn_activation=attn_activation,
                use_s2_act_attn=use_s2_act_attn,
                use_attn_renorm=use_attn_renorm,
                ffn_activation=ffn_activation,
                use_gate_act=use_gate_act,
                use_grid_mlp=use_grid_mlp,
                use_sep_s2_act=use_sep_s2_act,
                norm_type=norm_type,
                alpha_drop=alpha_drop,
                drop_path_rate=drop_path_rate,
                proj_drop=proj_drop,
            ))

        # ── output ────────────────────────────────────────────────────────────
        self.norm = get_normalization_layer(norm_type, lmax=max(lmax_list), num_channels=sphere_channels)
        self.energy_block = FeedForwardNetwork(
            sphere_channels, ffn_hidden_channels, 1, lmax_list, mmax_list, self.SO3_grid,
            ffn_activation, use_gate_act, use_grid_mlp, use_sep_s2_act,
        )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    # ─────────────────────────────────────────────────────────────────────────

    def _compute_rl_ij(self, edge_distance_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute edge spherical harmonics for l=1..lmax in the original frame.
        Detached — geometry defines the frame, not the energy landscape.
        """
        vec_det  = edge_distance_vec.detach()
        norm     = vec_det.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit_vec = vec_det / norm
        sh_full  = self._sphere(unit_vec)    # [E, (lmax+1)^2]
        return sh_full[:, 1:]                # [E, (lmax+1)^2 - 1], drop l=0

    def _init_t_ij(
        self,
        x_embedding:   torch.Tensor,   # [N, (L+1)^2, sphere_channels]  initial node SO3 emb
        edge_dist_feat: torch.Tensor,  # [E, num_rbf]  Gaussian smearing of distances
        edge_index:    torch.Tensor,   # [2, E]
    ) -> torch.Tensor:                 # [E, edge_channels]
        """
        Initialize t_ij for all edges.

        Paper equation:
            t_ij,init = (h_i + h_j) ⊙ (phi(r^0_ij) @ W_erp)

        where:
            h_i  = l=0 (scalar) slice of node i's SO3 embedding [sphere_channels]
            h_j  = l=0 (scalar) slice of node j's SO3 embedding [sphere_channels]
            phi(r^0_ij) = Gaussian smearing of ||r_ij|| [num_rbf]
            W_erp = learned projection [num_rbf → edge_channels]

        Intuition:
            (h_i + h_j) captures the chemical identity of the two endpoint atoms.
            phi(r^0_ij) @ W_erp captures the distance between them.
            Their element-wise product means both atom identity AND distance must
            jointly determine t_ij — the model can't ignore either.
        """
        # h_i, h_j: l=0 (invariant) slice of node embedding
        # x_embedding[:, 0, :] is the l=0 block for all nodes
        h_all = x_embedding[:, 0, :]                          # [N, sphere_C]
        h_i   = h_all[edge_index[0]]                          # [E, sphere_C]
        h_j   = h_all[edge_index[1]]                          # [E, sphere_C]

        # project node scalars to edge_channels
        h_sum = self.h_proj(h_i + h_j)                        # [E, edge_C]

        # project Gaussian smearing of distances to edge_channels
        phi_r = self.W_erp(edge_dist_feat)                    # [E, edge_C]

        # element-wise product: atom identity gates distance encoding
        t_ij  = h_sum * phi_r                                  # [E, edge_C]
        return t_ij

    # ─────────────────────────────────────────────────────────────────────────

    def generate_graph(self, pos, batch, cell):
        device  = pos.device
        pos_det = pos.detach()
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
                if src.shape[0] == 0:
                    continue
                src_list.append(src)
                dst_list.append(dst)

            if not src_list:
                continue
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

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, data):
        self.batch_size    = len(data['natoms'])
        self.dtype         = data['pos'].dtype
        self.device        = data['pos'].device
        atomic_numbers     = data['atomic_numbers'].long()
        num_atoms          = len(atomic_numbers)

        pos = data['pos']

        edge_index, edge_distance, edge_distance_vec = self.generate_graph(
            pos, data['batch'], data['cell']
        )

        # ── rotation matrices (detached) ──────────────────────────────────────
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        # ── edge spherical harmonics (detached, original frame) ───────────────
        # r^(l)_ij in the paper's Eq. 7 — the degree-l SH of the edge direction.
        # Detached because direction geometry (like rotation matrices) is not
        # part of the energy landscape; grad flows through radial distances.
        rl_ij = self._compute_rl_ij(edge_distance_vec)       # [E, (lmax+1)^2 - 1]

        # ── node embedding ────────────────────────────────────────────────────
        x = SO3_Embedding(num_atoms, self.lmax_list, self.sphere_channels, self.device, self.dtype)
        offset_res = offset = 0
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers) \
                if self.num_resolutions == 1 else \
                self.sphere_embedding(atomic_numbers)[:, offset:offset + self.sphere_channels]
            offset     += self.sphere_channels
            offset_res += int((self.lmax_list[i] + 1) ** 2)

        # ── edge distance features ────────────────────────────────────────────
        edge_dist_feat = self.distance_expansion(edge_distance)   # [E, num_rbf]
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            edge_dist_feat = torch.cat((
                edge_dist_feat,
                self.source_embedding(atomic_numbers[edge_index[0]]),
                self.target_embedding(atomic_numbers[edge_index[1]]),
            ), dim=1)

        x.embedding = x.embedding + self.edge_degree_embedding(
            atomic_numbers, edge_dist_feat, edge_index
        ).embedding

        # ── initialize t_ij ───────────────────────────────────────────────────
        # Paper: t_ij,init = (h_i + h_j) ⊙ (phi(r^0_ij) @ W_erp)
        #
        # This is done AFTER edge_degree_embedding updates x so that h_i, h_j
        # already contain some structural context from the initial embedding step.
        # t_ij then flows as a residual stream, refined by HTR at each layer.
        #
        # Note: we use self.distance_expansion(edge_distance) as phi(r^0_ij)
        # — the pure distance Gaussian smearing before any atom embedding concat.
        phi_r  = self.distance_expansion(edge_distance)           # [E, num_rbf] (pure distance)
        t_ij   = self._init_t_ij(x.embedding, phi_r, edge_index)  # [E, edge_channels]

        # ── transformer blocks ────────────────────────────────────────────────
        # t_ij is threaded through all blocks as a residual stream.
        # Each block: HTR refines t_ij, then GATA uses t_ij in the gate computation.
        for i in range(self.num_layers):
            x, t_ij = self.blocks[i](
                x,
                atomic_numbers,
                edge_dist_feat,
                edge_index,
                batch=data['batch'],
                t_ij=t_ij,      # ← refined in-place by HTR each layer
                rl_ij=rl_ij,    # ← fixed geometry, shared across all layers
            )

        # ── energy output ─────────────────────────────────────────────────────
        x.embedding  = self.norm(x.embedding)
        node_energy  = self.energy_block(x).embedding.narrow(1, 0, 1).view(-1)

        energy_total = torch.zeros(self.batch_size, device=self.device, dtype=node_energy.dtype)
        energy_total = energy_total.scatter_add(0, data['batch'], node_energy)
        energy_out   = (energy_total / data['natoms'].to(node_energy.dtype)).unsqueeze(1)

        return {
            'energy':       energy_out,
            'energy_total': energy_total,
            'pos':          pos,
        }

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                torch.nn.init.normal_(m.weight, 0, 1 / math.sqrt(m.in_features))
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
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
                    gname = f'{module_name}.{parameter_name}'
                    assert gname in named_parameters_list
                    no_wd_list.append(gname)
        return set(no_wd_list)