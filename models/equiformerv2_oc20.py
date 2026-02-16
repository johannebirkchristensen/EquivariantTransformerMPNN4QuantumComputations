"""
EquiformerV2 for OC20 with proper PBC support
Uses official OCP implementation exactly
"""
import math
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add OCP path
OCP_PATH = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/OC20/ocp/packages/fairchem-data-oc/src'
sys.path.insert(0, OCP_PATH)

from EquiformerV2Functions.edge_rot_mat import init_edge_rot_mat
from EquiformerV2Functions.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from EquiformerV2Functions.module_list import ModuleListInfo
from EquiformerV2Functions.radial_function import RadialFunction
from EquiformerV2Functions.layer_norm import get_normalization_layer
from EquiformerV2Functions.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2,
)
from EquiformerV2Functions.input_block import EdgeDegreeEmbedding

# Import official OCP graph generation (this handles everything correctly)
from fairchem.core.graph.compute import generate_graph

# OC20 statistics
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773


class GaussianSmearing(nn.Module):
    """Gaussian smearing for distance encoding"""
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, basis_width_scalar=1.0):
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
    @property
    def num_output(self):
        return self.num_gaussians


class EquiformerV2_OC20(nn.Module):
    """EquiformerV2 for OC20 with PBC support"""
    
    def __init__(
        self,
        num_atoms=None,
        bond_feat_dim=None,
        num_targets=1,
        use_pbc=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=20,
        max_radius=12.0,
        max_num_elements=90,
        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        norm_type='rms_norm_sh',
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=18,
        num_sphere_samples=128,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=600,
        attn_activation='silu',
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation='silu',
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
        weight_init='uniform'
    ):
        super().__init__()
        
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements
        
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution
        
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * sphere_channels
        
        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        self.weight_init = weight_init
        
        if share_atom_edge_embedding:
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = use_atom_edge_embedding
            
        # Embeddings
        self.sphere_embedding = nn.Embedding(max_num_elements, self.sphere_channels_all)
        
        # Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, max_radius, num_distance_basis, 2.0)
        
        self.edge_channels_list = [num_distance_basis, edge_channels, edge_channels]
        
        # Atom edge embeddings
        if share_atom_edge_embedding and use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(max_num_elements, edge_channels)
            self.target_embedding = nn.Embedding(max_num_elements, edge_channels)
            self.edge_channels_list[0] = num_distance_basis + 2 * edge_channels
        else:
            self.source_embedding = None
            self.target_embedding = None
        
        # SO3 modules
        self.SO3_rotation = nn.ModuleList([SO3_Rotation(lmax) for lmax in lmax_list])
        self.mappingReduced = CoefficientMappingModule(lmax_list, mmax_list)
        
        # SO3 grids
        self.SO3_grid = ModuleListInfo(f'({max(lmax_list)}, {max(lmax_list)})')
        for l in range(max(lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(l, m, resolution=grid_resolution, normalization='component'))
            self.SO3_grid.append(SO3_m_grid)
        
        # Edge degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels, lmax_list, mmax_list, self.SO3_rotation, self.mappingReduced,
            max_num_elements, self.edge_channels_list, self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransBlockV2(
                sphere_channels, attn_hidden_channels, num_heads,
                attn_alpha_channels, attn_value_channels, ffn_hidden_channels,
                sphere_channels, lmax_list, mmax_list, self.SO3_rotation, self.mappingReduced,
                self.SO3_grid, max_num_elements, self.edge_channels_list,
                self.block_use_atom_edge_embedding, use_m_share_rad,
                attn_activation, use_s2_act_attn, use_attn_renorm,
                ffn_activation, use_gate_act, use_grid_mlp, use_sep_s2_act,
                norm_type, alpha_drop, drop_path_rate, proj_drop
            ))
        
        # Output layers
        self.norm = get_normalization_layer(norm_type, lmax=max(lmax_list), num_channels=sphere_channels)
        self.energy_block = FeedForwardNetwork(
            sphere_channels, ffn_hidden_channels, 1, lmax_list, mmax_list,
            self.SO3_grid, ffn_activation, use_gate_act, use_grid_mlp, use_sep_s2_act
        )
        
        if regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                sphere_channels, attn_hidden_channels, num_heads,
                attn_alpha_channels, attn_value_channels, 1, lmax_list, mmax_list,
                self.SO3_rotation, self.mappingReduced, self.SO3_grid,
                max_num_elements, self.edge_channels_list, self.block_use_atom_edge_embedding,
                use_m_share_rad, attn_activation, use_s2_act_attn, use_attn_renorm,
                use_gate_act, use_sep_s2_act, alpha_drop=0.0
            )
        
        self.apply(self._init_weights)
    
    def forward(self, data):
        # Convert dict to object with required attributes
        class DataObj:
            pass
        data_obj = DataObj()
        for k, v in data.items():
            setattr(data_obj, k, v)
        
        # Add pbc attribute (required by generate_graph)
        if not hasattr(data_obj, 'pbc'):
            data_obj.pbc = torch.tensor([[True, True, True]] * len(data['natoms']), device=data['pos'].device)
        
        data = data_obj
        
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)
        
        # Use official OCP generate_graph - handles everything correctly
        graph_data = generate_graph(
            data=data,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            enforce_max_neighbors_strictly=False,
            radius_pbc_version=1,  # Use version 1 (standard)
            pbc=data.pbc[0] if len(data.pbc.shape) > 1 else data.pbc  # Extract first element if batched
        )
        
        edge_index = graph_data['edge_index']
        edge_distance = graph_data['edge_distance']
        edge_distance_vec = graph_data['edge_distance_vec']
        
        # Edge rotation matrices
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)
        
        # Node embeddings
        x = SO3_Embedding(num_atoms, self.lmax_list, self.sphere_channels, 
                         data.pos.device, data.pos.dtype)
        
        offset_res = 0
        offset = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels]
            offset += self.sphere_channels
            offset_res += int((self.lmax_list[i] + 1) ** 2)
        
        # Edge encoding
        edge_distance_encoded = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]
            target_element = atomic_numbers[edge_index[1]]
            edge_distance_encoded = torch.cat([
                edge_distance_encoded,
                self.source_embedding(source_element),
                self.target_embedding(target_element)
            ], dim=1)
        
        # Edge degree embedding
        edge_degree = self.edge_degree_embedding(atomic_numbers, edge_distance_encoded, edge_index)
        x.embedding = x.embedding + edge_degree.embedding
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, atomic_numbers, edge_distance_encoded, edge_index, batch=data.batch)
        
        x.embedding = self.norm(x.embedding)
        
        # Energy
        node_energy = self.energy_block(x).embedding.narrow(1, 0, 1)
        energy = torch.zeros(len(data.natoms), device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / _AVG_NUM_NODES
        
        # Forces
        if self.regress_forces:
            forces = self.force_block(x, atomic_numbers, edge_distance_encoded, edge_index)
            forces = forces.embedding.narrow(1, 1, 3).view(-1, 3)
            return energy, forces
        return energy
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if self.weight_init == 'uniform':
                std = 1 / math.sqrt(m.in_features)
                nn.init.uniform_(m.weight, -std, std)
            else:  # normal
                std = 1 / math.sqrt(m.in_features)
                nn.init.normal_(m.weight, 0, std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)