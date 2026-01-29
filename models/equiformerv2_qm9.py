"""
EquiformerV2 adapted for QM9 Dataset

CHANGES FROM ORIGINAL (equiformer_v2_oc20.py):
====================================================
- Changed: num_targets (1 for OC20 → 12 for QM9)
    Reason: QM9 has 12 properties to predict

- use_pbc: True → False
    Reason: QM9 molecules are isolated (no periodic boundaries)
   
- regress_forces: True → False
    Reason: QM9 doesn't have force labels, only molecular properties
   
- max_neighbors: 500 → 50
    Reason: QM9 molecules ~18 atoms, fewer neighbors than OC20 crystals
   
- max_num_elements: 90 → 10
     Reason: QM9 only has H, C, N, O, F (atomic numbers 1, 6, 7, 8, 9)
   
- num_layers: 12 → 8
     Reason: QM9 is simpler, fewer layers needed (faster, less overfitting)
   
- lmax_list: [6] → [4]
     Reason: Small molecules need less angular resolution
   
- _AVG_NUM_NODES: 77.81 → 18.0 (approximate)
     Reason: QM9 molecules average ~18 atoms vs OC20 ~78 atoms
   
   - _AVG_DEGREE: 23.4 → 6.0 (approximate, will compute from data)
     Reason: QM9 molecules have fewer neighbors per atom

4. OUTPUT HEADS (major change):
   - Removed: energy_block (single output)
   - Removed: force_block (SO2 attention for forces)
   - Added: output_blocks (ModuleList with 12 FFN heads)
   Reason: QM9 needs 12 separate property predictions

5. FORWARD METHOD:
   - Removed: Force computation logic
   - Changed: Single energy → 12 property predictions
   - Removed: Energy normalization by _AVG_NUM_NODES
   Reason: QM9 targets are molecular properties, not per-atom energies

All other components (SO3, attention, FFN, etc.) remain IDENTICAL.
This shows the generality of the EquiformerV2 architecture!
"""

import math
import torch
import torch.nn as nn

# Updated import paths for fairchem (you mentioned using fairchem.core)
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BackboneInterface as BaseModel

# These imports stay the same (internal modules)
from EquiformerV2Functions.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from EquiformerV2Functions.module_list import ModuleListInfo
from EquiformerV2Functions.so2_ops import SO2_Convolution
from EquiformerV2Functions.radial_function import RadialFunction
from EquiformerV2Functions.layer_norm import get_normalization_layer
from EquiformerV2Functions.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2, 
)
from EquiformerV2Functions.input_block import EdgeDegreeEmbedding
from EquiformerV2Functions.edge_rot_mat import init_edge_rot_mat

# For distance expansion
from fairchem.core.models.escaip.utils.smearing import GaussianSmearing

try:
    from e3nn import o3
except ImportError:
    pass


# CHANGE 1: Updated statistics for QM9
# Original (OC20): _AVG_NUM_NODES = 77.81, _AVG_DEGREE = 23.39
# QM9: Smaller molecules, fewer neighbors
_AVG_NUM_NODES_QM9 = 18.0  # Approximate, QM9 molecules have ~18 atoms on average
_AVG_DEGREE_QM9 = 6.0      # Approximate, will be computed from actual data


# CHANGE 2: Class name for clarity
# Original: EquiformerV2_OC20
# QM9: EquiformerV2_QM9
@registry.register_model("equiformer_v2_qm9")
class EquiformerV2_QM9(BaseModel, nn.Module): # we let also it inherit from nn.Module
    """
    EquiformerV2 adapted for QM9 molecular property prediction
    
    Key differences from OC20 version:
    - No periodic boundary conditions (isolated molecules)
    - No force regression (only molecular properties)
    - 12 output heads (one per QM9 property)
    - Smaller default architecture (fewer layers, lower lmax)
    - Adjusted for molecular scale (~18 atoms vs ~78 in OC20)
    
    Args:
        num_targets (int):    Number of prediction targets (12 for QM9)
        use_pbc (bool):       Use periodic boundary conditions (False for QM9)
        regress_forces (bool): Compute forces (False for QM9)
        
        [All other args same as EquiformerV2_OC20 - see original docstring]
    """
    
    def __init__(
        self,
        # CHANGE 3: num_targets now defaults to 12 (was 1 for OC20 energy)
        # Reason: QM9 has 12 properties: μ, α, ε_HOMO, ε_LUMO, Δε, ⟨R²⟩, ZPVE, U₀, U, H, G, c_v
        num_targets=12,
        
        # CHANGE 4: use_pbc defaults to False (was True for OC20)
        # Reason: QM9 molecules are isolated, not periodic crystals
        use_pbc=False,
        
        # CHANGE 5: regress_forces defaults to False (was True for OC20)
        # Reason: QM9 doesn't provide force labels
        regress_forces=False,
        
        otf_graph=True,  # Same: compute graph on-the-fly
        
        # CHANGE 6: max_neighbors reduced (500 → 50)
        # Reason: QM9 molecules much smaller (~18 atoms), need fewer neighbors
        max_neighbors=50,
        
        max_radius=5.0,  # Same: 5 Ångström cutoff works for both
        
        # CHANGE 7: max_num_elements reduced (90 → 10)
        # Reason: QM9 only has H(1), C(6), N(7), O(8), F(9)
        max_num_elements=10,
        
        # CHANGE 8: num_layers reduced (12 → 8)
        # Reason: Simpler task, avoid overfitting on smaller dataset
        num_layers=8,
        
        sphere_channels=128,      # Same: hidden dimension
        attn_hidden_channels=128, # Same
        num_heads=8,              # Same
        attn_alpha_channels=32,   # Same
        attn_value_channels=16,   # Same
        ffn_hidden_channels=512,  # Same
        
        norm_type='rms_norm_sh',  # Same: RMSNorm variant
        
        # CHANGE 9: lmax_list reduced ([6] → [4])
        # Reason: Small molecules need less angular resolution
        # lmax=4 gives up to g-orbitals, sufficient for organic molecules
        lmax_list=[4],
        
        mmax_list=[2],            # Same: order truncation for speed
        grid_resolution=None,     # Same: auto-compute grid resolution
        
        num_sphere_samples=128,   # Same
        
        edge_channels=128,        # Same
        use_atom_edge_embedding=True,    # Same
        share_atom_edge_embedding=False, # Same
        use_m_share_rad=False,    # Same
        distance_function="gaussian",    # Same
        num_distance_basis=512,   # Same
        
        attn_activation='scaled_silu',   # Same
        use_s2_act_attn=False,   # Same
        use_attn_renorm=True,    # Same
        ffn_activation='scaled_silu',    # Same
        use_gate_act=False,      # Same
        use_grid_mlp=False,      # Same
        use_sep_s2_act=True,     # Same: separable S2 activation (default)
        
        alpha_drop=0.1,          # Same
        drop_path_rate=0.05,     # Same (could reduce for QM9 if overfitting)
        proj_drop=0.0,           # Same
        
        weight_init='normal'     # Same
    ):
        super().__init__()
        
        
        # Store all config (same as original)
        self.num_targets = num_targets
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements
        
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution
        
        self.num_sphere_samples = num_sphere_samples
        
        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis
        
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
        assert self.weight_init in ['normal', 'uniform']
        
        self.device = 'cpu'
        
        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # ================================================================
        # IDENTICAL TO ORIGINAL: Core architecture components
        # (sphere_embedding, distance_expansion, SO3_rotation, etc.)
        # ================================================================
        
        # Atomic number embedding
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, 
            self.sphere_channels_all
        )
        
        # Distance basis functions
        assert self.distance_function in ['gaussian']
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,  # Number of Gaussians
                2.0,  # Width
            )
        else:
            raise ValueError
        
        # Edge channel sizes
        self.edge_channels_list = [
            int(self.distance_expansion.num_output)
        ] + [self.edge_channels] * 2
        
        # Atom edge embedding (if sharing)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, 
                self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, 
                self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # SO3 rotation modules
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))
        
        # Coefficient mapping
        self.mappingReduced = CoefficientMappingModule(
            self.lmax_list, 
            self.mmax_list
        )
        
        # SO3 grid for S2 activation
        self.SO3_grid = ModuleListInfo(
            '({}, {})'.format(max(self.lmax_list), max(self.lmax_list))
        )
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)
        
        # Edge-degree embedding
        # CHANGE 10: Use QM9 average degree for rescaling
        # Original: rescale_factor=_AVG_DEGREE (23.4 for OC20)
        # QM9: rescale_factor=_AVG_DEGREE_QM9 (6.0, approximate)
        # Reason: QM9 molecules have fewer neighbors per atom
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE_QM9  # CHANGED from _AVG_DEGREE
        )
        
        # Transformer blocks (IDENTICAL to original)
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)
        
        # ================================================================
        # MAJOR CHANGE: Output heads for QM9
        # ================================================================
        
        # Final normalization (same as original)
        self.norm = get_normalization_layer(
            self.norm_type, 
            lmax=max(self.lmax_list), 
            num_channels=self.sphere_channels
        )
        
        # CHANGE 11: Replace single energy_block with 12 property blocks
        # Original had:
        #   self.energy_block = FeedForwardNetwork(...)  # 1 output
        #   self.force_block = SO2EquivariantGraphAttention(...)  # 3 outputs
        #
        # QM9 needs:
        #   self.output_blocks = ModuleList([FFN, FFN, ...])  # 12 outputs
        #
        # Reason: QM9 predicts 12 different molecular properties
        # Each property gets its own prediction head (no parameter sharing)
        # This allows each head to learn property-specific transformations
        
        self.output_blocks = nn.ModuleList()
        for target_idx in range(self.num_targets):
            # Each target gets its own FeedForwardNetwork
            # Input: sphere_channels (128)
            # Hidden: ffn_hidden_channels (512)  
            # Output: 1 (scalar prediction)
            output_block = FeedForwardNetwork(
                self.sphere_channels,
                self.ffn_hidden_channels, 
                1,  # Single scalar output per property
                self.lmax_list,
                self.mmax_list,
                self.SO3_grid,  
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act
            )
            self.output_blocks.append(output_block)
        
        # REMOVED: force_block (not needed for QM9)
        # Original had:
        #   if self.regress_forces:
        #       self.force_block = SO2EquivariantGraphAttention(...)
        # Reason: QM9 doesn't have force labels
        
        # Weight initialization (same as original)
        
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)
    
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        """
        Forward pass for QM9 molecules
        
        CHANGES FROM ORIGINAL:
        - Removed force computation
        - Changed from single energy output to 12 property outputs
        - Removed energy normalization by _AVG_NUM_NODES
        
        Args:
            data: Batch with attributes:
                - atomic_numbers: [num_atoms] atomic numbers
                - pos: [num_atoms, 3] positions
                - batch: [num_atoms] batch assignment
                - natoms: [batch_size] number of atoms per graph
                - (optionally) edge_index if pre-computed
        
        Returns:
            predictions: [batch_size, 12] predicted molecular properties
        """
        
        self.batch_size = len(data['natoms'])
        self.dtype = data['pos'].dtype
        self.device = data['pos'].device
        
        atomic_numbers = data['atomic_numbers'].long()
        num_atoms = len(atomic_numbers)
        pos = data['pos']
        
        # ================================================================
        # IDENTICAL TO ORIGINAL: Graph construction and initialization
        # ================================================================
        
        # Generate graph (edges within cutoff)
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,  # Will be None for QM9 (no PBC)
            _,
            neighbors,
        ) = self.generate_graph(data)
        
        # Compute edge rotation matrices
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )
        
        # Set Wigner-D matrices
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)
        
        # Initialize node embeddings
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )
        
        # Set l=0, m=0 coefficients from atomic embeddings
        offset_res = 0
        offset = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)
        
        # Edge features (distance + optional atomic embeddings)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]
            target_element = atomic_numbers[edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), 
                dim=1
            )
        
        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index
        )
        x.embedding = x.embedding + edge_degree.embedding
        
        # ================================================================
        # IDENTICAL TO ORIGINAL: Transformer blocks
        # ================================================================
        
        for i in range(self.num_layers):
            x = self.blocks[i](
                x,
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data['batch']
            )
        
        # Final normalization
        x.embedding = self.norm(x.embedding)
        
        # ================================================================
        # MAJOR CHANGE: Output prediction for QM9
        # ================================================================
        
        # CHANGE 12: Predict 12 molecular properties instead of energy/forces
        # Original code:
        #   node_energy = self.energy_block(x)
        #   energy = sum over nodes, divide by _AVG_NUM_NODES
        #   if regress_forces:
        #       forces = self.force_block(x, ...)
        #   return energy, forces
        #
        # QM9 code:
        #   For each property:
        #       node_contributions = output_block[i](x)
        #       property[i] = sum over nodes (graph-level property)
        #   return all 12 properties
        #
        # Reason: 
        # - QM9 properties are molecular (graph-level), not atomic
        # - No force prediction needed
        # - Each property gets its own prediction head
        # - Sum over atoms (each atom contributes to molecular property)
        # - NO normalization by _AVG_NUM_NODES (properties have different scales)
        
        predictions = []
        
        for target_idx in range(self.num_targets):
            # Pass through property-specific FFN
            # Input: x (SO3_Embedding with all geometric info)
            # Output: node_contribution (what each atom contributes to this property)
            node_contribution = self.output_blocks[target_idx](x)
            
            # Extract scalar (l=0, m=0) component
            # Shape: [num_atoms, 1, 1] → [num_atoms]
            node_contribution = node_contribution.embedding.narrow(1, 0, 1)
            node_contribution = node_contribution.view(-1)
            
            # Sum contributions over atoms in each molecule
            # This gives graph-level prediction
            property_pred = torch.zeros(
                len(data['natoms']), 
                device=self.device, 
                dtype=self.dtype
            )
            property_pred.index_add_(0, data['batch'], node_contribution)
            
            # IMPORTANT: NO division by _AVG_NUM_NODES
            # Reason: QM9 properties have their own natural scales
            # - Some scale with molecule size (e.g., polarizability)
            # - Some don't (e.g., HOMO-LUMO gap)
            # - Let the model learn the right scaling
            
            predictions.append(property_pred)
        
        # Stack all predictions
        # Shape: [batch_size, 12]
        predictions = torch.stack(predictions, dim=1)
        
        return predictions
    
    
    # IDENTICAL TO ORIGINAL: Helper methods
    
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
    
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    
    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2)):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)
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
        """Same as original - parameters that shouldn't have weight decay"""
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
            ):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)


# ================================================================
# SUMMARY OF ALL CHANGES
# ================================================================
"""
ARCHITECTURAL CHANGES (what's different):
1. Output heads: 1 energy + 1 force → 12 property predictions
2. No force regression block

HYPERPARAMETER CHANGES (adapted for QM9):
1. use_pbc: True → False (isolated molecules)
2. regress_forces: True → False (no force labels)
3. max_neighbors: 500 → 50 (smaller molecules)
4. max_num_elements: 90 → 10 (only H, C, N, O, F)
5. num_layers: 12 → 8 (simpler task)
6. lmax_list: [6] → [4] (less angular resolution needed)
7. rescale_factor: 23.4 → 6.0 (fewer neighbors)

FORWARD PASS CHANGES:
1. Output: energy/forces → 12 properties
2. No normalization by _AVG_NUM_NODES (properties have different scales)
3. Each property: sum over node contributions

EVERYTHING ELSE IS IDENTICAL:
- All core components (SO3, SO2, attention, FFN)
- All transformer blocks
- All initialization
- All helper methods

This shows the generality of EquiformerV2!
Only output and dataset-specific parameters change.
"""