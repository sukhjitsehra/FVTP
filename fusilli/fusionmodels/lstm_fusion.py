# File: fusilli/fusionmodels/tabularfusion/lstm_fusion.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fusilli.fusionmodels.base_model import ParentFusionModel

# class LSTMFusion(ParentFusionModel):
#     """
#     LSTM-based fusion model that processes temporal (sequence) data
#     and fuses it with static (vehicle-specific) features.
    
#     Expected Input in the forward method:
#       batch['temporal'] => [B, T, temporal_dim]
#       batch['static']   => [B, static_dim]
#       batch['target']   => [B, 2] (used outside this model for loss calculation)
#     """

#     def __init__(
#         self,
#         prediction_task: str = 'regression',
#         temporal_dim: int = 5,
#         static_dim: int = 3,
#         hidden_dim: int = 64,
#         output_dim: int = 2,    # predicting [Local_X, Local_Y]
#         sequence_length: int = 10,
#         num_layers: int = 1,
#         dropout: float = 0.0,
#         multiclass_dimensions: int = None,
#         **kwargs
#     ):
#         """
#         Args:
#             prediction_task (str): 'regression' or 'binary' or 'multiclass'
#             temporal_dim (int): Number of features in the time-series input.
#             static_dim (int): Number of features in the static input.
#             hidden_dim (int): LSTM hidden size.
#             output_dim (int): Size of the final output (2 if predicting [X, Y]).
#             sequence_length (int): Number of timesteps in each sequence.
#             num_layers (int): Number of LSTM layers.
#             dropout (float): Dropout probability (only applied if num_layers > 1).
#             multiclass_dimensions (int): Number of classes if doing classification.
#         """
#         # Initialize the ParentFusionModel to handle prediction task logic
#         super().__init__(
#             prediction_task=prediction_task,
#             mod1_dim=temporal_dim,  # Using the "mod1" slot for temporal data
#             mod2_dim=None,          # Not using a second tabular modality
#             img_dim=None,           # Not using image data
#             multiclass_dimensions=multiclass_dimensions
#         )

#         self.sequence_length = sequence_length
#         self.temporal_dim = temporal_dim
#         self.static_dim = static_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         # Define the LSTM for time-series data
#         self.lstm = nn.LSTM(
#             input_size=self.temporal_dim,
#             hidden_size=self.hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0
#         )

#         # Fuse LSTM output with static features
#         self.fusion_fc = nn.Linear(self.hidden_dim + self.static_dim, self.hidden_dim)

#         # Final output layer to predict 2D position (Local_X, Local_Y)
#         self.output_fc = nn.Linear(self.hidden_dim, self.output_dim)

#     def forward(self, batch, *args, **kwargs):
#         """
#         Forward pass for the LSTM fusion model.

#         Args:
#             batch (dict): 
#                 {
#                   'temporal': [B, T, temporal_dim],
#                   'static':   [B, static_dim],
#                   'target':   [B, 2] (used outside this model for loss)
#                 }

#         Returns:
#             torch.Tensor: Predicted [Local_X, Local_Y] of shape [B, 2].
#         """
#         temporal = batch['temporal']  # [B, T, temporal_dim]
#         static = batch['static']      # [B, static_dim]

#         # Pass temporal data through the LSTM
#         # hn shape => [num_layers, B, hidden_dim]
#         _, (hn, _) = self.lstm(temporal)
#         # Take the last layer's hidden state => [B, hidden_dim]
#         hn = hn[-1]

#         # Concatenate LSTM hidden state with static features
#         fused = torch.cat([hn, static], dim=-1)  # [B, hidden_dim + static_dim]
#         fused = self.fusion_fc(fused)           # [B, hidden_dim]
#         fused = F.relu(fused)

#         # Final linear layer to predict 2D position
#         out = self.output_fc(fused)             # [B, 2]
#         return out

#_____________________________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F
from fusilli.fusionmodels.base_model import ParentFusionModel

class LSTMFusion(nn.Module, ParentFusionModel):
    """
    LSTM-based model that processes:
      - temporal data of the target vehicle
      - spatio-temporal neighbor context (aggregated)
    and fuses their representations for final prediction.
    
    Expected batch keys:
      batch['temporal'] => [B, T, temporal_dim]
      batch['spatial']  => [B, T, max_neighbors, spatial_dim]
      batch['target']   => [B, 2] (used outside this model for loss)
    """

    def __init__(
        self,
        prediction_task: str = 'regression',
        temporal_dim: int = 6,     # e.g., [Global_Time, Local_X, Local_Y, v_Vel, v_Acc, Lane_ID]
        spatial_dim: int = 6,      # e.g., [Δx, Δy, Δv, Δacc, lane_diff], or however you define neighbors
        hidden_dim: int = 64,
        output_dim: int = 2,       # predicting [Local_X, Local_Y]
        sequence_length: int = 10,
        num_layers: int = 1,
        dropout: float = 0.0,
        multiclass_dimensions: int = None,
        **kwargs
    ):
        """
        Args:
            prediction_task (str): 'regression', 'binary', or 'multiclass'
            temporal_dim (int): Number of features in the target vehicle time-series.
            spatial_dim (int): Number of features for each neighbor.
            hidden_dim (int): LSTM hidden size.
            output_dim (int): Final output dimension (2 for [Local_X, Local_Y]).
            sequence_length (int): Timesteps in each sequence window.
            num_layers (int): LSTM layers.
            dropout (float): Dropout for LSTM if num_layers > 1.
            multiclass_dimensions (int): Classes if doing classification.
        """
        nn.Module.__init__(self)

        data_dims = [temporal_dim, None, None]
        
        ParentFusionModel.__init__(
            self,
            prediction_task=prediction_task,
            data_dims=data_dims,
            multiclass_dimensions=multiclass_dimensions
        )

        self.sequence_length = sequence_length
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LSTM for target vehicle's temporal data
        self.lstm_temporal = nn.LSTM(
            input_size=self.temporal_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # LSTM for aggregated spatial neighbor data
        self.lstm_spatial = nn.LSTM(
            input_size=self.spatial_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fuse the two hidden states
        self.fusion_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Final output layer
        self.output_fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batch, *args, **kwargs):
        """
        Args:
            batch (dict): 
                {
                  'temporal': [B, T, temporal_dim],
                  'spatial':  [B, T, max_neighbors, spatial_dim],
                  'target':   [B, 2]
                }

        Returns:
            torch.Tensor: Predicted [Local_X, Local_Y] of shape [B, 2].
        """
        temporal = batch['temporal']   # [B, T, temporal_dim]
        spatial = batch['spatial']     # [B, T, max_neighbors, spatial_dim]

        # 1) Aggregate neighbor data across the max_neighbors dimension
        #    e.g., simple mean to get shape [B, T, spatial_dim]
        spatial_avg = spatial.mean(dim=2)  # [B, T, spatial_dim]

        # 2) Pass the target vehicle's temporal data through LSTM
        _, (hn_temp, _) = self.lstm_temporal(temporal)  # hn_temp shape: [num_layers, B, hidden_dim]
        hn_temp = hn_temp[-1]  # [B, hidden_dim]

        # 3) Pass the aggregated neighbor data through another LSTM
        _, (hn_spat, _) = self.lstm_spatial(spatial_avg) # hn_spat shape: [num_layers, B, hidden_dim]
        hn_spat = hn_spat[-1]  # [B, hidden_dim]

        # 4) Fuse the two representations
        fused = torch.cat([hn_temp, hn_spat], dim=-1)   # [B, hidden_dim * 2]
        fused = self.fusion_fc(fused)                   # [B, hidden_dim]
        fused = F.relu(fused)

        # 5) Final linear layer => [B, 2] for position prediction
        out = self.output_fc(fused)
        return out




########################################################################################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMLateFusion(nn.Module, ParentFusionModel):
    """
    LSTM-based late fusion model:
      - Processes target vehicle's temporal data in one LSTM
      - Processes neighbor data in a separate LSTM
      - Fuses their final hidden states at the last stage.

    Expected batch keys:
      batch['temporal']: [B, T, temporal_dim]
      batch['spatial']:  [B, T, max_neighbors, spatial_dim]
      batch['target']:   [B, 2] (not used directly in forward, but for loss)
    """

    def __init__(
        self,
        prediction_task: str = 'regression',
        temporal_dim: int = 6,      # e.g., [Global_Time, Local_X, Local_Y, v_Vel, v_Acc, Lane_ID]
        spatial_dim: int = 6,       # e.g., [Δx, Δy, Δv, Δacc, lane_diff], etc.
        hidden_dim: int = 64,
        output_dim: int = 2,        # final output: [Local_X, Local_Y]
        sequence_length: int = 10,
        max_neighbors: int = 5,     # used for shape references
        num_layers: int = 1,
        dropout: float = 0.0,
        multiclass_dimensions: int = None,
        **kwargs
    ):
        nn.Module.__init__(self)
        
        # We pass some dims to ParentFusionModel; it doesn't do PyTorch stuff
        data_dims = [temporal_dim, None, None]
        ParentFusionModel.__init__(
            self,
            prediction_task=prediction_task,
            data_dims=data_dims,
            multiclass_dimensions=multiclass_dimensions
        )

        self.sequence_length = sequence_length
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_neighbors = max_neighbors

        # LSTM for target vehicle's time-series
        self.lstm_temporal = nn.LSTM(
            input_size=self.temporal_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # LSTM for neighbor data => flatten (T, max_neighbors) => T * max_neighbors
        self.lstm_spatial = nn.LSTM(
            input_size=self.spatial_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Late fusion: combine final hidden states from each LSTM
        self.fusion_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Final output
        self.output_fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batch, *args, **kwargs):
        """
        batch = {
          'temporal': [B, T, temporal_dim],
          'spatial':  [B, T, max_neighbors, spatial_dim],
          'target':   [B, 2]
        }
        Returns: [B, 2] => next position
        """

        # 1) LSTM for the target vehicle
        temporal = batch['temporal']  # shape: [B, T, temporal_dim]
        _, (hn_temp, _) = self.lstm_temporal(temporal)  # hn_temp: [num_layers, B, hidden_dim]
        hn_temp = hn_temp[-1]  # [B, hidden_dim]

        # 2) LSTM for neighbor data
        # Flatten dimension T * max_neighbors => treat it as a longer time sequence
        spatial = batch['spatial']  # [B, T, max_neighbors, spatial_dim]
        B, T, N, Sdim = spatial.shape  # e.g., B, 10, 5, 6

        # Flatten => [B, T * N, spatial_dim]
        # This means we treat each time step & neighbor as a single "time" axis
        # for the second LSTM
        spatial_flat = spatial.view(B, T*N, Sdim)  # [B, T*N, 6]

        _, (hn_spat, _) = self.lstm_spatial(spatial_flat)  # [num_layers, B, hidden_dim]
        hn_spat = hn_spat[-1]  # [B, hidden_dim]

        # 3) Late Fusion
        fused = torch.cat([hn_temp, hn_spat], dim=-1)  # [B, hidden_dim * 2]
        fused = self.fusion_fc(fused)
        fused = F.relu(fused)

        # 4) Final output => [B, 2]
        out = self.output_fc(fused)
        return out



##########################################################################################################################################################################################


