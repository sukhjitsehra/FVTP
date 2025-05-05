# import torch
# from torch.utils.data import DataLoader, Subset
# import numpy as np
# from dat a import SpatioTemporalNGSIMDataset
# from fusionmodels.tabularfusion.lstm_fusion import LSTMFusion  # Update import based on actual file structure
# print("DEBUG: LSTMFusion MRO =", LSTMFusion.__mro__)

# print("DEBUG: LSTMFusion module =", LSTMFusion.__module__)
# def test_lstm_fusion():
#     # Define test parameters
#     subset_size = 100  
#     batch_size = 4     

#     print("=== Testing LSTMFusion model ===")

#     # Load dataset
#     dataset = SpatioTemporalNGSIMDataset(
#         csv_file='./ngsim_subset.csv',
#         sequence_length=5,
#         target_horizon=1,
#         max_neighbors=5,
#         spatial_radius=30.0,
#         stride=1
#     )

#     # Check if dataset is empty
#     if len(dataset) == 0:
#         print("Dataset is empty, skipping test.")
#         return

#     # Create a subset for testing
#     indices = np.random.choice(len(dataset), subset_size, replace=False)
#     subset_dataset = Subset(dataset, indices)
#     dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

#     # Load a batch
#     batch = next(iter(dataloader))
    
#     # Move to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Initialize model
#     model = LSTMFusion(
#         temporal_dim=6,
#         spatial_dim=6,
#         hidden_dim=64,
#         output_dim=2,
#         sequence_length=5,
#         num_layers=1
#     )

#     model.to(device)

#     # Move batch data to the same device as the model
#     batch = {key: value.to(device) for key, value in batch.items()}

#     # Forward pass
#     output = model(batch)

#     # Print results
#     print("Output shape:", output.shape)  # Expected: [batch_size, 2]
#     print("Output tensor:", output)

# if __name__ == "__main__":
#     test_lstm_fusion()


#-------------------------------------------------------------

#Performance Testing based on the baseline and model evaluation - this is not too complex to capture the essence of the code thats why we need hyperparameters and the model to be defined in the code
# import torch
# from torch.utils.data import DataLoader, Subset
# import numpy as np
# from data import SpatioTemporalNGSIMDataset
# from fusionmodels.tabularfusion.lstm_fusion import LSTMFusion
# from sklearn.metrics import mean_squared_error

# def evaluate_model(model, dataloader, device):
#     model.eval()
#     all_targets = []
#     all_predictions = []

#     with torch.no_grad():
#         for batch in dataloader:
#             batch = {key: value.to(device) for key, value in batch.items()}
#             output = model(batch)
#             all_predictions.append(output.cpu().numpy())
#             all_targets.append(batch['target'].cpu().numpy())

#     all_predictions = np.concatenate(all_predictions, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)
#     mse = mean_squared_error(all_targets, all_predictions)
#     return mse

# def evaluate_baseline(dataloader):
#     all_targets = []
#     all_predictions = []

#     with torch.no_grad():
#         for batch in dataloader:
#             all_targets.append(batch['target'].numpy())
#             # Baseline prediction: predict the mean of the target values
#             mean_prediction = np.mean(batch['target'].numpy(), axis=0)
#             all_predictions.append(np.tile(mean_prediction, (batch['target'].shape[0], 1)))

#     all_predictions = np.concatenate(all_predictions, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)
#     mse = mean_squared_error(all_targets, all_predictions)
#     return mse

# def main():
#     subset_size = 100
#     batch_size = 4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load dataset
#     dataset = SpatioTemporalNGSIMDataset(
#         csv_file='./ngsim_subset.csv',
#         sequence_length=5,
#         target_horizon=1,
#         max_neighbors=5,
#         spatial_radius=30.0,
#         stride=1
#     )

#     # Create a subset for testing
#     indices = np.random.choice(len(dataset), subset_size, replace=False)
#     subset_dataset = Subset(dataset, indices)
#     dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

#     # Initialize model
#     model = LSTMFusion(
#         prediction_task='regression',
#         temporal_dim=6,
#         spatial_dim=6,
#         hidden_dim=64,
#         output_dim=2,
#         sequence_length=5,
#         num_layers=1
#     ).to(device)

#     # Evaluate model
#     mse_model = evaluate_model(model, dataloader, device)
#     print(f"Mean Squared Error (Model): {mse_model}")

#     # Evaluate baseline
#     mse_baseline = evaluate_baseline(dataloader)
#     print(f"Mean Squared Error (Baseline): {mse_baseline}")

# if __name__ == "__main__":
#     main()

# import torch
# from torch.utils.data import DataLoader, Subset
# import numpy as np
# from data import SpatioTemporalNGSIMDataset
# from fusionmodels.tabularfusion.lstm_fusion import LSTMFusion, LSTMLateFusion
# from sklearn.metrics import mean_squared_error
# from itertools import product

# def evaluate_model(model, dataloader, device):
#     model.eval()
#     all_targets = []
#     all_predictions = []

#     with torch.no_grad():
#         for batch in dataloader:
#             batch = {key: value.to(device) for key, value in batch.items()}
#             output = model(batch)
#             all_predictions.append(output.cpu().numpy())
#             all_targets.append(batch['target'].cpu().numpy())

#     all_predictions = np.concatenate(all_predictions, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)
#     mse = mean_squared_error(all_targets, all_predictions)
#     return mse

# def main():
#     subset_size = 100
#     batch_size = 4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load dataset
#     dataset = SpatioTemporalNGSIMDataset(
#         csv_file='./ngsim_subset.csv',
#         sequence_length=5,
#         target_horizon=1,
#         max_neighbors=5,
#         spatial_radius=30.0,
#         stride=1
#     )

#     # Create a subset for testing
#     indices = np.random.choice(len(dataset), subset_size, replace=False)
#     subset_dataset = Subset(dataset, indices)
#     dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

#     # Hyperparameter grid
#     hidden_dims = [32, 64, 128]
#     num_layers_list = [1, 2, 3]
#     dropouts = [0.0, 0.2, 0.5]

#     best_mse_fusion = float('inf')
#     best_params_fusion = None

#     best_mse_late_fusion = float('inf')
#     best_params_late_fusion = None

#     for hidden_dim, num_layers, dropout in product(hidden_dims, num_layers_list, dropouts):
#         # Initialize LSTMFusion model with different hyperparameters
#         model_fusion = LSTMFusion(
#             prediction_task='regression',
#             temporal_dim=6,
#             spatial_dim=6,
#             hidden_dim=hidden_dim,
#             output_dim=2,
#             sequence_length=5,
#             num_layers=num_layers,
#             dropout=dropout
#         ).to(device)

#         # Evaluate LSTMFusion model
#         mse_fusion = evaluate_model(model_fusion, dataloader, device)
#         print(f"LSTMFusion - Hidden Dim: {hidden_dim}, Num Layers: {num_layers}, Dropout: {dropout}, MSE: {mse_fusion}")

#         if mse_fusion < best_mse_fusion:
#             best_mse_fusion = mse_fusion
#             best_params_fusion = (hidden_dim, num_layers, dropout)

#         # Initialize LSTMLateFusion model with different hyperparameters
#         model_late_fusion = LSTMLateFusion(
#             prediction_task='regression',
#             temporal_dim=6,
#             spatial_dim=6,
#             hidden_dim=hidden_dim,
#             output_dim=2,
#             sequence_length=5,
#             max_neighbors=5,
#             num_layers=num_layers,
#             dropout=dropout
#         ).to(device)

#         # Evaluate LSTMLateFusion model
#         mse_late_fusion = evaluate_model(model_late_fusion, dataloader, device)
#         print(f"LSTMLateFusion - Hidden Dim: {hidden_dim}, Num Layers: {num_layers}, Dropout: {dropout}, MSE: {mse_late_fusion}")

#         if mse_late_fusion < best_mse_late_fusion:
#             best_mse_late_fusion = mse_late_fusion
#             best_params_late_fusion = (hidden_dim, num_layers, dropout)

#     print(f"Best MSE (LSTMFusion): {best_mse_fusion} with params: Hidden Dim: {best_params_fusion[0]}, Num Layers: {best_params_fusion[1]}, Dropout: {best_params_fusion[2]}")
#     print(f"Best MSE (LSTMLateFusion): {best_mse_late_fusion} with params: Hidden Dim: {best_params_late_fusion[0]}, Num Layers: {best_params_late_fusion[1]}, Dropout: {best_params_late_fusion[2]}")

# if __name__ == "__main__":
#     main()

    #-------------------------------------------------------------
    # over the whole dataset

import torch
from torch.utils.data import DataLoader
import numpy as np
from data import SpatioTemporalNGSIMDataset
from fusionmodels.tabularfusion.lstm_fusion import LSTMFusion, LSTMLateFusion
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def threshold_accuracy(all_targets, all_predictions, threshold=2.0):
    """
    Computes the fraction of samples whose Euclidean distance
    between prediction and target is below 'threshold'.
    Adjust 'threshold' based on how strict you want the measure to be.
    """
    distances = np.sqrt(np.sum((all_targets - all_predictions)**2, axis=1))
    correct = np.sum(distances < threshold)
    return correct / len(distances)

def evaluate_model(model, dataloader, device, threshold=2.0):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch)
            all_predictions.append(output.cpu().numpy())
            all_targets.append(batch['target'].cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    acc = threshold_accuracy(all_targets, all_predictions, threshold=threshold)

    return mse, mae, r2, acc

def main():
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the spatio-temporal dataset
    dataset = SpatioTemporalNGSIMDataset(
        csv_file='./ngsim_subset.csv',
        sequence_length=5,
        target_horizon=1,
        max_neighbors=5,
        spatial_radius=30.0,
        stride=1
    )

    # Check if the dataset is empty
    if len(dataset) == 0:
        print("The dataset is empty after preprocessing.")
        return

    # Use the entire dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Hyperparameter grid to test
    hidden_dims = [32, 64, 128]
    num_layers_list = [1, 2, 3]
    dropouts = [0.0, 0.2, 0.5]

    best_mse_fusion = float('inf')
    best_params_fusion = None

    best_mse_late = float('inf')
    best_params_late = None

    for hidden_dim, num_layers, dropout in (
        (hd, nl, do) for hd in hidden_dims for nl in num_layers_list for do in dropouts
    ):
        # Early fusion model
        model_fusion = LSTMFusion(
            prediction_task='regression',
            temporal_dim=6,
            spatial_dim=6,
            hidden_dim=hidden_dim,
            output_dim=2,
            sequence_length=5,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        mse_fusion, mae_fusion, r2_fusion, acc_fusion = evaluate_model(
            model_fusion, dataloader, device, threshold=2.0
        )
        print(
            f"LSTMFusion - H: {hidden_dim}, L: {num_layers}, D: {dropout} | "
            f"MSE: {mse_fusion:.4f}, MAE: {mae_fusion:.4f}, R2: {r2_fusion:.4f}, Acc: {acc_fusion:.4f}"
        )

        if mse_fusion < best_mse_fusion:
            best_mse_fusion = mse_fusion
            best_params_fusion = (hidden_dim, num_layers, dropout)

        # Late fusion model
        model_late = LSTMLateFusion(
            prediction_task='regression',
            temporal_dim=6,
            spatial_dim=6,
            hidden_dim=hidden_dim,
            output_dim=2,
            sequence_length=5,
            max_neighbors=5,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        mse_late, mae_late, r2_late, acc_late = evaluate_model(
            model_late, dataloader, device, threshold=2.0
        )
        print(
            f"LSTMLateFusion - H: {hidden_dim}, L: {num_layers}, D: {dropout} | "
            f"MSE: {mse_late:.4f}, MAE: {mae_late:.4f}, R2: {r2_late:.4f}, Acc: {acc_late:.4f}"
        )

        if mse_late < best_mse_late:
            best_mse_late = mse_late
            best_params_late = (hidden_dim, num_layers, dropout)

    print("\nBest LSTMFusion MSE: {:.4f} with params: H={}, L={}, D={}"
          .format(best_mse_fusion, best_params_fusion[0], best_params_fusion[1], best_params_fusion[2]))
    print("Best LSTMLateFusion MSE: {:.4f} with params: H={}, L={}, D={}"
          .format(best_mse_late, best_params_late[0], best_params_late[1], best_params_late[2]))

if __name__ == "__main__":
    main()
