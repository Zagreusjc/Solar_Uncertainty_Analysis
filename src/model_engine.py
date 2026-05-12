import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# --- THE BAYESIAN NEURAL NETWORK ARCHITECTURE ---
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dimensions):
        super().__init__()
        self.layer1 = nn.Linear(input_dimensions, 64)
        # Dropout acts as our Bayesian approximator (Monte Carlo Dropout)
        self.dropout1 = nn.Dropout(0.2) 
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        return self.output_layer(x)

def execute_modeling_pipeline():
    # 1. Load Data
    solar_dataframe = pd.read_csv('data/processed_solar_data.csv')
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'hour_of_day']
    target = 'DC_POWER'

    features_train, features_test, target_train, target_test = train_test_split(
        solar_dataframe[features], solar_dataframe[target], test_size=0.2, random_state=42
    )

    # 2. Scale and Convert to PyTorch Tensors
    data_scaler = StandardScaler()
    x_train_scaled = data_scaler.fit_transform(features_train)
    x_test_scaled = data_scaler.transform(features_test)

    x_train_tensor = torch.FloatTensor(x_train_scaled)
    y_train_tensor = torch.FloatTensor(target_train.values).view(-1, 1)
    x_test_tensor = torch.FloatTensor(x_test_scaled)

    # 3. Train the Bayesian Neural Network
    print("Training Bayesian Neural Network (PyTorch)...")
    bnn_model = BayesianNeuralNetwork(input_dimensions=x_train_scaled.shape[1])
    optimizer = optim.Adam(bnn_model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    bnn_model.train() # Keep dropout active for training
    for epoch in range(500):
        optimizer.zero_grad()
        predictions = bnn_model(x_train_tensor)
        loss = loss_function(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

    # 4. Bayesian Inference (Monte Carlo Sampling)
    print("Performing Bayesian Inference (Monte Carlo Sampling)...")
    # We leave the model in .train() mode to keep Dropout active, simulating uncertainty
    mc_samples = 100 
    bnn_predictions = []
    
    with torch.no_grad():
        for _ in range(mc_samples):
            bnn_predictions.append(bnn_model(x_test_tensor).numpy())
            
    bnn_predictions = np.array(bnn_predictions).squeeze()
    
    # Calculate Mean and 90% Confidence Interval for the BNN
    bnn_mean_preds = bnn_predictions.mean(axis=0)
    bnn_lower_bound = np.percentile(bnn_predictions, 5, axis=0)
    bnn_upper_bound = np.percentile(bnn_predictions, 95, axis=0)

    # 5. Train Gradient Boosted Quantile Regressor (GBQR)
    print("Training Gradient Boosted Quantile Regressor...")
    gbqr_upper = GradientBoostingRegressor(loss='quantile', alpha=0.95, random_state=42).fit(features_train, target_train)
    gbqr_lower = GradientBoostingRegressor(loss='quantile', alpha=0.05, random_state=42).fit(features_train, target_train)
    
    gbqr_upper_bounds = gbqr_upper.predict(features_test)
    gbqr_lower_bounds = gbqr_lower.predict(features_test)

    # 6. Analytics Output
    print("\n" + "="*50)
    print(" METRICS FOR MANUSCRIPT TABLES")
    print("="*50)
    
    bnn_mae = mean_absolute_error(target_test, bnn_mean_preds)
    bnn_rmse = root_mean_squared_error(target_test, bnn_mean_preds)
    bnn_r2 = r2_score(target_test, bnn_mean_preds)
    print(f"Bayesian Neural Net -> MAE: {bnn_mae:.2f} | RMSE: {bnn_rmse:.2f} | R-Squared: {bnn_r2:.4f}")

    bnn_picp = np.mean(np.logical_and(target_test >= bnn_lower_bound, target_test <= bnn_upper_bound)) * 100
    gbqr_picp = np.mean(np.logical_and(target_test >= gbqr_lower_bounds, target_test <= gbqr_upper_bounds)) * 100
    
    print(f"\nBNN Coverage Probability (PICP): {bnn_picp:.2f}%")
    print(f"GBQR Coverage Probability (PICP): {gbqr_picp:.2f}%")
    print("="*50 + "\n")

    # 7. The Hero Visual
    plt.figure(figsize=(12, 6))
    plt.plot(target_test.values[:100], label='Actual DC Power', color='black', linewidth=1.5)
    plt.plot(bnn_mean_preds[:100], label='BNN Mean Prediction', color='red', linestyle='--', alpha=0.8)
    
    # Plotting the Bayesian Uncertainty Ribbon
    plt.fill_between(
        range(100), bnn_lower_bound[:100], bnn_upper_bound[:100], 
        color='blue', alpha=0.2, label='BNN Epistemic Uncertainty (90%)'
    )
    plt.title("Bayesian Neural Network: Predictive Mean and Epistemic Uncertainty")
    plt.xlabel("Test Set Samples (Time Sequential)")
    plt.ylabel("DC Power Output (kW)")
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    execute_modeling_pipeline()