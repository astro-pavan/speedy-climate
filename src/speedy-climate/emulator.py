import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def make_GP_emulator(data_path):

    data = pd.read_csv(data_path)

    # print(data.columns)

    input_features = ['Instellation (W/m^2)', 'P_CO2 (Pa)', 'P_H2O (Pa)', 'Albedo']
    output_targets = ['Surface_Temp (K)']

    X = data[input_features].values
    y = data[output_targets].values

    # log scale P_CO2 and P_H2O
    X[:, 1] = np.log10(X[:, 1])
    X[:, 2] = np.log10(X[:, 2])


    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # FIT *ONLY* ON THE TRAINING DATA
    x_scaler.fit(X_train)
    y_scaler.fit(y_train)

    # TRANSFORM all three datasets
    X_train_scaled = x_scaler.transform(X_train)
    X_val_scaled   = x_scaler.transform(X_val)
    X_test_scaled  = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.transform(y_train)
    y_val_scaled   = y_scaler.transform(y_val)
    y_test_scaled  = y_scaler.transform(y_test)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e5)) * RBF(length_scale=[1.0, 1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2)) 
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )

    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1e-10)

    gaussian_process.fit(X_train_scaled, y_train_scaled)

    print(f"Learned kernel: {gaussian_process.kernel_}")
    print(f"Log-marginal-likelihood: {gaussian_process.log_marginal_likelihood(gaussian_process.kernel_.theta):.3f}")

    # 1. Score the model on the Validation set
    print(f"Training R2: {gaussian_process.score(X_train_scaled, y_train_scaled):.4f}")
    print(f"Validation R2: {gaussian_process.score(X_val_scaled, y_val_scaled):.4f}")

    # 2. Visual Check (Optional but recommended)

    # Predict on validation data
    y_val_pred, y_val_std = gaussian_process.predict(X_val_scaled, return_std=True)

    # Plot Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_scaler.inverse_transform(y_val_pred.reshape(-1, 1)), alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2) # Perfect fit line
    plt.xlabel("Actual Surface Temp (K)")
    plt.ylabel("Predicted Surface Temp (K)")
    plt.title("Model Accuracy Check")
    plt.savefig("accuracy_check.png")
    print("Saved accuracy_check.png")

if __name__ == '__main__':

    make_GP_emulator('climate_data/helios_runs_earth_tidally_locked.csv')