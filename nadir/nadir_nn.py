"""
Medium Neural Network for Nadir Vector Prediction
~10-15k parameters for better accuracy while still being efficient

Architecture: Medium feedforward network
Input: 16 sensors + albedo + sun_nadir_angle + sun_vector (3D) = 21 features
Output: 3D nadir vector (Nadir_X, Nadir_Y, Nadir_Z)
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

# device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


def compute_angular_error(predicted, actual):
    """Compute angular error between predicted and actual vectors"""
    pred_norm = predicted / (torch.norm(predicted, dim=1, keepdim=True) + 1e-8)
    actual_norm = actual / (torch.norm(actual, dim=1, keepdim=True) + 1e-8)
    dot_product = torch.sum(pred_norm * actual_norm, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle_rad = torch.acos(dot_product)
    angle_deg = angle_rad * 180.0 / np.pi
    return angle_deg


def cosine_loss(pred, actual):
    """Cosine loss for vector prediction"""
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    actual_norm = actual / (torch.norm(actual, dim=1, keepdim=True) + 1e-8)
    cos_sim = torch.sum(pred_norm * actual_norm, dim=1)
    loss = 1 - cos_sim
    return loss.mean()


class MediumNadirNet(nn.Module):
    """
    Medium-sized neural network for nadir vector prediction.
    
    Architecture:
    - 21 -> 96 -> 64 -> 32 -> 3
    - Total params: ~10k
    """
    def __init__(self):
        super(MediumNadirNet, self).__init__()

        self.fc1 = nn.Linear(21, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)

        # normalize to unit vector
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x


def run_train(train, model, optimizer):
    """Training loop"""
    model.train()
    loss_function = cosine_loss

    total_loss = 0.0
    total_samples = 0

    for sensor_batch, nadir_vector_batch in train:
        sensor_batch = sensor_batch.to(device)
        nadir_vector_batch = nadir_vector_batch.to(device)

        optimizer.zero_grad()
        pred = model(sensor_batch)
        loss = loss_function(pred, nadir_vector_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * nadir_vector_batch.size(0)
        total_samples += nadir_vector_batch.size(0)

    return total_loss / total_samples


def run_test(test, model):
    """Evaluation loop"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_nadir_errors = []

    with torch.no_grad():
        for sensor_batch, nadir_vector_batch in test:
            sensor_batch = sensor_batch.to(device)
            nadir_vector_batch = nadir_vector_batch.to(device)

            pred = model(sensor_batch)
            loss = cosine_loss(pred, nadir_vector_batch)
            total_loss += loss.item() * nadir_vector_batch.size(0)
            total_samples += nadir_vector_batch.size(0)

            nadir_errors = compute_angular_error(pred, nadir_vector_batch)
            all_nadir_errors.append(nadir_errors.cpu().numpy())

    all_nadir_errors = np.concatenate(all_nadir_errors)
    return total_loss / total_samples, all_nadir_errors


def add_sun_vector_noise(sun_vectors, noise_angle_deg=2.0):
    """Add Gaussian noise to sun vectors with target angular error."""
    noise_angle_rad = noise_angle_deg * np.pi / 180.0
    noisy_vectors = sun_vectors.copy()

    for i in range(len(sun_vectors)):
        vec = sun_vectors[i]
        random_vec = np.random.randn(3)
        perpendicular = random_vec - np.dot(random_vec, vec) * vec
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)
        angle_error = np.random.normal(0, noise_angle_rad)
        noisy_vec = vec * np.cos(angle_error) + perpendicular * np.sin(angle_error)
        noisy_vectors[i] = noisy_vec / (np.linalg.norm(noisy_vec) + 1e-8)

    return noisy_vectors


def load_data():
    """Load and normalize data"""
    train_df = pd.read_csv('data/train.csv')
    sensor_cols = [f'Sensor_{i}' for i in range(1, 17)]

    # normalize sensors
    train_sensors = train_df[sensor_cols].values
    sensor_mean = train_sensors.mean(axis=0, keepdims=True)
    sensor_std = train_sensors.std(axis=0, keepdims=True)
    train_sensors = (train_sensors - sensor_mean) / (sensor_std + 1e-8)

    # normalize albedo
    train_albedo = train_df['Albedo'].values.reshape(-1, 1)
    albedo_mean = train_albedo.mean()
    albedo_std = train_albedo.std()
    train_albedo = (train_albedo - albedo_mean) / (albedo_std + 1e-8)

    # normalize angle
    train_angle = train_df['Sun_Nadir_Angle'].values.reshape(-1, 1)
    angle_mean = train_angle.mean()
    angle_std = train_angle.std()
    train_angle = (train_angle - angle_mean) / (angle_std + 1e-8)

    # Add noisy sun vector
    train_sun_vectors = train_df[['Sun_X', 'Sun_Y', 'Sun_Z']].values
    train_sun_noisy = add_sun_vector_noise(train_sun_vectors, noise_angle_deg=2.0)
    sun_mean = train_sun_noisy.mean(axis=0, keepdims=True)
    sun_std = train_sun_noisy.std(axis=0, keepdims=True)
    train_sun_normalized = (train_sun_noisy - sun_mean) / (sun_std + 1e-8)

    # combine features
    train_features_combined = np.concatenate([train_sensors, train_albedo, train_angle, train_sun_normalized], axis=1)
    train_features = torch.FloatTensor(train_features_combined)
    train_labels = torch.FloatTensor(train_df[['Nadir_X', 'Nadir_Y', 'Nadir_Z']].values)
    train_dataset = list(zip(train_features, train_labels))

    # load test data
    test_df = pd.read_csv('data/test.csv')
    test_sensors = test_df[sensor_cols].values
    test_sensors = (test_sensors - sensor_mean) / (sensor_std + 1e-8)

    test_albedo = test_df['Albedo'].values.reshape(-1, 1)
    test_albedo = (test_albedo - albedo_mean) / (albedo_std + 1e-8)

    test_angle = test_df['Sun_Nadir_Angle'].values.reshape(-1, 1)
    test_angle = (test_angle - angle_mean) / (angle_std + 1e-8)

    test_sun_vectors = test_df[['Sun_X', 'Sun_Y', 'Sun_Z']].values
    test_sun_noisy = add_sun_vector_noise(test_sun_vectors, noise_angle_deg=2.0)
    test_sun_normalized = (test_sun_noisy - sun_mean) / (sun_std + 1e-8)

    test_features_combined = np.concatenate([test_sensors, test_albedo, test_angle, test_sun_normalized], axis=1)
    test_features = torch.FloatTensor(test_features_combined)
    test_labels = torch.FloatTensor(test_df[['Nadir_X', 'Nadir_Y', 'Nadir_Z']].values)
    test_dataset = list(zip(test_features, test_labels))

    # Create output directories
    os.makedirs('models', exist_ok=True)

    # save normalization parameters
    norm_params = {
        'sensor_mean': sensor_mean.tolist(),
        'sensor_std': sensor_std.tolist(),
        'albedo_mean': float(albedo_mean),
        'albedo_std': float(albedo_std),
        'angle_mean': float(angle_mean),
        'angle_std': float(angle_std),
        'sun_mean': sun_mean.tolist(),
        'sun_std': sun_std.tolist()
    }
    with open('models/medium_nadir_normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    train = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train, test


def plot_errors(nadir_errors):
    """Plot error distribution"""
    os.makedirs('plots', exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(nadir_errors, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Angular Error (degrees)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Medium Nadir NN Angular Error Distribution', fontsize=14, fontweight='bold')
    ax.axvline(nadir_errors.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {nadir_errors.mean():.2f}°')
    ax.axvline(np.median(nadir_errors), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.median(nadir_errors):.2f}°')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/medium_nadir_error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'plots/medium_nadir_error_distribution.png'")
    plt.show()


def export_weights_for_c():
    """Export model weights in C-friendly format"""
    model = MediumNadirNet()
    model.load_state_dict(torch.load('models/medium_nadir_model.pth'))
    model.eval()

    with open('models/medium_nadir_weights.h', 'w') as f:
        f.write("/* Auto-generated weights for Medium Nadir NN */\n")
        f.write("/* ~10k parameters */\n\n")
        f.write("#ifndef MEDIUM_NADIR_WEIGHTS_H\n")
        f.write("#define MEDIUM_NADIR_WEIGHTS_H\n\n")

        for name, param in model.named_parameters():
            safe_name = name.replace('.', '_')
            tensor = param.detach().cpu().numpy()

            if len(tensor.shape) == 2:
                rows, cols = tensor.shape
                f.write(f"const float {safe_name}[{rows}][{cols}] = {{\n")
                for i in range(rows):
                    f.write("    {")
                    f.write(", ".join([f"{tensor[i][j]:.6f}f" for j in range(cols)]))
                    f.write("},\n" if i < rows - 1 else "}\n")
                f.write("};\n\n")
            else:
                size = tensor.shape[0]
                f.write(f"const float {safe_name}[{size}] = {{\n    ")
                f.write(", ".join([f"{tensor[i]:.6f}f" for i in range(size)]))
                f.write("\n};\n\n")

        f.write("#endif // MEDIUM_NADIR_WEIGHTS_H\n")

    print("Exported weights to 'models/medium_nadir_weights.h'")


def main():
    """Main training loop"""
    parser = argparse.ArgumentParser(description='Train Medium Nadir NN')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()
    
    train, test = load_data()
    print(f"Training examples: {len(train.dataset)}")
    print(f"Test examples: {len(test.dataset)}")

    model = MediumNadirNet()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Estimated model size: {total_params * 4 / 1024:.2f} KB (float32)")

    # test untrained model
    test_loss, nadir_errors = run_test(test, model)
    print(f"\nUntrained - Test Loss: {test_loss:.6f}, Mean Error: {nadir_errors.mean():.2f}°")

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    # train
    print("\nTraining...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = run_train(train, model, optimizer)
        test_loss, nadir_errors = run_test(test, model)
        
        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'models/medium_nadir_model.pth')
            print(f"Epoch {epoch+1:3d} | Test Loss: {test_loss:.6f} | Train Loss: {train_loss:.6f} | "
                  f"Mean Error: {nadir_errors.mean():5.2f}° | LR: {optimizer.param_groups[0]['lr']:.6f} - BEST")
        else:
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1:3d} | Test Loss: {test_loss:.6f} | Train Loss: {train_loss:.6f} | "
                      f"Mean Error: {nadir_errors.mean():5.2f}° | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # load best model
    model.load_state_dict(torch.load('models/medium_nadir_model.pth'))
    print("\nLoaded best model")

    # final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION - MEDIUM NADIR NN")
    print("="*70)
    final_loss, nadir_errors = run_test(test, model)
    print(f"\nFinal Test Loss: {final_loss:.6f}")
    print(f"\nNadir Vector Angular Error Statistics:")
    print(f"  Mean:            {nadir_errors.mean():.4f}°")
    print(f"  Median:          {np.median(nadir_errors):.4f}°")
    print(f"  Std:             {nadir_errors.std():.4f}°")
    print(f"  Min:             {nadir_errors.min():.4f}°")
    print(f"  Max:             {nadir_errors.max():.4f}°")
    print(f"  90th percentile: {np.percentile(nadir_errors, 90):.4f}°")
    print(f"  95th percentile: {np.percentile(nadir_errors, 95):.4f}°")
    print(f"  99th percentile: {np.percentile(nadir_errors, 99):.4f}°")

    # export for embedded
    export_weights_for_c()

    # plot
    plot_errors(nadir_errors)


if __name__ == "__main__":
    main()
