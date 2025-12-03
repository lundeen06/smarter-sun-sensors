"""
RANSAC-based Sun Vector Estimation
Replicating the flight code C++ implementation for Python

This approach uses Random Sample Consensus (RANSAC) to robustly estimate
the sun vector from noisy sensor readings with potential outliers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import json
import os

# RANSAC parameters
MAX_ITERATIONS = 1000
INLIER_THRESHOLD = 0.05  # relative error threshold (5%)
MIN_INLIERS = 3  # minimum number of inliers required

# Shadow detection parameters
ACTIVE_THRESHOLD = 1300
SUN_SENSOR_CLIP_VALUE = 3102  # from your test data max


def ransac_sun_vector(
    normals: np.ndarray,
    signals: np.ndarray,
    n_sensors: int
) -> Tuple[Optional[np.ndarray], int]:
    """
    RANSAC implementation for robust sun vector estimation (sun-only model)

    The sensor model is: I = α × max(0, n·sun)
    where α is an unknown intensity scaling factor

    Args:
        normals: Array of sensor normal vectors [n_sensors, 3]
        signals: Array of sensor readings [n_sensors]
        n_sensors: Number of valid sensors

    Returns:
        best_sun_vector: Best sun vector found (normalized), or None if failed
        best_inlier_count: Number of inliers for best solution
    """
    if n_sensors < 3:
        return None, 0

    best_inlier_count = 0
    best_sun_vector = None
    best_alpha = 0

    # Maximum possible unique combinations
    max_possible_iterations = min(
        MAX_ITERATIONS,
        n_sensors * (n_sensors - 1) * (n_sensors - 2) // 6
    )

    for iter in range(max_possible_iterations):
        # Sample 3 unique sensors
        indices = [
            iter % n_sensors,
            (iter + 1) % n_sensors,
            (iter + 2) % n_sensors
        ]

        # Make sure indices are unique
        if len(set(indices)) != 3:
            continue

        # Build 3x3 system: N * (α * sun_vector) = b
        # We'll solve for (α * sun) then normalize
        N = np.zeros((3, 3))
        b = np.zeros(3)

        for i, sensor_idx in enumerate(indices):
            N[i, :] = normals[sensor_idx, :]
            b[i] = signals[sensor_idx]

        # Check for singularity by computing determinant
        det = np.linalg.det(N)
        if abs(det) < 1e-6:
            continue

        try:
            # Solve for α * sun_vector
            scaled_sun = np.linalg.solve(N, b)
        except np.linalg.LinAlgError:
            continue

        # Extract magnitude (this is α) and direction (sun_vec)
        alpha = np.linalg.norm(scaled_sun)
        if alpha < 1e-6:
            continue

        sun_vec = scaled_sun / alpha

        # Count inliers using the sun-only model: I = α × max(0, n·sun)
        inlier_count = 0
        total_error = 0.0
        for i in range(n_sensors):
            normal = normals[i, :]
            expected_signal = alpha * max(0.0, np.dot(sun_vec, normal))
            actual_signal = signals[i]

            # Relative error (percentage)
            if expected_signal > 10:  # avoid division by small numbers
                relative_error = abs(expected_signal - actual_signal) / expected_signal
                if relative_error < INLIER_THRESHOLD:
                    inlier_count += 1
                    total_error += relative_error
            elif abs(expected_signal - actual_signal) < 50:  # both small
                inlier_count += 1

        # Update best solution if this one has more inliers
        if inlier_count > best_inlier_count and inlier_count >= MIN_INLIERS:
            best_inlier_count = inlier_count
            best_sun_vector = sun_vec
            best_alpha = alpha

    if best_sun_vector is not None:
        return best_sun_vector, best_inlier_count
    else:
        return None, best_inlier_count


def estimate_sun_vector_ransac(
    sensor_readings: np.ndarray,
    sensor_normals: np.ndarray
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Estimate sun vector from sensor readings using RANSAC (sun-only model)

    Args:
        sensor_readings: Array of 16 sensor readings
        sensor_normals: Array of sensor normal vectors [16, 3]

    Returns:
        sun_vector: Estimated sun vector (normalized), or None if failed
        valid: Whether the estimation is valid
    """
    # Find max intensity
    I_max = np.max(sensor_readings)

    # Check for saturation
    if I_max >= SUN_SENSOR_CLIP_VALUE:
        return None, False

    # Define opposite sensor pairs (from C++ code)
    opposite_pairs = [
        (0, 6), (1, 5), (2, 4), (3, 7),  # pyramid opposites
        (8, 10), (9, 11),  # Y+ vs Y- pairs
        (12, 14), (13, 15)  # Z+ vs Z- pairs
    ]

    # Mark sensors with impossible readings for exclusion
    exclude_sensor = np.zeros(16, dtype=bool)
    for sensor1, sensor2 in opposite_pairs:
        if (sensor_readings[sensor1] > ACTIVE_THRESHOLD and
            sensor_readings[sensor2] > ACTIVE_THRESHOLD):
            exclude_sensor[sensor1] = True
            exclude_sensor[sensor2] = True

    # Create unique sensor readings (excluding problematic sensors)
    sensor_readings_unique = np.zeros(12)

    # First 8 pyramid sensors
    for i in range(8):
        sensor_readings_unique[i] = 0.0 if exclude_sensor[i] else sensor_readings[i]

    # Average redundant +Y/-Y sensors (indices 8,9 and 10,11)
    sensor_readings_unique[8] = (sensor_readings[8] + sensor_readings[9]) / 2
    sensor_readings_unique[9] = (sensor_readings[10] + sensor_readings[11]) / 2

    # Average +Z/-Z sensors (indices 12,13 and 14,15)
    sensor_readings_unique[10] = (sensor_readings[12] + sensor_readings[13]) / 2
    sensor_readings_unique[11] = (sensor_readings[14] + sensor_readings[15]) / 2

    # Corresponding normal indices
    normals_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14]

    # Only consider readings above threshold
    sensor_readings_unique[sensor_readings_unique < ACTIVE_THRESHOLD] = 0.0

    # Create matrix of normals and signals excluding zero sensors
    valid_mask = sensor_readings_unique > 1e-6
    valid_normals = sensor_normals[normals_idx][valid_mask]
    valid_signals = sensor_readings_unique[valid_mask]
    valid_count = len(valid_signals)

    # Check for degenerate case
    if valid_count < 3:
        return None, False

    # Use RANSAC for robust estimation
    sun_vector, inlier_count = ransac_sun_vector(
        valid_normals, valid_signals, valid_count
    )

    if sun_vector is not None:
        # Check 1: Reject if opposite sensors are both bright (albedo contamination)
        opposite_pairs = [
            (0, 6), (1, 5), (2, 4), (3, 7),  # pyramid opposites
            (8, 10), (9, 11),  # Y+ vs Y-
            (12, 14), (13, 15)  # Z+ vs Z-
        ]

        for s1, s2 in opposite_pairs:
            # If both sensors in opposite pair are bright, likely albedo
            if sensor_readings[s1] > ACTIVE_THRESHOLD and sensor_readings[s2] > ACTIVE_THRESHOLD:
                return sun_vector, False

        # Check 2: Brightest sensors should align with estimated sun direction
        brightest_idx = np.argsort(-sensor_readings)[:4]  # top 4 brightest
        brightest_normals = sensor_normals[brightest_idx]

        alignment_scores = []
        for normal in brightest_normals:
            alignment = np.dot(normal, sun_vector)
            alignment_scores.append(alignment)

        mean_alignment = np.mean(alignment_scores)

        # If brightest sensors don't align well with sun direction, reject
        if mean_alignment < 0.5:  # require better alignment (< 60° angle)
            return sun_vector, False

        return sun_vector, True
    else:
        return None, False


def evaluate_ransac_on_dataset(csv_path: str, sensor_normals: np.ndarray):
    """
    Evaluate RANSAC method on a dataset

    Args:
        csv_path: Path to CSV file with sensor data
        sensor_normals: Array of sensor normal vectors [16, 3]
    """
    df = pd.read_csv(csv_path)
    sensor_cols = [f'Sensor_{i}' for i in range(1, 17)]

    errors = []
    error_data = []  # Store (error, idx, row) for analysis
    valid_count = 0
    failed_count = 0

    for idx, row in df.iterrows():
        # Get sensor readings
        sensor_readings = row[sensor_cols].values

        # Get ground truth
        true_sun = np.array([row['Sun_X'], row['Sun_Y'], row['Sun_Z']])
        true_sun_normalized = true_sun / np.linalg.norm(true_sun)

        # Estimate sun vector (sun-only model)
        estimated_sun, valid = estimate_sun_vector_ransac(
            sensor_readings, sensor_normals
        )

        if valid and estimated_sun is not None:
            # Compute angular error
            dot_product = np.clip(np.dot(estimated_sun, true_sun_normalized), -1.0, 1.0)
            angle_error = np.arccos(dot_product) * 180.0 / np.pi
            errors.append(angle_error)
            error_data.append((angle_error, idx, row, estimated_sun, true_sun_normalized))
            valid_count += 1
        else:
            failed_count += 1

    errors = np.array(errors)

    print("\n" + "="*70)
    print("RANSAC SUN VECTOR ESTIMATION RESULTS")
    print("="*70)
    print(f"\nTotal samples: {len(df)}")
    print(f"Valid estimations: {valid_count} ({valid_count/len(df)*100:.1f}%)")
    print(f"Failed estimations: {failed_count} ({failed_count/len(df)*100:.1f}%)")

    if len(errors) > 0:
        print(f"\nAngular Error Statistics:")
        print(f"  Mean:            {errors.mean():.4f}°")
        print(f"  Median:          {np.median(errors):.4f}°")
        print(f"  Std:             {errors.std():.4f}°")
        print(f"  Min:             {errors.min():.4f}°")
        print(f"  Max:             {errors.max():.4f}°")
        print(f"  90th percentile: {np.percentile(errors, 90):.4f}°")
        print(f"  95th percentile: {np.percentile(errors, 95):.4f}°")
        print(f"  99th percentile: {np.percentile(errors, 99):.4f}°")

        # Find and print worst case
        worst_idx = np.argmax(errors)
        worst_error, worst_row_idx, worst_row, worst_est, worst_true = error_data[worst_idx]

        print(f"\n" + "="*70)
        print(f"WORST CASE ANALYSIS (Error: {worst_error:.4f}°)")
        print("="*70)
        print(f"Row index: {worst_row_idx}")
        print(f"\nTrue sun vector:      [{worst_true[0]:7.4f}, {worst_true[1]:7.4f}, {worst_true[2]:7.4f}]")
        print(f"Estimated sun vector: [{worst_est[0]:7.4f}, {worst_est[1]:7.4f}, {worst_est[2]:7.4f}]")

        if 'Nadir_X' in worst_row:
            nadir = np.array([worst_row['Nadir_X'], worst_row['Nadir_Y'], worst_row['Nadir_Z']])
            nadir_norm = nadir / np.linalg.norm(nadir)
            print(f"Nadir vector:         [{nadir_norm[0]:7.4f}, {nadir_norm[1]:7.4f}, {nadir_norm[2]:7.4f}]")

            # Sun-nadir angle
            sun_nadir_angle = np.arccos(np.clip(np.dot(worst_true, nadir_norm), -1, 1)) * 180 / np.pi
            print(f"Sun-Nadir angle:      {sun_nadir_angle:.2f}°")

        if 'Albedo' in worst_row:
            print(f"Albedo:               {worst_row['Albedo']:.4f}")

        if 'Sun_Nadir_Angle' in worst_row:
            print(f"Sun-Nadir angle (CSV): {worst_row['Sun_Nadir_Angle']:.2f}°")

        print(f"\nSensor readings:")
        for i in range(1, 17):
            print(f"  Sensor {i:2d}: {int(worst_row[f'Sensor_{i}']):4d}", end="")
            if i % 4 == 0:
                print()  # newline every 4 sensors

        # Plot error distribution
        os.makedirs('plots', exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_xlabel('Angular Error (degrees)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('RANSAC Sun Vector Angular Error Distribution', fontsize=14, fontweight='bold')
        ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {errors.mean():.2f}°')
        ax.axvline(np.median(errors), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(errors):.2f}°')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/ransac_sun_error_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as 'plots/ransac_sun_error_distribution.png'")
        plt.show()

    return errors


def main():
    """Main function to run RANSAC evaluation"""

    # Actual sensor normals from sensor_config.py
    SQRT_2_INV = 1.0 / np.sqrt(2.0)

    sensor_normals = np.array([
        # Pyramid sensors (0-7) - 45 degree angles
        [SQRT_2_INV, 0, SQRT_2_INV],           # 0: +X/+Z diagonal
        [SQRT_2_INV, SQRT_2_INV, 0],           # 1: +X/+Y diagonal
        [SQRT_2_INV, 0, -SQRT_2_INV],          # 2: +X/-Z diagonal
        [SQRT_2_INV, -SQRT_2_INV, 0],          # 3: +X/-Y diagonal
        [-SQRT_2_INV, 0, SQRT_2_INV],          # 4: -X/+Z diagonal
        [-SQRT_2_INV, -SQRT_2_INV, 0],         # 5: -X/-Y diagonal
        [-SQRT_2_INV, 0, -SQRT_2_INV],         # 6: -X/-Z diagonal
        [-SQRT_2_INV, SQRT_2_INV, 0],          # 7: -X/+Y diagonal
        # Y-axis sensors (8-11)
        [0, 1, 0],                              # 8: +Y
        [0, 1, 0],                              # 9: +Y (redundant)
        [0, -1, 0],                             # 10: -Y
        [0, -1, 0],                             # 11: -Y (redundant)
        # Z-axis sensors (12-15)
        [0, 0, 1],                              # 12: +Z
        [0, 0, 1],                              # 13: +Z (redundant)
        [0, 0, -1],                             # 14: -Z
        [0, 0, -1],                             # 15: -Z (redundant)
    ])

    # Normalize all normals
    for i in range(len(sensor_normals)):
        sensor_normals[i] = sensor_normals[i] / np.linalg.norm(sensor_normals[i])

    # Save sensor normals for reference
    os.makedirs('models', exist_ok=True)
    np.save('models/sensor_normals.npy', sensor_normals)

    # Evaluate on test set
    print("Evaluating RANSAC on test set...")
    test_errors = evaluate_ransac_on_dataset('data/test.csv', sensor_normals)


if __name__ == "__main__":
    main()
