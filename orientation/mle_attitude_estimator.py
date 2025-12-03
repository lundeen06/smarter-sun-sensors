import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sun.ransac_sun_vector import estimate_sun_vector_ransac
from nadir.entropy_estimator import EntropyEstimator

SQRT_2_INV = 1 / np.sqrt(2)

# Satellite sensor normal vectors (body frame)
SENSOR_NORMALS = np.array([
    [SQRT_2_INV, 0, SQRT_2_INV], [SQRT_2_INV, SQRT_2_INV, 0], [SQRT_2_INV, 0, -SQRT_2_INV], [SQRT_2_INV, -SQRT_2_INV, 0],
    [-SQRT_2_INV, 0, SQRT_2_INV], [-SQRT_2_INV, -SQRT_2_INV, 0], [-SQRT_2_INV, 0, -SQRT_2_INV], [-SQRT_2_INV, SQRT_2_INV, 0],
    [0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, -1],
])

def quaternion_to_rotation_matrix(q):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    rot = R.from_matrix(R)
    q = rot.as_quat() # returns (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]]) # Convert to (w, x, y, z)

def quaternion_angular_error(q1, q2):
    """Compute angular error between two quaternions in degrees."""
    if len(q1.shape) == 1:
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, 0, 1)
        return 2.0 * np.degrees(np.arccos(dot))
    else:
        dot = np.abs(np.sum(q1 * q2, axis=1))
        dot = np.clip(dot, 0, 1)
        return 2.0 * np.degrees(np.arccos(dot))

def angular_error_between_vectors(v1, v2):
    """Compute angular error between two unit vectors in degrees."""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
    return np.rad2deg(np.arccos(cos_angle))

class MLEAttitudeEstimator:
    def __init__(self, lut_path):
        self.entropy_estimator = EntropyEstimator(lut_path=lut_path)
        self.nadir_sigma = None
        self.sun_sigma = None
        self.sigmas_estimated = False

    def estimate_sigmas(self, df):
        """Estimate sun and nadir measurement uncertainties from data."""
        print(f"\nEstimating measurement uncertainties from data...")

        sun_errors = []
        nadir_errors = []

        # Collect all errors
        for i, row in df.iterrows():
            if i % 100 == 0:
                print(f"  Processing: {i}/{len(df)}...", end='\r')

            sensor_cols = [f'Sensor_{k+1}' for k in range(16)]
            sensors = row[sensor_cols].values.astype(float)
            sun_body_true = row[['Sun_X', 'Sun_Y', 'Sun_Z']].values.astype(float)
            nadir_body_true = row[['Nadir_X', 'Nadir_Y', 'Nadir_Z']].values.astype(float)
            albedo = row['Albedo']
            sun_nadir_angle = row['Sun_Nadir_Angle']

            # Sun estimate (RANSAC)
            sun_body_est, valid = estimate_sun_vector_ransac(sensors, SENSOR_NORMALS)
            if not valid or sun_body_est is None:
                continue

            # Nadir estimate (Entropy)
            nadir_est, _, _ = self.predict_nadir(sensors, albedo, sun_nadir_angle, sun_body_est)
            if nadir_est is None:
                continue

            sun_err = angular_error_between_vectors(sun_body_est, sun_body_true)
            nadir_err = angular_error_between_vectors(nadir_est, nadir_body_true)

            sun_errors.append(sun_err)
            nadir_errors.append(nadir_err)

        sun_errors = np.array(sun_errors)
        nadir_errors = np.array(nadir_errors)

        # Compute standard deviations
        self.sun_sigma = np.std(sun_errors)
        self.nadir_sigma = np.std(nadir_errors)
        self.sigmas_estimated = True

        print(f"\n")
        print("="*70)
        print("ESTIMATED MEASUREMENT UNCERTAINTIES")
        print("="*70)
        print(f"Sun σ:   {self.sun_sigma:.4f}°")
        print(f"Nadir σ: {self.nadir_sigma:.4f}°")
        print("="*70 + "\n")

    def predict_nadir(self, sensors, albedo, sun_nadir_angle, sun_body):
        """Predict nadir vector using Entropy Estimator.

        Returns:
            nadir_pred: (3,) unit vector or None if failed
            entropy: Posterior entropy (nats)
            probs: Always None (not returned in current implementation)
        """
        try:
            nadir_pred, entropy = self.entropy_estimator.get_nadir(
                sensors, sun_body, albedo, sun_nadir_angle
            )
            return nadir_pred, entropy, None
        except Exception as e:
            # If estimation fails, return None
            return None, None, None

    def negative_log_likelihood(self, rotation_params, measurements_body, measurements_eci, sigmas):
        rot = R.from_rotvec(rotation_params)
        R_matrix = rot.as_matrix()
        nll = 0.0
        for v_body, v_eci, sigma in zip(measurements_body, measurements_eci, sigmas):
            v_eci_norm = v_eci / np.linalg.norm(v_eci)
            v_body_norm = v_body / np.linalg.norm(v_body)
            v_body_pred = R_matrix @ v_eci_norm
            cos_angle = np.clip(np.dot(v_body_norm, v_body_pred), -1, 1)
            angular_error_deg = np.rad2deg(np.arccos(cos_angle))
            nll += 0.5 * (angular_error_deg / sigma)**2
        return nll

    def estimate_attitude(self, sensors, sun_body, sun_eci, nadir_eci, albedo, sun_nadir_angle):

        # Get nadir prediction
        nadir_body, entropy, _ = self.predict_nadir(sensors, albedo, sun_nadir_angle, sun_body)
        if nadir_body is None: return None

        # Check if sigmas have been estimated
        if not self.sigmas_estimated:
            raise ValueError("Sigmas have not been estimated. Call estimate_sigmas() first.")

        # Use estimated sigmas
        sun_sigma = self.sun_sigma
        nadir_sigma = self.nadir_sigma

        # Measurements (Sun + Nadir only)
        measurements_body = [sun_body, nadir_body]
        measurements_eci = [sun_eci, nadir_eci]
        sigmas = [sun_sigma, nadir_sigma]

        # Initial guess: TRIAD (Sun + Nadir)
        v1_b = sun_body / np.linalg.norm(sun_body)
        v2_b = nadir_body / np.linalg.norm(nadir_body)
        v1_e = sun_eci / np.linalg.norm(sun_eci)
        v2_e = nadir_eci / np.linalg.norm(nadir_eci)

        t1_b = v1_b
        t2_b = np.cross(v1_b, v2_b)
        if np.linalg.norm(t2_b) < 1e-4: return None
        t2_b = t2_b / np.linalg.norm(t2_b)
        t3_b = np.cross(t1_b, t2_b)

        t1_e = v1_e
        t2_e = np.cross(v1_e, v2_e)
        if np.linalg.norm(t2_e) < 1e-4: return None
        t2_e = t2_e / np.linalg.norm(t2_e)
        t3_e = np.cross(t1_e, t2_e)

        M_body = np.column_stack([t1_b, t2_b, t3_b])
        M_eci = np.column_stack([t1_e, t2_e, t3_e])
        R_init = M_body @ M_eci.T

        # Optimization
        rot_init = R.from_matrix(R_init)
        x0 = rot_init.as_rotvec()

        result = minimize(
            self.negative_log_likelihood,
            x0,
            args=(measurements_body, measurements_eci, sigmas),
            method='BFGS',
            options={'maxiter': 50}
        )

        rot_optimal = R.from_rotvec(result.x)
        q_optimal = rot_optimal.as_quat() # (x, y, z, w)
        return np.array([q_optimal[3], q_optimal[0], q_optimal[1], q_optimal[2]]) # (w, x, y, z)

def plot_results(results):
    """Plot results - simple 3-plot layout."""
    all_errors = results['angular_errors']
    sun_errors = results['sun_errors']
    nadir_errors = results['nadir_errors']

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Attitude errors
    ax1.hist(all_errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(all_errors.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {all_errors.mean():.2f}°')
    ax1.axvline(np.median(all_errors), color='blue', linestyle=':', linewidth=2,
               label=f'Median: {np.median(all_errors):.2f}°')
    ax1.set_xlabel('Angular Error (degrees)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title(f'Attitude Error (n={len(all_errors)})', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Sun errors (RANSAC)
    ax2.hist(sun_errors, bins=50, edgecolor='black', alpha=0.7, color='gold')
    ax2.axvline(sun_errors.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {sun_errors.mean():.2f}°')
    ax2.axvline(np.median(sun_errors), color='blue', linestyle=':', linewidth=2,
               label=f'Median: {np.median(sun_errors):.2f}°')
    ax2.set_xlabel('Sun Vector Error (degrees)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title(f'RANSAC Sun Vector Error (n={len(sun_errors)})', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Nadir errors (Entropy)
    ax3.hist(nadir_errors, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    ax3.axvline(nadir_errors.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {nadir_errors.mean():.2f}°')
    ax3.axvline(np.median(nadir_errors), color='blue', linestyle=':', linewidth=2,
               label=f'Median: {np.median(nadir_errors):.2f}°')
    ax3.set_xlabel('Nadir Vector Error (degrees)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title(f'Entropy Nadir Vector Error (n={len(nadir_errors)})', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('MLE (RANSAC + Entropy Nadir) Attitude Estimation',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    save_path = 'plots/mle_attitude_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{save_path}'")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=None, help='Number of samples')
    args = parser.parse_args()

    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_csv = os.path.join(project_root, 'data/test.csv')
    lut_path = os.path.join(project_root, 'data/sensor_occlusion_lut.csv')

    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found")
        return

    print(f"Loading data from {test_csv}...")
    df = pd.read_csv(test_csv)
    if args.samples: df = df.head(args.samples)

    # Check if dataset has quaternions and ECI data
    has_attitude_data = 'Quat_W' in df.columns and 'Sun_ECI_X' in df.columns

    if not has_attitude_data:
        print("\n" + "="*70)
        print("NOTE: Dataset doesn't contain ECI/quaternion data.")
        print("      Running COMPONENT ESTIMATION ONLY (Sun + Nadir vectors)")
        print("      To test full MLE attitude: regenerate data/test.csv")
        print("="*70 + "\n")

    estimator = MLEAttitudeEstimator(lut_path)

    # Estimate measurement uncertainties
    estimator.estimate_sigmas(df)

    sun_errors = []
    nadir_errors = []
    attitude_errors = []
    valid_indices = []

    mode = "MLE Attitude Estimation" if has_attitude_data else "Component Estimation"
    print(f"\nEvaluating {mode} on {len(df)} samples...")

    for i, row in df.iterrows():
        if i % 100 == 0: print(f"  Processing {i}/{len(df)}...", end='\r')

        sensor_cols = [f'Sensor_{k+1}' for k in range(16)]
        sensors = row[sensor_cols].values.astype(float)
        sun_body_true = row[['Sun_X', 'Sun_Y', 'Sun_Z']].values.astype(float)
        nadir_body_true = row[['Nadir_X', 'Nadir_Y', 'Nadir_Z']].values.astype(float)
        albedo = row['Albedo']
        sun_nadir_angle = row['Sun_Nadir_Angle']

        # 1. Estimate Sun (RANSAC)
        sun_body_est, valid = estimate_sun_vector_ransac(sensors, SENSOR_NORMALS)
        if not valid or sun_body_est is None: continue

        # 2. Estimate Nadir (Entropy)
        nadir_est, _, _ = estimator.predict_nadir(sensors, albedo, sun_nadir_angle, sun_body_est)
        if nadir_est is None: continue

        # Compute component errors
        sun_err = angular_error_between_vectors(sun_body_est, sun_body_true)
        nadir_err = angular_error_between_vectors(nadir_est, nadir_body_true)
        sun_errors.append(sun_err)
        nadir_errors.append(nadir_err)

        # 3. MLE Attitude Estimation (if data available)
        if has_attitude_data:
            sun_eci = row[['Sun_ECI_X', 'Sun_ECI_Y', 'Sun_ECI_Z']].values.astype(float)
            nadir_eci = row[['Nadir_ECI_X', 'Nadir_ECI_Y', 'Nadir_ECI_Z']].values.astype(float)
            quat_true = row[['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']].values.astype(float)

            q_est = estimator.estimate_attitude(
                sensors, sun_body_est, sun_eci, nadir_eci, albedo, sun_nadir_angle
            )
            if q_est is None: continue

            att_err = quaternion_angular_error(q_est, quat_true)
            attitude_errors.append(att_err)
            valid_indices.append(i)

    sun_errors = np.array(sun_errors)
    nadir_errors = np.array(nadir_errors)

    # Print results
    print("\n" + "="*70)
    if has_attitude_data:
        attitude_errors = np.array(attitude_errors)
        print("MLE ATTITUDE ESTIMATION RESULTS")
        print("="*70)
        print(f"Valid Samples: {len(attitude_errors)}/{len(df)} ({100*len(attitude_errors)/len(df):.1f}%)\n")
        print(f"Attitude Error:")
        print(f"  Mean:   {attitude_errors.mean():.4f}°")
        print(f"  Median: {np.median(attitude_errors):.4f}°")
        print(f"  Std:    {attitude_errors.std():.4f}°")
        print(f"  95th:   {np.percentile(attitude_errors, 95):.4f}°")
    else:
        print("COMPONENT ESTIMATION RESULTS")
        print("="*70)
        print(f"Valid Samples: {len(sun_errors)}/{len(df)} ({100*len(sun_errors)/len(df):.1f}%)\n")

    print(f"\nSun Vector (RANSAC):")
    print(f"  Mean:   {sun_errors.mean():.4f}°")
    print(f"  Median: {np.median(sun_errors):.4f}°")
    print(f"  Std:    {sun_errors.std():.4f}°")
    print(f"  95th:   {np.percentile(sun_errors, 95):.4f}°")
    print(f"\nNadir Vector (Entropy):")
    print(f"  Mean:   {nadir_errors.mean():.4f}°")
    print(f"  Median: {np.median(nadir_errors):.4f}°")
    print(f"  Std:    {nadir_errors.std():.4f}°")
    print(f"  95th:   {np.percentile(nadir_errors, 95):.4f}°")
    print("="*70)

    # Plotting
    if has_attitude_data:
        # 3-plot layout: Attitude + Sun + Nadir
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Attitude errors
        ax1.hist(attitude_errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(attitude_errors.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {attitude_errors.mean():.2f}°')
        ax1.axvline(np.median(attitude_errors), color='blue', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(attitude_errors):.2f}°')
        ax1.set_xlabel('Angular Error (degrees)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title(f'Attitude Error (n={len(attitude_errors)})', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Sun errors
        ax2.hist(sun_errors, bins=50, edgecolor='black', alpha=0.7, color='gold')
        ax2.axvline(sun_errors.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {sun_errors.mean():.2f}°')
        ax2.axvline(np.median(sun_errors), color='blue', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(sun_errors):.2f}°')
        ax2.set_xlabel('Sun Vector Error (degrees)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title(f'RANSAC Sun Error (n={len(sun_errors)})', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Nadir errors
        ax3.hist(nadir_errors, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        ax3.axvline(nadir_errors.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {nadir_errors.mean():.2f}°')
        ax3.axvline(np.median(nadir_errors), color='blue', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(nadir_errors):.2f}°')
        ax3.set_xlabel('Nadir Vector Error (degrees)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title(f'Entropy Nadir Error (n={len(nadir_errors)})', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.suptitle('MLE Attitude Estimation (RANSAC + Entropy + MLE)',
                    fontsize=14, fontweight='bold')
        save_name = 'mle_attitude_results.png'
    else:
        # 2-plot layout: Sun + Nadir only
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Sun errors
        ax1.hist(sun_errors, bins=50, edgecolor='black', alpha=0.7, color='gold')
        ax1.axvline(sun_errors.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {sun_errors.mean():.2f}°')
        ax1.axvline(np.median(sun_errors), color='blue', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(sun_errors):.2f}°')
        ax1.set_xlabel('Sun Vector Error (degrees)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title(f'RANSAC Sun Error (n={len(sun_errors)})', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Nadir errors
        ax2.hist(nadir_errors, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        ax2.axvline(nadir_errors.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {nadir_errors.mean():.2f}°')
        ax2.axvline(np.median(nadir_errors), color='blue', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(nadir_errors):.2f}°')
        ax2.set_xlabel('Nadir Vector Error (degrees)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title(f'Entropy Nadir Error (n={len(nadir_errors)})', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Component Estimation (RANSAC Sun + Entropy Nadir)',
                    fontsize=14, fontweight='bold')
        save_name = 'component_estimation_results.png'

    plt.tight_layout()

    # Use absolute path to plots folder
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    main()
