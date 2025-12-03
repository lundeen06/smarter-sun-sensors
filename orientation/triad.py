import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sun.ransac_sun_vector import estimate_sun_vector_ransac
from nadir.entropy_estimator import EntropyEstimator
from common.constants import SENSOR_NORMALS

def triad_algorithm(s_body, n_body, s_eci, n_eci):
    """
    TRIAD Algorithm to compute rotation matrix R (ECI to Body).
    s_body = R * s_eci
    n_body = R * n_eci
    """
    # Normalize
    s_body = s_body / np.linalg.norm(s_body)
    n_body = n_body / np.linalg.norm(n_body)
    s_eci = s_eci / np.linalg.norm(s_eci)
    n_eci = n_eci / np.linalg.norm(n_eci)
    
    # Construct Triad Frames
    # Body Frame
    t1b = s_body
    t2b = np.cross(s_body, n_body)
    if np.linalg.norm(t2b) < 1e-4: return None # Parallel vectors
    t2b = t2b / np.linalg.norm(t2b)
    t3b = np.cross(t1b, t2b)
    
    # ECI Frame
    t1r = s_eci
    t2r = np.cross(s_eci, n_eci)
    if np.linalg.norm(t2r) < 1e-4: return None
    t2r = t2r / np.linalg.norm(t2r)
    t3r = np.cross(t1r, t2r)
    
    # Rotation Matrix
    # R = [t1b t2b t3b] * [t1r t2r t3r]^T
    M_body = np.column_stack((t1b, t2b, t3b))
    M_ref = np.column_stack((t1r, t2r, t3r))
    
    R = M_body @ M_ref.T
    return R

def rotation_matrix_to_quaternion(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = np.sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)

def quaternion_angular_error(q1, q2):
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.degrees(np.arccos(dot))

def plot_results(results):
    """Plot results - simple 3-plot layout."""
    all_errors = results['all_errors']
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

    plt.suptitle('RANSAC + Entropy Nadir → TRIAD Attitude Estimation',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    save_path = 'plots/triad_results.png'
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
    lut_path = os.path.join(project_root, 'data/occlusion_lut.csv')
    
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found")
        return
        
    print(f"Loading data from {test_csv}...")
    df = pd.read_csv(test_csv)
    if args.samples: df = df.head(args.samples)
    
    # Initialize Entropy Estimator
    estimator = EntropyEstimator(lut_path=lut_path)
    
    all_errors = []
    sun_errors = []
    nadir_errors = []
    
    print(f"Evaluating on {len(df)} samples...")
    
    for i, row in df.iterrows():
        if i % 100 == 0: print(f"  Processing {i}/{len(df)}...", end='\r')
            
        sensor_cols = [f'Sensor_{k+1}' for k in range(16)]
        sensors = row[sensor_cols].values.astype(float)
        sun_body_true = row[['Sun_X', 'Sun_Y', 'Sun_Z']].values.astype(float)
        nadir_body_true = row[['Nadir_X', 'Nadir_Y', 'Nadir_Z']].values.astype(float)
        sun_eci = row[['Sun_ECI_X', 'Sun_ECI_Y', 'Sun_ECI_Z']].values.astype(float)
        nadir_eci = row[['Nadir_ECI_X', 'Nadir_ECI_Y', 'Nadir_ECI_Z']].values.astype(float)
        quat_true = row[['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']].values.astype(float)
        albedo = row['Albedo']
        sun_nadir_angle = row['Sun_Nadir_Angle']
        
        # 1. Estimate Sun (RANSAC)
        sun_body_est, valid = estimate_sun_vector_ransac(sensors, SENSOR_NORMALS)
        if not valid or sun_body_est is None: continue
        
        # 2. Estimate Nadir (Entropy) using ESTIMATED Sun
        nadir_body_est, _, _ = estimator.estimate(
            sensors, sun_body_est, albedo, sun_nadir_angle,
            min_albedo=0.15, max_entropy=3.8
        )
        if nadir_body_est is None: continue
        
        # 3. TRIAD
        R = triad_algorithm(sun_body_est, nadir_body_est, sun_eci, nadir_eci)
        if R is None: continue
        
        q_pred = rotation_matrix_to_quaternion(R)
        
        # Errors
        att_err = quaternion_angular_error(q_pred, quat_true)
        sun_err = np.degrees(np.arccos(np.clip(np.dot(sun_body_est, sun_body_true), -1, 1)))
        nadir_err = np.degrees(np.arccos(np.clip(np.dot(nadir_body_est, nadir_body_true), -1, 1)))
        
        all_errors.append(att_err)
        sun_errors.append(sun_err)
        nadir_errors.append(nadir_err)
        
    all_errors = np.array(all_errors)
    sun_errors = np.array(sun_errors)
    nadir_errors = np.array(nadir_errors)
    
    print("\n" + "="*70)
    print("RANSAC + ENTROPY NADIR -> TRIAD RESULTS")
    print("="*70)
    print(f"Valid Samples: {len(all_errors)}/{len(df)} ({100*len(all_errors)/len(df):.1f}%)")
    print(f"Attitude Error: Mean={all_errors.mean():.2f}°, Median={np.median(all_errors):.2f}°")
    print(f"Sun Error:      Mean={sun_errors.mean():.2f}°, Median={np.median(sun_errors):.2f}°")
    print(f"Nadir Error:    Mean={nadir_errors.mean():.2f}°, Median={np.median(nadir_errors):.2f}°")
    
    plot_results({
        'all_errors': all_errors,
        'sun_errors': sun_errors,
        'nadir_errors': nadir_errors
    })

if __name__ == "__main__":
    main()
