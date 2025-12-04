import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@dataclass
class SearchConfig:
    """Multi-stage search configuration for coarse-to-fine nadir estimation.

    Progressive refinement strategy:
    - Stage 1: Search full 360° circle around sun vector
    - Stage 2: Zoom into ±15° window around best candidate
    - Stage 3: Final precision search in ±3° window
    """
    coarse_samples: int = 180      # Stage 1: Full circle search
    medium_samples: int = 100      # Stage 2: Medium refinement
    fine_samples: int = 100        # Stage 3: Fine refinement
    medium_window_deg: float = 15.0  # Stage 2 search window
    fine_window_deg: float = 3.0     # Stage 3 search window

# ==================
#   CONSTANTS
# ==================

SQRT_2_INV = 1.0 / np.sqrt(2.0)

"""
Satellite altitude (m)

Affects...
(1) the angular size of the Earth disk
(2) the solid angle of the Earth,
(3) how bright the reflected light is
"""
ALTITUDE = 500e3

# Earth radius (m)
EARTH_RADIUS = 6371e3

# Maximum sensor value (i.e., when sun is head-on)
MAX_SENSOR_VALUE = 4095.0

# Satellite sensor normal vectors (body frame)
SENSOR_NORMALS = np.array([
    [SQRT_2_INV, 0, SQRT_2_INV], [SQRT_2_INV, SQRT_2_INV, 0], [SQRT_2_INV, 0, -SQRT_2_INV], [SQRT_2_INV, -SQRT_2_INV, 0],
    [-SQRT_2_INV, 0, SQRT_2_INV], [-SQRT_2_INV, -SQRT_2_INV, 0], [-SQRT_2_INV, 0, -SQRT_2_INV], [-SQRT_2_INV, SQRT_2_INV, 0],
    [0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, -1],
])

"""
Characteristic noise of sun sensors

The sun sensors output a voltage, and to convert this voltage to
a digital number, it goes through a device called an ADC (Analog-to-Digital Converter). 

The first 8 sun sensors (on the "sun pyramids") connect to a higher-grade ADC with a higher
reference voltage, so they have lower noise. The last 8 sun sensors (on the "sun disks")
connect to a lower-grade ADC with a lower reference voltage, so they have higher noise. 

Assuming Guassian noise. TODO: measure noise on flight hardware 
"""
STD_SUN_SENSORS_PYRAMID = 10
STD_SUN_SENSORS_YZ = 20
    
# ======================
#   NADIR ESTIMATOR
# ======================

class EntropyEstimator:
    """
    Bayesian nadir vector estimator using L1 (Laplace) likelihood.

    Strategy:
    1. Subtract predicted sun signal from sensor readings → Earth albedo residual
    2. Multi-stage search (coarse → medium → fine) around sun-nadir constraint circle
    3. For each candidate: predict Earth signal, compute L1 likelihood
    4. Return posterior mean (Bayes estimator minimizing expected angular error)

    Probability Model:
        P(nadir | sensors) ∝ P(sensors | nadir) × P(nadir)

        Likelihood: P(sensors | nadir) = ∏ᵢ Laplace(sensor_i | predicted_i, σ_i)
                                        = ∏ᵢ (1/2σ_i) exp(-|sensor_i - predicted_i| / σ_i)

        Log-likelihood: log P(sensors | nadir) = -Σᵢ |error_i| / σ_i  (up to constant)
    """

    def __init__(self, lut_path: str = "data/sensor_occlusion_lut.csv",
                        num_samples_earth: int = 100,
                        sigma_noise: float = 10.0,
                        search_config: Optional[SearchConfig] = None,
                        max_entropy: Optional[float] = 4.5):
        self.num_samples_earth = num_samples_earth
        self.sigma_noise = sigma_noise
        self.config = search_config if search_config else SearchConfig()
        self.max_entropy = max_entropy
       
       # Load LUT (used to check if sensors are in view of a given light vector)
        if not os.path.exists(lut_path):
             # Try relative to project root if not found
             project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
             lut_path = os.path.join(project_root, lut_path)
             
        print(f"Loading Occlusion LUT from {lut_path}...")
        df = pd.read_csv(lut_path)
        self.lut_points = df[['dx', 'dy', 'dz']].values
        self.lut_visibility = df[[f'vis_{i+1}' for i in range(16)]].values
        self.kdtree = KDTree(self.lut_points)

        # Precompute deterministic Fibonacci spiral samples for Earth disk
        # This gives better coverage than random sampling (quasi-Monte Carlo)
        indices = np.arange(0, self.num_samples_earth, dtype=float) + 0.5
        r = np.sqrt(indices / self.num_samples_earth)  # Uniform area distribution
        theta = np.pi * (1 + 5**0.5) * indices  # Golden angle
        self.disk_samples = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

        # Earth angular radius ~62 deg at 500km altitude (low earth orbit)
        self.earth_angular_radius = np.radians(62)
        self.earth_solid_angle = 2 * np.pi * (1 - np.cos(self.earth_angular_radius)) 

    def get_visibility(self, light_vector):
        """Given all 16 sun sensors, which can see a given light vector (sun, Earth, etc.)"""
        dist, idx = self.kdtree.query(light_vector)
        return self.lut_visibility[idx]

    def sample_earth_disk(self, nadir_vector):
        """Generate Earth surface points for a single nadir vector to help estimate Earth albedo"""
        M = self.num_samples_earth

        # Basis vectors
        z_axis = nadir_vector
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-6:
            x_axis = np.cross(np.array([0, 1, 0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Project disk samples
        sin_alpha = np.sin(self.earth_angular_radius) * self.disk_samples[:, 0]
        cos_alpha = np.cos(self.earth_angular_radius) * self.disk_samples[:, 1]
        z_comp = np.sqrt(1 - sin_alpha**2 - cos_alpha**2) # (M,)

        # (M, 3) = (3,) * (M, 1) + (3,) * (M, 1) + (3,) * (M, 1)
        points = (x_axis * sin_alpha[:, np.newaxis] +
                  y_axis * cos_alpha[:, np.newaxis] +
                  z_axis * z_comp[:, np.newaxis])
        return points
    
    def get_earth_signal(self, nadir_vector, sun_vector, albedo):
        """Predict contribution of the Earth's reflected light to the sun sensors

        For each sampled point on Earth's visible disk:
        1. Calculate distance from satellite (depends on altitude and angle)
        2. Check if sensors can see that direction (using occlusion LUT)
        3. Calculate reflected light intensity (Lambertian reflectance + inverse square law)
        4. Average contributions across all samples
        """
        # Sample M points on the visible Earth disk
        earth_directions = self.sample_earth_disk(nadir_vector)  # (M, 3)
        M = len(earth_directions)

        # Calculate distance to each Earth surface point using law of cosines
        # distance^2 = h^2 + R^2 + 2*h*R*cos(angle_from_nadir)
        cos_angles = np.dot(earth_directions, nadir_vector)  # (M,)
        distances_sq = ALTITUDE**2 + EARTH_RADIUS**2 + 2*ALTITUDE*EARTH_RADIUS*cos_angles
        distances = np.sqrt(distances_sq)

        # Sunlight hitting each Earth point
        sun_on_earth = np.maximum(0, -np.dot(earth_directions, sun_vector))  # (M,)

        # How much each sensor sees each Earth point (before checking for satellite body occlusion)
        sensor_view_angles = np.maximum(0, np.dot(earth_directions, SENSOR_NORMALS.T))  # (M, 16)

        # Which sensors can see each Earth light vector
        visibility = np.zeros((M, 16))
        for i in range(M):
            visibility[i] = self.get_visibility(earth_directions[i])  # (16,)

        # Calculate reflected light intensity reaching each sensor
        # Lambertian reflectance: albedo/π * (sun_angle) * (sensor_angle)
        # No distance term - absorbed in solid angle normalization
        contributions = (sun_on_earth[:, np.newaxis] *
                        sensor_view_angles *
                        visibility)

        # Average over all M samples (Monte Carlo integration)
        avg_contribution = np.mean(contributions, axis=0)  # (16,)

        # Scale by Earth solid angle and albedo to get sensor reading
        I_earth = avg_contribution * albedo * (self.earth_solid_angle / np.pi) * MAX_SENSOR_VALUE
        return I_earth
    
    def get_sun_signal(self, sun_vector):
        """Predict contribution of direct sunlight to the sun sensors,
        assuming the sun is a point source
        """
        # Cosine law for relative intensity (1 if normal to sun)
        sensor_view_angles = np.maximum(0, np.dot(SENSOR_NORMALS, sun_vector))  # (16,)
        visibility = self.get_visibility(sun_vector)  # (16,)
        I_sun = sensor_view_angles * visibility * MAX_SENSOR_VALUE
        return I_sun

    def get_total_signal(self, sun_vector, nadir_vector, albedo):
        """Predict total signal from sun and Earth"""
        I_earth = self.get_earth_signal(nadir_vector, sun_vector, albedo)
        I_sun = self.get_sun_signal(sun_vector)
        return I_earth + I_sun

    def get_nadir(self, sensors: np.ndarray, sun_vector: np.ndarray, albedo: float, sun_nadir_angle_deg: float) -> Tuple[Optional[np.ndarray], float]:
        """Estimate nadir vector using multi-stage coarse-to-fine search.

        Strategy:
        1. Coarse stage: Search full 360° circle at sun-nadir constraint angle
        2. Medium stage: Refine in ±15° window around best coarse candidate
        3. Fine stage: Precise search in ±3° window around best medium candidate
        4. Return posterior mean of fine-stage distribution

        Args:
            sensors: (16,) array of sensor readings
            sun_vector: (3,) sun direction in body frame
            albedo: Earth albedo coefficient [0, 1]
            sun_nadir_angle_deg: Constraint angle between sun and nadir (degrees)

        Returns:
            nadir_estimate: (3,) unit vector pointing to nadir (posterior mean), or None if rejected by entropy filter
            entropy: Shannon entropy of the final posterior distribution (nats)
        """
        # Normalize sun vector
        sun_vector = sun_vector / np.linalg.norm(sun_vector)

        # Subtract sun signal from sensor readings to isolate Earth albedo signal
        I_nadir = sensors - self.get_sun_signal(sun_vector)

        # Basis vectors for the constraint circle (nadir is at fixed angle from sun)
        sun_nadir_angle = np.radians(sun_nadir_angle_deg)
        z_axis = sun_vector
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([0, 1, 0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        sin_angle = np.sin(sun_nadir_angle)
        cos_angle = np.cos(sun_nadir_angle)

        def get_candidates(thetas):
            """Generate nadir candidates at specific angles around the constraint circle."""
            return (x_axis[np.newaxis, :] * sin_angle * np.cos(thetas)[:, np.newaxis] +
                    y_axis[np.newaxis, :] * sin_angle * np.sin(thetas)[:, np.newaxis] +
                    z_axis[np.newaxis, :] * cos_angle)

        # ============================
        # STAGE 1: Coarse Search
        # ============================
        thetas_coarse = np.linspace(0, 2*np.pi, self.config.coarse_samples, endpoint=False)
        candidates_coarse = get_candidates(thetas_coarse)
        probs_coarse = self._evaluate_candidates_vectorized(candidates_coarse, sun_vector, albedo, I_nadir)
        best_theta = thetas_coarse[np.argmax(probs_coarse)]

        # ============================
        # STAGE 2: Medium Refinement
        # ============================
        window_medium = np.radians(self.config.medium_window_deg)
        thetas_medium = np.linspace(best_theta - window_medium,
                                    best_theta + window_medium,
                                    self.config.medium_samples)
        candidates_medium = get_candidates(thetas_medium)
        probs_medium = self._evaluate_candidates_vectorized(candidates_medium, sun_vector, albedo, I_nadir)
        best_theta = thetas_medium[np.argmax(probs_medium)]

        # ============================
        # STAGE 3: Fine Refinement
        # ============================
        window_fine = np.radians(self.config.fine_window_deg)
        thetas_fine = np.linspace(best_theta - window_fine,
                                  best_theta + window_fine,
                                  self.config.fine_samples)
        candidates_fine = get_candidates(thetas_fine)
        probs_fine = self._evaluate_candidates_vectorized(candidates_fine, sun_vector, albedo, I_nadir)

        # Posterior mean (Bayes estimator - minimizes expected angular error)
        mean_vec = np.sum(candidates_fine * probs_fine[:, np.newaxis], axis=0)
        mean_vec /= np.linalg.norm(mean_vec)

        # Compute entropy of final distribution (in nats)
        entropy = -np.sum(probs_fine * np.log(probs_fine + 1e-10))

        # Apply entropy filtering if threshold is set
        if self.max_entropy is not None and entropy > self.max_entropy:
            return None, entropy

        return mean_vec, entropy

    def _evaluate_candidates_vectorized(self, candidates: np.ndarray, sun_vector: np.ndarray, albedo: float, I_nadir: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of all candidate nadir vectors using L1 likelihood.

        Probability Model (Laplace/L1):
            P(sensors | nadir) = ∏ᵢ (1/2σ_i) exp(-|sensor_i - predicted_i| / σ_i)

            Log-likelihood:
            log P(sensors | nadir) = -Σᵢ |sensor_i - predicted_i| / σ_i + const

        Args:
            candidates: (N, 3) array of candidate nadir vectors
            sun_vector: (3,) sun direction vector
            albedo: Earth albedo coefficient
            I_nadir: (16,) residual sensor signal after removing sun

        Returns:
            probs: (N,) array of probabilities for each candidate (normalized posterior)
        """
        N = len(candidates)
        M = self.num_samples_earth

        # Vectorized Earth disk sampling for all candidates
        all_earth_directions = self._sample_earth_disk_vectorized(candidates)  # (N, M, 3)

        # Reshape for batch processing: (N*M, 3)
        earth_dirs_flat = all_earth_directions.reshape(N * M, 3)

        # Calculate distances for all samples (currently unused, but kept for future physics-based scaling)
        cos_angles = np.einsum('ij,ij->i', earth_dirs_flat,
                              np.repeat(candidates, M, axis=0))  # (N*M,)
        distances_sq = ALTITUDE**2 + EARTH_RADIUS**2 + 2*ALTITUDE*EARTH_RADIUS*cos_angles

        # Sunlight hitting each Earth point (Lambertian: proportional to cos(sun_angle))
        sun_on_earth = np.maximum(0, -np.dot(earth_dirs_flat, sun_vector))  # (N*M,)

        # Sensor view angles for all Earth points
        sensor_view_angles = np.maximum(0, np.dot(earth_dirs_flat, SENSOR_NORMALS.T))  # (N*M, 16)

        # Batch query KDTree for visibility (MAJOR optimization!)
        _, indices = self.kdtree.query(earth_dirs_flat)
        visibility = self.lut_visibility[indices]  # (N*M, 16)

        # Calculate contributions (no distance term - absorbed in solid angle normalization)
        contributions = (sun_on_earth[:, np.newaxis] *
                        sensor_view_angles *
                        visibility)

        # Reshape back to (N, M, 16) and average over M samples (Monte Carlo integration)
        contributions = contributions.reshape(N, M, 16)
        avg_contributions = np.mean(contributions, axis=1)  # (N, 16)

        # Scale by Earth solid angle and albedo (Lambertian BRDF: albedo/π)
        I_earth_predicted = avg_contributions * albedo * (self.earth_solid_angle / np.pi) * MAX_SENSOR_VALUE

        # === CRITICAL: Proper L1 Likelihood ===
        # Compute absolute error per sensor
        errors_per_sensor = np.abs(I_nadir[np.newaxis, :] - I_earth_predicted)  # (N, 16)

        # Divide by sensor noise (different noise for pyramid vs YZ sensors)
        # First 8 sensors: σ = 10, Last 8 sensors: σ = 20
        sigmas = np.array([self.sigma_noise] * 8 + [self.sigma_noise * 2.0] * 8)
        weighted_errors = errors_per_sensor / sigmas  # (N, 16) - THIS IS THE FIX!

        # Sum across sensors to get total negative log-likelihood
        total_errors = np.sum(weighted_errors, axis=1)  # (N,)

        # Convert to probabilities via softmax (numerically stable)
        log_likelihoods = -total_errors
        log_likelihoods -= np.max(log_likelihoods)  # Prevent overflow
        probs = np.exp(log_likelihoods)
        probs /= np.sum(probs)  # Normalize to sum to 1

        return probs

    def _sample_earth_disk_vectorized(self, nadir_vectors: np.ndarray) -> np.ndarray:
        """Vectorized Earth disk sampling for multiple nadir vectors

        Args:
            nadir_vectors: (N, 3) array of nadir direction vectors

        Returns:
            points: (N, M, 3) array of Earth surface points for each nadir vector
        """
        N = len(nadir_vectors)
        M = self.num_samples_earth

        # Compute basis vectors for all nadirs at once
        # x_axis = cross(z_ref, nadir), with fallback for parallel cases
        z_ref = np.array([0, 0, 1])
        x_axes = np.cross(z_ref, nadir_vectors)  # (N, 3)
        norms = np.linalg.norm(x_axes, axis=1, keepdims=True)  # (N, 1)

        # Fallback for parallel cases
        parallel_mask = (norms < 1e-6).squeeze()
        if np.any(parallel_mask):
            y_ref = np.array([0, 1, 0])
            x_axes[parallel_mask] = np.cross(y_ref, nadir_vectors[parallel_mask])

        x_axes = x_axes / np.linalg.norm(x_axes, axis=1, keepdims=True)  # (N, 3)
        y_axes = np.cross(nadir_vectors, x_axes)  # (N, 3)

        # Disk samples (reuse same deterministic samples for all candidates)
        # NOTE: Both use SIN(earth_angular_radius) to properly scale the disk!
        sin_alpha = np.sin(self.earth_angular_radius) * self.disk_samples[:, 0]  # (M,)
        cos_alpha = np.sin(self.earth_angular_radius) * self.disk_samples[:, 1]  # (M,) - YES, SIN not COS!
        z_comp = np.sqrt(1 - sin_alpha**2 - cos_alpha**2)  # (M,)

        # Broadcast and combine: (N, M, 3) = (N, 1, 3) * (1, M, 1) + ...
        points = (x_axes[:, np.newaxis, :] * sin_alpha[np.newaxis, :, np.newaxis] +
                  y_axes[:, np.newaxis, :] * cos_alpha[np.newaxis, :, np.newaxis] +
                  nadir_vectors[:, np.newaxis, :] * z_comp[np.newaxis, :, np.newaxis])

        return points

    def _sample_cone_ring(self, axis: np.ndarray, angle: float, num_samples: int) -> np.ndarray:
        """Generate uniform samples on a cone ring at fixed angle from axis

        Creates a ring of vectors all at exactly 'angle' radians from 'axis'
        """
        # Create two orthogonal vectors perpendicular to axis
        perp1 = np.cross(axis, np.array([0, 0, 1]))
        if np.linalg.norm(perp1) < 1e-6:
            perp1 = np.cross(axis, np.array([0, 1, 0]))
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)
        perp2 /= np.linalg.norm(perp2)

        # Generate samples around the ring
        thetas = np.linspace(0, 2*np.pi, num_samples, endpoint=False)

        # Each sample = axis*cos(angle) + perpendicular_component*sin(angle)
        # perpendicular_component rotates around the ring
        samples = np.zeros((num_samples, 3))
        for i, theta in enumerate(thetas):
            perp_component = perp1 * np.cos(theta) + perp2 * np.sin(theta)
            samples[i] = axis * np.cos(angle) + perp_component * np.sin(angle)

        return samples

def compute_angular_error(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Compute angular error between two vectors (numpy arrays)"""
    dot = np.sum(pred * actual, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

def plot_error_distribution(nadir_errors: np.ndarray, save_path: str):
    """Plot error distribution histogram"""
    plt.figure(figsize=(10, 6))
    plt.hist(nadir_errors, bins=50, edgecolor='black', alpha=0.7, color='purple')
    plt.xlabel('Angular Error (degrees)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Nadir Vector via Entropy Estimator', fontsize=14, fontweight='bold')
    plt.axvline(nadir_errors.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {nadir_errors.mean():.2f}°')
    plt.axvline(np.median(nadir_errors), color='blue', linestyle=':', linewidth=2,
               label=f'Median: {np.median(nadir_errors):.2f}°')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close()

def plot_entropy_correlation(entropies: np.ndarray, nadir_errors: np.ndarray, save_path: str):
    """Plot entropy vs error correlation scatter plot"""
    plt.figure(figsize=(10, 6))

    # Compute correlation
    correlation = np.corrcoef(entropies, nadir_errors)[0, 1]

    # Scatter plot
    plt.scatter(entropies, nadir_errors, alpha=0.4, s=20, color='darkblue', edgecolors='none')

    plt.xlabel('Posterior Entropy (nats)', fontsize=12)
    plt.ylabel('Angular Error (degrees)', fontsize=12)
    plt.title(f'Entropy vs Error Correlation (ρ = {correlation:.4f})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    print(f"Entropy-Error Correlation: ρ = {correlation:.4f}")
    plt.close()

def main():
    # Command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--max_entropy', type=float, default=4.5, help='Maximum allowed entropy for accepting estimates')
    parser.add_argument('--no_filter', action='store_true', help='Disable entropy filtering')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_csv = os.path.join(project_root, 'data/test.csv')
    lut_path = os.path.join(project_root, 'data/sensor_occlusion_lut.csv')

    # Load test data
    print(f"Loading data from {test_csv}...")
    df = pd.read_csv(test_csv)
    if args.samples: df = df.head(args.samples)

    # Initialize estimator with max_entropy if filtering is enabled
    max_entropy_threshold = None if args.no_filter else args.max_entropy
    estimator = EntropyEstimator(lut_path, max_entropy=max_entropy_threshold)

    nadir_errors = []
    entropies = []
    rejected_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Estimating"):
        # Extract data
        sun_vector = np.array([row['Sun_X'], row['Sun_Y'], row['Sun_Z']])
        nadir_actual = np.array([row['Nadir_X'], row['Nadir_Y'], row['Nadir_Z']])
        sensors = np.array([row[f'Sensor_{i}'] for i in range(1, 17)])
        albedo = row['Albedo']
        sun_nadir_angle = row['Sun_Nadir_Angle']

        # Estimate nadir (now returns entropy too)
        nadir_pred, entropy = estimator.get_nadir(sensors, sun_vector, albedo, sun_nadir_angle)

        # Check if nadir was rejected by entropy filter
        if nadir_pred is None:
            rejected_count += 1
            continue

        # Compute error
        error = compute_angular_error(nadir_pred.reshape(1, -1), nadir_actual.reshape(1, -1))[0]
        nadir_errors.append(error)
        entropies.append(entropy)

    nadir_errors = np.array(nadir_errors)
    entropies = np.array(entropies)

    # Print statistics
    print(f"\n{'='*50}")
    print(f"Nadir Estimation Results")
    print(f"{'='*50}")
    if not args.no_filter:
        print(f"Entropy Filtering: ENABLED (max_entropy={args.max_entropy})")
        print(f"Rejected Samples: {rejected_count}/{len(df)} ({100*rejected_count/len(df):.1f}%)")
        print(f"Accepted Samples: {len(nadir_errors)}/{len(df)} ({100*len(nadir_errors)/len(df):.1f}%)")
    else:
        print(f"Entropy Filtering: DISABLED")
    print(f"-"*50)
    print(f"Mean Error:   {nadir_errors.mean():.2f}°")
    print(f"Median Error: {np.median(nadir_errors):.2f}°")
    print(f"Std Dev:      {nadir_errors.std():.2f}°")
    print(f"Min Error:    {nadir_errors.min():.2f}°")
    print(f"Max Error:    {nadir_errors.max():.2f}°")
    print(f"{'='*50}\n")

    # Generate plots - use absolute path to plots folder
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("\nGenerating plots...")

    # Plot 1: Error distribution
    error_plot_path = os.path.join(plots_dir, 'nadir_error_distribution.png')
    plot_error_distribution(nadir_errors, error_plot_path)

    # Plot 2: Entropy vs Error correlation
    correlation_plot_path = os.path.join(plots_dir, 'nadir_entropy_correlation.png')
    plot_entropy_correlation(entropies, nadir_errors, correlation_plot_path)

if __name__ == '__main__':
    main()