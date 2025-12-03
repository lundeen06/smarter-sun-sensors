"""
Generate synthetic sensor data using the LUT-based physics model.

This matches the estimator's physics model exactly, so there's no model mismatch.
Much faster than Blender simulation since it uses KDTree lookups instead of raycasting.
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==================
#   CONSTANTS
# ==================

SQRT_2_INV = 1.0 / np.sqrt(2.0)

# Satellite altitude (m)
ALTITUDE = 500e3

# Earth radius (m)
EARTH_RADIUS = 6371e3

# Maximum sensor value
MAX_SENSOR_VALUE = 4095.0

# Satellite sensor normal vectors (body frame)
SENSOR_NORMALS = np.array([
    [SQRT_2_INV, 0, SQRT_2_INV], [SQRT_2_INV, SQRT_2_INV, 0], [SQRT_2_INV, 0, -SQRT_2_INV], [SQRT_2_INV, -SQRT_2_INV, 0],
    [-SQRT_2_INV, 0, SQRT_2_INV], [-SQRT_2_INV, -SQRT_2_INV, 0], [-SQRT_2_INV, 0, -SQRT_2_INV], [-SQRT_2_INV, SQRT_2_INV, 0],
    [0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, -1],
])

# Sensor noise standard deviations
STD_SUN_SENSORS_PYRAMID = 10  # First 8 sensors
STD_SUN_SENSORS_YZ = 20       # Last 8 sensors

# Earth angular radius at 500km altitude
EARTH_ANGULAR_RADIUS = np.radians(62.0)
EARTH_SOLID_ANGLE = 2 * np.pi * (1 - np.cos(EARTH_ANGULAR_RADIUS))

# ==================
#   PHYSICS MODEL
# ==================

class SensorSimulator:
    def __init__(self, lut_path: str, num_samples_earth: int = 100):
        """Initialize simulator with occlusion LUT"""
        self.num_samples_earth = num_samples_earth

        # Load LUT
        if not os.path.exists(lut_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            lut_path = os.path.join(project_root, lut_path)

        print(f"Loading Occlusion LUT from {lut_path}...")
        df = pd.read_csv(lut_path)
        self.lut_points = df[['dx', 'dy', 'dz']].values
        self.lut_visibility = df[[f'vis_{i+1}' for i in range(16)]].values
        self.kdtree = KDTree(self.lut_points)

        # Precompute deterministic disk samples (Fibonacci spiral for better coverage)
        indices = np.arange(0, num_samples_earth, dtype=float) + 0.5
        r = np.sqrt(indices / num_samples_earth)
        theta = np.pi * (1 + 5**0.5) * indices
        self.disk_samples = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    def get_visibility(self, light_vector):
        """Query LUT for visibility of 16 sensors to given light direction"""
        _, idx = self.kdtree.query(light_vector)
        return self.lut_visibility[idx]

    def sample_earth_disk(self, nadir_vector):
        """Sample points on visible Earth disk"""
        # Basis vectors
        z_axis = nadir_vector
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-6:
            x_axis = np.cross(np.array([0, 1, 0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Project disk samples
        sin_alpha = np.sin(EARTH_ANGULAR_RADIUS) * self.disk_samples[:, 0]
        cos_alpha = np.sin(EARTH_ANGULAR_RADIUS) * self.disk_samples[:, 1]
        z_comp = np.sqrt(1 - sin_alpha**2 - cos_alpha**2)

        points = (x_axis * sin_alpha[:, np.newaxis] +
                  y_axis * cos_alpha[:, np.newaxis] +
                  z_axis * z_comp[:, np.newaxis])
        return points

    def get_sun_signal(self, sun_vector):
        """Compute direct sun signal for all 16 sensors"""
        sun_vector = sun_vector / np.linalg.norm(sun_vector)
        sensor_view_angles = np.maximum(0, np.dot(SENSOR_NORMALS, sun_vector))
        visibility = self.get_visibility(sun_vector)
        I_sun = sensor_view_angles * visibility * MAX_SENSOR_VALUE
        return I_sun

    def get_earth_signal(self, nadir_vector, sun_vector, albedo):
        """Compute Earth albedo signal for all 16 sensors"""
        earth_directions = self.sample_earth_disk(nadir_vector)
        M = len(earth_directions)

        # Sunlight hitting each Earth point
        sun_on_earth = np.maximum(0, -np.dot(earth_directions, sun_vector))

        # Sensor view angles
        sensor_view_angles = np.maximum(0, np.dot(earth_directions, SENSOR_NORMALS.T))

        # Visibility for each Earth point
        visibility = np.zeros((M, 16))
        for i in range(M):
            visibility[i] = self.get_visibility(earth_directions[i])

        # Contributions (Lambertian BRDF)
        contributions = (sun_on_earth[:, np.newaxis] *
                        sensor_view_angles *
                        visibility)

        # Monte Carlo integration
        avg_contribution = np.mean(contributions, axis=0)

        # Scale by albedo and solid angle
        I_earth = avg_contribution * albedo * (EARTH_SOLID_ANGLE / np.pi) * MAX_SENSOR_VALUE
        return I_earth

    def simulate_sensors(self, sun_vector, nadir_vector, albedo, add_noise=True):
        """
        Simulate all 16 sensor readings

        Args:
            sun_vector: (3,) unit vector pointing to sun
            nadir_vector: (3,) unit vector pointing to Earth center
            albedo: Earth albedo (0-1)
            add_noise: whether to add Gaussian noise

        Returns:
            sensors: (16,) array of sensor readings
        """
        # Normalize inputs
        sun_vector = sun_vector / np.linalg.norm(sun_vector)
        nadir_vector = nadir_vector / np.linalg.norm(nadir_vector)

        # Compute total signal
        I_sun = self.get_sun_signal(sun_vector)
        I_earth = self.get_earth_signal(nadir_vector, sun_vector, albedo)
        I_total = I_sun + I_earth

        # Add noise if requested
        if add_noise:
            noise = np.concatenate([
                np.random.normal(0, STD_SUN_SENSORS_PYRAMID, 8),
                np.random.normal(0, STD_SUN_SENSORS_YZ, 8)
            ])
            I_total = I_total + noise

        # Clamp to valid range
        I_total = np.clip(I_total, 0, MAX_SENSOR_VALUE)

        return I_total

# ==================
#   DATA GENERATION
# ==================

def random_unit_vector():
    """Generate random unit vector uniformly on sphere"""
    theta = np.random.uniform(0, 2*np.pi)
    cos_phi = np.random.uniform(-1, 1)
    phi = np.arccos(cos_phi)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.array([x, y, z])

def random_quaternion():
    """Generate random unit quaternion (uniform on SO(3))"""
    # Using Shoemake's method for uniform random rotation
    u1, u2, u3 = np.random.uniform(0, 1, 3)

    q = np.array([
        np.sqrt(1-u1) * np.sin(2*np.pi*u2),
        np.sqrt(1-u1) * np.cos(2*np.pi*u2),
        np.sqrt(u1) * np.sin(2*np.pi*u3),
        np.sqrt(u1) * np.cos(2*np.pi*u3)
    ])

    return q  # (x, y, z, w) scipy convention

def is_eclipsed(sun_vector, nadir_vector):
    """Check if sun is eclipsed by Earth"""
    # Angular separation between sun and nadir
    cos_sep = np.dot(sun_vector, nadir_vector)
    cos_sep = np.clip(cos_sep, -1, 1)
    separation = np.arccos(cos_sep)

    # If separation is less than Earth's angular radius, sun is eclipsed
    return separation < EARTH_ANGULAR_RADIUS

def generate_dataset(simulator, n_samples, output_path, min_albedo=0.1, max_albedo=0.7):
    """
    Generate synthetic dataset with full attitude information

    The generation process:
    1. Generate random sun/nadir vectors in ECI frame
    2. Generate random attitude quaternion
    3. Rotate ECI vectors to body frame using quaternion
    4. Simulate sensor readings from body-frame vectors
    5. Save both ECI and body-frame vectors, plus quaternion

    Args:
        simulator: SensorSimulator instance
        n_samples: number of samples to generate
        output_path: path to save CSV
        min_albedo: minimum Earth albedo
        max_albedo: maximum Earth albedo
    """
    print(f"\nGenerating {n_samples} samples...")
    print(f"  Earth samples per sensor: {simulator.num_samples_earth}")
    print(f"  Albedo range: [{min_albedo:.2f}, {max_albedo:.2f}]")
    print(f"  Noise: {STD_SUN_SENSORS_PYRAMID} std (sensors 1-8), {STD_SUN_SENSORS_YZ} std (sensors 9-16)")
    print(f"  Includes: Quaternions and ECI reference vectors for MLE attitude estimation")

    data = []
    samples_generated = 0

    pbar = tqdm(total=n_samples, desc="Generating")

    while samples_generated < n_samples:
        # ===== ECI Frame =====
        # Generate random sun and nadir vectors in ECI frame
        sun_eci = random_unit_vector()
        nadir_eci = random_unit_vector()

        # Skip if eclipsed (check in ECI frame)
        if is_eclipsed(sun_eci, nadir_eci):
            continue

        # ===== Attitude Quaternion =====
        # Generate random attitude (rotation from ECI to Body frame)
        q_xyzw = random_quaternion()  # scipy format (x,y,z,w)
        rot = R.from_quat(q_xyzw)
        R_eci_to_body = rot.as_matrix()

        # Convert to (w,x,y,z) format for storage
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        # ===== Body Frame =====
        # Rotate ECI vectors to body frame
        sun_body = R_eci_to_body @ sun_eci
        nadir_body = R_eci_to_body @ nadir_eci

        # Random albedo
        albedo = np.random.uniform(min_albedo, max_albedo)

        # ===== Sensor Simulation =====
        # Simulate sensors using body-frame vectors
        sensors = simulator.simulate_sensors(sun_body, nadir_body, albedo, add_noise=True)

        # Calculate sun-nadir angle (same in both frames)
        cos_angle = np.dot(sun_eci, nadir_eci)
        cos_angle = np.clip(cos_angle, -1, 1)
        sun_nadir_angle = np.degrees(np.arccos(cos_angle))

        # ===== Store Sample =====
        row = {
            # Body frame vectors (what sensors measure)
            'Sun_X': sun_body[0],
            'Sun_Y': sun_body[1],
            'Sun_Z': sun_body[2],
            'Nadir_X': nadir_body[0],
            'Nadir_Y': nadir_body[1],
            'Nadir_Z': nadir_body[2],

            # ECI frame reference vectors (ground truth)
            'Sun_ECI_X': sun_eci[0],
            'Sun_ECI_Y': sun_eci[1],
            'Sun_ECI_Z': sun_eci[2],
            'Nadir_ECI_X': nadir_eci[0],
            'Nadir_ECI_Y': nadir_eci[1],
            'Nadir_ECI_Z': nadir_eci[2],

            # Attitude quaternion (w, x, y, z)
            'Quat_W': q_wxyz[0],
            'Quat_X': q_wxyz[1],
            'Quat_Y': q_wxyz[2],
            'Quat_Z': q_wxyz[3],
        }

        # Sensor readings
        for i in range(16):
            row[f'Sensor_{i+1}'] = int(round(sensors[i]))

        row['Albedo'] = round(albedo, 4)
        row['Sun_Nadir_Angle'] = round(sun_nadir_angle, 2)

        data.append(row)
        samples_generated += 1
        pbar.update(1)

    pbar.close()

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Albedo: mean={df['Albedo'].mean():.3f}, std={df['Albedo'].std():.3f}")
    print(f"  Sun-Nadir angle: mean={df['Sun_Nadir_Angle'].mean():.1f}°, std={df['Sun_Nadir_Angle'].std():.1f}°")

# ==================
#   MAIN
# ==================

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    lut_path = os.path.join(project_root, 'data/sensor_occlusion_lut.csv')

    # Check if LUT exists
    if not os.path.exists(lut_path):
        print(f"ERROR: Occlusion LUT not found at {lut_path}")
        print("Please generate it first using generate_sensor_occlusion_lut.py")
        sys.exit(1)

    # Initialize simulator
    simulator = SensorSimulator(lut_path, num_samples_earth=100)

    # Generate training set
    train_path = os.path.join(project_root, 'data/train.csv')
    generate_dataset(simulator, n_samples=10000, output_path=train_path)

    # Generate test set
    test_path = os.path.join(project_root, 'data/test.csv')
    generate_dataset(simulator, n_samples=100000, output_path=test_path)

    print("\nAll done!")
