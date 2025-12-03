import bpy
import math
import mathutils
import csv
import random
import numpy as np

# ----------------------------
#   User settings
# ----------------------------

satellite = bpy.data.objects["Satellite"]

# Get sensor positions and normals from Sensor objects in the scene
sensor_data = []
for i in range(1, 17):  # Sensors 1-16
    sensor_obj = bpy.data.objects[f"Sensor {i}"]
    # Convert world position back to local coordinates relative to satellite
    local_pos = satellite.matrix_world.inverted() @ sensor_obj.location
    # Get sensor normal (assuming sensor faces +Z in local space)
    local_normal = satellite.matrix_world.inverted().to_quaternion() @ (
        sensor_obj.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, 1))
    )
    sensor_data.append((local_pos, local_normal.normalized()))

# ----------------------------
#   Physical Constants
# ----------------------------

EARTH_RADIUS_KM = 6371.0  # Earth radius in km
ORBITAL_ALTITUDE_KM = 500.0  # Satellite altitude
ORBITAL_RADIUS_KM = EARTH_RADIUS_KM + ORBITAL_ALTITUDE_KM

# Solar constant (W/m^2) - doesn't matter for relative intensities
SOLAR_IRRADIANCE = 1361.0

# Max sensor reading
MAX_SENSOR_VALUE = 3102

# Noise parameters (standard deviations)
NOISE_STD_PYRAMID = 10.0  # First 8 sensors (pyramid)
NOISE_STD_ORTHOGONAL = 20.0  # Last 8 sensors (Y and Z axis)

# Number of Monte Carlo samples for extended Earth source
N_EARTH_SAMPLES = 100  # Increase for more accuracy, decrease for speed

# ----------------------------
#   Helper Functions
# ----------------------------

def round_small(val, tol=1e-6):
    return 0.0 if abs(val) < tol else val

def random_unit_vector():
    """Generate a random unit vector uniformly distributed on sphere"""
    theta = random.uniform(0, 2 * math.pi)
    cos_phi = random.uniform(-1, 1)
    phi = math.acos(cos_phi)

    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)

    return mathutils.Vector((x, y, z)).normalized()

def angular_separation(v1, v2):
    """Calculate angular separation in radians between two unit vectors"""
    dot = max(-1.0, min(1.0, v1.dot(v2)))
    return math.acos(dot)

def is_eclipsed(sun_dir, nadir_dir):
    """
    Check if the sun is eclipsed by Earth from satellite's perspective

    Args:
        sun_dir: unit vector pointing to sun
        nadir_dir: unit vector pointing to Earth center (nadir)

    Returns:
        True if eclipsed, False otherwise
    """
    # Angular radius of Earth as seen from satellite
    earth_angular_radius = math.asin(EARTH_RADIUS_KM / ORBITAL_RADIUS_KM)

    # Angular separation between sun and nadir
    separation = angular_separation(sun_dir, nadir_dir)

    # If sun direction is within Earth's angular disk, it's eclipsed
    return separation < earth_angular_radius

def sample_point_on_earth_disk(nadir_dir, earth_angular_radius):
    """
    Sample a random point on the visible Earth disk as seen from the satellite.

    Args:
        nadir_dir: unit vector pointing to Earth center
        earth_angular_radius: angular radius of Earth as seen from satellite

    Returns:
        unit vector pointing to a random point on Earth's visible disk
    """
    # Sample uniformly within a cone using the solid angle method
    # Random angle around the nadir axis
    azimuth = random.uniform(0, 2 * math.pi)

    # Random radial distance from nadir (uniformly distributed in solid angle)
    # This ensures uniform distribution on the spherical cap
    cos_max_angle = math.cos(earth_angular_radius)
    cos_angle = random.uniform(cos_max_angle, 1.0)
    angle = math.acos(cos_angle)

    # Create a perpendicular basis to nadir
    # Find a vector not parallel to nadir
    if abs(nadir_dir.z) < 0.9:
        arbitrary = mathutils.Vector((0, 0, 1))
    else:
        arbitrary = mathutils.Vector((1, 0, 0))

    # Construct orthonormal basis
    tangent1 = nadir_dir.cross(arbitrary).normalized()
    tangent2 = nadir_dir.cross(tangent1).normalized()

    # Construct the sample point
    sin_angle = math.sin(angle)
    sample = (
        nadir_dir * cos_angle +
        tangent1 * sin_angle * math.cos(azimuth) +
        tangent2 * sin_angle * math.sin(azimuth)
    )

    return sample.normalized()

def compute_sensor_intensity_extended_earth(sensor_pos, sensor_normal, sun_dir, nadir_dir, albedo):
    """
    Compute sensor intensity from both direct sunlight and Earth albedo.

    KEY IMPROVEMENT: Earth albedo is computed by Monte Carlo integration over
    the visible Earth disk, treating Earth as an extended source rather than
    a point source at nadir.

    - Uses sensor_normal but ignores its sign via abs(dot).
    - Occlusion is checked by raycasting.
    - If line of sight to sun/earth is blocked by the 'Satellite' mesh,
      that contribution is zero.
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    total_intensity = 0.0

    # Normalize directions
    n = sensor_normal.normalized()
    s = sun_dir.normalized()
    d = nadir_dir.normalized()

    # Satellite origin in world coords
    sat_center = satellite.matrix_world.translation

    # A direction approximately "outward" from the body, from center to sensor
    outward = (sensor_pos - sat_center).normalized()
    if outward.length == 0:
        outward = mathutils.Vector((0.0, 0.0, 1.0))  # fallback

    # Small offset distance to move just outside the body
    EPS = 0.01

    # ------------------
    # Direct sunlight (point source)
    # ------------------
    sun_dot = abs(n.dot(s))  # ignore sign of normal
    if sun_dot > 0.0:
        # Start slightly outside the satellite along the outward direction,
        # then cast toward the sun.
        ray_origin = sensor_pos + outward * EPS
        ray_dir = s

        hit, loc, norm, index, obj, matrix = scene.ray_cast(depsgraph, ray_origin, ray_dir)

        # If we hit the satellite, sun is blocked for this sensor
        if not (hit and obj == satellite):
            total_intensity += sun_dot * SOLAR_IRRADIANCE

    # ------------------
    # Earth albedo (EXTENDED SOURCE - Monte Carlo integration)
    # ------------------
    if albedo > 0.0:
        # Angular radius of Earth as seen from satellite
        earth_angular_radius = math.asin(EARTH_RADIUS_KM / ORBITAL_RADIUS_KM)

        # Monte Carlo integration: sample random points on Earth's visible disk
        albedo_contribution = 0.0
        visible_samples = 0

        for _ in range(N_EARTH_SAMPLES):
            # Sample a random point on the visible Earth disk
            earth_point_dir = sample_point_on_earth_disk(nadir_dir, earth_angular_radius)

            # Check if this point on Earth is visible to the sensor
            earth_dot = abs(n.dot(earth_point_dir))

            if earth_dot > 0.0:
                # Check occlusion
                ray_origin = sensor_pos + outward * EPS
                ray_dir = earth_point_dir

                hit, loc, norm, index, obj, matrix = scene.ray_cast(depsgraph, ray_origin, ray_dir)

                if not (hit and obj == satellite):
                    # This point on Earth is visible to the sensor
                    # The Earth reflects sunlight according to Lambert's law
                    # Intensity from this patch depends on:
                    # 1. How much sunlight hits this part of Earth: max(0, -earth_point_dir.dot(s))
                    # 2. The sensor's view angle: earth_dot
                    # 3. The albedo factor

                    sun_on_earth = max(0.0, -earth_point_dir.dot(s))

                    if sun_on_earth > 0.0:
                        albedo_contribution += earth_dot * sun_on_earth
                        visible_samples += 1

        # Average the contributions and scale by albedo and solar irradiance
        if visible_samples > 0:
            # Solid angle of Earth
            earth_solid_angle = 2 * math.pi * (1 - math.cos(earth_angular_radius))

            # Average contribution per sample
            avg_contribution = albedo_contribution / N_EARTH_SAMPLES

            # Scale by solid angle and albedo
            # The factor of π accounts for Lambertian reflection
            albedo_intensity = avg_contribution * albedo * SOLAR_IRRADIANCE * earth_solid_angle / math.pi

            total_intensity += albedo_intensity

    # Normalize to sensor range [0, MAX_SENSOR_VALUE]
    normalized_intensity = (total_intensity / SOLAR_IRRADIANCE) * MAX_SENSOR_VALUE

    # Clamp to max value
    return min(normalized_intensity, MAX_SENSOR_VALUE)

def add_sensor_noise(sensor_intensities):
    """
    Add Gaussian noise to sensor readings.

    First 8 sensors (pyramid): 10.0 std
    Last 8 sensors (orthogonal): 20.0 std

    Args:
        sensor_intensities: list of 16 sensor intensities

    Returns:
        list of 16 sensor intensities with noise added
    """
    noisy_intensities = []

    for i, intensity in enumerate(sensor_intensities):
        if i < 8:
            # Pyramid sensors (0-7)
            noise_std = NOISE_STD_PYRAMID
        else:
            # Orthogonal sensors (8-15)
            noise_std = NOISE_STD_ORTHOGONAL

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std)
        noisy_intensity = intensity + noise

        # Clamp to valid range [0, MAX_SENSOR_VALUE]
        noisy_intensity = max(0.0, min(noisy_intensity, MAX_SENSOR_VALUE))

        noisy_intensities.append(int(round(noisy_intensity)))

    return noisy_intensities

# ----------------------------
#   Main Simulation
# ----------------------------

def generate_dataset(n_samples, output_csv):
    """
    Generate dataset with random sun and nadir vectors

    IMPROVEMENTS:
    1. Earth albedo computed as extended source (Monte Carlo integration)
    2. Gaussian noise added to all sensor readings (10 std for pyramid, 20 std for orthogonal)

    Args:
        n_samples: number of samples to generate
        output_csv: output file path
    """
    print(f"Generating {n_samples} samples with extended Earth source and noise...")
    print(f"  Earth sampling: {N_EARTH_SAMPLES} Monte Carlo samples per sensor")
    print(f"  Noise: {NOISE_STD_PYRAMID} std (pyramid), {NOISE_STD_ORTHOGONAL} std (orthogonal)")

    # Compute sensor world positions and normals
    world_sensor_data = []
    for sensor_pos, sensor_normal in sensor_data:
        world_pos = satellite.matrix_world @ sensor_pos
        world_normal = satellite.matrix_world.to_quaternion() @ sensor_normal
        world_sensor_data.append((world_pos, world_normal.normalized()))

    samples_generated = 0

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Header
        header = ["Sun_X", "Sun_Y", "Sun_Z", "Nadir_X", "Nadir_Y", "Nadir_Z"]
        header += [f"Sensor_{i+1}" for i in range(len(sensor_data))]
        header += ["Albedo", "Sun_Nadir_Angle"]
        writer.writerow(header)

        while samples_generated < n_samples:
            # Generate random sun direction
            sun_dir = random_unit_vector()

            # Generate random nadir direction
            nadir_dir = random_unit_vector()

            # Check if eclipsed - skip if true
            if is_eclipsed(sun_dir, nadir_dir):
                continue

            # Random Earth albedo between 0.1 and 0.7
            albedo = random.uniform(0.1, 0.7)

            # Compute sensor intensities (WITHOUT noise first)
            sensor_intensities = []
            for sensor_pos, sensor_normal in world_sensor_data:
                intensity = compute_sensor_intensity_extended_earth(
                    sensor_pos, sensor_normal, sun_dir, nadir_dir, albedo
                )
                sensor_intensities.append(intensity)

            # Add Gaussian noise to all sensors
            sensor_intensities = add_sensor_noise(sensor_intensities)

            # Round small values in sun and nadir vectors
            sun_rounded = [
                round_small(sun_dir.x),
                round_small(sun_dir.y),
                round_small(sun_dir.z),
            ]
            nadir_rounded = [
                round_small(nadir_dir.x),
                round_small(nadir_dir.y),
                round_small(nadir_dir.z),
            ]

            # Calculate sun-nadir angle in degrees
            sun_nadir_angle = math.degrees(angular_separation(sun_dir, nadir_dir))

            # Write row: sun, nadir, sensors, albedo, angle
            row = (
                sun_rounded
                + nadir_rounded
                + sensor_intensities
                + [round(albedo, 4), round(sun_nadir_angle, 2)]
            )
            writer.writerow(row)

            samples_generated += 1

            if samples_generated % 100 == 0:
                print(f"  Generated {samples_generated}/{n_samples} samples...")

    print(f"Done! Dataset saved to {output_csv}")

# ----------------------------
#   Run Simulation
# ----------------------------

# Generate training set
output_train = "/Users/lundeencahilly/Desktop/github/cs109/project/smarter_sun_sensor/data/train_extended.csv"
generate_dataset(50000, output_train)

# Generate test set
output_test = "/Users/lundeencahilly/Desktop/github/cs109/project/smarter_sun_sensor/data/test_extended.csv"
generate_dataset(10000, output_test)

print("\nAll done!")
