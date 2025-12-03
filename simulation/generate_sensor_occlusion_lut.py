import bpy
import math
import mathutils
import csv
import numpy as np
import os

# ----------------------------
#   Settings
# ----------------------------

OUTPUT_FILE = "/Users/lundeencahilly/Desktop/github/cs109/project/smarter_sun_sensor/data/occlusion_lut.csv"
NUM_SAMPLES = 10000
SATELLITE_NAME = "Satellite"

# ----------------------------
#   Helper Functions
# ----------------------------

def fibonacci_sphere(samples=1000):
    """
    Generate points on a sphere using the Fibonacci spiral method.
    Returns a list of (x, y, z) tuples.
    """
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

# ----------------------------
#   Main Script
# ----------------------------

def main():
    print(f"Generating Occlusion LUT with {NUM_SAMPLES} samples...")
    
    # 1. Get Objects
    satellite = bpy.data.objects[SATELLITE_NAME]
    
    # 2. Get Sensor Data (Local Pos, Local Normal)
    # We need world positions for raycasting, but we'll compute them per ray
    # Actually, simpler: Rotate the *direction* into body frame?
    # No, the satellite might be rotated in the scene.
    # Best approach:
    # - Get sensor positions in Body Frame.
    # - We want to test directions in Body Frame.
    # - So we can place the satellite at identity rotation/location for the test?
    # - Or just transform our test vectors from Body to World.
    
    # Let's assume we want the LUT to map Body Vector -> Visibility.
    # So we generate test vectors in Body Frame.
    # Then we transform them to World Frame to cast rays.
    
    sensor_data = []
    for i in range(1, 17):
        sensor_name = f"Sensor {i}"
        if sensor_name not in bpy.data.objects:
            print(f"Error: {sensor_name} not found!")
            return
            
        sensor_obj = bpy.data.objects[sensor_name]
        
        # Local pos relative to satellite
        # P_body = M_sat^-1 @ P_world
        local_pos = satellite.matrix_world.inverted() @ sensor_obj.matrix_world.translation
        
        # We also need the sensor normal to check if it's facing away
        # Normal in body frame
        # R_sat^-1 @ R_sensor @ Z_axis
        local_normal = satellite.matrix_world.inverted().to_quaternion() @ (
            sensor_obj.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, 1))
        )
        
        sensor_data.append({
            'name': sensor_name,
            'pos_body': local_pos,
            'normal_body': local_normal.normalized()
        })
        
    # 3. Generate Directions (Body Frame)
    directions_body = fibonacci_sphere(NUM_SAMPLES)
    
    # 4. Raycast
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    results = []
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        header = ['dx', 'dy', 'dz'] + [f'vis_{i+1}' for i in range(16)]
        writer.writerow(header)
        
        for idx, (dx, dy, dz) in enumerate(directions_body):
            if idx % 100 == 0:
                print(f"Processing {idx}/{NUM_SAMPLES}...")
                
            dir_body = mathutils.Vector((dx, dy, dz)).normalized()
            
            # Transform direction to World Frame for raycasting
            # D_world = R_sat @ D_body
            dir_world = satellite.matrix_world.to_quaternion() @ dir_body
            
            row = [dx, dy, dz]
            
            for sensor in sensor_data:
                # Check 1: Is the sensor facing this direction?
                # If normal . direction < 0, it's self-occluded by the mounting face itself
                # (or just facing away).
                # Lambert's law handles the cosine falloff, but for pure binary visibility
                # we usually care about "is there an obstacle".
                # However, if it's facing away, it can't see it regardless of obstacles.
                # Let's record "geometric visibility" (raycast) AND "facing" separately?
                # The user wants to subtract the signal.
                # Signal = S0 * max(0, n.s) * Visibility.
                # So if n.s <= 0, signal is 0. Visibility doesn't matter.
                # If n.s > 0, we need to know if it's blocked.
                
                # Raycast
                # Origin: Sensor World Pos + epsilon * Normal (to avoid self-intersection)
                # Actually, origin should be Sensor World Pos.
                # But we need to avoid hitting the sensor itself?
                # Sensors are usually on the surface.
                # Let's push out slightly along the sensor normal?
                # Or just push out along the ray direction?
                
                sensor_pos_world = satellite.matrix_world @ sensor.get('pos_body')
                
                # Push out slightly to avoid self-intersection with the exact surface point
                EPS = 0.001
                ray_origin = sensor_pos_world + dir_world * EPS
                
                # Cast ray to infinity (or large distance)
                # Blender ray_cast takes origin and direction
                hit, loc, norm, index, obj, matrix = scene.ray_cast(depsgraph, ray_origin, dir_world)
                
                is_occluded = False
                if hit:
                    # If we hit the satellite itself, it's occluded
                    if obj == satellite:
                        is_occluded = True
                    # If we hit other parts (like solar panels if they are separate objects), handle that
                    # For now assuming "Satellite" object is the main occluder
                
                # Visibility is 1 if not occluded, 0 if occluded
                # Note: This does NOT account for "facing away".
                # The estimator will handle max(0, n.s).
                # We only care if *structure* blocks the view.
                vis = 0 if is_occluded else 1
                row.append(vis)
                
            writer.writerow(row)
            
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
