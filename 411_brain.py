#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import SimpleITK as sitk
from collections import defaultdict

# Load the segmentation NIfTI file
file_path = "/Users/seandmello/Downloads/OASIS-TRT-20-10_DKT31_CMA_labels_in_MNI152.nii.gz"
image = sitk.ReadImage(file_path)

# Convert to NumPy array
data = sitk.GetArrayFromImage(image)

# Define start and end points for the beam
start_point = (0, 0, 0)  # Start (z, y, x)
end_point = (100, 218, 100)  # End (z, y, x)

# Function to perform Bresenham's line algorithm for 3D
def bresenham_line_3d(start, end):
    points = []
    x1, y1, z1 = start
    x2, y2, z2 = end

    # Calculate deltas
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    # Determine the direction of movement
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    # Driving axis is x-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            points.append((x1, y1, z1))
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz

    # Driving axis is y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            points.append((x1, y1, z1))
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz

    # Driving axis is z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            points.append((x1, y1, z1))
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx

    # Add the last point
    points.append((x2, y2, z2))
    return points

# Add surrounding points within a given radius
def add_radius_around_points(points, radius, shape):
    expanded_points = set()
    for z, y, x in points:
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dz**2 + dy**2 + dx**2 <= radius**2:  # Within the sphere
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                            expanded_points.add((nz, ny, nx))
    return expanded_points

print("Tracing the beam...")
# Trace the beam path
beam_points = bresenham_line_3d(start_point, end_point)

# Expand the beam with a radius
radius = 5  # Adjustable radius size
expanded_beam_points = add_radius_around_points(beam_points, radius, data.shape)

# Assign Gaussian intensity
sigma = 10  # Standard deviation for Gaussian
I_0 = 1.0  # Peak intensity
intensity_per_segmentation = defaultdict(float)

for z, y, x in expanded_beam_points:
    label = data[z, y, x]
    # Calculate distance from the beam center
    distances = [np.sqrt((z - bz)**2 + (y - by)**2 + (x - bx)**2) for bz, by, bx in beam_points]
    min_distance = min(distances)
    intensity = I_0 * np.exp(-min_distance**2 / (2 * sigma**2))
    intensity_per_segmentation[label] += intensity

# Output the total intensity for each segmentation
for label, intensity in intensity_per_segmentation.items():
    print(f"Segmentation {label}: Total intensity = {intensity:.2f}")


# In[ ]:




