from multiprocessing import Pool
import numpy as np
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk
import random

np.random.seed(42)

def objective_function_with_dv_penalty(
        weights, intensity_matrix, tumor_voxels, normal_voxels, segmentation_map_data, 
        min_tumor_dose, max_tumor_dose, dose_threshold, dv_fraction, min_max_penalty_weight, dv_penalty_weight
    ):
        """
        Objective function with penalties for violating tumor dose and dose-volume constraints.

        Args:
            weights (np.ndarray): Beam weights.
            intensity_matrix (np.ndarray): 2D array (num_beams, num_voxels).
            tumor_voxels (np.ndarray): Boolean mask for tumor voxels.
            normal_voxels (np.ndarray): Flattened indices of normal tissue voxels.
            segmentation_map_data (np.ndarray): Segmentation map identifying voxel classes.
            min_tumor_dose (float): Minimum dose for tumor tissue.
            max_tumor_dose (float): Maximum dose for tumor tissue.
            dose_threshold (float): Dose threshold for DV constraint.
            dv_fraction (float): Maximum allowed fraction of overdosed voxels in each class.
            min_max_penalty_weight (float): Weight for penalty terms.

        Returns:
            float: Penalized objective value.
        """
        # Dose calculations
        dose_to_tumor = np.dot(weights, intensity_matrix[:, tumor_voxels])
        dose_to_normal = np.dot(weights, intensity_matrix[:, normal_voxels])

        # Penalty for tumor dose violations
        min_max_penalty = 0
        dv_penalty = 0

        if np.min(dose_to_tumor) < min_tumor_dose:
            min_max_penalty += (min_tumor_dose - np.min(dose_to_tumor)) ** 2
        if np.max(dose_to_tumor) > max_tumor_dose:
            min_max_penalty += (np.max(dose_to_tumor) - max_tumor_dose) ** 2

        # Dose-volume constraint penalty
        for cls in np.unique(segmentation_map_data):
            # Find indices of normal voxels in the current class
            class_voxels = np.where(segmentation_map_data.flatten() == cls)[0]
            class_normal_voxels = np.intersect1d(class_voxels, normal_voxels)

            if len(class_normal_voxels) > 0:
                doses = np.dot(weights, intensity_matrix[:, class_normal_voxels])
                overdosed_voxels = np.sum(doses > dose_threshold)
                max_allowed_overdosed = int(dv_fraction * len(class_normal_voxels))
                
                # Penalize violations of the DV constraint
                if overdosed_voxels > max_allowed_overdosed:
                    dv_penalty += (overdosed_voxels - max_allowed_overdosed) ** 2

        # Objective: minimize normal tissue dose + penalties
        return np.sum(dose_to_normal) + min_max_penalty_weight * min_max_penalty + dv_penalty_weight * dv_penalty

# Move this function to the global scope
def evaluate_particle(args):
    """
    Wrapper function for parallel evaluation of particles.
    """
    particle, intensity_matrix, tumor_voxels, normal_voxels, segmentation_map_data, \
        min_tumor_dose, max_tumor_dose, dose_threshold, dv_fraction, \
        min_max_penalty_weight, dv_penalty_weight = args

    return objective_function_with_dv_penalty(
        particle, intensity_matrix, tumor_voxels, normal_voxels, segmentation_map_data,
        min_tumor_dose, max_tumor_dose, dose_threshold, dv_fraction, min_max_penalty_weight, dv_penalty_weight
    )

# Example usage
if __name__ == "__main__":
    # Load the segmentation data
    segmentation_map_file_path = "OASIS-TRT-20-10_DKT31_CMA_labels_in_MNI152.nii.gz"
    segmentation_map_image = sitk.ReadImage(segmentation_map_file_path)
    segmentation_map_data = sitk.GetArrayFromImage(segmentation_map_image) # np.unique(segmentation_map_data) == [0,4,5,6,7,10,11,12,13,14,15,16,...,2030,2031,2034,2035]

    # Load the label map of brain
    brain_mask_file_path = "Segmentation_8-BrainLabelMap-label_1.nrrd"
    brain_mask_image = sitk.ReadImage(brain_mask_file_path)
    brain_mask_data = sitk.GetArrayFromImage(brain_mask_image) # np.unique(brain_mask_data) == [0,1]

    def add_tumor_to_data(_segmentation_map_data: np.ndarray, _brain_mask_data: np.ndarray, tumor_size: int = None):
        # Find all brain coordinates (label == 1)
        brain_coords = np.argwhere(_brain_mask_data == 1)  # Assuming 1 represents brain tissue in your segmentation
        
        # Randomly select a point as the tumor center
        tumor_center = tuple(brain_coords[np.random.randint(len(brain_coords))])
        
        # Get the shape of the data matrix
        shape = _segmentation_map_data.shape
        
        # Initialize tumor voxels set and add the center voxel
        tumor_voxels = set()
        tumor_voxels.add(tumor_center)
        _segmentation_map_data[tumor_center] = 999  # Assign label 999 for the tumor

        # Maintain a set of valid neighbors
        valid_neighbors = set()
        
        # All 26 directions (adjacent + diagonal neighbors)
        directions = [
            (-1, 0, 0), (1, 0, 0),  # x-axis neighbors
            (0, -1, 0), (0, 1, 0),  # y-axis neighbors
            (0, 0, -1), (0, 0, 1),  # z-axis neighbors
            (-1, -1, 0), (1, 1, 0),  # diagonal in xy-plane
            (-1, 0, -1), (1, 0, 1),  # diagonal in xz-plane
            (0, -1, -1), (0, 1, 1),  # diagonal in yz-plane
            (-1, -1, -1), (1, 1, 1), # diagonal in all 3 planes
            (-1, 1, -1), (1, -1, 1), # diagonal in all 3 planes
            (-1, 1, 0), (1, -1, 0),  # diagonal x-y plane
            (0, -1, 1), (0, 1, -1),  # diagonal y-z plane
            (-1, 0, 1), (1, 0, -1),  # diagonal x-z plane
        ]

        # Add initial neighbors of the tumor center
        for direction in directions:
            neighbor = tuple(np.array(tumor_center) + np.array(direction))
            z, y, x = neighbor
            if (
                0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]
                and _brain_mask_data[z, y, x] == 1
            ):
                valid_neighbors.add(neighbor)
        
        # Function to calculate distance from the tumor center
        def distance_from_center(voxel):
            return np.sqrt((voxel[0] - tumor_center[0]) ** 2 + (voxel[1] - tumor_center[1]) ** 2 + (voxel[2] - tumor_center[2]) ** 2)
        
        # Expand the tumor until the desired size is reached
        while len(tumor_voxels) < tumor_size if tumor_size else True:
            if not valid_neighbors:
                break  # Stop if no valid neighbors are left

            # Select a neighbor to expand to, preferring closer ones
            distances = {neighbor: distance_from_center(neighbor) for neighbor in valid_neighbors}
            min_distance = min(distances.values())
            close_neighbors = [neighbor for neighbor, dist in distances.items() if dist == min_distance]
            new_voxel = random.choice(close_neighbors)

            # Add the new voxel to the tumor
            tumor_voxels.add(new_voxel)
            _segmentation_map_data[new_voxel] = 999  # Assign label 999 for the tumor
            valid_neighbors.remove(new_voxel)

            # Add its neighbors to the valid neighbors list
            for direction in directions:
                neighbor = tuple(np.array(new_voxel) + np.array(direction))
                z, y, x = neighbor
                if (
                    0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]
                    and _brain_mask_data[z, y, x] == 1
                    and neighbor not in tumor_voxels
                ):
                    valid_neighbors.add(neighbor)
        
        return _segmentation_map_data, tumor_center

    def create_single_sided_beams(tumor_center, _brain_mask_data: np.ndarray, num_beams=10, distance_threshold=2):
        """
        Generate single-sided beams originating from points outside the brain and directed toward the tumor center.
        Each beam has a random direction.
        """
        beams = []
        data_shape = _brain_mask_data.shape
        distance_to_brain = np.array(distance_transform_edt(1 - _brain_mask_data)).astype(float)
        for i in range(num_beams):
            valid_start = False
            while not valid_start:
                # Randomly generate beam starting positions within the data shape
                start = np.array([
                    np.random.randint(0, data_shape[0]),  # Random x position
                    np.random.randint(0, data_shape[1]),  # Random y position
                    np.random.randint(0, data_shape[2]),  # Random z position
                ])

                # Check if the point is outside the brain (_brain_mask_data == 0)
                if _brain_mask_data[start[0], start[1], start[2]] == 0:
                    if distance_to_brain[start[0], start[1], start[2]] <= distance_threshold:
                        valid_start = True 

            # Direction is towards the tumor center
            direction = tumor_center - start
            direction = direction / np.linalg.norm(direction)  # Normalize direction vector
            
            beams.append({"start": start, "direction": direction})
        
        return beams

    def longitudinal_intensity_profile(tumor_center, projected_position, beam_width_longitudinal_std):
        """
        Customizable intensity profile in the longitudinal direction.
        Uses the Euclidean distance between the projected position and the tumor center.
        """
        distance = np.linalg.norm(projected_position - tumor_center)  # Euclidean distance
        intensity = np.exp(-0.5 * (distance / beam_width_longitudinal_std) ** 2)  # Gaussian profile
        return intensity

    def transverse_intensity_profile(voxel, projected_position, transverse_std_dev):
        """
        Compute the Gaussian intensity in the transverse plane defined by the beam direction.
        The intensity is based on the perpendicular distance from the beam direction.
        """
        perpendicular_distance = np.linalg.norm(voxel - projected_position)
        intensity = np.exp(-0.5 * (perpendicular_distance / transverse_std_dev) ** 2)
        return intensity

    def compute_intensity_matrix(beams, _brain_mask_data: np.ndarray, tumor_center, transverse_std_dev=0.5, beam_width_longitudinal_std=0.5):
        """
        Compute the intensity matrix for all voxels, considering both transverse and longitudinal intensity profiles.
        """
        data_shape = _brain_mask_data.shape
        intensity_matrix = np.zeros((len(beams), np.prod(data_shape)))  # Flattened voxel space
        for beam_index, beam in enumerate(beams):
            print(f'Beam Index: {beam_index}')
            start = np.array(beam["start"])
            direction = np.array(beam["direction"]) / np.linalg.norm(beam["direction"])
            
            # Loop over all voxels in the grid
            for z in range(data_shape[0]):
                for y in range(data_shape[1]):
                    for x in range(data_shape[2]):
                        if _brain_mask_data[z, y, x] == 0:
                            continue
                        voxel = np.array([z, y, x])
                        # Vector from the beam start to the voxel
                        relative_position = voxel - start
                        projection_length = np.dot(relative_position, direction)
                        if projection_length <= 0:
                            continue
                        projected_position = start + projection_length * direction

                        # Compute transverse and longitudinal intensities
                        intensity_transverse = transverse_intensity_profile(voxel, projected_position, transverse_std_dev)
                        intensity_longitudinal = longitudinal_intensity_profile(tumor_center, projected_position, beam_width_longitudinal_std)
                        if intensity_transverse < 0.01 or intensity_longitudinal < 0.01: continue
                        # Combine intensities
                        intensity = intensity_transverse * intensity_longitudinal
                        intensity_matrix[beam_index, z * data_shape[1] * data_shape[2] + y * data_shape[2] + x] = intensity

        return intensity_matrix

    num_beams = 10
    segmentation_map_data_with_tumor, tumor_center = add_tumor_to_data(segmentation_map_data, brain_mask_data, tumor_size=50)

    # Define tumor and normal tissue masks
    tumor_mask = (segmentation_map_data_with_tumor == 999)
    normal_mask = (segmentation_map_data_with_tumor != 0) & (segmentation_map_data_with_tumor != 999)

    tumor_voxels = np.where(tumor_mask.flatten())[0]
    normal_voxels = np.where(normal_mask.flatten())[0]

    # Generate beams
    beams = create_single_sided_beams(tumor_center, brain_mask_data, num_beams=num_beams)

    # Compute intensity matrix
    intensity_matrix = compute_intensity_matrix(beams, brain_mask_data, tumor_center, transverse_std_dev=1, beam_width_longitudinal_std=50)

    # Main function for PSO optimization
    def solve_beam_optimization_with_dv_PSO_jupyter(
        intensity_matrix, tumor_voxels, normal_voxels, segmentation_map_data_with_tumor,
        num_particles=30, num_iterations=100, max_beam_weight=1,
        min_tumor_dose=9.8, max_tumor_dose=10.2, dose_threshold=2.5, dv_fraction=0.75
    ):
        num_beams = intensity_matrix.shape[0]

        # Initialize particles and velocities
        particles = np.random.uniform(0, max_beam_weight, size=(num_particles, num_beams))
        velocities = np.zeros_like(particles)
        personal_best = particles.copy()
        personal_best_scores = np.full(num_particles, np.inf)
        global_best = None
        global_best_score = np.inf

        # PSO hyperparameters
        inertia = 1
        cognitive = 1
        social = 1
        min_max_penalty_weight = 100
        dv_penalty_weight = 100

        for iteration in range(num_iterations):
            # Prepare arguments for parallel evaluation
            args = [
                (
                    particle, intensity_matrix, tumor_voxels, normal_voxels, 
                    segmentation_map_data_with_tumor, min_tumor_dose, max_tumor_dose, 
                    dose_threshold, dv_fraction, min_max_penalty_weight, dv_penalty_weight
                ) for particle in particles
            ]

            # Use multiprocessing pool for parallel evaluation
            with Pool() as pool:
                scores = pool.map(evaluate_particle, args)

            # Update personal best and global best
            for i, score in enumerate(scores):
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = particles[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best = particles[i]

            # Update particles
            for i in range(num_particles):
                r1 = np.random.random(size=num_beams)
                r2 = np.random.random(size=num_beams)
                velocities[i] = (
                    inertia * velocities[i]
                    + cognitive * r1 * (personal_best[i] - particles[i])
                    + social * r2 * (global_best - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], 0, max_beam_weight)

            # Print progress
            print(f"Iteration {iteration + 1}/{num_iterations}, Best Score: {global_best_score}")

        return global_best

    
    # Example data for testing
    optimized_beam_weights = solve_beam_optimization_with_dv_PSO_jupyter(
        intensity_matrix=intensity_matrix,
        tumor_voxels=tumor_voxels,
        normal_voxels=normal_voxels,
        segmentation_map_data_with_tumor=segmentation_map_data_with_tumor,
    )

    print("Optimized beam weights:", optimized_beam_weights)
