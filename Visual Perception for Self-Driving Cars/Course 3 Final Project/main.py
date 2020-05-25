import numpy as np
import cv2
import matplotlib.pyplot as plt
from m6bk import *

def xy_from_depth(depth, k):
    # Get shape of depth tensor
    depth_height, depth_width = depth.shape

    # Grab required parameters from K matrix
    focal_length = k[0, 0]
    u0 = k[0, 2]
    v0 = k[1, 2]

    # Generate a grid of coordinates corresponding to the shape of depth map
    x_range = np.arange(1, depth_width+1)
    y_range = np.arange(1, depth_height+1)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Compute x and y coordinates
    x = (x_grid - u0) * depth / focal_length
    y = (y_grid - v0) * depth / focal_length

    return x, y

def ransac_plane_fit(xyz_data):
    num_itr = 100
    min_num_inliers = xyz_data.shape[1] / 2
    distance_threshold = 0.1
    largest_num_inliers = 0

    for _ in range(num_itr):
        # Choose a random of 3 points from xyz data at random
        random_points = xyz_data[:, np.random.choice(xyz_data.shape[1], 15, replace=False)]

        # Compute plane model
        plane_parameters = compute_plane(random_points)

        # Find number of inliers
        distances = np.abs(dist_to_plane(plane_parameters, xyz_data[0], xyz_data[1], xyz_data[2]))
        num_inliers = np.sum(distances < distance_threshold)

        # Check if the current number of inliers is greater than all previous iterations
        # and keep the inlier set with the largest number of points.
        if num_inliers > largest_num_inliers:
            largest_num_inliers = num_inliers
            largest_inlier_set = xyz_data[:, distances < distance_threshold]

        # Check if stopping criterion is satisfied and break
        if num_inliers > min_num_inliers:
            break

    # Recompute the model parameters using largest inlier set
    output_plane = compute_plane(largest_inlier_set)

    return output_plane

def estimate_lane_lines(segmentation_output):
    # Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    lane_boundaries = np.zeros_like(segmentation_output, dtype=np.uint8)
    lane_boundaries[segmentation_output == 6] = 255
    lane_boundaries[segmentation_output == 8] = 255

    # Perform Edge Detection
    blurred_lane_boundaries = cv2.GaussianBlur(lane_boundaries, (5, 5), 1)
    edges = cv2.Canny(blurred_lane_boundaries, 100, 110, apertureSize=3)

    # Perform Line estimation
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=25)

    lines = np.squeeze(lines, axis=1)

    return lines

def merge_lane_lines(lines):
    # Define thresholds
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3

    # Get slope and intercept of lines
    slopes, intercepts = get_slope_intecept(lines)

    # Determine lines with slope less than horizontal slope threshold
    slope_horizontal_filter = np.abs(slopes) > min_slope_threshold

    # Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    clusters = []
    current_inds = []
    itr = 0
    for slope, intercept in zip(slopes, intercepts):
        exists_in_clusters = np.array([itr in current for current in current_inds])
        if not exists_in_clusters.any():
            slope_cluster = np.logical_and(
                slopes < (slope + slope_similarity_threshold),
                slopes > (slope - slope_similarity_threshold))
            intercept_cluster = np.logical_and(
                intercepts < (intercept + intercept_similarity_threshold),
                intercepts > (intercept - intercept_similarity_threshold))
            inds = np.argwhere(
                slope_cluster & intercept_cluster & slope_horizontal_filter).T
            if inds.size:
                current_inds.append(inds.flatten())
                clusters.append(lines[inds])
        itr += 1

    # Merge all lines in clusters using mean averaging
    merged_lines = [np.mean(cluster, axis=1) for cluster in clusters]
    merged_lines = np.squeeze(np.array(merged_lines), axis=1)

    return merged_lines

def filter_detections_by_segmentation(detections, segmentation_output):
    ratio_threshold = 0.3
    filtered_detections = []

    for detection in detections:
        if detection[0] == 'Car' or detection[0]=='Cyclist':
            Class = 10
        elif detection[0] == 'Pedestrian':
            Class = 4

        x_min = int(float(detection[1]))
        y_min = int(float(detection[2]))
        x_max = int(float(detection[3]))
        y_max = int(float(detection[4]))

        # Compute number of pixels belonging to the category for every detection.
        num_pixels = (segmentation_output[y_min:y_max, x_min:x_max] == Class).sum()
        # Divide the computed number of pixels by the area of the bounding box (total number of pixels).
        box_area = (y_max - y_min) * (x_max - x_min)
        ratio = num_pixels / box_area

        # If the ratio is greater than a threshold keep the detection. Else, remove the detection from the list of detections.
        if ratio > ratio_threshold:
            filtered_detections.append(detection)

    return filtered_detections


def find_min_distance_to_detection(detections, x, y, z):

    min_distances = []
    for detection in detections:
        x_min = int(float(detection[1]))
        y_min = int(float(detection[2]))
        x_max = int(float(detection[3]))
        y_max = int(float(detection[4]))
        min_distance = np.inf
        # Step 1: Compute distance of every pixel in the detection bounds
        for y_box in range(y_min, y_max):
            for x_box in range(x_min, x_max):
                distance = np.sqrt(x[y_box, x_box] ** 2 + y[y_box, x_box] ** 2 + z[y_box, x_box] ** 2)
                # Step 2: Find minimum distance
                if distance < min_distance:
                    min_distance = distance
        min_distances.append(min_distance)

    return min_distances

def main():
    # Load dataset
    dataset_handler = DatasetHandler()
    dataset_handler.set_frame(2)
    segmentation = dataset_handler.segmentation
    detections = dataset_handler.object_detection
    z = dataset_handler.depth
    k = dataset_handler.k

    x, y = xy_from_depth(z, k)
    road_mask = np.zeros_like(segmentation)
    road_mask[segmentation == 7] = 1
    x_ground = x[road_mask == 1]
    y_groud = y[road_mask == 1]
    z_ground = z[road_mask == 1]
    xyz_ground = np.stack((x_ground, y_groud, z_ground))
    plane_parameters = ransac_plane_fit(xyz_ground)

    lane_lines = estimate_lane_lines(segmentation)
    merged_lane_lines = merge_lane_lines(lane_lines)
    max_y = dataset_handler.image.shape[0]
    min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

    extrapolated_lines = extrapolate_lines(merged_lane_lines, max_y, min_y)
    final_lanes = find_closest_lines(extrapolated_lines, dataset_handler.lane_midpoint)

    filtered_detections = filter_detections_by_segmentation(detections, segmentation)
    min_distances = find_min_distance_to_detection(filtered_detections, x, y, z)

    # Original Image
    plt.imshow(dataset_handler.image)
    plt.show()

    # Part I
    dist = np.abs(dist_to_plane(plane_parameters, x, y, z))

    ground_mask = np.zeros(dist.shape)

    ground_mask[dist < 0.1] = 1
    ground_mask[dist > 0.1] = 0

    plt.imshow(ground_mask)
    plt.show()

    plt.imshow(dataset_handler.vis_lanes(final_lanes))
    plt.show()

    # Part III
    font = {'family': 'serif', 'color': 'red', 'weight': 'normal', 'size': 12}

    im_out = dataset_handler.vis_object_detection(filtered_detections)

    for detection, min_distance in zip(filtered_detections, min_distances):
        bounding_box = np.asfarray(detection[1:5])
        plt.text(bounding_box[0], bounding_box[1] - 20, 'Distance to Impact:' + str(np.round(min_distance, 2)) + ' m',
                 fontdict=font)

    plt.imshow(im_out)
    plt.show()

if __name__ == "__main__":
    main()