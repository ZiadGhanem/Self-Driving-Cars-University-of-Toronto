import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches


def compute_left_disparity_map(img_left, img_right):
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=16 * 6,
                                   blockSize=11,
                                   P1=8 * 3 * 6 ** 2,
                                   P2=32 * 3 * 6 ** 2,
                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    disp_left = stereo.compute(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16

    return disp_left


def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = t / t[3]

    return k, r, t


def calc_depth_map(disp_left, k_left, t_left, t_right):

    # Get the focal length from the K matrix
    f = k_left[0, 0]

    # Get the distance between the cameras from the t matrices (baseline)
    b = t_left[1] - t_right[1]

    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1
    print(disp_left)
    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disp_left.shape, np.single)

    # Calculate the depths
    depth_map[:] = f * b / disp_left[:]

    return depth_map


def locate_obstacle_in_image(image, obstacle_image):
    # Run the template matching from OpenCV
    cross_corr_map = cv2.matchTemplate(image, obstacle_image, method=cv2.TM_CCOEFF)

    # Locate the position of the obstacle using the minMaxLoc function from OpenCV
    _, _, _, obstacle_location = cv2.minMaxLoc(cross_corr_map)

    return cross_corr_map, obstacle_location


def calculate_nearest_point(depth_map, obstacle_location, obstacle_img):

    # Gather the relative parameters of the obstacle box
    obstacle_width = obstacle_img.shape[0]
    obstacle_height = obstacle_img.shape[1]
    obstacle_min_x_pos = obstacle_location[1]
    obstacle_max_x_pos = obstacle_location[1] + obstacle_width
    obstacle_min_y_pos = obstacle_location[0]
    obstacle_max_y_pos = obstacle_location[0] + obstacle_height

    # Get the depth of the pixels within the bounds of the obstacle image, find the closest point in this rectangle
    obstacle_depth = depth_map[obstacle_min_x_pos:obstacle_max_x_pos, obstacle_min_y_pos:obstacle_max_y_pos]
    closest_point_depth = obstacle_depth.min()

    # Create the obstacle bounding box
    obstacle_bbox = patches.Rectangle((obstacle_min_y_pos, obstacle_min_x_pos), obstacle_height, obstacle_width,
                                      linewidth=1, edgecolor='r', facecolor='none')

    return closest_point_depth, obstacle_bbox

def main():
    p_left = np.array([[640.0, 0.0, 640.0, 2176.0],
                       [0.0, 480.0, 480.0, 552.0],
                       [0.0, 0.0, 1.0, 1.4]])
    p_right = np.array([[640.0, 0.0, 640.0, 2176.0],
                        [0.0, 480.0, 480.0, 792.0],
                        [0.0, 0.0, 1.0, 1.4]])

    img_left = cv2.imread("stereo_set/frame_00077_1547042741L.png")
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right = cv2.imread("stereo_set/frame_00077_1547042741R.png")
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

    obstacle_img = img_left[479:509, 547:593, :]

    disp_left = compute_left_disparity_map(img_left, img_right)

    k_left, r_left, t_left = decompose_projection_matrix(p_left)
    k_right, r_right, t_right = decompose_projection_matrix(p_right)

    depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)

    cross_corr_map, obstacle_location = locate_obstacle_in_image(img_left, obstacle_img)

    closest_point_depth, obstacle_bbox = calculate_nearest_point(depth_map_left, obstacle_location, obstacle_img)

    fig, ax = plt.subplots(1)
    ax.imshow(img_left)
    ax.add_patch(obstacle_bbox)
    plt.show()

    # Print the depth of the nearest point
    print("closest_point_depth {0:0.3f}".format(closest_point_depth))

if __name__ == "__main__":
    main()