import numpy as np
import cv2
import matplotlib.pyplot as plt
from DatasetHandler import *

def extract_features_dataset(images):
    kp_list = []
    des_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for image in images:
        kp, des = sift.detectAndCompute(image, None)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list


def match_features_dataset(des_list):
    matches = []
    FLANN_INDEX_KDTREE = 0
    index_params = dict(
        algorithm=FLANN_INDEX_KDTREE,
        trees=5
    )
    search_params = dict(
        checks=50
    )
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for des1, des2 in zip(des_list[:-1], des_list[1:]):
        matches.append(flann.knnMatch(des1, des2, k=2))

    return matches

def filter_matches_dataset(matches, dist_threshold):
    filtered_matches = []
    for match in matches:
        filtered_matches.append([f[0] for f in match if f[0].distance / f[1].distance < dist_threshold])
    return filtered_matches

def estimate_motion(match, kp1, kp2, k):
    image1_points = np.array([kp1[m.queryIdx].pt for m in match])
    image2_points = np.array([kp2[m.trainIdx].pt for m in match])
    E, _ = cv2.findEssentialMat(image1_points, image2_points, k)
    _, rmat, tvec, _ = cv2.recoverPose(E, image1_points, image2_points, k)

    return rmat, tvec

def estimate_trajectory(matches, kp_list, k):
    trajectory = [np.array([0, 0, 0])]

    T = np.eye(4)

    for i, match in enumerate(matches):
        rmat, tvec = estimate_motion(match, kp_list[i], kp_list[i + 1], k)
        Ti = np.eye(4)
        Ti[:3, :4] = np.c_[rmat.T, -rmat.T @ tvec]
        T = T @ Ti
        trajectory.append(T[:3, 3])

    return np.array(trajectory).T

def main():
    DIST_THRESHOLD = 0.6
    dataset_handler = DatasetHandler()
    images = dataset_handler.images
    kp_list, des_list = extract_features_dataset(images)
    matches = match_features_dataset(des_list)
    filtered_matches = filter_matches_dataset(matches, DIST_THRESHOLD)
    trajectory = estimate_trajectory(filtered_matches, kp_list, dataset_handler.k)
    visualize_trajectory(trajectory)

if __name__=="__main__":
    main()