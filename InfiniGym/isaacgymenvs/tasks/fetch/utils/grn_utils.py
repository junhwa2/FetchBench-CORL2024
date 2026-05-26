import math
import numpy as np

def compute_distance(pose1, pose2):
    return np.linalg.norm(np.array(pose1) - np.array(pose2))

def compute_threshold(dim1, dim2):
    return (max(dim1) + max(dim2)+0.6) / 2

def is_neighbor(dim1, dim2, pose1, pose2):
    distance = compute_distance(pose1[:2], pose2[:2])
    threshold = compute_threshold(dim1[:2], dim2[:2])
    if distance > threshold:
        return False
    else:
        return True
