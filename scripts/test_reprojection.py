import numpy as np
from src.metrics.reprojection_error import compute_reprojection_error


K = np.array([
    [1000, 0, 800],
    [0, 1000, 450],
    [0, 0, 1]
])

R1 = np.eye(3)
t1 = np.zeros(3)

R2 = np.eye(3)
t2 = np.array([1, 0, 0])

points3D = [
    np.array([10, 0, 50]),
    np.array([5, 2, 40]),
]

pixels2 = [
    np.array([820, 450]),
    np.array([810, 460]),
]

error = compute_reprojection_error(
    K,
    R1,
    t1,
    R2,
    t2,
    points3D,
    pixels2
)

print("Reprojection error:", error)