import numpy as np
from src.geometry.reprojection import reprojection_error


def compute_reprojection_error(K, R1, t1, R2, t2, points3D, pixels2):

    errors = []

    for p3d, p2 in zip(points3D, pixels2):

        err = reprojection_error(
            K,
            R1,
            t1,
            R2,
            t2,
            p3d,
            p2
        )

        errors.append(err)

    return np.mean(errors)