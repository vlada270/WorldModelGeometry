import numpy as np


def quaternion_to_rotation(q):
    """
    Convert quaternion [w, x, y, z] to rotation matrix
    """
    w, x, y, z = q

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    return R


def project_point(K, R, t, point3D):
    """
    Project 3D point into image
    """

    point_cam = R @ point3D + t
    point_img = K @ point_cam

    point_img /= point_img[2]

    return point_img[:2]


def reprojection_error(K, R1, t1, R2, t2, point3D, pixel2):
    """
    Compute reprojection error
    """

    pred_pixel = project_point(K, R2, t2, point3D)

    error = np.linalg.norm(pred_pixel - pixel2)

    return error