import numpy as np
import mujoco
from ctypes import c_int, addressof
from math import sqrt


def raycast(model: mujoco.MjModel, data: mujoco.MjData, geom1_id=None, geom2_id=None, pt1=None, pt2=None, geom_group=None) -> (float, c_int):
    """

    Args:
        model: Mujoco model object
        data: Mujoco data object
        geom1_id (int): id of geom ray originates from
        geom2_id (int): id of geom ray points to
        pt1 (np.ndarray[3]): 3D point ray originates from
        pt2 (np.ndarray[3]): 3D point ray points to
        geom_group: one-hot list determining which of the five geom groups should be visible to the raycast

    Returns:
        TODO: find return types
    """
    assert (geom1_id is None) != (pt1 is None), "geom1_id or p1 must be specified"
    assert (geom2_id is None) != (pt2 is None), "geom2_id or p2 must be specified"
    if geom1_id is not None:
        pt1 = data.geom_xpos[geom1_id]
        body1 = model.geom_bodyid[geom1_id]
    else:
        # Don't exclude any bodies if we originate ray from a point
        body1 = np.max(model.geom_bodyid) + 1
    if geom2_id is not None:
        pt2 = data.geom_xpos[geom2_id]

    ray_direction = pt2 - pt1
    ray_direction /= sqrt(ray_direction[0] ** 2 + ray_direction[1] ** 2 + ray_direction[2] ** 2)

    if geom_group is not None:
        geom_group = np.array(geom_group).astype(np.uint8)
    else:
        geom_group = np.array([1, 1, 1, 1, 1]).astype(np.uint8)  # This is the default geom group

    # Setup int array
    c_arr = (c_int*1)(0)
    dist: float = mujoco.mj_ray(model,
                            data,
                            pt1,
                            ray_direction,
                            geom_group,
                            np.array([[0]]).astype(np.uint8),  # flg_static. TODO idk what this is
                            body1,  # Bodyid to exclude
                            addressof(c_arr))
    collision_geom = c_arr[0] if c_arr[0] != -1 else None
    return dist, collision_geom
