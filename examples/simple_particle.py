import numpy as np
from mujoco_worldgen import Env, WorldParams, WorldBuilder, Floor, ObjFromXML


def get_reward(model, data):
    object_xpos = data.site("object").xpos
    target_xpos = data.site("target").xpos
    ctrl = np.sum(np.square(data.ctrl))
    return -np.sum(np.square(object_xpos - target_xpos)) - 1e-3 * ctrl


def get_sim(seed):
    world_params = WorldParams(size=(4., 4., 2.5))
    builder = WorldBuilder(world_params, seed)
    floor = Floor()
    builder.append(floor)
    obj = ObjFromXML("particle")
    floor.append(obj)
    obj.mark("object")
    floor.mark("target", (.5, .5, 0.05))
    return builder.get_sim()


def make_env():
    return Env(get_sim, get_reward=get_reward, render_mode="human", horizon=30)
