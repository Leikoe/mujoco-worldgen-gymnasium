from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium
from gymnasium.spaces import Box, Dict

from mujoco_worldgen import Floor, WorldBuilder, Geom, ObjFromXML, WorldParams, MujocoEnv


def update_obs_space(env, delta):
    print(env.observation_space)
    spaces = env.observation_space.spaces.copy()
    for key, shape in delta.items():
        spaces[key] = Box(-np.inf, np.inf, shape, np.float32)
    print(Dict(spaces))
    return Dict(spaces)


def rand_pos_on_floor(model, data, n=1):
    world_size = model.geom_size[model.geom('floor0').id] * 2
    new_pos = np.random.uniform(np.array([[0.2, 0.2] for _ in range(n)]),
                                np.array([world_size[:2] - 0.2 for _ in range(n)]))
    return new_pos


class GatherEnv(MujocoEnv):
    def __init__(self, n_food=3, horizon=200, n_substeps=10,
                 floorsize=4., deterministic_mode=False):
        self.n_food = n_food
        self.horizon = horizon
        self.n_substeps = n_substeps
        self.floorsize = floorsize

        super().__init__(
            frame_skip=4,
            get_sim=self._get_sim,
            get_obs=self._get_obs,
            render_mode="human",
            # action_space=(-1.0, 1.0),
            horizon=horizon,
            # deterministic_mode=deterministic_mode
        )

    def _get_obs(self, model, data):
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qpos_qvel = np.concatenate([qpos, qvel], -1)
        return {'qpos': qpos, 'qvel': qvel, 'qpos_qvel': qpos_qvel}

    def _get_sim(self, seed):
        self.model, self.data = self._get_new_sim(seed)

        self.data.qpos[0:2] = rand_pos_on_floor(self.model, self.data)
        return self.model, self.data

    def _get_new_sim(self, seed):
        world_params = WorldParams(size=(self.floorsize, self.floorsize, 2.5),
                                   num_substeps=self.n_substeps)
        builder = WorldBuilder(world_params, seed)
        floor = Floor()
        builder.append(floor)
        # Walls
        wallsize = 0.1
        wall = Geom('box', (wallsize, self.floorsize, 0.5), name="wall1")
        wall.mark_static()
        floor.append(wall, placement_xy=(0, 0))
        wall = Geom('box', (wallsize, self.floorsize, 0.5), name="wall2")
        wall.mark_static()
        floor.append(wall, placement_xy=(1, 0))
        wall = Geom('box', (self.floorsize - wallsize * 2, wallsize, 0.5), name="wall3")
        wall.mark_static()
        floor.append(wall, placement_xy=(1 / 2, 0))
        wall = Geom('box', (self.floorsize - wallsize * 2, wallsize, 0.5), name="wall4")
        wall.mark_static()
        floor.append(wall, placement_xy=(1 / 2, 1))
        # Add agents
        obj = ObjFromXML("particle", name="agent0")
        floor.append(obj)
        obj.mark(f"object0")
        # Add food sites
        for i in range(self.n_food):
            floor.mark(f"food{i}", (.5, .5, 0.05), rgba=(0., 1., 0., 1.))
        model, data = builder.get_sim()

        # Cache constants for quicker lookup later
        self.food_ids = np.array([model.site(f'food{i}').id for i in range(self.n_food)])
        return model, data


class FoodHealthWrapper(gymnasium.ObservationWrapper):
    '''
        Adds food health to underlying env.
        Manages food levels.
        Randomizes food positions.
    '''

    def __init__(self, env, max_food_health=10):
        super().__init__(env)
        self.unwrapped.max_food_health = self.max_food_health = max_food_health
        self.unwrapped.max_food_size = self.max_food_size = 0.1
        self.observation_space = update_obs_space(env,
                                                  {'food_obs': (self.unwrapped.n_food, 4),
                                                   'food_pos': (self.unwrapped.n_food, 3),
                                                   'food_health': (self.unwrapped.n_food, 1)})
        self.food_healths = None

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset food healths
        self.food_healths = np.array([self.max_food_health
                                      for _ in range(self.unwrapped.n_food)])
        # Randomize food positions
        new_pos = rand_pos_on_floor(self.unwrapped.model, self.unwrapped.data, self.unwrapped.n_food)
        sites_offset = (self.unwrapped.data.site_xpos -
                        self.unwrapped.model.site_pos).copy()
        self.unwrapped.model.site_pos[self.unwrapped.food_ids, :2] = \
            new_pos - sites_offset[self.unwrapped.food_ids, :2]

        # Reset food size
        self.unwrapped.model.site_size[self.unwrapped.food_ids] = self.max_food_size

        return self.observation(obs), info

    def observation(self, obs):
        # Add food position and healths to obersvations
        food_pos = self.unwrapped.data.site_xpos[self.unwrapped.food_ids]
        food_health = self.food_healths
        obs['food_pos'] = food_pos
        obs['food_health'] = np.expand_dims(food_health, 1)
        obs['food_obs'] = np.concatenate([food_pos, np.expand_dims(food_health, 1)], 1)
        return obs

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        assert np.all(self.food_healths >= 0), \
            f"There is a food health below 0: {self.food_healths}"
        obs = self.observation(obs)
        return obs, rew, done, truncated, info


class ProcessEatFood(gymnasium.Wrapper):
    """
        Manage food health. Resize food based on health.
        Expects a binary vector as input detailing which
    """

    def __init__(self, env, eat_thresh=0.7):
        super().__init__(env)
        self.n_food = self.unwrapped.n_food
        self.eat_thresh = eat_thresh

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        return self.env.reset(seed=seed, options=options)

    def observation(self, obs):
        return obs

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        obs = self.observation(obs)

        # Eat food that is close enough
        agent_food_diff = obs['food_pos'] - np.expand_dims(obs['qpos'], axis=0)
        dist_to_food = np.linalg.norm(agent_food_diff, axis=-1)
        eat = np.logical_and(dist_to_food < self.eat_thresh, self.food_healths > 0)

        # Update food healths and sizes
        self.food_healths = self.food_healths - eat
        health_diff = np.expand_dims(eat, 1)
        size_diff = health_diff * (self.unwrapped.max_food_size / self.unwrapped.max_food_health)
        size = self.unwrapped.model.site_size[self.unwrapped.food_ids] - size_diff
        size = np.maximum(0, size)
        self.unwrapped.model.site_size[self.unwrapped.food_ids] = size

        rew += np.sum(eat)
        return obs, rew, done, truncated, info


def make_env(n_food=3, horizon=50, floorsize=4.):
    env = GatherEnv(horizon=horizon, floorsize=floorsize, n_food=n_food)
    print(env.reset())
    env = FoodHealthWrapper(env)
    env = ProcessEatFood(env)
    env.reset()
    return env
