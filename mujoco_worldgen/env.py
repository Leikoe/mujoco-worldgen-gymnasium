import inspect
import logging
import hashlib
from typing import Optional

import gymnasium

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

import numpy as np
from gymnasium import error, logger, spaces
from gymnasium import Space
from gymnasium.envs.mujoco.mujoco_env import BaseMujocoEnv

from mujoco_worldgen.util.types import enforce_is_callable
from mujoco_worldgen.util.sim_funcs import (
    empty_get_info,
    flatten_get_obs,
    false_get_diverged,
    ctrl_set_action,
    zero_get_reward,
)

# logger = logging.getLogger(__name__)

DEFAULT_SIZE = 500


class EmptyEnvException(Exception):
    pass


class Env(BaseMujocoEnv):
    """Superclass for MuJoCo environments."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
            self,
            get_sim,
            frame_skip=0,
            model_path="",
            get_obs=flatten_get_obs,
            get_reward=zero_get_reward,
            get_info=empty_get_info,
            get_diverged=false_get_diverged,
            set_action=ctrl_set_action,
            action_space=None,

            render_mode: Optional[str] = None,
            horizon=100,
            width: int = DEFAULT_SIZE,
            height: int = DEFAULT_SIZE,
            camera_id: Optional[int] = None,
            camera_name: Optional[str] = None,
            default_camera_config: Optional[dict] = None,

            start_seed=None,
            deterministic_mode=False
    ):
        # TODO: fix this doc
        """
        Env is a Gym environment subclass tuned for robotics learning
        research.
        Args:
        - get_sim (callable): a callable that returns an MjSim.
        - get_obs (callable): callable with an MjSim object as the sole
            argument and should return observations.
        - set_action (callable): callable which takes an MjSim object and
            updates its data and buffer directly.
        - get_reward (callable): callable which takes an MjSim object and
            returns a scalar reward.
        - get_info (callable): callable which takes an MjSim object and
            returns info (dictionary).
        - get_diverged (callable): callable which takes an MjSim object
            and returns a (bool, float) tuple. First value is True if
            simulator diverged and second value is the reward at divergence.
        - action_space: a space of allowed actions or a two-tuple of a ranges
            if number of actions is unknown until the simulation is instantiated
        - horizon (int): horizon of environment (i.e. max number of steps).
        - start_seed (int or string): seed for random state generator (None for random seed).
            Strings will be hashed.  A non-None value implies deterministic_mode=True.
            This argument allows us to run a deterministic series of goals/randomizations
            for a given policy.  Then applying the same seed to another policy will allow the
            comparison of results more accurately.  The reason a string is allowed is so
            that we can more easily find and share seeds that are farther from 0,
            which is the default starting point for deterministic_mode, and thus have
            more likelihood of getting a performant sequence of goals.
        """

        if MUJOCO_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(
                f"{MUJOCO_IMPORT_ERROR}. (HINT: you need to install mujoco)"
            )

        self.get_sim = enforce_is_callable(get_sim, (
            'get_sim should be callable and should return an (MjModel, MjData) object'))
        self.get_obs = enforce_is_callable(get_obs, (
            'get_obs should be callable with an MjSim object as the sole '
            'argument and should return observations'))
        self.set_action = enforce_is_callable(set_action, (
            'set_action should be a callable which takes an MjSim object and '
            'updates its data and buffer directly'))
        self.get_reward = enforce_is_callable(get_reward, (
            'get_reward should be a callable which takes an MjSim object and '
            'returns a scalar reward'))
        self.get_info = enforce_is_callable(get_info, (
            'get_info should be a callable which takes an MjSim object and '
            'returns a dictionary'))
        self.get_diverged = enforce_is_callable(get_diverged, (
            'get_diverged should be a callable which takes an MjSim object '
            'and returns a (bool, float) tuple. First value is whether '
            'simulator is diverged (or done) and second value is the reward at '
            'that time.'))

        super().__init__(
            model_path,
            frame_skip,
            None,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
        )

        self.horizon = horizon
        self.t = None

        self.deterministic_mode = deterministic_mode

        # Numpy Random State
        if isinstance(start_seed, str):
            start_seed = int(hashlib.sha1(start_seed.encode()).hexdigest(), 16) % (2 ** 32)
            self.deterministic_mode = True
        elif isinstance(start_seed, int):
            self.deterministic_mode = True
        else:
            start_seed = 0 if self.deterministic_mode else np.random.randint(2 ** 32)
        self._random_state = np.random.RandomState(start_seed)
        # Seed that will be used on next _reset()
        self._next_seed = start_seed
        # Seed that was used in last _reset()
        self._current_seed = None

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        # For rendering
        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, default_camera_config
        )

        obs = self.get_obs(self.model, self.data)
        self.observation_space = gym_space_from_arrays(obs)
        # if provided with action_space, use it
        if action_space is not None:
            self.action_space = action_space

    def step(self, action):
        action = np.asarray(action)
        assert self.action_space.contains(action), (
                'Action should be in action_space:\nSPACE=%s\nACTION=%s' %
                (self.action_space, action))

        self.do_simulation(action, self.frame_skip)
        self.t += 1

        reward = self.get_reward(self.model, self.data)
        if not isinstance(reward, float):
            raise TypeError("The return value of get_reward must be a float")

        obs = self.get_obs(self.model, self.data)
        diverged, divergence_reward = self.get_diverged(self.model, self.data)

        if not isinstance(diverged, bool):
            raise TypeError(
                "The first return value of get_diverged must be boolean")
        if not isinstance(divergence_reward, float):
            raise TypeError(
                "The second return value of get_diverged must be float")

        if diverged:
            done = True
            if divergence_reward is not None:
                reward = divergence_reward
        elif self.horizon is not None:
            done = (self.t >= self.horizon)
        else:
            done = False

        info = self.get_info(self.model, self.data)
        info["diverged"] = divergence_reward

        if self.render_mode == "human":
            self.render()

        # Return value as required by Gym
        return obs, reward, done, False, info

    def _initialize_simulation(self):
        self.model, self.data = self.get_sim(0)  # TODO: proper random seeding

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def reset_model(self):
        qpos = self.init_qpos
        qvel = (
            self.init_qvel
        )
        self.set_state(qpos, qvel)
        self.t = 0

        observation = self.get_obs(self.model, self.data)
        self.observation_space = gym_space_from_arrays(observation)
        return observation

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def render(self):
        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos


# Helpers
###############################################################################


class Spec(object):
    # required by gym.wrappers.Monitor

    def __init__(self, max_episode_steps=np.inf, timestep_limit=np.inf):
        self.id = "worldgen.env"
        self.max_episode_steps = max_episode_steps
        self.timestep_limit = timestep_limit


def gym_space_from_arrays(arrays):
    if isinstance(arrays, np.ndarray):
        ret = spaces.Box(-np.inf, np.inf, arrays.shape, np.float32)
        ret.flatten_dim = np.prod(ret.shape)
    elif isinstance(arrays, (tuple, list)):
        ret = spaces.Tuple([gym_space_from_arrays(arr) for arr in arrays])
    elif isinstance(arrays, dict):
        ret = spaces.Dict(dict([(k, gym_space_from_arrays(v)) for k, v in arrays.items()]))
    else:
        raise TypeError("Array is of unsupported type.")
    return ret
