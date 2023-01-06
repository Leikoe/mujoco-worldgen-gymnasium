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


class Env(gymnasium.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self,
                 get_sim,
                 get_obs=flatten_get_obs,
                 get_reward=zero_get_reward,
                 get_info=empty_get_info,
                 get_diverged=false_get_diverged,
                 set_action=ctrl_set_action,
                 action_space=None,
                 horizon=100,
                 start_seed=None,
                 render_mode: Optional[str] = None,
                 width: int = DEFAULT_SIZE,
                 height: int = DEFAULT_SIZE,
                 camera_id: Optional[int] = None,
                 camera_name: Optional[str] = None,
                 deterministic_mode=False):
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
        if (horizon is not None) and not isinstance(horizon, int):
            raise TypeError('horizon must be an int')

        self.get_sim = enforce_is_callable(get_sim, (
            'get_sim should be callable and should return an MjSim object'))
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

        # self.sim = None
        self.model: mujoco.MjModel = None
        self.data: mujoco.MjData = None

        self.horizon = horizon
        self.t = None
        self.deterministic_mode = deterministic_mode

        self.width = width
        self.height = height
        self._initialize_simulation()  # may use width and height

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

        # For rendering
        self.mujoco_renderer = None

        # These are required by Gym
        self.observation_space = None
        self._set_action_space()

        self._spec = Spec(max_episode_steps=horizon, timestep_limit=horizon)
        self._name = None

    # This is to mitigate issues with old/new envs
    @property
    def unwrapped(self):
        return self

    @property
    def name(self):
        if self._name is None:
            name = str(inspect.getfile(self.get_sim))
            if name.endswith(".py"):
                name = name[:-3]
            self._name = name
        return self._name

    def _initialize_simulation(self):
        """
        Initialize MuJoCo simulation data structures mjModel and mjData.
        """
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    # TODO: remove this properly
    # def set_state(self, state, call_forward=True):
    #     """
    #     Sets the state of the enviroment to the given value. It does not
    #     set time.
    #
    #     Warning: This only sets the MuJoCo state by setting qpos/qvel
    #         (and the user-defined state "udd_state"). It doesn't set
    #         the state of objects which don't have joints.
    #
    #     Args:
    #     - state (MjSimState): desired state.
    #     - call_forward (bool): if True, forward simulation after setting
    #         state.
    #     """
    #     if not isinstance(state, MjSimState):
    #         raise TypeError("state must be an MjSimState")
    #     if self.sim is None:
    #         raise EmptyEnvException(
    #             "You must call reset() or reset_to_state() before setting the "
    #             "state the first time")
    #
    #     # Call forward to write out values in the MuJoCo data.
    #     # Note: if udd_callback is set on the MjSim instance, then the
    #     # user will need to call forward() manually before calling step.
    #     self.sim.set_state(state)
    #     if call_forward:
    #         self.sim.forward()

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """
                Set the joints position qpos and velocity qvel of the model. Override this method depending on the MuJoCo bindings used.
                """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> (np.ndarray, np.ndarray):
        """
        Returns a copy of the current environment state.

        Returns:
            # TODO: FIX THIS
        - qpos (ndarray): state of the environment's MjSim object.
        - qvel (np.ndarray):
        """
        if self.sim is None:
            raise EmptyEnvException(
                "You must call reset() or reset_to_state() before accessing "
                "the state the first time")
        return self.self.data.qpos[:], self.data.qvel[:]

    def get_xml(self):
        '''
        :return: full state of the simulator serialized as XML (won't contain
                 meshes, textures, and data information).
        '''
        raise NotImplementedError
        # return self.model.get_xml()

    def get_mjb(self):
        '''
        :return: full state of the simulator serialized as mjb.
        '''
        raise NotImplementedError
        # return self.model.get_mjb()

    # def reset_to_state(self, state, call_forward=True):
    #     """
    #     Reset to given state.
    #
    #     Args:
    #     - state (MjSimState): desired state.
    #     """
    #     if not isinstance(state, MjSimState):
    #         raise TypeError(
    #             "You must reset to an explicit state (MjSimState).")
    #
    #     if self.sim is None:
    #         if self._current_seed is None:
    #             self._update_seed()
    #
    #         self.sim = self.get_sim(self._current_seed)
    #     else:
    #         # Ensure environment state not captured in MuJoCo's qpos/qvel
    #         # is reset to the state defined by the model.
    #         self.sim.reset()
    #
    #     self.set_state(state, call_forward=call_forward)
    #
    #     self.t = 0
    #     return self._reset_sim_and_spaces()

    # def _update_seed(self, force_seed=None):
    #     if force_seed is not None:
    #         self._next_seed = force_seed
    #     self._current_seed = self._next_seed
    #     assert self._current_seed is not None
    #     # if in deterministic mode, then simply increment seed, otherwise randomize
    #     if self.deterministic_mode:
    #         self._next_seed = self._next_seed + 1
    #     else:
    #         self._next_seed = np.random.randint(2 ** 32)
    #     # immediately update the seed in the random state object
    #     self._random_state.seed(self._current_seed)

    # TODO: remove this
    # @property
    # def current_seed(self):
    #     # Note: this is a property rather than just instance variable
    #     # for legacy and backwards compatibility reasons.
    #     return self._current_seed

    def _reset_sim_and_spaces(self):
        # TODO: port this
        obs = self.get_obs(self.sim)

        # Mocaps are defined by 3-dim position and 4-dim quaternion
        if isinstance(self._action_space, tuple):
            assert len(self._action_space) == 2
            self._action_space = spaces.Box(
                self._action_space[0], self._action_space[1],
                (self.sim.model.nmocap * 7 + self.sim.model.nu,), np.float32)
        elif self._action_space is None:
            self._action_space = spaces.Box(
                -np.inf, np.inf, (self.sim.model.nmocap * 7 + self.sim.model.nu,), np.float32)
        self._action_space.flatten_dim = np.prod(self._action_space.shape)

        self._observation_space = gym_space_from_arrays(obs)
        if self.viewer is not None:
            self.viewer.update_sim(self.sim)

        return obs

    #
    # Custom pickling
    #

    def __getstate__(self):
        excluded_attrs = frozenset(
            ("model", "data", "mujoco_renderer", "_monitor"))
        attr_values = {k: v for k, v in self.__dict__.items()
                       if k not in excluded_attrs}
        if self.sim is not None:
            attr_values['sim_state'] = self.get_state()
        return attr_values

    def __setstate__(self, attr_values):
        for k, v in attr_values.items():
            if k != 'sim_state':
                self.__dict__[k] = v

        self.sim = None
        self.viewer = None
        if 'sim_state' in attr_values:
            if self.sim is None:
                assert self._current_seed is not None
                self.sim = self.get_sim(self._current_seed)
            self.set_state(attr_values['sim_state'])
            self._reset_sim_and_spaces()

        return self

    def logs(self):
        logs = []
        if hasattr(self.env, 'logs'):
            logs += self.env.logs()
        return logs

    #
    # GYM REQUIREMENTS: these are methods required to be compatible with Gym
    #

    @property
    def action_space(self):
        if self._action_space is None:
            raise EmptyEnvException(
                "You have to reset environment before accessing action_space.")
        return self._action_space

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise EmptyEnvException(
                "You have to reset environment before accessing "
                "observation_space.")
        return self._observation_space

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)

        self._reset_simulation()

        ob = self.reset_model()
        if self.render_mode == "human":
            self.render()
        return ob, {}

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def seed(self, seed=None):
        """
        Use `env.seed(some_seed)` to set the seed that'll be used in
        `env.reset()`. More specifically, this is the seed that will
        be passed into `env.get_sim` during `env.reset()`. The seed
        will then be incremented in consequent calls to `env.reset()`.
        For example:

            env.seed(0)
            env.reset() -> gives seed(0) world
            env.reset() -> gives seed(1) world
            ...
            env.seed(0)
            env.reset() -> gives seed(0) world
        """
        if isinstance(seed, list):
            # Support list of seeds as required by Gym.
            assert len(seed) == 1, "Only a single seed supported."
            self._next_seed = seed[0]
        elif isinstance(seed, int):
            self._next_seed = seed
        elif seed is not None:
            # If seed is None, we just return current seed.
            raise ValueError("Seed must be an integer.")

        # Return list of seeds to conform to Gym specs
        return [self._next_seed]

    def step(self, action):
        action = np.asarray(action)
        action = np.minimum(action, self.action_space.high)
        action = np.maximum(action, self.action_space.low)
        assert self.action_space.contains(action), (
                'Action should be in action_space:\nSPACE=%s\nACTION=%s' %
                (self.action_space, action))
        self.set_action(self.sim, action)
        self.sim.step()
        # Need to call forward() so that sites etc are updated,
        # since they're used in the reward computations.
        self.sim.forward()
        self.t += 1

        reward = self.get_reward(self.sim)
        if not isinstance(reward, float):
            raise TypeError("The return value of get_reward must be a float")

        obs = self.get_obs(self.sim)
        diverged, divergence_reward = self.get_diverged(self.sim)

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

        info = self.get_info(self.sim)
        info["diverged"] = divergence_reward
        # Return value as required by Gym
        return obs, reward, done, info

    def observe(self):
        """ Gets a new observation from the environment. """
        self.sim.forward()
        return self.get_obs(self.sim)

    def render(self, mode='human', close=False):
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        if close:
            # TODO: actually close the inspection viewer
            return
        if mode == 'human' or mode == 'rgb_array':
            # Use a nicely-interactive version of the mujoco viewer
            if self.mujoco_renderer is None:
                # Inline import since this is only relevant on platforms
                # which have GLFW support.
                from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
                self.mujoco_renderer = MujocoRenderer(
                    self.model, self.data, None
                )
            self.mujoco_renderer.render(mode, self.camera_id, self.camera_name)
        else:
            raise ValueError("Unsupported mode %s" % mode)

    def close(self):
        """Close all processes like rendering contexts"""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def get_body_com(self, body_name: str):
        """Return the cartesian position of a body frame"""
        return self.data.body(body_name).xpos


class EmptyEnvException(Exception):
    pass


class MujocoEnv(BaseMujocoEnv):
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
            frame_skip,
            model_path="",
            get_obs=flatten_get_obs,
            get_reward=zero_get_reward,
            get_info=empty_get_info,
            get_diverged=false_get_diverged,
            set_action=ctrl_set_action,
            render_mode: Optional[str] = None,
            horizon=100,
            width: int = DEFAULT_SIZE,
            height: int = DEFAULT_SIZE,
            camera_id: Optional[int] = None,
            camera_name: Optional[str] = None,
            default_camera_config: Optional[dict] = None,
    ):
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

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, default_camera_config
        )
        self.observation_space = gym_space_from_arrays(self.flatten_get_obs())

    def flatten_get_obs(self):
        if self.data.qpos is None:
            return np.zeros(0)
        return np.concatenate([self.data.qpos, self.data.qvel])

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

        observation = self.flatten_get_obs()

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
