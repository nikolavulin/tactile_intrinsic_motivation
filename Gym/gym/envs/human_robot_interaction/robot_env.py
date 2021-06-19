import os
import copy
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
import csv
import pdb

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))

FINGERTIP_SITE_NAMES = [
    'robot1:S_palm',
    'robot1:S_fftip',
    'robot1:S_mftip',
    'robot1:S_rftip',
    'robot1:S_lftip',
    'robot1:S_thtip',
]

DATA_JOINT_NAMES = [
    'robot0:WRJ1',
    'robot0:WRJ0',
    'robot0:FFJ3',
    'robot0:FFJ2',
    'robot0:FFJ1',
    'robot0:MFJ3',
    'robot0:MFJ2',
    'robot0:MFJ1',
    'robot0:LFJ4',
    'robot0:LFJ3',
    'robot0:LFJ2',
    'robot0:LFJ1',
    'robot0:RFJ3',
    'robot0:RFJ2',
    'robot0:RFJ1',
]


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, training_range, lift_angle_range,
                 flex_angle_range,
                 time_offset_range, imi_flag, demonstr_idxs, interaction_t_correction):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self.step_counter = 0
        self.epoch = 0
        self.reset_counter = 0
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        qposs_list = []
        for name in DATA_JOINT_NAMES:
            qposs_list.append(self.sim.data.get_joint_qpos(name))

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.initial_goal_pos = np.array(
            [self.sim.data.get_joint_qpos('robot1:HMJX'), self.sim.data.get_joint_qpos('robot1:HMJY'),
             self.sim.data.get_joint_qpos('robot1:HMJZ')])

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.goal = self._update_goal()
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done = False
        self.step_counter += 1
        info = {'is_success': self._is_success(obs['achieved_goal'], self.goal), 'done': obs['done']}
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        self.step_counter = 0
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def render(self, mode='human'):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self._viewer_setup()
        return self.viewer

    # Extension methods
    # ----------------------------
    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _update_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
