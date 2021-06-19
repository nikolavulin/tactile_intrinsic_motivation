import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'stack.xml')


class FetchStackEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'robot0:shoulder_pan_joint': 0.0,
            'robot0:shoulder_lift_joint': -0.7,
            'robot0:upperarm_roll_joint': 0.0,
            'robot0:elbow_flex_joint': 1.55,
            'robot0:forearm_roll_joint': 0.0,
            'robot0:wrist_flex_joint': 0.75,
            'robot0:wrist_roll_joint': 0.0,
            'robot0:l_gripper_finger_joint': 0.05,
            'robot0:r_gripper_finger_joint': 0.05,
            'object0:joint': [1.25, 0.53, 0.42, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=np.array([0.5, 0.0, 0.1]),
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            force_range=1, force_reward_weight=0.25, task='stack')
        utils.EzPickle.__init__(self)
