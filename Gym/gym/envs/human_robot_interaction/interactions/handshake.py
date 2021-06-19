from gym import utils
from gym.envs.human_robot_interaction import hri_env


class HriHandshakeEnv(hri_env.HRIEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', **kwargs):
        initial_qpos = {}
        # Offsets in x, y, z directions
        training_range = [[0.0, 0.05], [-0.05, 0.05], [0.0, 0.10]]
        hri_env.HRIEnv.__init__(
            self, 'human_robot_interaction/handshake.xml', n_substeps=20, initial_qpos=initial_qpos,
            training_range=training_range,
            lift_angle_range=[-0.5, 0.5], flex_angle_range=[-0.15, 0], time_offset_range=[30, 50], imi_flag=0,
            demonstr_idxs=[0, 1, 2, 7, 8, 9], interaction_t_correction=None, yaml_name=kwargs['yaml_name'])
        utils.EzPickle.__init__(self)
