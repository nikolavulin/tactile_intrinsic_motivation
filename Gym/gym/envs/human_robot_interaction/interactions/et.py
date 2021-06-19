from gym import utils
from gym.envs.human_robot_interaction import hri_env


class HriEtEnv(hri_env.HRIEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', **kwargs):
        initial_qpos = {
        }
        training_range = [[0.0, 0.05], [-0.05, 0.05], [0.0, 0.10]]
        hri_env.HRIEnv.__init__(
            self, 'human_robot_interaction/et.xml', n_substeps=20, initial_qpos=initial_qpos,
            training_range=training_range,
            lift_angle_range=[-0.5, 0.5], flex_angle_range=[-0.15, 0], time_offset_range=[60, 61], imi_flag=1,
            demonstr_idxs=[0, 1, 3, 4], interaction_t_correction=[0, 70, 0, 30, 70], yaml_name=kwargs['yaml_name'])
        utils.EzPickle.__init__(self)
