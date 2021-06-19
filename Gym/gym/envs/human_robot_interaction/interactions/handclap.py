from gym import utils
from gym.envs.human_robot_interaction import hri_env


class HriHandclapEnv(hri_env.HRIEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', **kwargs):
        initial_qpos = {
        }
        training_range = [[-0.05, 0.05], [-0.05, 0.05], [0.0, 0.15]]
        hri_env.HRIEnv.__init__(
            self, 'human_robot_interaction/handclap.xml', n_substeps=20, initial_qpos=initial_qpos,
            training_range=training_range,
            lift_angle_range=[0, 0.5], flex_angle_range=[-0.55, -0.5], time_offset_range=[150, 180], imi_flag=0,
            demonstr_idxs=[1, 2, 3, 5, 7, 8], interaction_t_correction=[0, 0, 0, 20, 40, 40, 0, 0, 0], yaml_name=kwargs['yaml_name'])
        utils.EzPickle.__init__(self)
