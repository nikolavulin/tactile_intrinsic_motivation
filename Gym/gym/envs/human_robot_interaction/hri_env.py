import numpy as np
import pdb
import csv
import pyquaternion
import yaml
import os

from gym.envs.human_robot_interaction import rotations, robot_env, utils


def reward_function_eval(achieved_goal, goal, idx_goals, interaction, robot0_joint_names, achieved_force):
    """
    Evaluate different goals

    Args:
        achieved_goal: the current state of the robot
        goal: the goal state of the robot
    Ret:
        position error, force error, angle error, imitation angle error, imitation position error
    """

    assert achieved_goal.shape == goal.shape

    # Final state position error
    dist_reshaped = np.reshape(achieved_goal[idx_goals[1]:idx_goals[2]] - goal[idx_goals[1]:idx_goals[2]], (-1, 5, 3))
    # Final state angle error
    fc_angle_error = np.sum(np.abs(achieved_goal[idx_goals[2]:idx_goals[3]] - goal[idx_goals[2]:idx_goals[3]]), axis=-1)
    # Imitation state angle error
    imi_angle_error = np.sum(np.abs(achieved_goal[idx_goals[3]:idx_goals[4]] - goal[idx_goals[3]:idx_goals[4]]),
                             axis=-1)
    # Imitation state position error
    imi_pos_error = np.reshape(achieved_goal[idx_goals[-1]:] - goal[idx_goals[-1]:], (-1, 5, 3))

    if interaction == 'handclap':
        force_err = np.abs((np.sum(goal[:idx_goals[1]][1:], axis=-1)
                            - np.sum(achieved_goal[:idx_goals[1]][1:], axis=-1))) / np.sum(goal[:idx_goals[1]])
        if force_err < 0.1 and not achieved_force:
            achieved_force = True
    else:
        force_err = np.linalg.norm(goal[:idx_goals[1]] - achieved_goal[:idx_goals[1]]) / np.linalg.norm(goal[:idx_goals[1]])

    return np.sum(np.linalg.norm(dist_reshaped, axis=-1), axis=-1), force_err, fc_angle_error / len(robot0_joint_names), \
           imi_angle_error / len(robot0_joint_names), np.sum(np.linalg.norm(imi_pos_error, axis=-1), axis=-1), achieved_force


def reward_function(achieved_goal, goal, w_p, idx_goals, reward_weights, force_goals):
    """
        Calculate reward given the achieved goal and the desired goal

        Args:
            achieved_goal: the current state of the robot
            goal: the goal state of the robot
        Ret:
            cost: the total reward (negative)
        """
    assert achieved_goal.shape == goal.shape
    # If we are evaluating during training, the goals have different shape, hence the try/except
    try:
        # =========================
        # contact error
        # =========================
        contact_goal = (np.array(goal[:, :idx_goals[1]]) > 0.0).astype('Float32')
        contact_achieved_goal = (np.array(achieved_goal[:, :idx_goals[1]]) > 0.0).astype('Float32')
        force_err = np.sum(np.abs(contact_goal - contact_achieved_goal), axis=-1)

        # =========================
        # l2-norm force error
        # =========================
        force_err += len(force_goals) * np.linalg.norm(goal[:, :idx_goals[1]] - achieved_goal[:, :idx_goals[1]], axis=-1) / np.linalg.norm(force_goals)

        # =========================
        # l1-norm force error
        # =========================
        # force_err += len(FORCE_GOALS) * np.sum(np.abs(goal[:, :idx_goals[1]] - achieved_goal[:, :idx_goals[1]]), axis=-1) / np.sum(FORCE_GOALS)

        # =========================
        # force reward for handclap: compare sum of forces
        # =========================
        # force_clap = np.sum(goal[:, :idx_goals[1]], axis=-1) - np.sum(achieved_goal[:, :idx_goals[1]], axis=-1)
        # force_err += len(FORCE_GOALS) * force_clap / np.sum(FORCE_GOALS)

        # Final state position error
        dist_reshaped = np.reshape(achieved_goal[:, idx_goals[1]:idx_goals[2]] - goal[:, idx_goals[1]:idx_goals[2]], (-1, 5, 3))

        # =========================
        # force reward for ET: wait until on right position
        # =========================
        # alpha = get_alpha(pos_err=np.linalg.norm(dist_reshaped[:, 1, :], axis=-1), inner_bound=0.02, outer_bound=0.03)
        # force_err_int = np.linalg.norm(goal[:, :idx_goals[1]] - achieved_goal[:, :idx_goals[1]], axis=-1)
        # force_err_int = len(FORCE_GOALS) * force_err_int / np.linalg.norm(FORCE_GOALS)
        # force_err += len(FORCE_GOALS)*(1-alpha) + alpha*force_err_int

        # =========================
        # =========================
        # Final state angle error
        fc_angle_error = np.sum(np.abs(achieved_goal[:, idx_goals[2]:idx_goals[3]] - goal[:, idx_goals[2]:idx_goals[3]]), axis=-1)

        # Imitation state angle error
        imi_angle_error = np.sum(np.abs(achieved_goal[:, idx_goals[3]:idx_goals[4]] - goal[:, idx_goals[3]:idx_goals[4]]), axis=-1)

        # Imitation state position error
        imi_pos_error = np.reshape(achieved_goal[:, idx_goals[-1]:] - goal[:, idx_goals[-1]:], (-1, 5, 3))

    except (IndexError, ValueError):
        # =========================
        # contact error
        # =========================
        contact_goal = (np.array(goal[:idx_goals[1]]) > 0.0).astype('Float32')
        contact_achieved_goal = (np.array(achieved_goal[:idx_goals[1]]) > 0.0).astype('Float32')
        force_err = np.sum(np.abs(contact_goal - contact_achieved_goal), axis=-1)
        print(achieved_goal[:idx_goals[1]])
        # print('goal: {}'.format(goal[:idx_goals[1]]))

        # =========================
        # l2-norm force error
        # =========================
        force_err += len(force_goals) * np.linalg.norm(goal[:idx_goals[1]] - achieved_goal[:idx_goals[1]], axis=-1) / np.linalg.norm(force_goals)

        # =========================
        # l1-norm force error
        # =========================
        # force_err += len(FORCE_GOALS) * np.sum(np.abs(goal[:idx_goals[1]] - achieved_goal[:idx_goals[1]]), axis=-1) / np.sum(FORCE_GOALS)

        # =========================
        # force reward for handclap: compare sum of forces
        # =========================
        # force_clap = np.sum(goal[:idx_goals[1]], axis=-1) - np.sum(achieved_goal[:idx_goals[1]], axis=-1)
        # force_err += len(FORCE_GOALS) * force_clap / np.sum(FORCE_GOALS)

        # Final state position error
        dist_reshaped = np.reshape(achieved_goal[idx_goals[1]:idx_goals[2]] - goal[idx_goals[1]:idx_goals[2]], (-1, 5, 3))

        # =========================
        # force reward for ET: wait until on right position
        # =========================
        # print(np.linalg.norm(dist_reshaped[:, 1, :], axis=-1))
        # alpha = get_alpha(pos_err=np.linalg.norm(dist_reshaped[:, 1, :], axis=-1), inner_bound=0.02, outer_bound=0.03)
        # force_err_int = np.linalg.norm(goal[:idx_goals[1]] - achieved_goal[:idx_goals[1]], axis=-1)
        # force_err_int = len(FORCE_GOALS) * force_err_int / np.linalg.norm(FORCE_GOALS)
        # force_err += len(FORCE_GOALS)*(1-alpha) + alpha*force_err_int

        # =========================
        # =========================
        # Final state angle error
        fc_angle_error = np.sum(np.abs(achieved_goal[idx_goals[2]:idx_goals[3]] - goal[idx_goals[2]:idx_goals[3]]), axis=-1)

        # Imitation state angle error
        imi_angle_error = np.sum(np.abs(achieved_goal[idx_goals[3]:idx_goals[4]] - goal[idx_goals[3]:idx_goals[4]]), axis=-1)

        # Imitation state position error
        imi_pos_error = np.reshape(achieved_goal[idx_goals[-1]:] - goal[idx_goals[-1]:], (-1, 5, 3))

    # don't forget to change this to have unit force_err
    force_err = force_err / 2

    # print('W_pos: {}'.format(reward_weights[0] * np.sum(np.linalg.norm(dist_reshaped, axis=-1) * w_p.transpose()), axis=-1))
    # print('W_force: {}'.format(reward_weights[2] * force_err))
    # print('W_ang: {}'.format(reward_weights[1] * fc_angle_error))

    final_state_reward = -(reward_weights[0] * np.sum(np.linalg.norm(dist_reshaped, axis=-1) * w_p.transpose(),
                                                      axis=-1)) - reward_weights[2] * force_err - reward_weights[1] * fc_angle_error

    imi_state_reward = -reward_weights[4] * imi_angle_error - reward_weights[3] * np.sum(np.linalg.norm(imi_pos_error, axis=-1)
                                                                                         * w_p.transpose(), axis=-1)

    cost = final_state_reward + imi_state_reward

    return cost


# ========================================================================================#
# Environment
# ========================================================================================#
class HRIEnv(robot_env.RobotEnv):
    """Superclass for all HRI environments.
    """

    def __init__(
            self, model_path, n_substeps, initial_qpos, training_range, lift_angle_range, flex_angle_range,
            time_offset_range, imi_flag, demonstr_idxs, interaction_t_correction, yaml_name):
        """Initializes a new HRI environment.

        Args: model_path (string): path to the environments XML file n_substeps (int): number of substeps the
        simulation runs on every call to step initial_qpos (dict): a dictionary of joint names and values that define
        the initial configuration training_range (ndarray (3,2)): the rows indicating x,y,z and the columns
        indicating min and max offset of target hand lift_angle_range(tuple (1,2): range of initial arm lift angle of
        robot for randomization of starting configuration flex_angle_range(tuple (1,2): range of initial elbow flex
        angle of robot for randomization of starting configuration time_offset_range (tuple (1,2)): range of starting
        times on the trajectory relative to the final goal state of the interaction imi_flag (int): whether training
        should be started from positions on the demonstration trajectory (1) or not (0) demonstr_idxs(list):
        selection of which provided demonstrations should be used for training interaction_t_correction(list): some
        interactions need a manual time offset for the imitation trajectories, can be None
        """
        # ========================================================================================#
        # Loading configurations depending on the different hand interactions
        # ========================================================================================#
        self.os_path_dir = os.path.dirname(__file__)
        self.yaml_name = yaml_name
        with open(self.os_path_dir + '/config/' + self.yaml_name + '.yaml', 'r') as stream:
            self.data_loaded = yaml.safe_load(stream)

        self.force_goals = self.data_loaded['FORCE_GOALS']
        self.fingertip_site_names = self.data_loaded['FINGERTIP_SITE_NAMES']
        self.robot1_body_names = self.data_loaded['ROBOT1_BODY_NAMES']
        self.robot1_joint_names = self.data_loaded['ROBOT1_JOINT_NAMES']
        self.robot0_joint_names = self.data_loaded['ROBOT0_JOINT_NAMES']
        self.robot_pos_sites = self.data_loaded['ROBOT_POS_SITES']
        self.touch_sensors = self.data_loaded['TOUCH_SENSORS']
        self.robot0_control_joints = self.data_loaded['ROBOT0_CONTROL_JOINTS']
        self.interaction = self.data_loaded['INTERACTION_TYPE'][0]
        self.w_pos = np.float(self.data_loaded['W_pos'][0])
        self.w_ang = np.float(self.data_loaded['W_ang'][0])
        self.w_force = np.float(self.data_loaded['W_f'][0])
        self.w_imi_pos = np.float(self.data_loaded['W_ipos'][0])
        self.w_imi_ang = np.float(self.data_loaded['W_iang'][0])
        self.reward_weights = [self.w_pos, self.w_ang, self.w_force, self.w_imi_pos, self.w_imi_ang]

        # ========================================================================================#
        # Initialization of indices for reward calculation
        # ========================================================================================#
        self.idx_goals = [0]
        self.idx_goals.append(self.idx_goals[-1] + len(self.force_goals))
        self.idx_goals.append(self.idx_goals[-1] + len(self.robot_pos_sites) * 3)
        self.idx_goals.append(self.idx_goals[-1] + len(self.robot0_joint_names))
        self.idx_goals.append(self.idx_goals[-1] + len(self.robot0_joint_names))

        self.force_imitation = False
        self.goal_conditioned = False
        self.random_force = False
        self.increase_epoch = False

        self.achieved_force = False

        self.time_offset_range = time_offset_range
        self.demonstr_indices = demonstr_idxs
        self.training_range = np.array(training_range)
        self.lift_angle_range = lift_angle_range
        self.flex_angle_range = flex_angle_range
        self.imi_flag = imi_flag

        self.time_offset = 0
        self.body_pos = []
        self.speed_pertubation = 0
        self.x_offset = 0
        self.y_offset = 0
        self.z_offset = 0
        self.track_point = 0
        self.rnd_interaction = 0

        self.offset_inactive_joints_r0 = 0
        self.offset_inactive_joints_r1 = 0

        # ========================================================================================#
        # Loading datasets from human-human demonstrations
        # ========================================================================================#
        os_path_interaction = os.path.join(self.os_path_dir, 'optitrack_arrays', self.interaction)

        # Array of distances between tracked positions of the two target hands in Optitrack in
        # the shape: (timesteps, tracked points h1, tracked points h2, 3)
        self.distance_array = np.load(os_path_interaction + '_distance_array.npy')
        # Array of distances between the two palms in Optitrack global frame
        self.palm_distance = np.load(os_path_interaction + '_distance_array_palms.npy')
        # Array of timesteps when the interactions happen
        self.interaction_t = np.load(os_path_interaction + '_interaction_t.npy')
        if interaction_t_correction is not None:
            self.interaction_t[:, 0] += interaction_t_correction

        # Array of rotations between local joint coordinate frames of the target hand and Optitrack global frame
        self.human_rot_array = np.load(os_path_interaction + '_rot_human.npy').reshape((-1, 6, 4))
        # Array of rotations between local joint coordinate frames of the robot hand and Optitrack global frame
        self.robot_rot_array = np.load(os_path_interaction + '_rot_robot.npy')

        # Array of joint angles of target hand
        self.human_angles = np.load(os_path_interaction + '_joint_angles_human.npy')
        # Array of joint angles of robot hands
        self.robot_angles = np.load(os_path_interaction + '_joint_angles_robot.npy')

        self.closest_pt_idx = np.argmax(np.linalg.norm(self.distance_array, axis=-1), axis=-1)

        super(HRIEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=24, initial_qpos=initial_qpos,
            training_range=training_range,
            lift_angle_range=lift_angle_range, flex_angle_range=flex_angle_range, time_offset_range=time_offset_range,
            imi_flag=imi_flag, demonstr_idxs=demonstr_idxs, interaction_t_correction=interaction_t_correction)

    # ========================================================================================#
    # GoalEnv Methods
    # ========================================================================================#
    def _get_achieved_goal(self):
        """Calculates the current state relevant for the reward function

        Ret:
            goal (array): array containing the current state of the robot
        """

        # Calculate current contacts
        gripper_force = []
        for name in self.touch_sensors:
            gripper_force.append(self.sim.data.get_sensor(name))

        # Calculate current positions of goal sites
        robot_pos = []
        for site in self.robot_pos_sites:
            robot_pos.append(self.sim.data.get_site_xpos(site))

        # Calculate current joint positions of robot joints
        qpos_list = []
        for name in self.robot0_joint_names:
            qpos_list.append(self.sim.data.get_joint_qpos(name))

        # gripper_array = (np.array(gripper_force) > 0.0).astype('Float32')

        # gripper_force instead of gripper_array: Continuous instead of binary
        goal = np.hstack((gripper_force, np.array(robot_pos).flatten()))
        goal = np.hstack((goal, np.array(qpos_list).flatten()))
        goal = np.hstack((goal, np.array(qpos_list).flatten()))
        goal = np.hstack((goal, np.array(robot_pos).flatten()))
        return np.array(goal).flatten()

    def compute_reward(self, achieved_goal, goal, info):
        """Call of reward function

        Args:
            achieved_goal: the current state of the robot
            goal: the desired state of the robot
        Ret:
            reward: the complete reward
        """
        cost = reward_function(achieved_goal, goal, self.weights, self.idx_goals, self.reward_weights, self.force_goals)
        return cost

    # ========================================================================================#
    # RobotEnv Methods
    # ========================================================================================#

    def _set_action(self, action):

        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.

        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        # Converting actions to control inputs
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range

        # Clipping actions according to allowed ctrl_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _get_obs(self):

        gripper_force = []
        body_pos = []

        done = False

        # Collect force sensor information
        for name in self.touch_sensors:
            gripper_force.append(self.sim.data.get_sensor(name))

        # Collect body positions of target hand
        for name in self.robot1_body_names:
            body_id = self.sim.model.body_name2id(name)
            body_pos.append(self.sim.data.body_xpos[body_id])

        # Collect joint angles and joint angle velocities of
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        ctrl_joints = self.robot0_control_joints
        joint_qpos = np.array([self.sim.data.get_joint_qpos(name) for name in ctrl_joints])
        joint_vels = np.array([self.sim.data.get_joint_qvel(name) for name in ctrl_joints])
        joint_vels = joint_vels * dt

        goal_force = self.goal[:self.idx_goals[1]].copy()

        # Concatenate all relevant observations
        if self.goal_conditioned:
            obs = np.concatenate([joint_qpos, joint_vels, np.array(body_pos).flatten(),
                                  np.array(goal_force), np.array(gripper_force).flatten()])
        else:
            obs = np.concatenate([joint_qpos, joint_vels, np.array(body_pos).flatten(), np.array(gripper_force).flatten()])

        achieved_goal = self._get_achieved_goal().ravel()

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'done': done,
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot1:palm')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.4
        self.viewer.cam.azimuth = -90.
        self.viewer.cam.elevation = -24.

    def _render_callback(self):
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        for i in range(5):
            site_name = 'target_{}'.format(i)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = np.reshape(self.goal.copy()[self.idx_goals[1]:self.idx_goals[2]], (-1, 3))[i] - \
                                               sites_offset[site_id]  # - self.data_array[self.itty, i, idx, :]

    def _reset_sim(self):
        """
        Here we setup a new training episode
        The following steps are taken:
            1. set target hand at random position and angle configuration
            2. select a demonstration for the reward function parameters and the imitation trajectories
            3. if imitation on: put robot on a position of the trajectory
               if imitation off: put robot into random configuration
            4. set angles of robot hand to preposition given by demonstration data
        """
        self.sim.set_state(self.initial_state)
        # Randomize start position of target hand.
        goal_pos = self.initial_goal_pos.copy()
        goal_pos[0] = self.initial_goal_pos[0] + self.np_random.uniform(self.training_range[0, 0],
                                                                        self.training_range[0, 1], size=1)
        # Z - direction
        goal_pos[1] = self.initial_goal_pos[1] + self.np_random.uniform(self.training_range[1, 0],
                                                                        self.training_range[1, 1], size=1)
        # Y - direction
        goal_pos[2] = self.initial_goal_pos[2] + self.np_random.uniform(self.training_range[2, 0],
                                                                        self.training_range[2, 1], size=1)

        self.sim.data.set_joint_qpos('robot1:HMJX', goal_pos[0])
        self.sim.data.set_joint_qpos('robot1:HMJY', goal_pos[1])
        self.sim.data.set_joint_qpos('robot1:HMJZ', goal_pos[2])

        self.rnd_interaction = np.random.choice(self.demonstr_indices)

        ctrlrange = self.sim.model.actuator_ctrlrange
        clipped = np.clip(np.array(self.human_angles[self.interaction_t[self.rnd_interaction, 0]], 'Float64'),
                          ctrlrange[4:, 0], ctrlrange[4:, 1])

        for i, name in enumerate(self.robot1_joint_names):
            self.sim.data.set_joint_qpos(name, clipped[i + self.offset_inactive_joints_r1])

        self.sim.forward()
        self.time_offset = np.random.randint(self.time_offset_range[0], self.time_offset_range[1])

        if self.imi_flag:

            start_indx = self.interaction_t[self.rnd_interaction, 0]
            goal_pos_global = - self.palm_distance[self.interaction_t[self.rnd_interaction, 0] - self.time_offset, 0, 0,
                                :]

            quat_human_palm_optitrack = pyquaternion.Quaternion(self.human_rot_array[start_indx, 0, 3],
                                                                self.human_rot_array[start_indx, 0, 0],
                                                                self.human_rot_array[start_indx, 0, 1],
                                                                self.human_rot_array[start_indx, 0, 2])
            quat_human_palm_optitrack_inv = quat_human_palm_optitrack.inverse
            goal_pos_local_optitrack = quat_human_palm_optitrack_inv.rotate(goal_pos_global)
            goal_pos_local_sim = np.zeros(3)
            goal_pos_local_sim[0] = goal_pos_local_optitrack[2]
            goal_pos_local_sim[1] = goal_pos_local_optitrack[1]
            goal_pos_local_sim[2] = -goal_pos_local_optitrack[0]

            body_id = self.sim.model.body_name2id('robot1:palm')
            quat_human_palm_sim = self.sim.data.body_xquat[body_id]
            quat_human_palm_sim = pyquaternion.Quaternion(quat_human_palm_sim)

            goal_pos_global_sim = quat_human_palm_sim.rotate(goal_pos_local_sim) + self.sim.data.get_body_xpos(
                'robot1:palm')

            self.goal_pos_global = goal_pos_global_sim

            joint_angles = self.ik_jacobian(goal_pos_global_sim)

            body_id = self.sim.model.body_name2id('robot1:palm')
            q_palm_target = self.sim.data.body_xquat[body_id]
            quat_palm_target = pyquaternion.Quaternion(w=q_palm_target[0], x=q_palm_target[1], y=q_palm_target[2],
                                                       z=q_palm_target[3])

            body_id = self.sim.model.body_name2id('robot0:forearm')
            q_elbow = self.sim.data.body_xquat[body_id]
            quat_elbow = pyquaternion.Quaternion(w=q_elbow[0], x=q_elbow[1], y=q_elbow[2], z=q_elbow[3])

            quat_human_rot_optitrack = pyquaternion.Quaternion(self.human_rot_array[start_indx, 0, 3],
                                                               self.human_rot_array[start_indx, 0, 0],
                                                               self.human_rot_array[start_indx, 0, 1],
                                                               self.human_rot_array[start_indx, 0, 2])
            quat_robot_rot_optitrack = pyquaternion.Quaternion(self.robot_rot_array[start_indx, 3],
                                                               self.robot_rot_array[start_indx, 0],
                                                               self.robot_rot_array[start_indx, 1],
                                                               self.robot_rot_array[start_indx, 2])
            quat_rotation_optitrack2sim = pyquaternion.Quaternion(axis=[0.0, 1.0, 0.0], angle=-np.pi / 2)

            quat_human_rot_sim = quat_human_rot_optitrack * quat_rotation_optitrack2sim
            quat_robot_rot_sim = quat_robot_rot_optitrack * quat_rotation_optitrack2sim
            quat_relative = quat_human_rot_sim.inverse * quat_robot_rot_sim

            quat_target = quat_palm_target * quat_relative
            quat_wrist = quat_elbow.inverse * quat_target
            rotm = quat_wrist.rotation_matrix
            pitch = np.arctan2(-rotm[1, 2], rotm[1, 1])
            yaw = np.arctan2(-rotm[2, 0], rotm[0, 0])
            pitch = np.arctan2(rotm[2, 1], rotm[2, 2])
            yaw = np.arcsin(-rotm[2, 0])

            self.sim.data.set_joint_qpos('robot0:WRJ1', yaw)
            self.sim.data.set_joint_qpos('robot0:WRJ0', pitch)
            self.sim.data.set_joint_qpos('robot0:shoulder_pan_joint', joint_angles[0])
            self.sim.data.set_joint_qpos('robot0:shoulder_lift_joint', joint_angles[1])
            self.sim.data.set_joint_qpos('robot0:upperarm_roll_joint', joint_angles[2])
            self.sim.data.set_joint_qpos('robot0:elbow_flex_joint', joint_angles[3])

            # Moving away target hand such that a feasible approaching angle can be achieved
            goal_pos = self.sim.data.get_joint_qpos('robot1:HMJZ') + self.np_random.uniform(.04, 0.07, size=1)
            self.sim.data.set_joint_qpos('robot1:HMJZ', goal_pos)

        else:
            # Setting the initial configuration of robot arm
            lift_angle = self.np_random.uniform(self.lift_angle_range[0], self.lift_angle_range[1], size=1)
            flex_angle = self.np_random.uniform(self.flex_angle_range[0], self.flex_angle_range[1], size=1)

            self.sim.data.set_joint_qpos('robot0:shoulder_lift_joint', lift_angle)
            self.sim.data.set_joint_qpos('robot0:elbow_flex_joint', flex_angle)

        # Setting the initial configuration of robot hand
        clipped_robot = np.clip(np.array(self.robot_angles[self.interaction_t[self.rnd_interaction, 0] - 0], 'Float64'),
                                ctrlrange[4:, 0], ctrlrange[4:, 1])

        for i, name in enumerate(self.robot0_joint_names):
            self.sim.data.set_joint_qpos(name, clipped_robot[i + self.offset_inactive_joints_r0])

        self.sim.forward()
        self.sim.step()

        return True

    def _update_goal(self):
        """
        This method is required for 2 things:
            1. as the target hand might move with the interaction, the target points should also move and thus we need
             to update them
            2. if we have imitation reward active, we need to update the goal depending on the current timestep
        """

        ftip_pos = [self.sim.data.get_site_xpos(name) for name in self.fingertip_site_names]
        ctrlrange = self.sim.model.actuator_ctrlrange

        if self.interaction_t[
            self.rnd_interaction, 0] - self.time_offset + 4 * self.step_counter < self.sample_position:
            clipped_angle_imi = np.clip(
                np.array(self.robot_angles[
                         self.interaction_t[self.rnd_interaction, 0] - self.time_offset + self.step_counter * 4, :],
                         'Float64').flatten(), ctrlrange[-20:, 0], ctrlrange[-20:, 1])
        else:
            clipped_angle_imi = np.clip(np.array(self.robot_angles[self.sample_position, :], 'Float64').flatten(),
                                        ctrlrange[-20:, 0], ctrlrange[-20:, 1])

        self.goal[self.idx_goals[3]:self.idx_goals[4]] = clipped_angle_imi[self.offset_inactive_joints_r0:]

        goal_pos, goal_imi_positions = [], []
        int_idx = self.interaction_t[self.rnd_interaction, 0]

        quat_rot = pyquaternion.Quaternion(self.human_rot_array[int_idx, self.track_point, 3],
                                           self.human_rot_array[int_idx, self.track_point, 0],
                                           self.human_rot_array[int_idx, self.track_point, 1]
                                           , self.human_rot_array[int_idx, self.track_point, 2])
        quat_rot = quat_rot.inverse
        body_id = self.sim.model.body_name2id('robot1:palm')
        body_quat = self.sim.data.body_xquat[body_id]

        quat_rot_palm = pyquaternion.Quaternion(body_quat)

        for i in range(len(self.robot_pos_sites)):
            rel_pos_temp = quat_rot.rotate(self.distance_array[self.sample_position, i, self.track_point])

            reordered_h = np.zeros(len(rel_pos_temp))
            reordered_h[0] = rel_pos_temp[2]
            reordered_h[1] = rel_pos_temp[1]
            reordered_h[2] = -rel_pos_temp[0]

            rel_rotated_temp = quat_rot_palm.rotate(reordered_h)
            goal_pos.append(ftip_pos[self.track_point] - rel_rotated_temp)

            if self.interaction_t[
                self.rnd_interaction, 0] - self.time_offset + 6 * self.step_counter < self.sample_position:
                rel_imi_pos_temp = quat_rot.rotate(self.distance_array[self.interaction_t[self.rnd_interaction, 0] -
                                                                       self.time_offset + 6 * self.step_counter, i,
                                                   self.track_point, :])
            else:
                rel_imi_pos_temp = quat_rot.rotate(self.distance_array[self.sample_position, i,
                                                   self.track_point, :])

            reordered_h = np.zeros(len(rel_imi_pos_temp))
            reordered_h[0] = rel_imi_pos_temp[2]
            reordered_h[1] = rel_imi_pos_temp[1]
            reordered_h[2] = -rel_imi_pos_temp[0]

            rel_rotated_imi_temp = quat_rot_palm.rotate(reordered_h)

            goal_imi_positions.append(ftip_pos[self.track_point] - rel_rotated_imi_temp)

        self.goal[self.idx_goals[1]:self.idx_goals[2]] = np.array(goal_pos).flatten()
        self.goal[self.idx_goals[-1]:] = np.array(goal_imi_positions).flatten()

        if self.achieved_force and self.force_imitation:
            self.goal[:self.idx_goals[1]] = np.where(np.array(self.goal[:self.idx_goals[1]]) == 0, 0, 1)

        self.sim.forward()

        return self.goal.copy()

    def _sample_goal(self):
        """
        Create a goal state
        """

        # Collect fingertip positions of target hand
        ftip_pos = [self.sim.data.get_site_xpos(name) for name in self.fingertip_site_names]
        goal_positions, goal_imi_positions = [], []

        # Randomly select a sample point within the contact phase of the interaction
        if self.interaction_t[self.rnd_interaction, 1] == self.interaction_t[self.rnd_interaction, 2]:
            self.sample_position = self.interaction_t[self.rnd_interaction, 1]
        else:
            self.sample_position = np.random.randint(self.interaction_t[self.rnd_interaction, 1],
                                                     self.interaction_t[self.rnd_interaction, 2])

        # Choose the reference point on the target hand (leave out thumb for now due to inaccuracies)
        self.track_point = np.argmin(np.linalg.norm(self.distance_array[self.sample_position], axis=-1)) % len(
            self.fingertip_site_names)
        if self.track_point == 1:
            self.track_point = 0

        dmin = np.min(np.linalg.norm(self.distance_array[self.sample_position, :, self.track_point], axis=-1))
        self.weights = dmin / np.linalg.norm(self.distance_array[self.sample_position, :, self.track_point], axis=-1)

        int_idx = self.interaction_t[self.rnd_interaction, 0]
        quat_rot = pyquaternion.Quaternion(self.human_rot_array[int_idx, self.track_point, 3],
                                           self.human_rot_array[int_idx, self.track_point, 0],
                                           self.human_rot_array[int_idx, self.track_point, 1]
                                           , self.human_rot_array[int_idx, self.track_point, 2])

        quat_rot = quat_rot.inverse

        body_id = self.sim.model.body_name2id('robot1:palm')
        body_quat = self.sim.data.body_xquat[body_id]
        quat_rot_palm = pyquaternion.Quaternion(body_quat)

        for i in range(len(self.robot_pos_sites)):
            rel_pos_temp = quat_rot.rotate(self.distance_array[self.sample_position, i, self.track_point, :])

            reordered_h = np.zeros(len(rel_pos_temp))
            reordered_h[0] = rel_pos_temp[2]
            reordered_h[1] = rel_pos_temp[1]
            reordered_h[2] = -rel_pos_temp[0]

            rel_rotated_temp = quat_rot_palm.rotate(reordered_h)

            goal_positions.append(ftip_pos[self.track_point] - rel_rotated_temp)  # +pos_offset

            rel_imi_pos_temp = quat_rot.rotate(
                self.distance_array[self.interaction_t[self.rnd_interaction, 0] - self.time_offset, i, self.track_point,
                :])

            reordered_h = np.zeros(len(rel_imi_pos_temp))
            reordered_h[0] = rel_imi_pos_temp[2]
            reordered_h[1] = rel_imi_pos_temp[1]
            reordered_h[2] = -rel_imi_pos_temp[0]

            rel_rotated_imi_temp = quat_rot_palm.rotate(reordered_h)

            goal_imi_positions.append(ftip_pos[self.track_point] - rel_rotated_imi_temp)

        goal_force = np.array(self.force_goals)
        self.achieved_force = False
        if self.random_force:
            goal_force = goal_force + np.random.randint(-70, 71)

        if self.increase_epoch:
            goal_force = goal_force + 10 * int(self.epoch / 10)

        # if self.epoch == 5:
        #     print('epoch: {}'.format(self.epoch))
        #     self.reward_weights[0] = 0.0
        #     self.reward_weights[1] = 0.0
        #     print(self.reward_weights)
        #
        # if self.epoch == 6:
        #     print('epoch: {}'.format(self.epoch))
        #     print(self.reward_weights)

        ctrlrange = self.sim.model.actuator_ctrlrange
        clipped_imi_angles = np.clip(
            np.array(self.robot_angles[self.interaction_t[self.rnd_interaction, 0] - self.time_offset], 'Float64'),
            ctrlrange[-20:, 0], ctrlrange[-20:, 1])

        clipped_fc_angles = np.clip(np.array(self.robot_angles[self.sample_position], 'Float64'),
                                    ctrlrange[-20:, 0], ctrlrange[-20:, 1])

        goal = np.hstack((goal_force, np.array(goal_positions).flatten()))
        goal = np.hstack((goal, np.array(clipped_fc_angles[self.offset_inactive_joints_r0:]).flatten()))
        goal = np.hstack((goal, np.array(clipped_imi_angles[self.offset_inactive_joints_r0:]).flatten()))
        goal = np.hstack((goal, np.array(goal_imi_positions).flatten()))
        return goal.copy().flatten()

    def _is_success(self, achieved_goal, desired_goal):

        pos_error, force_error, fc_angle_error, imi_angle_error, imi_pos_error, self.achieved_force = reward_function_eval(
            achieved_goal, desired_goal, self.idx_goals, self.interaction, self.robot0_joint_names, self.achieved_force)
        return -pos_error, -force_error, -fc_angle_error, -imi_angle_error, -imi_pos_error

    def _env_setup(self, initial_qpos):
        self.initial_goal = self._get_achieved_goal().copy()

        self.sim.data.set_joint_qpos('robot0:shoulder_lift_joint', -0.0)
        self.sim.data.set_joint_qpos('robot0:elbow_flex_joint', -0.6)
        self.sim.forward()

        # Setting handshake of human arm
        q_pan = self.sim.data.get_joint_qpos('robot0:shoulder_pan_joint')
        q_lift = self.sim.data.get_joint_qpos('robot0:shoulder_lift_joint')
        q_roll = self.sim.data.get_joint_qpos('robot0:upperarm_roll_joint')
        q_flex = self.sim.data.get_joint_qpos('robot0:elbow_flex_joint')
        q_vec = np.array([q_pan, q_lift, q_roll, q_flex])

        self.joint_angles_start = q_vec

        # -4 because we don't care about the arm joints
        self.offset_inactive_joints_r0 = self.sim.data.ctrl.shape[0] - len(self.robot0_joint_names) - 4
        self.offset_inactive_joints_r1 = self.sim.data.ctrl.shape[0] - len(self.robot1_joint_names) - 4

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:S_palm').copy()

    # ========================================================================================#
    # Inverse Kinematics Calculation of arm starting position (by Stefan)
    # ========================================================================================#
    def ik_arm(self):

        l_1 = 0.06
        l_2 = 0.117
        l_3 = 0.352
        l_4 = 0.356

        p_x = l_2 + l_3
        p_y = 0.0
        p_z = l_1 - 0.3

        T = np.array([[1, 0, 0, p_x],
                      [0, 1, 0, p_y],
                      [0, 0, 1, p_z],
                      [0, 0, 0, 1]])

        # define the input vars
        r_11 = T[0, 0]
        r_12 = T[0, 1]
        r_13 = T[0, 2]
        r_21 = T[1, 0]
        r_22 = T[1, 1]
        r_23 = T[1, 2]
        r_31 = T[2, 0]
        r_32 = T[2, 1]
        r_33 = T[2, 2]
        Px = T[0, 3]
        Py = T[1, 3]
        Pz = T[2, 3]

        #
        # Caution:    Generated code is not yet validated
        #

        solvable_pose = True

        # Variable:  th_1
        th_1s1 = np.arctan2(Py - l_4 * r_21, Px - l_4 * r_11)
        th_1s2 = np.arctan2(-Py + l_4 * r_21, -Px + l_4 * r_11)

        # Variable:  th_2
        th_2s2 = np.arctan2(-(Pz - l_1 - l_4 * r_31) / l_3,
                            (Px - l_2 * np.cos(th_1s2) - l_4 * r_11) / (l_3 * np.cos(th_1s2)))
        th_2s1 = np.arctan2(-(Pz - l_1 - l_4 * r_31) / l_3,
                            (Px - l_2 * np.cos(th_1s1) - l_4 * r_11) / (l_3 * np.cos(th_1s1)))

        th_3s2 = np.arctan2(r_32 / np.cos(th_2s1), -r_12 * np.sin(th_1s1) + r_22 * np.cos(th_1s1))
        th_3s1 = np.arctan2(r_32 / np.cos(th_2s2), -r_12 * np.sin(th_1s2) + r_22 * np.cos(th_1s2))

        # Variable:  th_4
        th_4s1 = np.arctan2((-r_11 * np.sin(th_1s1) + r_21 * np.cos(th_1s1)) / np.sin(th_3s2),
                            -(-r_13 * np.sin(th_1s1) + r_23 * np.cos(th_1s1)) / np.sin(th_3s2))
        th_4s2 = np.arctan2((-r_11 * np.sin(th_1s2) + r_21 * np.cos(th_1s2)) / np.sin(th_3s1),
                            -(-r_13 * np.sin(th_1s2) + r_23 * np.cos(th_1s2)) / np.sin(th_3s1))

        ##################################
        #
        # package the solutions into a list for each set
        #
        ###################################

        solution_list = []
        # (note trailing commas allowed in python
        solution_list.append([th_1s2, th_2s2, th_3s1, th_4s2, ])
        # (note trailing commas allowed in python
        solution_list.append([th_1s1, th_2s1, th_3s2, th_4s1, ])

        return solution_list

    def ik_jacobian(self, goal_pos):

        ik_loop = True
        counter = 0
        while ik_loop:
            J = self.sim.data.get_body_jacp('robot0:palm')
            jac = np.array([J[:4], J[55: 55 + 4], J[2 * 55: 2 * 55 + 4]])
            palm_pos = self.sim.data.get_body_xpos('robot0:palm')
            # print(palm_pos)
            d_e = goal_pos - palm_pos
            q_pan = self.sim.data.get_joint_qpos('robot0:shoulder_pan_joint')
            q_lift = self.sim.data.get_joint_qpos('robot0:shoulder_lift_joint')
            q_roll = self.sim.data.get_joint_qpos('robot0:upperarm_roll_joint')
            q_flex = self.sim.data.get_joint_qpos('robot0:elbow_flex_joint')
            q_vec = np.array([q_pan, q_lift, q_roll, q_flex])

            # jac.transpose()
            d_q = np.matmul(np.linalg.pinv(jac), d_e)

            gama = 0.5
            q_vec += gama * d_q

            self.sim.data.set_joint_qpos('robot0:shoulder_pan_joint', q_vec[0])
            self.sim.data.set_joint_qpos('robot0:shoulder_lift_joint', q_vec[1])
            self.sim.data.set_joint_qpos('robot0:upperarm_roll_joint', q_vec[2])
            self.sim.data.set_joint_qpos('robot0:elbow_flex_joint', q_vec[3])

            self.sim.step()
            counter += 1
            if counter > 20:
                ik_loop = False
                q_vec = self.joint_angles_start.copy()

            if (np.linalg.norm(d_e) < 0.3):
                ik_loop = False

        return q_vec
