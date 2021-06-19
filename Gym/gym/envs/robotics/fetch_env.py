import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    try:
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    except(IndexError, ValueError):
        return np.linalg.norm(goal_a - goal_b)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
            self, model_path, n_substeps, gripper_extra_height, block_gripper,
            has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type, force_range, force_reward_weight, task,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        # new init
        self.force_range = force_range
        self.force_reward_weight = force_reward_weight
        self.ymg = 0.91
        self.baseline = False
        if self.block_gripper:
            self.n_action = 7
            self.joint_names = ["robot0:shoulder_pan_joint",
                                "robot0:shoulder_lift_joint",
                                "robot0:upperarm_roll_joint",
                                "robot0:elbow_flex_joint",
                                "robot0:forearm_roll_joint",
                                "robot0:wrist_flex_joint",
                                "robot0:wrist_roll_joint"]
        else:
            self.n_action = 8
            self.joint_names = ["robot0:shoulder_pan_joint",
                                "robot0:shoulder_lift_joint",
                                "robot0:upperarm_roll_joint",
                                "robot0:elbow_flex_joint",
                                "robot0:forearm_roll_joint",
                                "robot0:wrist_flex_joint",
                                "robot0:wrist_roll_joint",
                                "robot0:l_gripper_finger_joint",
                                "robot0:r_gripper_finger_joint"]



        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=self.n_action,
            initial_qpos=initial_qpos, task=task)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # weights
        force_weight = self.force_reward_weight
        position_weight = 1 - force_weight

        # Compute rewards
        intrinsic = info['intrinsic_sum_force']

        force_reward = -(np.squeeze(intrinsic) < self.force_range).astype(np.float32)
        d_pos = goal_distance(achieved_goal, goal)
        position_reward = -(d_pos > self.distance_threshold).astype(np.float32)
        if self.baseline:
            return position_reward
        else:
            return force_weight * force_reward + position_weight * position_reward

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (self.n_action,)
        # ensure that we don't change the action outside of this scope
        if self.block_gripper:
            action = action.copy()
        else:
            action = np.hstack([action.copy(), action[-1].copy()])

        # control range of actuator
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.

        # Converting actions to control inputs
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range

        # Clipping actions according to allowed ctrl_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        # forces
        force = self.force
        sum_force = self.sum_force

        # achieved goal
        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        self.achieved_goal = achieved_goal.copy()

        if self.baseline:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel])
        else:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, force.ravel(), sum_force.ravel()])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        # pick
        # self.viewer.cam.distance = 1.5
        # self.viewer.cam.azimuth = 135.
        # self.viewer.cam.elevation = -20.

        # push
        # self.viewer.cam.distance = 1.8
        # self.viewer.cam.azimuth = 115.
        # self.viewer.cam.elevation = -22.

        # slide
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 115.
        self.viewer.cam.elevation = -25.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        if self.success_log:
            self.sim.model.site_rgba[site_id] = np.array([0, 1, 0, 1])
        else:
            self.sim.model.site_rgba[site_id] = np.array([1, 0, 0, 1])
        self.sim.forward()

    def randomize_initial_state(self):
        if self.block_gripper:
            lower_ranges = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
            upper_ranges = [0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.1]
        else:
            lower_ranges = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0, 0.0]
            upper_ranges = [0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0]

        for i, actuator in enumerate(self.joint_names):
            actuator_state = self.sim.data.get_joint_qpos(actuator)
            self.sim.data.set_joint_qpos(actuator, actuator_state + self.np_random.uniform(lower_ranges[i], upper_ranges[i], size=1))

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.randomize_initial_state()

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                if self.task == 'slide':
                    object_xpos[0] += self.obj_range
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal[2] = self.height_offset
            goal += self.target_offset
            if self.target_in_the_air:
                goal[2] += self.np_random.uniform(0, 0.35)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
