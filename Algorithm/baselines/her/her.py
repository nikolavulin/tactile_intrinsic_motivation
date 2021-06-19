import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Up sample transitions with achieved minimal force. 5x higher probability
        force_obs = episode_batch['o'][:, :, -1]  # force_obs = array(buffer_size x T x 1)
        # force_obs_probability: 5 if force_obs > 1, else 1, then normalize
        force_obs_probability = 9 * (force_obs > 1).astype(np.float32) + 1

        # Compute probabilities for episode ...
        force_obs_probability_episode = np.sum(force_obs_probability, axis=1)
        force_obs_probability_episode = force_obs_probability_episode / np.sum(force_obs_probability_episode)
        episode_idxs = np.random.choice(rollout_batch_size, size=batch_size, p=force_obs_probability_episode)

        # pick steps conditioned on episode
        force_obs_probability_step = force_obs_probability[episode_idxs, :]
        force_obs_probability_step = np.delete(force_obs_probability_step, -1, axis=1)
        row_sums = np.sum(force_obs_probability_step, axis=1)
        force_obs_probability_step = force_obs_probability_step / row_sums[:, np.newaxis]
        t_samples = np.array([np.random.choice(T, size=1, p=force_obs_probability_step_row) for force_obs_probability_step_row
                              in list(force_obs_probability_step)])
        t_samples = np.array(t_samples).squeeze()

        # Select which episodes and time steps to use.
        # episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # t_samples = np.random.randint(T, size=batch_size)

        # Select past (!) time indexes
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        past_offset = np.random.uniform(size=batch_size) * t_samples
        past_offset = past_offset.astype(int)
        future_t = (t_samples + 1)[her_indexes]

        transitions = {key: episode_batch[key][episode_idxs, past_offset].copy()
                       for key in episode_batch.keys()}

        # # Select future time indexes proportional with probability future_p. These
        # # will be used for HER replay by substituting in future goals.
        # her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        # future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        # future_offset = future_offset.astype(int)
        # future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
