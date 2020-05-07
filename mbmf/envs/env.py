import gym

HALF_CHEETAH_RUN = "HalfCheetahRun"
HALF_CHEETAH_FLIP = "HalfCheetahFlip"
DM_REACHER = "DeepMindReacher"
DM_CATCH = "DeepMindCatch"


class Env(object):
    def __init__(self, env_name, max_episode_len=10 ** 4, action_repeat=1, seed=None):
        self._env = self._get_env_object(env_name)
        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        if seed is not None:
            self._env.seed(seed)
        self._t = 0

    def reset(self):
        self._t = 0
        state = self._env.reset()
        return state

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            state, reward_k, done, info = self._env.step(action)
            reward += reward_k
            self._t += 1
            done = done or self._t == self.max_episode_len
            if done:
                break
        return state, reward, done, info

    def sample_action(self):
        return self._env.action_space.sample()

    def render(self, mode="human"):
        self._env.render(mode)

    def close(self):
        self._env.close()

    def batch_reward_function(self, states, actions=None):
        return self._env.batch_reward_function(states, actions)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def unwrapped(self):
        return self._env

    def _get_env_object(self, env_name):
        if env_name == HALF_CHEETAH_RUN:
            from mbmf.envs.envs.half_cheetah import HalfCheetahRunEnv

            return HalfCheetahRunEnv()

        elif env_name == HALF_CHEETAH_FLIP:
            from mbmf.envs.envs.half_cheetah import HalfCheetahFlipEnv

            return HalfCheetahFlipEnv()

        elif env_name == DM_CATCH:
            from mbmf.envs.dm_wrapper import DeepMindWrapper

            return DeepMindWrapper(domain="ball_in_cup", task="catch")

        elif env_name == DM_REACHER:
            from mbmf.envs.dm_wrapper import DeepMindWrapper

            return DeepMindWrapper(domain="reacher", task="easy")
        else:
            return gym.make(env_name)
