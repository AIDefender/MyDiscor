import gym
from gym.envs.registration import register
from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS
from numpy.random import randn

gym.logger.set_level(40)

def assert_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert hasattr(env, '_max_episode_steps')

def make_env(env_id, seed, reward_noise_scale=None):
    try:
        env = gym.make(env_id)
    except gym.error.UnregisteredEnv:
        register(
            id=env_id,
            entry_point=ALL_V1_ENVIRONMENTS[env_id],
            max_episode_steps=150)
        print("Registered env", env_id)
        env = gym.make(env_id)
        assert_env(env)
    env.seed(seed)
    setattr(env, 'is_metaworld', env_id in ALL_V1_ENVIRONMENTS.keys())
    if reward_noise_scale:
        from gym.wrappers import TransformReward
        max_step_bk = env._max_episode_steps
        env = TransformReward(env, lambda r: r + reward_noise_scale * randn())
        setattr(env, "_max_episode_steps", max_step_bk)
    return env

def make_dmc_env(domain_name, task_name, seed=0):
    import dmc2gym
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed)
    setattr(env, 'is_metaworld', False)
    return env
