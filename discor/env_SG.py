import gym
from gym.envs.registration import register
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

gym.logger.set_level(40)


def assert_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert hasattr(env, '_max_episode_steps')


task_names = (
    'hammer-v2', 'stick-push-v2', 'push-wall-v2',
    'stick-pull-v2', 'dial-turn-v2', 'peg-insert-side-v2',
    'door-open-v2', 'drawer-open-v2', 'button-press-v2')

# Single observable Goal
METAWORLD_TASKS = [i+"-goal-observable" for i in task_names]

for task, name in zip(METAWORLD_TASKS, task_names):
    register(
        id=name,
        entry_point=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task],
        max_episode_steps=150)
    assert_env(gym.make(name))


def make_env_SG(env_id, seed=0):
    # Single Goal Env
    env = gym.make(env_id, seed=seed)
    setattr(env, 'is_metaworld', env_id in METAWORLD_TASKS)
    return env
