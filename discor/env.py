import gym
from gym.envs.registration import register
from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS
import metaworld
import random

gym.logger.set_level(40)


def assert_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert hasattr(env, '_max_episode_steps')


METAWORLD_TASKS = (
    'hammer-v2', 'stick-push-v2', 'push-wall-v2',
    'stick-pull-v2', 'dial-turn-v2', 'peg-insert-side-v2')

def make_env(env_id):
    if env_id in METAWORLD_TASKS:
        ml1 = metaworld.MT1(env_id)
        env = ml1.train_classes[env_id]()
        # task = random.choice(ml1.train_tasks)
        task = ml1.train_tasks[0]
        env.set_task(task)
        setattr(env, '_max_episode_steps', 150)
    else:
        env = gym.make(env_id)
    setattr(env, 'is_metaworld', env_id in METAWORLD_TASKS)
    assert_env(env)
    return env

if __name__ == '__main__':
    env = make_env("stick-pull-v2")
    print(env.reset())