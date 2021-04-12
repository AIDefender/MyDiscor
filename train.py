import os
import yaml
import argparse
from datetime import datetime
import torch

from discor.env import make_env
from discor.algorithm import SAC, DisCor
from discor.agent import Agent
import gym


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.num_steps is not None:
        config['Agent']['num_steps'] = args.num_steps

    # Create environments.
    env = make_env(args.env_id)
    test_env = make_env(args.env_id)

    # Device to use.
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{args.exp_name}-seed{args.seed}-{time}')

    try:
        state_dim = env.observation_space.shape[0]
    except TypeError:
        # gym-fetch env
        env, test_env = gym.wrappers.FlattenObservation(env), gym.wrappers.FlattenObservation(test_env)
        env.env.reward_type="dense"
        test_env.env.reward_type="dense"
        setattr(env, '_max_episode_steps', 150)
        setattr(test_env, '_max_episode_steps', 150)
        state_dim = env.observation_space.shape[0]
        print("==========state dim: %d========"%state_dim)

    horizon = None
    if args.TP:
        if args.dyna_h:
            raise NotImplementedError
        else:
            horizon = env._max_episode_steps

        print("=========horizon:%d========="%horizon)

    if args.algo == 'discor':
        # Discor algorithm.
        algo = DisCor(
            state_dim=state_dim,
            action_dim=env.action_space.shape[0],
            device=device, seed=args.seed, 
            tau_scale = args.tau_scale, horizon = horizon,
            hard_tper_weight=args.hard_weight,
            **config['SAC'], **config['DisCor'])
    elif args.algo == 'sac':
        # SAC algorithm.
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=args.seed, **config['SAC'])
    else:
        raise Exception('You need to set "--algo sac" or "--algo discor".')

    agent = Agent(
        env=env, test_env=test_env, algo=algo, log_dir=log_dir, horizon=horizon, temperature=args.tper_t,
        device=device, seed=args.seed, **config['Agent'])
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'metaworld.yaml'))
    parser.add_argument('--num_steps', type=int, required=False)
    parser.add_argument('--env_id', type=str, default='hammer-v1')
    parser.add_argument('--exp_name', type=str, default='discor')
    parser.add_argument('--algo', choices=['sac', 'discor'], default='discor')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--TP', action='store_true') # Temporal PER. Reweight according to length to done in the trajectory.
    parser.add_argument('--dyna_h', action='store_true') # whether to determine horizon length dynamically
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tau_scale', type=float, default=1.0)
    parser.add_argument('--hard_weight', type=float, default=0.4)
    parser.add_argument('--tper_t', type=int, default=3e4)
    args = parser.parse_args()
    run(args)
