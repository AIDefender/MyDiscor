import os
import yaml
import argparse
from datetime import datetime
import torch

from discor.algorithm import SAC, DisCor, DQN
from discor.agent import Agent
import gym
from discor.envs.continuous_grid import ContinuousGridEnv

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.num_steps is not None:
        config['Agent']['num_steps'] = args.num_steps

    # Create environments.
    if "SG" in args.env_id:
        from discor.env_SG import make_env_SG
        env = make_env_SG(args.env_id.split("_")[0], args.seed)
        test_env = make_env_SG(args.env_id.split("_")[0], args.seed)
    elif args.dmc:
        from discor.env import make_dmc_env
        env = make_dmc_env(args.domain_name, args.task_name, args.seed)
        test_env = make_dmc_env(args.domain_name, args.task_name, args.seed)
    else:
        from discor.env import make_env
        env = make_env(args.env_id, args.seed, args.reward_noise_scale)
        test_env = make_env(args.env_id, args.seed)

    # Device to use.
    device = torch.device(
        "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
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

    reweigh_hyper = {
        "linear": args.linear_hp,
        "adaptive_linear": args.adaptive_scheme,
        "exp": args.exp_hp,
    }
    if args.algo == 'discor':
        # Discor algorithm.
        algo = DisCor(
            state_dim=state_dim,
            action_dim=env.action_space.shape[0],
            device=device, seed=args.seed, 
            tau_scale = args.tau_scale, 
            discor=args.discor,
            lfiw=args.lfiw,
            tper=args.tper,
            hard_tper_weight=args.hard_weight,
            use_backward_timestep=args.bk_step,
            log_dir=log_dir,
            env=test_env,
            eval_tper = args.eval_tper,
            reweigh_type = args.reweigh_type,
            reweigh_hyper = reweigh_hyper,
            **config['SAC'], **config['DisCor'])
    elif args.algo == 'sac':
        # SAC algorithm.
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env = test_env, # for computing Q_pi
            eval_tper = args.eval_tper,
            log_dir = log_dir,
            device=device, seed=args.seed, **config['SAC'])
    # elif args.algo == 'dqn':
    #     algo = DQN(
    #         state_dim=env.observation_space.shape[0]
    #     )
    else:
        raise Exception('You need to set "--algo sac" or "--algo discor".')

    agent = Agent(
        env=env, test_env=test_env, algo=algo, log_dir=log_dir, use_tper=args.tper,
        device=device, seed=args.seed, use_backward_steps=args.bk_step, save_model_interval=args.save_interval,
        eval_tper=args.eval_tper,
         **config['Agent'])
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
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--discor', action='store_true')
    parser.add_argument('--lfiw', action='store_true')
    parser.add_argument('--tper', action='store_true') # Temporal PER. Reweight according to length to done in the trajectory.
    parser.add_argument('--eval_tper', action='store_true')
    parser.add_argument('--bk_step', action='store_false')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=0) # 0 means only saving last and best model
    parser.add_argument('--tau_scale', type=float, default=1.0)
    parser.add_argument('--reweigh_type', 
                        choices=['linear', 'adaptive_linear', 'done_cnt_linear', 
                                 'exp', 'adaptive_exp',
                                 'hard'], 
                        default='hard')
    # hyperparameters for low, high, k, b
    parser.add_argument("--linear_hp", type=float, nargs='*', default=[0.6, 1.5, 3., -0.3])
    # hyperparameters for low for lower weight, high for lower weight, 
    # low for higher_weight, high for higher weight, timestep_start, timestep_end
    parser.add_argument('--adaptive_scheme', type=float, nargs="*", default=[0.4, 0.8, 1.2, 1.6, 0, 1e6])
    # hyperparameters for exponential reweight: k, gamma
    parser.add_argument('--exp-hp', type=float, nargs='*', default=[1., 0.996])
    parser.add_argument('--hard_weight', type=float, default=0.4)
    parser.add_argument('--dmc', action='store_true')
    parser.add_argument('--domain_name', type=str)
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--reward_noise_scale', type=float, default=0)
    args = parser.parse_args()
    run(args)
