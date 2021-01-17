import os
import sys

sh = True
if sh:
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(os.path.split(rootPath)[0])
    proj_path = os.path.abspath(os.path.join(os.getcwd()))
else:
    proj_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

import time
import argparse
import datetime
import torch
import numpy as np
import random
from agents import *
from torch.utils.tensorboard import SummaryWriter
from envs.naive_v6_1_1.sampler import Sampler, make_env
from common.utils import write_config
from common.tensorboard import TensorBoard, show_agent
from common.networks import *
from common.buffers import *

import envs.naive_v6_1_1
import envs.multi_lane_v1
from envs.naive_v6_1_1.config import Config

config = Config()

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in sumo')
parser.add_argument('--env', type=str, default='sumo_env-v6',
                    help=' environment')
if sh:
    parser.add_argument('--env_path', type=str, default=sys.path[0] + '/exp.sumocfg',
                        help=' environment scenario path')
else:
    parser.add_argument('--env_path', type=str, default='exp.sumocfg',
                        help=' environment scenario path')
parser.add_argument('--reward_model', type=int, default=2,
                    help='reward_model')
parser.add_argument('--human_model', default=False, action="store_true",
                    help='human_model')
parser.add_argument('--gui', default=False, action="store_true",
                    help='gui')
parser.add_argument('--seed', type=int, default=0,
                    help='seed for random number generators')

parser.add_argument('--algo', type=str, default='ppo2',
                    help=' algorithm cppo ppo2 ddpg td3')
parser.add_argument('--load_path', type=str, default=None,
                    help='rl model path')
parser.add_argument('--episodes', type=int, default=100,
                    help='evel episodes')
parser.add_argument('--gpu_index', type=int, default=0)

# parser.add_argument('--model_path', type=str, default='tests/save_model/sumo_sac_s_0_ep_3650_tr_-6.29_er_-9.2.pt')

args = parser.parse_args()
config.gpu_index = args.gpu_index
config.env['id'] = args.env
config.env['path'] = args.env_path
config.env['reward_model'] = args.reward_model
config.env['human_model'] = args.human_model
config.env['gui'] = args.gui
config.seed = args.seed
config.agent_gpu_index = args.gpu_index
config.select_agent(args.algo)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
random.seed(config.seed)
torch.cuda.manual_seed(config.seed)

if config.gpu_index == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda', index=config.agent_gpu_index) if torch.cuda.is_available() else torch.device('cpu')

curr_time = datetime.datetime.now().strftime("%H-%M-%S")
curr_day = datetime.datetime.now().strftime("%Y-%m-%d")
# init save model
save_path = config.create_save_path()
write_config(save_path, config)

print(vars(config))
# init tensorboard
writer = TensorBoard.get_writer(load_path=save_path)

env = make_env(config.env, config.seed, 'test', )

policys = {}
if args.algo == 'cppo':
    policys[args.algo] = CPPOModel(env.observation_space.shape[0], env.action_space.shape[0])
elif args.algo == 'ppo2':
    policys[args.algo] = PPOModel(env.observation_space.shape[0], env.action_space.shape[0])
elif args.algo == 'ddpg' or args.algo == 'td3':
    policys[args.algo] = DDPGActor(env.observation_space.shape[0], env.action_space.shape[0],
                                   hidden_sizes=(128, 128),
                                   output_activation=torch.tanh)
elif args.algo == 'human':
    pass
else:
    policys[args.algo] = SACActor(env.observation_space.shape[0], env.action_space.shape[0],
                                  hidden_sizes=(64, 64), )

if not args.algo == 'human':
    policys[args.algo].load_state_dict(torch.load(args.load_path, map_location=device))


def evaluation(algo_name):
    total_result = {
        'time': 0,
        'speed': 0,
        'collision': 0,
        'success': 0,
        'fuel': 0,
    }
    total_time = 160
    for i in range(total_time):
        results = {}
        results['time'] = 0
        results['speed'] = 0
        results['collision'] = 0
        results['success'] = 0
        results['fuel'] = 0
        print('-----------------', i, '---------------------')
        for epi in range(args.episodes):
            env.set_startstep(i)
            obs = env.reset()
            done = False
            speed = 0
            step = 0
            if args.algo == 'human':
                while not done:
                    step += 1
                    obs_next, reward, done, info = env.step()
                    obs = obs_next
                    speed += info['speed']
                    results['collision'] += info['crash']
                    results['success'] += info['success']
                    results['fuel'] += info['fuel']
            else:
                while not done:
                    step += 1
                    obs = torch.Tensor(obs).unsqueeze(0)
                    action = policys[algo_name](obs)
                    action = action.detach().cpu().numpy()[0]
                    obs_next, reward, done, info = env.step(action)
                    obs = obs_next
                    speed += info['speed']
                    results['collision'] += info['crash']
                    results['success'] += info['success']
                    results['fuel'] += info['fuel']

                if config.env['gui']:
                    time.sleep(0.005)
            results['time'] += 0.2 * step
            results['speed'] += speed / step

        writer.add_scalar('Evel/DoneRate', results['success'] / args.episodes, i)
        writer.add_scalar('Evel/CollisionRate', results['collision'] / args.episodes, i)
        writer.add_scalar('Evel/AverageTime', results['time'] / args.episodes, i)
        writer.add_scalar('Evel/AverageSpeed', results['speed'] / args.episodes, i)
        writer.add_scalar('Evel/AverageFuel', results['fuel'] / args.episodes, i)

        print('success rate:', results['success'] / args.episodes)
        print('collision rate:', results['collision'] / args.episodes)
        print('average time:', results['time'] / args.episodes)
        print('average speed:', results['speed'] / args.episodes)
        print('fuel:', results['fuel'] / args.episodes)
        total_result['success'] += results['success'] / args.episodes
        total_result['collision'] += results['collision'] / args.episodes
        total_result['time'] += results['time'] / args.episodes
        total_result['speed'] += results['speed'] / args.episodes
        total_result['fuel'] += results['fuel'] / args.episodes

    total_result['success'] = total_result['success'] / total_time
    total_result['collision'] = total_result['collision'] / total_time
    total_result['time'] = total_result['time'] / total_time
    total_result['speed'] = total_result['speed'] / total_time
    total_result['fuel'] = total_result['fuel'] / total_time
    print(total_result)


if __name__ == "__main__":
    print('Evaluation:', args.algo)
    print(vars(args))
    evaluation(args.algo)
